use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};

use crate::catalog::runs::{self, RunStats};
use crate::catalog::{self, images};
use crate::config::{Config, Paths};
use crate::faces::{cluster, fetch};
use crate::index;
use crate::lock::IndexLock;

/// Read-only image catalog and viewer.
///
/// Without a subcommand, opens the viewer on PATH or the configured roots.
#[derive(Debug, Parser)]
#[command(version, args_conflicts_with_subcommands = true)]
pub struct Cli {
    /// Config file, instead of vitrine/config.toml in the XDG config dir
    #[arg(long, global = true, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Folder to open instead of the configured roots
    pub path: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Index the configured roots, or PATH
    Index { path: Option<PathBuf> },
    /// Manage face recognition
    Faces {
        #[command(subcommand)]
        command: FacesCommand,
    },
    /// Show config, storage locations, catalog counts and the last index run
    Status,
}

#[derive(Debug, Subcommand)]
pub enum FacesCommand {
    /// Download the face models after showing their license
    Fetch,
    /// Rerun clustering over all faces not assigned by hand
    Recluster,
}

pub fn run(cli: Cli) -> Result<()> {
    let mut paths = Paths::from_env()?;
    if let Some(config_file) = cli.config {
        ensure!(
            config_file.is_file(),
            "config file {} not found",
            config_file.display()
        );
        paths.config_file = config_file;
    }
    let config = Config::load(&paths.config_file, &paths.home_dir)?;

    match cli.command {
        None => viewer(&paths, &config, cli.path),
        Some(Command::Index { path }) => index(&paths, &config, path),
        Some(Command::Faces {
            command: FacesCommand::Fetch,
        }) => faces_fetch(&paths, &config),
        Some(Command::Faces {
            command: FacesCommand::Recluster,
        }) => faces_recluster(&paths, &config),
        Some(Command::Status) => status(&paths, &config),
    }
}

fn viewer(paths: &Paths, config: &Config, path: Option<PathBuf>) -> Result<()> {
    let roots = match path {
        Some(path) => vec![
            path.canonicalize()
                .with_context(|| format!("cannot open {}", path.display()))?,
        ],
        // the catalog stores canonical paths; an unreachable root keeps its configured form
        None => config
            .roots
            .iter()
            .map(|root| root.canonicalize().unwrap_or_else(|_| root.clone()))
            .collect(),
    };
    crate::gui::run(paths, &roots, config)
}

fn index(paths: &Paths, config: &Config, path: Option<PathBuf>) -> Result<()> {
    let roots = match path {
        Some(path) => vec![
            path.canonicalize()
                .with_context(|| format!("cannot index {}", path.display()))?,
        ],
        None if config.roots.is_empty() => bail!(
            "no roots configured in {}; add `roots` there or run `vitrine index PATH`",
            paths.config_file.display()
        ),
        None => config.roots.clone(),
    };
    let summary = index::run(
        paths,
        &roots,
        &config.faces,
        &AtomicBool::new(false),
        |_| {},
    )?;
    for failure in &summary.failures {
        eprintln!("{}: {}", failure.path.display(), failure.error);
    }
    print_stats(&summary.stats);
    Ok(())
}

fn print_stats(stats: &RunStats) {
    for (name, value) in [
        ("files", stats.files),
        ("hashed", stats.hashed),
        ("added", stats.added),
        ("changed", stats.changed),
        ("moved", stats.moved),
        ("restored", stats.restored),
        ("vanished", stats.vanished),
        ("failed", stats.failed),
        ("thumbnailed", stats.thumbnailed),
        ("faces scanned", stats.faces_scanned),
        ("faces found", stats.faces_found),
        ("people created", stats.people_created),
        ("faces assigned", stats.faces_assigned),
    ] {
        println!("  {:<16}{value}", format!("{name}:"));
    }
}

fn faces_fetch(paths: &Paths, config: &Config) -> Result<()> {
    println!("{}\n", fetch::LICENSE_NOTE);
    let dir = paths.models_dir();
    println!("models: {}", dir.display());
    for (name, outcome) in fetch::fetch(&dir)? {
        let state = match outcome {
            fetch::Outcome::Downloaded => "downloaded, sha256 verified",
            fetch::Outcome::Present => "present, sha256 verified",
        };
        println!("  {name}: {state}");
    }
    let conn = catalog::open(&paths.catalog_db())?;
    catalog::faces::register_model(
        &conn,
        fetch::MODEL_NAME,
        fetch::DETECTOR_NAME,
        fetch::EMBEDDING_DIM,
    )?;
    if !config.faces.enabled {
        println!(
            "Set `faces.enabled = true` in {} to index faces.",
            paths.config_file.display()
        );
    }
    Ok(())
}

fn faces_recluster(paths: &Paths, config: &Config) -> Result<()> {
    ensure!(
        config.faces.enabled,
        "faces are disabled; set `faces.enabled = true` in {}",
        paths.config_file.display()
    );
    let _lock = IndexLock::acquire(&paths.index_lock())?;
    let mut conn = catalog::open(&paths.catalog_db())?;
    let Some(model) = catalog::faces::model_by_name(&conn, fetch::MODEL_NAME)? else {
        bail!(
            "face model {} is not registered; run `vitrine faces fetch`",
            fetch::MODEL_NAME
        );
    };
    let outcome = cluster::recluster(&mut conn, model, &config.faces)?;
    println!("  {:<16}{}", "people created:", outcome.people_created);
    println!("  {:<16}{}", "faces assigned:", outcome.faces_assigned);
    println!("  {:<16}{}", "people merged:", outcome.people_merged);
    Ok(())
}

fn status(paths: &Paths, config: &Config) -> Result<()> {
    println!("config:  {}", paths.config_file.display());
    println!("catalog: {}", paths.catalog_db().display());
    println!("models:  {}", paths.models_dir().display());
    println!("thumbs:  {}", paths.thumbs_db().display());
    if config.roots.is_empty() {
        println!("roots:   none");
    }
    for root in &config.roots {
        println!("root:    {}", root.display());
    }
    println!(
        "faces:   {}",
        if config.faces.enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    let catalog_db = paths.catalog_db();
    if !catalog_db.exists() {
        println!("indexed: never");
        return Ok(());
    }
    let conn = catalog::open(&catalog_db)?;
    let counts = images::counts(&conn)?;
    println!("images:  {}", counts.images);
    println!("videos:  {}", counts.videos);
    println!("missing: {}", counts.missing);
    println!("unhashed: {}", counts.unhashed);
    match runs::last(&conn)? {
        Some(run) => {
            println!("indexed: {}", run.finished_at);
            print_stats(&run.stats);
        }
        None => println!("indexed: never"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn bare_path_opens_viewer() {
        let cli = Cli::try_parse_from(["vitrine", "/photos"]).unwrap();
        assert!(cli.command.is_none());
        assert_eq!(cli.path, Some(PathBuf::from("/photos")));
    }

    #[test]
    fn subcommands_parse() {
        let cli = Cli::try_parse_from(["vitrine", "index", "/photos"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Index { path: Some(_) })
        ));
        let cli =
            Cli::try_parse_from(["vitrine", "faces", "recluster", "--config", "/c.toml"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Faces {
                command: FacesCommand::Recluster
            })
        ));
        assert_eq!(cli.config, Some(PathBuf::from("/c.toml")));
    }
}
