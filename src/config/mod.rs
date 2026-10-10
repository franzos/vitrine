mod paths;

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::Deserialize;

pub use paths::Paths;

use crate::faces::detect::SCORE_THRESHOLD;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("no home directory found")]
    NoHome,
    #[error("failed to read {}", path.display())]
    Read { path: PathBuf, source: io::Error },
    #[error("failed to parse {}", path.display())]
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
    #[error("root `{}` is not an absolute path", .0.display())]
    RelativeRoot(PathBuf),
    #[error("faces.cluster_threshold must be in (0, 1], got {0}")]
    ClusterThreshold(f32),
    #[error("faces.min_neighbours must be at least 1")]
    MinNeighbours,
    #[error("faces.merge_threshold must be in (0, 1], got {0}")]
    MergeThreshold(f32),
    #[error("faces.min_score must be in [{SCORE_THRESHOLD}, 1], got {0}")]
    MinScore(f32),
    #[error("viewer.external must name a command")]
    EmptyExternal,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub roots: Vec<PathBuf>,
    pub faces: Faces,
    pub viewer: Viewer,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Faces {
    pub enabled: bool,
    pub device: Device,
    /// Largest cosine distance (1 - cosine similarity) at which two faces count as neighbours.
    pub cluster_threshold: f32,
    /// Other faces within `cluster_threshold` a face needs to form a new person.
    pub min_neighbours: usize,
    /// Largest cosine distance between the mean embeddings of two people for a clustering pass to merge them.
    pub merge_threshold: f32,
    /// Minimum detection score for a face to count; the detector itself keeps everything from 0.5 up.
    pub min_score: f32,
    /// Minimum length of the shorter box side, in source pixels, for a face to count.
    pub min_face_size: u32,
}

impl Default for Faces {
    fn default() -> Self {
        Self {
            enabled: false,
            device: Device::Cpu,
            cluster_threshold: 0.62,
            min_neighbours: 2,
            merge_threshold: 0.25,
            min_score: SCORE_THRESHOLD,
            min_face_size: 40,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    #[default]
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Viewer {
    /// Command and leading arguments; image paths are appended.
    pub external: Vec<String>,
}

impl Default for Viewer {
    fn default() -> Self {
        Self {
            external: vec!["oculante".to_owned()],
        }
    }
}

impl Config {
    /// Expands `~` in roots against `home`; a missing file yields the defaults.
    pub fn load(path: &Path, home: &Path) -> Result<Self, ConfigError> {
        let config = match fs::read_to_string(path) {
            Ok(text) => toml::from_str(&text).map_err(|source| ConfigError::Parse {
                path: path.to_owned(),
                source,
            })?,
            Err(e) if e.kind() == io::ErrorKind::NotFound => Self::default(),
            Err(source) => {
                return Err(ConfigError::Read {
                    path: path.to_owned(),
                    source,
                });
            }
        };
        config.resolve(home)
    }

    fn resolve(mut self, home: &Path) -> Result<Self, ConfigError> {
        for root in &mut self.roots {
            *root = expand_tilde(root, home);
            if !root.is_absolute() {
                return Err(ConfigError::RelativeRoot(root.clone()));
            }
        }
        let threshold = self.faces.cluster_threshold;
        if !(threshold > 0.0 && threshold <= 1.0) {
            return Err(ConfigError::ClusterThreshold(threshold));
        }
        if self.faces.min_neighbours == 0 {
            return Err(ConfigError::MinNeighbours);
        }
        let merge = self.faces.merge_threshold;
        if !(merge > 0.0 && merge <= 1.0) {
            return Err(ConfigError::MergeThreshold(merge));
        }
        let min_score = self.faces.min_score;
        if !(SCORE_THRESHOLD..=1.0).contains(&min_score) {
            return Err(ConfigError::MinScore(min_score));
        }
        if self.viewer.external.first().is_none_or(String::is_empty) {
            return Err(ConfigError::EmptyExternal);
        }
        Ok(self)
    }
}

/// Expands a leading `~` component; `~user` is left as is.
#[must_use]
pub fn expand_tilde(path: &Path, home: &Path) -> PathBuf {
    match path.strip_prefix("~") {
        Ok(rest) => home.join(rest),
        Err(_) => path.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOME: &str = "/home/test";

    fn parse(text: &str) -> Result<Config, ConfigError> {
        toml::from_str::<Config>(text)
            .expect("test TOML parses")
            .resolve(Path::new(HOME))
    }

    #[test]
    fn empty_file_gives_defaults() {
        let config = parse("").unwrap();
        assert_eq!(config, Config::default());
        assert!(config.roots.is_empty());
        assert!(!config.faces.enabled);
        assert_eq!(config.faces.device, Device::Cpu);
        assert!((config.faces.cluster_threshold - 0.62).abs() < f32::EPSILON);
        assert_eq!(config.faces.min_neighbours, 2);
        assert!((config.faces.merge_threshold - 0.25).abs() < f32::EPSILON);
        assert!((config.faces.min_score - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.faces.min_face_size, 40);
        assert_eq!(config.viewer.external, ["oculante"]);
    }

    #[test]
    fn missing_file_gives_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::load(&dir.path().join("config.toml"), Path::new(HOME)).unwrap();
        assert_eq!(config, Config::default());
    }

    #[test]
    fn full_file_parses() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            r#"
            roots = ["~/Pictures", "/mnt/photos"]

            [faces]
            enabled = true
            device = "cuda"
            cluster_threshold = 0.5
            min_neighbours = 3
            min_score = 0.7
            min_face_size = 24

            [viewer]
            external = ["imv", "-f"]
            "#,
        )
        .unwrap();
        let config = Config::load(&path, Path::new(HOME)).unwrap();
        assert_eq!(
            config.roots,
            [
                PathBuf::from("/home/test/Pictures"),
                PathBuf::from("/mnt/photos")
            ]
        );
        assert!(config.faces.enabled);
        assert_eq!(config.faces.device, Device::Cuda);
        assert!((config.faces.cluster_threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.faces.min_neighbours, 3);
        assert!((config.faces.min_score - 0.7).abs() < f32::EPSILON);
        assert_eq!(config.faces.min_face_size, 24);
        assert_eq!(config.viewer.external, ["imv", "-f"]);
    }

    #[test]
    fn partial_section_keeps_other_defaults() {
        let config = parse("[faces]\nenabled = true\n").unwrap();
        assert!(config.faces.enabled);
        assert_eq!(config.faces.min_neighbours, 2);
    }

    #[test]
    fn tilde_expansion() {
        let home = Path::new(HOME);
        assert_eq!(expand_tilde(Path::new("~"), home), home);
        assert_eq!(
            expand_tilde(Path::new("~/a/b"), home),
            Path::new("/home/test/a/b")
        );
        assert_eq!(
            expand_tilde(Path::new("~other/a"), home),
            Path::new("~other/a")
        );
        assert_eq!(expand_tilde(Path::new("/abs"), home), Path::new("/abs"));
    }

    #[test]
    fn relative_root_rejected() {
        assert!(matches!(
            parse(r#"roots = ["Pictures"]"#),
            Err(ConfigError::RelativeRoot(_))
        ));
        assert!(matches!(
            parse(r#"roots = ["~other/Pictures"]"#),
            Err(ConfigError::RelativeRoot(_))
        ));
    }

    #[test]
    fn cluster_threshold_bounds() {
        for bad in ["0.0", "-0.1", "1.01", "nan"] {
            let text = format!("[faces]\ncluster_threshold = {bad}\n");
            assert!(
                matches!(parse(&text), Err(ConfigError::ClusterThreshold(_))),
                "{bad} accepted"
            );
        }
        assert!(parse("[faces]\ncluster_threshold = 1.0\n").is_ok());
    }

    #[test]
    fn min_neighbours_zero_rejected() {
        assert!(matches!(
            parse("[faces]\nmin_neighbours = 0\n"),
            Err(ConfigError::MinNeighbours)
        ));
    }

    #[test]
    fn min_score_bounds() {
        for bad in ["0.49", "1.01", "nan"] {
            let text = format!("[faces]\nmin_score = {bad}\n");
            assert!(
                matches!(parse(&text), Err(ConfigError::MinScore(_))),
                "{bad} accepted"
            );
        }
        assert!(parse("[faces]\nmin_score = 0.5\n").is_ok());
        assert!(parse("[faces]\nmin_score = 1.0\n").is_ok());
    }

    #[test]
    fn empty_external_rejected() {
        for bad in ["[]", r#"[""]"#] {
            let text = format!("[viewer]\nexternal = {bad}\n");
            assert!(matches!(parse(&text), Err(ConfigError::EmptyExternal)));
        }
    }

    #[test]
    fn unknown_device_and_keys_rejected() {
        assert!(toml::from_str::<Config>("[faces]\ndevice = \"rocm\"\n").is_err());
        assert!(toml::from_str::<Config>("[faces]\nmin_neighbors = 2\n").is_err());
    }

    #[test]
    fn invalid_toml_reports_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, "roots = [").unwrap();
        let err = Config::load(&path, Path::new(HOME)).unwrap_err();
        assert!(matches!(err, ConfigError::Parse { .. }));
        assert!(err.to_string().contains("config.toml"));
    }
}
