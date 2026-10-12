//! A rayon pool hashes and thumbnails; the calling thread is the only writer and commits in batches.
//! Rows whose file is gone are marked missing, never deleted; a new path with the same content takes over the row.

mod faces;
mod geo;
mod meta;
mod walk;
mod writer;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, mpsc};
use std::thread;

use rayon::prelude::*;

use crate::catalog::faces::ModelId;
use crate::catalog::images::{self, Image, ImageId, Kind, Status};
use crate::catalog::runs::RunStats;
use crate::config::{self, Paths};
use crate::db::Sha256;
use crate::faces::cluster;
use crate::faces::detect::Scrfd;
use crate::faces::embed::ArcFace;
use crate::faces::fetch::{DETECTOR_NAME, EMBEDDING_DIM, MODEL_NAME};
use crate::faces::runtime::{self, FaceRuntime};
use crate::geo::GeoDb;
use crate::lock::{IndexLock, LockError};
use crate::slots::Slots;
use crate::thumbs::{self, generate::BUCKETS};
use crate::video::{self, Tools};
use faces::{Models, Setup, Stage};
use meta::Meta;
use walk::{Found, Walk};
use writer::{Op, Stored, Writer};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("nothing to index: no roots given")]
    NoRoots,
    #[error(transparent)]
    Lock(#[from] LockError),
    #[error(transparent)]
    Db(#[from] crate::db::Error),
    #[error("index run cancelled")]
    Cancelled,
}

/// Emitted in commit order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexEvent {
    /// Each `pending` file ends in exactly one `Added`, `Changed` or `Moved`.
    Walked {
        files: usize,
        pending: usize,
    },
    Added {
        id: ImageId,
        path: PathBuf,
    },
    Changed {
        id: ImageId,
        path: PathBuf,
    },
    Moved {
        id: ImageId,
        from: PathBuf,
        to: PathBuf,
    },
    Restored {
        id: ImageId,
        path: PathBuf,
    },
    Missing {
        id: ImageId,
        path: PathBuf,
    },
    /// Each of `images` ends in exactly one `FacesScanned` unless the stage stops early.
    FacesPending {
        images: usize,
    },
    /// `faces` is 0 when the content was left pending.
    FacesScanned {
        sha256: Sha256,
        path: PathBuf,
        faces: usize,
    },
    /// Each of `images` ends in exactly one `GeoScanned` unless the stage stops early.
    GeoPending {
        images: usize,
    },
    /// `found` is `false` for content without usable GPS and for content left pending.
    GeoScanned {
        sha256: Sha256,
        path: PathBuf,
        found: bool,
    },
    /// A file that reached hashing still gets a row with what could be read.
    Failed {
        path: PathBuf,
        error: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    pub path: PathBuf,
    pub error: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary {
    pub stats: RunStats,
    pub failures: Vec<Failure>,
}

struct Job {
    path: PathBuf,
    size: u64,
    mtime_ns: i64,
    kind: Kind,
    existing: Option<ImageId>,
}

struct Donor {
    id: ImageId,
    path: PathBuf,
    meta: Meta,
}

/// Holds the indexer lock for the whole run. Only rows at or below `roots` are compared with the disk; an unresolvable root's rows are left alone.
/// Unreadable files and folders, and missing or unloadable face models, go to [`Summary::failures`], not `Err`.
/// `on_event` runs on the calling thread after each batch commits.
/// `cancel` stops once in-progress files are committed; the run isn't recorded and nothing is marked missing unless hashing finished.
pub fn run(
    paths: &Paths,
    roots: &[PathBuf],
    faces: &config::Faces,
    cancel: &AtomicBool,
    on_event: impl FnMut(&IndexEvent),
) -> Result<Summary, Error> {
    let models_dir = paths.models_dir();
    let load = || -> anyhow::Result<Models> {
        let runtime = Arc::new(FaceRuntime::new(&models_dir, faces.device)?);
        Ok(Models {
            detector: Box::new(Scrfd::new(Arc::clone(&runtime))),
            embedder: Box::new(ArcFace::new(runtime)),
        })
    };
    let stage = if !faces.enabled {
        Stage::Off
    } else if let Some(missing) = runtime::missing_model(&models_dir) {
        Stage::Unavailable(Failure {
            path: missing,
            error: "faces: model file missing; run `vitrine faces fetch`".to_owned(),
        })
    } else {
        Stage::On(Setup {
            name: MODEL_NAME,
            detector: DETECTOR_NAME,
            dim: EMBEDDING_DIM,
            models_dir: &models_dir,
            load: &load,
            config: faces,
        })
    };
    run_with(paths, roots, stage, &Tools::SYSTEM, cancel, on_event)
}

/// Without working `tools`, videos are stored unhashed and the run reports one failure.
fn run_with(
    paths: &Paths,
    roots: &[PathBuf],
    stage: Stage<'_>,
    tools: &Tools,
    cancel: &AtomicBool,
    mut on_event: impl FnMut(&IndexEvent),
) -> Result<Summary, Error> {
    if roots.is_empty() {
        return Err(Error::NoRoots);
    }
    let _lock = IndexLock::acquire(&paths.index_lock())?;
    let mut conn = crate::catalog::open(&paths.catalog_db())?;

    let mut walked = Walk::default();
    let mut scope = Vec::new();
    for root in roots {
        match root.canonicalize() {
            Ok(root) => {
                walk::walk(&root, &mut walked, cancel);
                scope.push(root);
            }
            Err(e) => walked.failures.push(Failure {
                path: root.clone(),
                error: format!("root: {e}"),
            }),
        }
    }

    let mut known = HashMap::new();
    for root in &scope {
        for row in images::under(&conn, root)? {
            known.insert(row.path.clone(), row);
        }
        if let Some(row) = images::get_by_path(&conn, root)? {
            known.insert(row.path.clone(), row);
        }
    }

    if cancel.load(Ordering::Relaxed) {
        return Err(Error::Cancelled);
    }

    let first_video = walked
        .files
        .iter()
        .find(|(_, found)| found.kind == Kind::Video)
        .map(|(path, _)| path.clone());
    let have_tools = match first_video.map(|path| (path, tools.check())) {
        Some((path, Err(e))) => {
            walked.failures.push(Failure {
                path,
                error: format!("video: {e}; videos are not thumbnailed"),
            });
            false
        }
        Some((_, Ok(()))) => true,
        None => false,
    };

    let files = walked.files.len();
    let mut jobs = Vec::new();
    let mut direct = Vec::new();
    for (path, found) in walked.files {
        let row = known.remove(&path);
        if let Some(row) = &row
            && row.size == found.size
            && row.mtime_ns == found.mtime_ns
            && (row.sha256.is_some() || (row.kind == Kind::Video && !have_tools))
        {
            if row.status == Status::Missing {
                direct.push(Op::Restore { id: row.id, path });
            }
            continue;
        }
        let existing = row.map(|r| r.id);
        if found.kind == Kind::Video && !have_tools {
            direct.push(Op::Store(video(path, &found, existing)));
        } else {
            jobs.push(Job {
                path,
                size: found.size,
                mtime_ns: found.mtime_ns,
                kind: found.kind,
                existing,
            });
        }
    }
    let pending = jobs.len()
        + direct
            .iter()
            .filter(|op| matches!(op, Op::Store(_)))
            .count();

    let gone: Vec<Image> = known
        .into_values()
        .filter(|row| !walked.unreadable.iter().any(|u| row.path.starts_with(u)))
        .collect();
    let (donors, thumbs_conn) = if jobs.is_empty() {
        (HashMap::new(), None)
    } else {
        (
            donors(&gone, &images::missing(&conn)?, &scope),
            Some(thumbs::open(&paths.thumbs_db())?),
        )
    };
    let thumbnails = Thumbnails {
        stored: thumbs_conn
            .as_ref()
            .map(thumbs::keys)
            .transpose()?
            .unwrap_or_default(),
        claimed: Mutex::default(),
        decodes: Slots::new(DECODE_SLOTS),
    };

    for failure in &walked.failures {
        on_event(&IndexEvent::Failed {
            path: failure.path.clone(),
            error: failure.error.clone(),
        });
    }
    on_event(&IndexEvent::Walked { files, pending });

    let stats = RunStats {
        files: files as u64,
        ..RunStats::default()
    };
    let mut writer = Writer::new(&mut conn, on_event, stats, cancel);
    writer.failures = walked.failures;
    writer.vanished = gone
        .into_iter()
        .filter(|row| row.status == Status::Active)
        .collect();
    writer.thumbs = thumbs_conn;

    let (tx, rx) = mpsc::channel();
    let hashed = thread::scope(|s| {
        let (donors, thumbnails) = (&donors, &thumbnails);
        s.spawn(move || {
            // a send fails once the writer stopped on an error; that ends the pool early
            _ = jobs.into_par_iter().try_for_each_with(tx, |tx, job| {
                if cancel.load(Ordering::Relaxed) {
                    return Err(());
                }
                let stored = hash(job, donors, thumbnails, tools, cancel).ok_or(())?;
                tx.send(Op::Store(stored)).map_err(drop)
            });
        });
        writer.drain(direct, &rx)
    })?;
    if !hashed {
        return Err(Error::Cancelled);
    }
    writer.mark_vanished()?;

    match stage {
        Stage::Off => {}
        Stage::Unavailable(failure) => writer.fail(failure),
        Stage::On(setup) => {
            let model = crate::catalog::faces::register_model(
                writer.conn,
                setup.name,
                setup.detector,
                setup.dim,
            )?;
            if !face_stage(&mut writer, &setup, model, &scope, cancel)? {
                return Err(Error::Cancelled);
            }
            match cluster::incremental(writer.conn, model, setup.config) {
                Ok(outcome) => {
                    writer.stats.people_created += outcome.people_created;
                    writer.stats.faces_assigned += outcome.faces_assigned;
                }
                Err(e) => writer.fail(Failure {
                    path: paths.catalog_db(),
                    error: format!("faces: clustering: {e}"),
                }),
            }
        }
    }
    if !geo_stage(&mut writer, &paths.geo_db(), &scope, cancel)? {
        return Err(Error::Cancelled);
    }
    let (stats, failures) = writer.finish()?;
    Ok(Summary { stats, failures })
}

/// `false` when cancelled.
fn face_stage<F: FnMut(&IndexEvent)>(
    writer: &mut Writer<'_, F>,
    setup: &Setup<'_>,
    model: ModelId,
    scope: &[PathBuf],
    cancel: &AtomicBool,
) -> Result<bool, Error> {
    let pending = crate::catalog::faces::pending(writer.conn, model, scope)?;
    if pending.is_empty() {
        return Ok(true);
    }
    let models = match (setup.load)() {
        Ok(models) => models,
        Err(e) => {
            writer.fail(Failure {
                path: setup.models_dir.to_owned(),
                error: format!("faces: {e:#}"),
            });
            return Ok(true);
        }
    };
    (writer.on_event)(&IndexEvent::FacesPending {
        images: pending.len(),
    });
    let (tx, rx) = mpsc::channel();
    let complete = thread::scope(|s| {
        s.spawn(move || faces::scan_all(pending, &models, model, cancel, &tx));
        writer.drain(Vec::new(), &rx)
    })?;
    Ok(complete)
}

/// `false` when cancelled. Without `geo_db` coordinates stay pending.
fn geo_stage<F: FnMut(&IndexEvent)>(
    writer: &mut Writer<'_, F>,
    geo_db: &Path,
    scope: &[PathBuf],
    cancel: &AtomicBool,
) -> Result<bool, Error> {
    let pending = crate::catalog::geo::unscanned(writer.conn, scope)?;
    if !pending.is_empty() {
        (writer.on_event)(&IndexEvent::GeoPending {
            images: pending.len(),
        });
        let (tx, rx) = mpsc::channel();
        let complete = thread::scope(|s| {
            s.spawn(move || geo::scan_all(pending, cancel, &tx));
            writer.drain(Vec::new(), &rx)
        })?;
        if !complete {
            return Ok(false);
        }
    }
    match GeoDb::open(geo_db) {
        Ok(Some(gazetteer)) => {
            if writer.backfill_places(&gazetteer, geo_db)?.is_none() {
                return Ok(false);
            }
            Ok(writer.resolve(&gazetteer, geo_db)?)
        }
        Ok(None) => Ok(true),
        Err(e) => {
            writer.fail(geo::failure(geo_db, e));
            Ok(true)
        }
    }
}

fn video(path: PathBuf, found: &Found, existing: Option<ImageId>) -> Stored {
    Stored {
        path,
        size: found.size,
        mtime_ns: found.mtime_ns,
        kind: Kind::Video,
        existing,
        sha256: None,
        meta: Meta::default(),
        donors: Vec::new(),
        error: None,
        thumbs: Vec::new(),
    }
}

/// Caps memory: the pool has a thread per core and each full decode may hold up to the decode limit. DCT-scaled JPEG decodes don't take a slot.
const DECODE_SLOTS: usize = 4;

struct Thumbnails {
    stored: HashSet<(Sha256, u32)>,
    claimed: Mutex<HashSet<Sha256>>,
    decodes: Slots,
}

impl Thumbnails {
    /// Buckets to make for `sha256`; empty when all are stored or another worker claimed it.
    fn claim(&self, sha256: Sha256) -> Vec<u32> {
        let missing: Vec<u32> = BUCKETS
            .into_iter()
            .filter(|&bucket| !self.stored.contains(&(sha256, bucket)))
            .collect();
        let first = !missing.is_empty()
            && self
                .claimed
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(sha256);
        if first { missing } else { Vec::new() }
    }
}

/// Hashed rows whose file is gone, by content and kind: `gone` rows, plus `missing` rows outside `scope` whose path doesn't exist.
fn donors(
    gone: &[Image],
    missing: &[Image],
    scope: &[PathBuf],
) -> HashMap<(Sha256, Kind), Vec<Donor>> {
    let elsewhere = missing.iter().filter(|row| {
        !scope.iter().any(|root| row.path.starts_with(root))
            && fs::symlink_metadata(&row.path).is_err()
    });
    let mut donors: HashMap<(Sha256, Kind), Vec<Donor>> = HashMap::new();
    for row in gone.iter().chain(elsewhere) {
        if let Some(sha256) = row.sha256 {
            donors.entry((sha256, row.kind)).or_default().push(Donor {
                id: row.id,
                path: row.path.clone(),
                meta: Meta {
                    dimensions: row.width.zip(row.height),
                    taken_at: row.taken_at.clone(),
                    duration_ms: row.duration_ms,
                    codec: row.codec.clone(),
                },
            });
        }
    }
    donors
}

/// `None` when cancelled mid-file. Videos go through `tools` and take no decode slot.
fn hash(
    job: Job,
    donors: &HashMap<(Sha256, Kind), Vec<Donor>>,
    thumbnails: &Thumbnails,
    tools: &Tools,
    cancel: &AtomicBool,
) -> Option<Stored> {
    let mut stored = Stored {
        path: job.path,
        size: job.size,
        mtime_ns: job.mtime_ns,
        kind: job.kind,
        existing: job.existing,
        sha256: None,
        meta: Meta::default(),
        donors: Vec::new(),
        error: None,
        thumbs: Vec::new(),
    };
    let sha256 = match meta::sha256(&stored.path, cancel) {
        Ok(Some(sha256)) => sha256,
        Ok(None) => return None,
        Err(e) => {
            stored.error = Some(e.to_string());
            return Some(stored);
        }
    };
    stored.sha256 = Some(sha256);
    if let Some(found) = donors.get(&(sha256, stored.kind)) {
        stored.meta = found[0].meta.clone();
        stored.donors = found.iter().map(|d| (d.id, d.path.clone())).collect();
    } else if stored.kind == Kind::Video {
        match tools.probe(&stored.path, cancel) {
            Ok(probed) => stored.meta = probed.into(),
            Err(video::Error::Cancelled) => return None,
            Err(e) => stored.error = Some(format!("video: {e}")),
        }
    } else {
        let (meta, error) = meta::read(&stored.path);
        stored.meta = meta;
        stored.error = error.map(|e| format!("{e:#}"));
    }
    // an unreadable header or failed probe is already a failure and won't decode either
    if stored.error.is_none() {
        let buckets = thumbnails.claim(sha256);
        if !buckets.is_empty() {
            let made = if stored.kind == Kind::Video {
                let duration_ms = stored.meta.duration_ms;
                let largest = buckets.iter().copied().max().unwrap_or_default();
                let frame = |path: &Path| Ok(tools.frame(path, duration_ms, largest, cancel)?);
                thumbs::generate::generate_with(&stored.path, &buckets, frame)
            } else {
                let decode = |path: &Path| {
                    let _slot = thumbnails.decodes.acquire();
                    crate::decode::decode(path)
                };
                thumbs::generate::generate_with(&stored.path, &buckets, decode)
            };
            match made {
                Ok(made) => stored.thumbs = made.into_iter().map(|g| (g.bucket, g.thumb)).collect(),
                Err(e) if matches!(e.downcast_ref(), Some(video::Error::Cancelled)) => return None,
                Err(e) => stored.error = Some(format!("thumbnail: {e:#}")),
            }
        }
    }
    Some(stored)
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::AtomicUsize;
    use std::time::{Duration, Instant};

    use image::RgbImage;

    use super::*;
    use crate::catalog::geo::Source;
    use crate::catalog::{self, runs};
    use crate::faces::align::ARCFACE_DST;
    use crate::faces::{Detection, FaceDetector, FaceEmbedder};

    const NO_TOOLS: Tools = Tools {
        ffprobe: "/nonexistent/ffprobe",
        ffmpeg: "/nonexistent/ffmpeg",
    };

    fn off() -> config::Faces {
        config::Faces::default()
    }

    fn setup() -> (tempfile::TempDir, Paths, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let tree = root.join("photos");
        fs::create_dir_all(&tree).unwrap();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        for name in ["jpeg.jpg", "png.png"] {
            fs::copy(fixtures.join(name), tree.join(name)).unwrap();
        }
        let paths = Paths {
            home_dir: root.clone(),
            config_file: root.join("config.toml"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        (dir, paths, tree)
    }

    #[test]
    fn cancelled_before_the_walk_writes_nothing() {
        let (_dir, paths, tree) = setup();
        let err = run(&paths, &[tree], &off(), &AtomicBool::new(true), |_| {}).unwrap_err();
        assert!(matches!(err, Error::Cancelled), "{err:?}");
        let conn = crate::catalog::open(&paths.catalog_db()).unwrap();
        assert_eq!(images::counts(&conn).unwrap().images, 0);
        assert_eq!(runs::last(&conn).unwrap(), None);
    }

    #[test]
    fn cancelled_run_marks_nothing_missing_and_is_not_recorded() {
        let (_dir, paths, tree) = setup();
        let roots = [tree.clone()];
        run(&paths, &roots, &off(), &AtomicBool::new(false), |_| {}).unwrap();
        fs::remove_file(tree.join("png.png")).unwrap();
        fs::copy(tree.join("jpeg.jpg"), tree.join("copy.jpg")).unwrap();

        let cancel = AtomicBool::new(false);
        let err = run(&paths, &roots, &off(), &cancel, |event| {
            if matches!(event, IndexEvent::Walked { .. }) {
                cancel.store(true, Ordering::Relaxed);
            }
        })
        .unwrap_err();
        assert!(matches!(err, Error::Cancelled), "{err:?}");

        let conn = crate::catalog::open(&paths.catalog_db()).unwrap();
        let png = images::get_by_path(&conn, &tree.join("png.png"))
            .unwrap()
            .unwrap();
        assert_eq!(png.status, Status::Active);
        assert_eq!(
            images::get_by_path(&conn, &tree.join("copy.jpg")).unwrap(),
            None
        );
        assert_eq!(runs::last(&conn).unwrap().unwrap().stats.added, 2);

        let summary = run(&paths, &roots, &off(), &AtomicBool::new(false), |_| {}).unwrap();
        assert_eq!((summary.stats.added, summary.stats.vanished), (1, 1));
    }

    fn with_clip(tree: &Path) -> PathBuf {
        let clip = tree.join("clip.mp4");
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        fs::copy(fixtures.join("clip.mp4"), &clip).unwrap();
        clip
    }

    fn run_tools(
        paths: &Paths,
        roots: &[PathBuf],
        tools: &Tools,
        cancel: &AtomicBool,
    ) -> (Result<Summary, Error>, Vec<IndexEvent>) {
        let mut events = Vec::new();
        let result = run_with(paths, roots, Stage::Off, tools, cancel, |e| {
            events.push(e.clone());
        });
        (result, events)
    }

    #[test]
    fn missing_tools_leave_videos_unhashed_and_fail_each_run() {
        let (_dir, paths, tree) = setup();
        let clip = with_clip(&tree);
        let roots = [tree];
        let never = AtomicBool::new(false);
        let expected = [Failure {
            path: clip.clone(),
            error: "video: /nonexistent/ffprobe not found; videos are not thumbnailed".to_owned(),
        }];

        let (summary, _) = run_tools(&paths, &roots, &NO_TOOLS, &never);
        let summary = summary.unwrap();
        assert_eq!(summary.failures, expected);
        let stats = summary.stats;
        assert_eq!((stats.added, stats.hashed, stats.failed), (3, 2, 1));
        let conn = catalog::open(&paths.catalog_db()).unwrap();
        let row = images::get_by_path(&conn, &clip).unwrap().unwrap();
        assert_eq!((row.kind, row.sha256), (Kind::Video, None));
        assert_eq!(images::counts(&conn).unwrap().unhashed, 1);

        let (summary, events) = run_tools(&paths, &roots, &NO_TOOLS, &never);
        let summary = summary.unwrap();
        assert_eq!(summary.failures, expected);
        assert_eq!((summary.stats.added, summary.stats.changed), (0, 0));
        assert!(events.contains(&IndexEvent::Walked {
            files: 3,
            pending: 0
        }));
    }

    #[test]
    fn cancel_stops_a_running_tool_call() {
        let (dir, paths, tree) = setup();
        with_clip(&tree);
        let started = dir.path().join("started");
        let script = dir.path().join("tool");
        fs::write(
            &script,
            format!(
                "#!/bin/sh\n[ \"$1\" = -version ] && exit 0\ntouch '{}'\nexec sleep 30\n",
                started.display()
            ),
        )
        .unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
        let program: &'static str = Box::leak(script.to_str().unwrap().to_owned().into_boxed_str());
        let tools = Tools {
            ffprobe: program,
            ffmpeg: program,
        };

        let cancel = AtomicBool::new(false);
        let start = Instant::now();
        let result = thread::scope(|s| {
            s.spawn(|| {
                while !started.exists() && start.elapsed() < Duration::from_secs(20) {
                    thread::sleep(Duration::from_millis(10));
                }
                cancel.store(true, Ordering::Relaxed);
            });
            run_tools(&paths, std::slice::from_ref(&tree), &tools, &cancel).0
        });
        assert!(started.exists());
        assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "{:?}",
            start.elapsed()
        );
    }

    #[test]
    fn videos_are_hashed_probed_and_thumbnailed() {
        if !video::tools_present() {
            return;
        }
        let (_dir, paths, tree) = setup();
        let clip = with_clip(&tree);
        let roots = [tree.clone()];
        let never = AtomicBool::new(false);

        let (summary, events) = run_tools(&paths, &roots, &Tools::SYSTEM, &never);
        let summary = summary.unwrap();
        assert_eq!(summary.failures, []);
        let stats = summary.stats;
        assert_eq!((stats.added, stats.hashed, stats.thumbnailed), (3, 3, 3));
        // videos stay out of the geo and face stages
        assert!(events.contains(&IndexEvent::GeoPending { images: 2 }));
        let conn = catalog::open(&paths.catalog_db()).unwrap();
        let model =
            crate::catalog::faces::register_model(&conn, "fake", "fake-detector", 4).unwrap();
        assert_eq!(
            crate::catalog::faces::pending(&conn, model, &roots)
                .unwrap()
                .len(),
            2
        );

        let row = images::get_by_path(&conn, &clip).unwrap().unwrap();
        let sha256 = row.sha256.unwrap();
        assert_eq!(row.kind, Kind::Video);
        assert_eq!((row.width, row.height), (Some(72), Some(56)));
        assert_eq!(row.duration_ms, Some(1000));
        assert_eq!(row.codec.as_deref(), Some("h264"));
        assert!(
            row.taken_at.as_deref().unwrap().starts_with("2024-05-0"),
            "{row:?}"
        );
        let stored = thumbs::keys(&thumbs::open(&paths.thumbs_db()).unwrap()).unwrap();
        assert!(BUCKETS.iter().all(|&b| stored.contains(&(sha256, b))));
        assert_eq!(images::counts(&conn).unwrap().unhashed, 0);

        let (summary, _) = run_tools(&paths, &roots, &Tools::SYSTEM, &never);
        assert_eq!(summary.unwrap().stats.hashed, 0);

        let moved = tree.join("moved.mp4");
        fs::rename(&clip, &moved).unwrap();
        let (summary, _) = run_tools(&paths, &roots, &Tools::SYSTEM, &never);
        let stats = summary.unwrap().stats;
        assert_eq!(
            (stats.moved, stats.added, stats.vanished, stats.thumbnailed),
            (1, 0, 0, 0)
        );
        let after = images::get_by_path(&conn, &moved).unwrap().unwrap();
        assert_eq!((after.id, after.duration_ms), (row.id, row.duration_ms));
        assert_eq!(images::get_by_path(&conn, &clip).unwrap(), None);
    }

    #[test]
    fn tools_appearing_upgrade_placeholders() {
        if !video::tools_present() {
            return;
        }
        let (_dir, paths, tree) = setup();
        let clip = with_clip(&tree);
        let roots = [tree];
        let never = AtomicBool::new(false);
        run_tools(&paths, &roots, &NO_TOOLS, &never).0.unwrap();

        let (summary, _) = run_tools(&paths, &roots, &Tools::SYSTEM, &never);
        let summary = summary.unwrap();
        assert_eq!(summary.failures, []);
        assert_eq!((summary.stats.changed, summary.stats.hashed), (1, 1));
        let conn = catalog::open(&paths.catalog_db()).unwrap();
        let row = images::get_by_path(&conn, &clip).unwrap().unwrap();
        assert!(row.sha256.is_some() && row.duration_ms.is_some());
    }

    /// Every image yields one counted face and one too small to count.
    #[derive(Default)]
    struct Fake {
        loads: AtomicUsize,
        detections: Arc<AtomicUsize>,
        broken: bool,
    }

    struct FakeDetector {
        calls: Arc<AtomicUsize>,
        broken: bool,
    }

    impl FaceDetector for FakeDetector {
        fn detect(&self, _image: &RgbImage) -> anyhow::Result<Vec<Detection>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            anyhow::ensure!(!self.broken, "inference failed");
            let landmarks = ARCFACE_DST.map(|[x, y]| [x * 0.3 + 5.0, y * 0.3 + 5.0]);
            Ok(vec![
                Detection {
                    bbox: [2.0, 2.0, 42.0, 42.0],
                    landmarks,
                    score: 0.9,
                },
                Detection {
                    bbox: [0.0, 0.0, 10.0, 10.0],
                    landmarks,
                    score: 0.6,
                },
            ])
        }
    }

    struct FakeEmbedder;

    impl FaceEmbedder for FakeEmbedder {
        fn embed(&self, aligned: &[RgbImage]) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(aligned
                .iter()
                .map(|crop| {
                    let [r, g, b] = crop.get_pixel(56, 56).0.map(f32::from);
                    vec![r, g, b, 1.0]
                })
                .collect())
        }
    }

    fn run_fake(paths: &Paths, roots: &[PathBuf], fake: &Fake) -> Summary {
        run_fake_events(paths, roots, fake).0
    }

    fn run_fake_events(
        paths: &Paths,
        roots: &[PathBuf],
        fake: &Fake,
    ) -> (Summary, Vec<IndexEvent>) {
        let load = || -> anyhow::Result<Models> {
            fake.loads.fetch_add(1, Ordering::SeqCst);
            Ok(Models {
                detector: Box::new(FakeDetector {
                    calls: Arc::clone(&fake.detections),
                    broken: fake.broken,
                }),
                embedder: Box::new(FakeEmbedder),
            })
        };
        let models_dir = paths.models_dir();
        let stage = Stage::On(Setup {
            name: "fake",
            detector: "fake-detector",
            dim: 4,
            models_dir: &models_dir,
            load: &load,
            config: &config::Faces::default(),
        });
        let mut events = Vec::new();
        let summary = run_with(
            paths,
            roots,
            stage,
            &NO_TOOLS,
            &AtomicBool::new(false),
            |e| {
                events.push(e.clone());
            },
        )
        .unwrap();
        (summary, events)
    }

    fn face_events(events: &[IndexEvent]) -> (Option<usize>, Vec<(PathBuf, usize)>) {
        let pending = events.iter().find_map(|e| match e {
            IndexEvent::FacesPending { images } => Some(*images),
            _ => None,
        });
        let mut scanned: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                IndexEvent::FacesScanned { path, faces, .. } => Some((path.clone(), *faces)),
                _ => None,
            })
            .collect();
        scanned.sort();
        (pending, scanned)
    }

    fn model(paths: &Paths) -> crate::catalog::faces::ModelId {
        let conn = catalog::open(&paths.catalog_db()).unwrap();
        crate::catalog::faces::model_by_name(&conn, "fake")
            .unwrap()
            .unwrap()
    }

    fn contents(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
        let mut files: Vec<_> = fs::read_dir(dir)
            .unwrap()
            .map(|e| {
                let path = e.unwrap().path();
                let bytes = fs::read(&path).unwrap();
                (path, bytes)
            })
            .collect();
        files.sort();
        files
    }

    #[test]
    fn face_stage_scans_earlier_content_once() {
        let (_dir, paths, tree) = setup();
        let roots = [tree.clone()];
        let summary = run(&paths, &roots, &off(), &AtomicBool::new(false), |_| {}).unwrap();
        assert_eq!((summary.stats.added, summary.stats.faces_scanned), (2, 0));
        let before = contents(&tree);

        let fake = Fake::default();
        let (summary, events) = run_fake_events(&paths, &roots, &fake);
        assert_eq!(
            face_events(&events),
            (
                Some(2),
                vec![(tree.join("jpeg.jpg"), 2), (tree.join("png.png"), 2)]
            )
        );
        let scans = events
            .iter()
            .position(|e| matches!(e, IndexEvent::FacesScanned { .. }))
            .unwrap();
        let start = events
            .iter()
            .position(|e| matches!(e, IndexEvent::FacesPending { .. }))
            .unwrap();
        assert!(start < scans);
        assert_eq!(summary.failures, []);
        let stats = summary.stats;
        assert_eq!(
            (stats.hashed, stats.faces_scanned, stats.faces_found),
            (0, 2, 4)
        );
        assert_eq!(fake.loads.load(Ordering::SeqCst), 1);
        assert_eq!(fake.detections.load(Ordering::SeqCst), 2);
        assert_eq!(
            runs::last(&catalog::open(&paths.catalog_db()).unwrap())
                .unwrap()
                .unwrap()
                .stats,
            stats
        );

        let conn = catalog::open(&paths.catalog_db()).unwrap();
        let model = model(&paths);
        let counts = crate::catalog::faces::counts(&conn, model, 0.5, 40).unwrap();
        assert_eq!(counts.values().copied().collect::<Vec<_>>(), [1, 1]);
        let faces = crate::catalog::faces::for_clustering(&conn, model, 0.0, 0).unwrap();
        assert_eq!(faces.len(), 4);
        assert!(faces.iter().all(|f| f.embedding.len() == 4));
        let crop = crate::catalog::faces::crop(&conn, faces[0].id)
            .unwrap()
            .unwrap();
        let crop = image::load_from_memory(&crop).unwrap();
        assert_eq!(
            (crop.width(), crop.height()),
            (faces::CROP_SIZE, faces::CROP_SIZE)
        );
        let (w, h, score): (f32, f32, f32) = conn
            .query_row(
                "SELECT w, h, score FROM faces WHERE id = ?1",
                [faces[0].id.0],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!((w, h), (40.0, 40.0));
        assert!((score - 0.9).abs() < 1e-6);

        let summary = run_fake(&paths, &roots, &fake);
        assert_eq!(
            (summary.stats.faces_scanned, summary.stats.faces_found),
            (0, 0)
        );
        assert_eq!(fake.loads.load(Ordering::SeqCst), 1);
        assert_eq!(fake.detections.load(Ordering::SeqCst), 2);
        assert_eq!(contents(&tree), before);

        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        fs::copy(fixtures.join("webp.webp"), tree.join("webp.webp")).unwrap();
        fs::copy(fixtures.join("jpeg.jpg"), tree.join("copy.jpg")).unwrap();
        let summary = run_fake(&paths, &roots, &fake);
        assert_eq!((summary.stats.hashed, summary.stats.faces_scanned), (2, 1));
        assert_eq!(fake.detections.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn face_stage_is_limited_to_the_run_roots() {
        let (_dir, paths, tree) = setup();
        fs::create_dir_all(tree.join("b")).unwrap();
        fs::rename(tree.join("png.png"), tree.join("b/png.png")).unwrap();
        run(
            &paths,
            std::slice::from_ref(&tree),
            &off(),
            &AtomicBool::new(false),
            |_| {},
        )
        .unwrap();

        let fake = Fake::default();
        let summary = run_fake(&paths, &[tree.join("b")], &fake);
        assert_eq!(summary.stats.faces_scanned, 1);
        let summary = run_fake(&paths, &[tree.join("jpeg.jpg")], &fake);
        assert_eq!(summary.stats.faces_scanned, 1);
        let summary = run_fake(&paths, &[tree], &fake);
        assert_eq!(summary.stats.faces_scanned, 0);
    }

    #[test]
    fn undecodable_content_is_scanned_once() {
        let (_dir, paths, tree) = setup();
        fs::write(tree.join("broken.jpg"), b"not a jpeg").unwrap();
        let fake = Fake::default();
        let (summary, events) = run_fake_events(&paths, std::slice::from_ref(&tree), &fake);
        assert_eq!(summary.stats.faces_scanned, 3);
        assert_eq!(face_events(&events).1[0], (tree.join("broken.jpg"), 0));
        assert!(
            summary
                .failures
                .iter()
                .any(|f| f.path == tree.join("broken.jpg") && f.error.starts_with("faces: ")),
            "{:?}",
            summary.failures
        );
        let summary = run_fake(&paths, &[tree], &fake);
        assert_eq!(
            (summary.stats.faces_scanned, summary.failures.len()),
            (0, 0)
        );
    }

    #[test]
    fn model_failure_stops_the_stage_and_leaves_content_pending() {
        let (_dir, paths, tree) = setup();
        let roots = [tree];
        let broken = Fake {
            broken: true,
            ..Fake::default()
        };
        let summary = run_fake(&paths, &roots, &broken);
        assert_eq!((summary.stats.added, summary.stats.faces_scanned), (2, 0));
        assert!(
            summary
                .failures
                .iter()
                .all(|f| f.error.contains("inference failed"))
        );
        assert!(!summary.failures.is_empty());

        let summary = run_fake(&paths, &roots, &Fake::default());
        assert_eq!(summary.stats.faces_scanned, 2);
    }

    #[test]
    fn index_run_clusters_new_faces() {
        let (_dir, paths, tree) = setup();
        let people = tree.join("people");
        fs::create_dir_all(&people).unwrap();
        // one grey shade: one fake embedding; different sizes: different contents
        for side in [64, 65, 66] {
            RgbImage::from_pixel(side, side, image::Rgb([90, 90, 90]))
                .save(people.join(format!("{side}.png")))
                .unwrap();
        }
        let summary = run_fake(&paths, std::slice::from_ref(&people), &Fake::default());
        assert_eq!(
            (summary.stats.people_created, summary.stats.faces_assigned),
            (1, 3)
        );
        let conn = catalog::open(&paths.catalog_db()).unwrap();
        assert_eq!(runs::last(&conn).unwrap().unwrap().stats, summary.stats);
        assert_eq!(
            crate::catalog::people::list(&conn, model(&paths))
                .unwrap()
                .iter()
                .map(|p| p.faces)
                .collect::<Vec<_>>(),
            [3]
        );

        let summary = run_fake(&paths, &[people], &Fake::default());
        assert_eq!(
            (summary.stats.people_created, summary.stats.faces_assigned),
            (0, 0)
        );
    }

    #[test]
    fn unusable_models_fail_only_the_face_stage() {
        let (_dir, paths, tree) = setup();
        let roots = [tree];
        let enabled = config::Faces {
            enabled: true,
            ..config::Faces::default()
        };
        let summary = run(&paths, &roots, &enabled, &AtomicBool::new(false), |_| {}).unwrap();
        assert_eq!(summary.stats.added, 2);
        assert_eq!(summary.stats.failed, 1);
        assert!(
            summary.failures[0]
                .error
                .contains("run `vitrine faces fetch`"),
            "{:?}",
            summary.failures
        );
        let conn = catalog::open(&paths.catalog_db()).unwrap();
        assert_eq!(runs::last(&conn).unwrap().unwrap().stats, summary.stats);

        let load = || -> anyhow::Result<Models> { anyhow::bail!("no runtime") };
        let models_dir = paths.models_dir();
        let stage = Stage::On(Setup {
            name: "fake",
            detector: "fake-detector",
            dim: 4,
            models_dir: &models_dir,
            load: &load,
            config: &config::Faces::default(),
        });
        let summary = run_with(
            &paths,
            &roots,
            stage,
            &NO_TOOLS,
            &AtomicBool::new(false),
            |_| {},
        )
        .unwrap();
        assert_eq!(
            summary.failures,
            [Failure {
                path: models_dir,
                error: "faces: no runtime".to_owned()
            }]
        );
        let summary = run_fake(&paths, &roots, &Fake::default());
        assert_eq!(summary.stats.faces_scanned, 2);
    }

    fn with_gps(tree: &Path) {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        fs::copy(fixtures.join("gps.jpg"), tree.join("gps.jpg")).unwrap();
    }

    fn run_events(paths: &Paths, roots: &[PathBuf]) -> (Summary, Vec<IndexEvent>) {
        let mut events = Vec::new();
        let summary = run(paths, roots, &off(), &AtomicBool::new(false), |e| {
            events.push(e.clone());
        })
        .unwrap();
        (summary, events)
    }

    fn geo_scans(paths: &Paths) -> Vec<(PathBuf, Option<(f64, f64)>)> {
        let conn = catalog::open(&paths.catalog_db()).unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT min(i.path), s.lat, s.lon FROM geo_scans s
                 JOIN images i ON i.sha256 = s.sha256
                 GROUP BY s.sha256 ORDER BY 1",
            )
            .unwrap();
        stmt.query_map([], |r| {
            let lat: Option<f64> = r.get(1)?;
            let lon: Option<f64> = r.get(2)?;
            Ok((PathBuf::from(r.get::<_, String>(0)?), lat.zip(lon)))
        })
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
    }

    #[test]
    fn geo_stage_stores_coordinates_without_a_gazetteer() {
        let (_dir, paths, tree) = setup();
        with_gps(&tree);
        assert!(!paths.geo_db().exists());
        let (summary, events) = run_events(&paths, std::slice::from_ref(&tree));
        assert_eq!(summary.failures, []);
        assert_eq!(
            (
                summary.stats.added,
                summary.stats.gps_found,
                summary.stats.places_resolved
            ),
            (3, 1, 0)
        );

        let pending = events
            .iter()
            .position(|e| *e == IndexEvent::GeoPending { images: 3 })
            .unwrap();
        let scanned: Vec<usize> = events
            .iter()
            .enumerate()
            .filter(|(_, e)| matches!(e, IndexEvent::GeoScanned { .. }))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(scanned.len(), 3);
        assert!(scanned.iter().all(|&i| i > pending));
        assert!(events.iter().any(|e| matches!(
            e,
            IndexEvent::GeoScanned { path, found: true, .. } if *path == tree.join("gps.jpg")
        )));

        let scans = geo_scans(&paths);
        assert_eq!(scans.len(), 3);
        let (lat, lon) = scans[0].1.unwrap();
        assert_eq!(scans[0].0, tree.join("gps.jpg"));
        assert!((lat - 38.7223).abs() < 1e-4 && (lon + 9.1393).abs() < 1e-4);
        assert_eq!((scans[1].1, scans[2].1), (None, None));

        let conn = catalog::open(&paths.catalog_db()).unwrap();
        assert_eq!(runs::last(&conn).unwrap().unwrap().stats, summary.stats);
        let located: i64 = conn
            .query_row("SELECT count(*) FROM locations", [], |r| r.get(0))
            .unwrap();
        assert_eq!(located, 0);
    }

    #[test]
    fn geo_stage_backfills_unchanged_files_once() {
        let (_dir, paths, tree) = setup();
        with_gps(&tree);
        let roots = [tree.clone()];
        run(&paths, &roots, &off(), &AtomicBool::new(false), |_| {}).unwrap();
        // a catalog indexed before the stage existed
        catalog::open(&paths.catalog_db())
            .unwrap()
            .execute("DELETE FROM geo_scans", [])
            .unwrap();

        let (summary, events) = run_events(&paths, &roots);
        assert_eq!((summary.stats.hashed, summary.stats.gps_found), (0, 1));
        assert!(events.contains(&IndexEvent::GeoPending { images: 3 }));
        assert_eq!(geo_scans(&paths).len(), 3);

        let (summary, events) = run_events(&paths, &roots);
        assert_eq!(summary.stats.gps_found, 0);
        assert!(!events.iter().any(|e| matches!(
            e,
            IndexEvent::GeoPending { .. } | IndexEvent::GeoScanned { .. }
        )));
    }

    #[test]
    fn unopenable_file_stays_pending_for_gps() {
        let (_dir, paths, tree) = setup();
        let mut conn = catalog::open(&paths.catalog_db()).unwrap();
        let gone = tree.join("gone.jpg");
        images::upsert(
            &conn,
            &images::NewImage {
                path: &gone,
                size: 1,
                mtime_ns: 1,
                sha256: Some([7; 32]),
                kind: Kind::Image,
                width: None,
                height: None,
                taken_at: None,
                duration_ms: None,
                codec: None,
            },
        )
        .unwrap();
        let cancel = AtomicBool::new(false);
        let mut events = Vec::new();
        let mut writer = Writer::new(
            &mut conn,
            |e: &IndexEvent| events.push(e.clone()),
            RunStats::default(),
            &cancel,
        );
        assert!(
            geo_stage(
                &mut writer,
                &paths.geo_db(),
                std::slice::from_ref(&tree),
                &cancel
            )
            .unwrap()
        );
        let failures = std::mem::take(&mut writer.failures);
        drop(writer);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].path, gone);
        assert!(failures[0].error.starts_with("gps: "), "{failures:?}");
        assert!(events.contains(&IndexEvent::GeoScanned {
            sha256: [7; 32],
            path: gone.clone(),
            found: false,
        }));
        assert_eq!(
            crate::catalog::geo::unscanned(&conn, &[tree]).unwrap(),
            [([7; 32], gone)]
        );
    }

    #[test]
    fn cancelled_geo_stage_is_not_recorded() {
        let (_dir, paths, tree) = setup();
        let roots = [tree];
        let cancel = AtomicBool::new(false);
        let err = run(&paths, &roots, &off(), &cancel, |event| {
            if matches!(event, IndexEvent::GeoPending { .. }) {
                cancel.store(true, Ordering::Relaxed);
            }
        })
        .unwrap_err();
        assert!(matches!(err, Error::Cancelled), "{err:?}");
        let conn = catalog::open(&paths.catalog_db()).unwrap();
        assert_eq!(runs::last(&conn).unwrap(), None);
        assert_eq!(images::counts(&conn).unwrap().images, 2);
    }

    const LISBON: i64 = 2_267_057;

    fn cities() -> String {
        use crate::geo::fixtures::city;
        [
            city(
                LISBON, "Lisbon", "Lisbon", "Lisboa", 38.71667, -9.13333, "PT", "14", 517_802,
            ),
            city(
                2_735_943, "Porto", "Porto", "", 41.14961, -8.61099, "PT", "17", 249_633,
            ),
        ]
        .concat()
    }

    fn gazetteer(paths: &Paths) {
        fs::create_dir_all(&paths.data_dir).unwrap();
        crate::geo::fixtures::build(&paths.data_dir, &cities());
    }

    fn located(paths: &Paths, path: &Path) -> Option<(Option<i64>, Source)> {
        let conn = catalog::open(&paths.catalog_db()).unwrap();
        let sha256: Sha256 = conn
            .query_row(
                "SELECT sha256 FROM images WHERE path = ?1",
                [path.to_str().unwrap()],
                |r| r.get(0),
            )
            .unwrap();
        crate::catalog::geo::location_of(&conn, &sha256)
            .unwrap()
            .map(|l| (l.place.map(|p| p.geoname_id), l.source))
    }

    #[test]
    fn coordinates_resolve_once_a_gazetteer_appears() {
        let (_dir, paths, tree) = setup();
        with_gps(&tree);
        fs::copy(tree.join("gps.jpg"), tree.join("gps-copy.jpg")).unwrap();
        let roots = [tree.clone()];
        let (summary, _) = run_events(&paths, &roots);
        assert_eq!(
            (summary.stats.gps_found, summary.stats.places_resolved),
            (1, 0)
        );
        assert_eq!(located(&paths, &tree.join("gps.jpg")), None);

        gazetteer(&paths);
        let (summary, _) = run_events(&paths, &roots);
        assert_eq!(summary.failures, []);
        assert_eq!(
            (summary.stats.gps_found, summary.stats.places_resolved),
            (0, 1)
        );
        for name in ["gps.jpg", "gps-copy.jpg"] {
            assert_eq!(
                located(&paths, &tree.join(name)),
                Some((Some(LISBON), Source::Exif))
            );
        }
        let conn = catalog::open(&paths.catalog_db()).unwrap();
        assert_eq!(runs::last(&conn).unwrap().unwrap().stats, summary.stats);

        let (summary, _) = run_events(&paths, &roots);
        assert_eq!(summary.stats.places_resolved, 0);
    }

    #[test]
    fn index_run_backfills_place_coordinates_once() {
        use crate::catalog::geo::{tests::place, upsert_place};

        let (_dir, paths, tree) = setup();
        let mut conn = catalog::open(&paths.catalog_db()).unwrap();
        catalog::edit(&mut conn, |tx| {
            upsert_place(tx, &place(LISBON, "Lisbon"))?;
            upsert_place(tx, &place(9, "Elsewhere"))
        });
        drop(conn);
        gazetteer(&paths);
        let roots = [tree];
        let coords = |paths: &Paths| -> Vec<(i64, Option<f64>, Option<f64>)> {
            let conn = catalog::open(&paths.catalog_db()).unwrap();
            let mut stmt = conn
                .prepare("SELECT geoname_id, lat, lon FROM places ORDER BY geoname_id")
                .unwrap();
            stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap()
        };
        let expected = vec![(9, None, None), (LISBON, Some(38.71667), Some(-9.13333))];

        let (summary, _) = run_events(&paths, &roots);
        assert_eq!(summary.failures, []);
        assert_eq!(coords(&paths), expected);

        let (summary, _) = run_events(&paths, &roots);
        assert_eq!(summary.failures, []);
        assert_eq!(coords(&paths), expected);

        let gazetteer = GeoDb::open(&paths.geo_db()).unwrap().unwrap();
        let mut conn = catalog::open(&paths.catalog_db()).unwrap();
        assert_eq!(
            crate::catalog::geo::places_without_coords(&conn).unwrap(),
            [9]
        );
        let cancel = AtomicBool::new(false);
        let mut writer = Writer::new(&mut conn, |_: &IndexEvent| {}, RunStats::default(), &cancel);
        assert_eq!(
            writer.backfill_places(&gazetteer, &paths.geo_db()).unwrap(),
            Some(0)
        );
        cancel.store(true, Ordering::Relaxed);
        assert_eq!(
            writer.backfill_places(&gazetteer, &paths.geo_db()).unwrap(),
            None
        );
    }

    #[test]
    fn manual_locations_and_markers_win_over_exif() {
        use crate::catalog::geo::{clear, insert_scan, set_manual, tests::place};

        let (dir, mut conn) = catalog::temp();
        let gazetteer = crate::geo::fixtures::build(dir.path(), &cities());
        let (manual, marker, free, no_gps) = ([1; 32], [2; 32], [3; 32], [4; 32]);
        catalog::edit(&mut conn, |tx| {
            for sha in [manual, marker, free] {
                insert_scan(tx, &sha, Some((38.7223, -9.1393)))?;
            }
            insert_scan(tx, &no_gps, None)?;
            set_manual(tx, &manual, &place(9, "Elsewhere"), 10)?;
            clear(tx, &marker, 10)
        });
        let cancel = AtomicBool::new(false);
        let mut writer = Writer::new(&mut conn, |_: &IndexEvent| {}, RunStats::default(), &cancel);
        assert!(writer.resolve(&gazetteer, dir.path()).unwrap());
        assert_eq!(writer.stats.places_resolved, 1);
        assert_eq!(writer.failures, []);
        drop(writer);

        let at = |sha| {
            crate::catalog::geo::location_of(&conn, &sha)
                .unwrap()
                .map(|l| (l.place.map(|p| p.geoname_id), l.source))
        };
        assert_eq!(at(manual), Some((Some(9), Source::Manual)));
        assert_eq!(at(marker), Some((None, Source::Manual)));
        assert_eq!(at(free), Some((Some(LISBON), Source::Exif)));
        assert_eq!(at(no_gps), None);
    }

    #[test]
    fn resolve_pages_past_unplaceable_scans_and_stops_on_cancel() {
        use crate::geo::fixtures::{ADMIN1, COUNTRIES};

        let (dir, mut conn) = catalog::temp();
        let shas: Vec<Sha256> = (0..=500_u16)
            .map(|i| {
                let mut sha = [0; 32];
                sha[..2].copy_from_slice(&i.to_be_bytes());
                sha
            })
            .collect();
        catalog::edit(&mut conn, |tx| {
            shas.iter().try_for_each(|sha| {
                crate::catalog::geo::insert_scan(tx, sha, Some((38.7223, -9.1393)))
            })
        });
        let empty = dir.path().join("empty.db");
        crate::geo::build(
            &empty,
            vec![crate::geo::Rows::all("cities1000.txt", &b""[..])],
            ADMIN1.as_bytes(),
            COUNTRIES.as_bytes(),
            0,
        )
        .unwrap();
        let empty = GeoDb::open(&empty).unwrap().unwrap();
        let full = crate::geo::fixtures::build(dir.path(), &cities());

        let cancel = AtomicBool::new(true);
        let mut writer = Writer::new(&mut conn, |_: &IndexEvent| {}, RunStats::default(), &cancel);
        assert!(!writer.resolve(&full, dir.path()).unwrap());
        cancel.store(false, Ordering::Relaxed);
        assert!(writer.resolve(&empty, dir.path()).unwrap());
        assert_eq!(writer.stats.places_resolved, 0);
        assert!(writer.resolve(&full, dir.path()).unwrap());
        assert_eq!(writer.stats.places_resolved, 501);
        drop(writer);
        assert!(
            crate::catalog::geo::unresolved(&conn, None, 10)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn unusable_gazetteer_fails_and_keeps_coordinates() {
        let (_dir, paths, tree) = setup();
        with_gps(&tree);
        fs::create_dir_all(&paths.data_dir).unwrap();
        fs::write(paths.geo_db(), b"not a database").unwrap();
        let (summary, _) = run_events(&paths, std::slice::from_ref(&tree));
        assert_eq!(summary.stats.gps_found, 1);
        assert_eq!(summary.failures.len(), 1);
        assert_eq!(summary.failures[0].path, paths.geo_db());
        assert!(
            summary.failures[0].error.starts_with("geo: failed to open"),
            "{:?}",
            summary.failures
        );
        assert_eq!(located(&paths, &tree.join("gps.jpg")), None);
    }
}
