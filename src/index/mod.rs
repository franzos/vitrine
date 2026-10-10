//! A rayon pool hashes and thumbnails; the calling thread is the only writer and commits in batches.
//! Rows whose file is gone are marked missing, never deleted; a new path with the same content takes over the row.

mod faces;
mod meta;
mod walk;
mod writer;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError, mpsc};
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
use crate::lock::{IndexLock, LockError};
use crate::thumbs::{self, generate::BUCKETS};
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
    run_with(paths, roots, stage, cancel, on_event)
}

fn run_with(
    paths: &Paths,
    roots: &[PathBuf],
    stage: Stage<'_>,
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

    let files = walked.files.len();
    let mut jobs = Vec::new();
    let mut direct = Vec::new();
    for (path, found) in walked.files {
        let row = known.remove(&path);
        if let Some(row) = &row
            && row.size == found.size
            && row.mtime_ns == found.mtime_ns
            && (row.sha256.is_some() || row.kind == Kind::Video)
        {
            if row.status == Status::Missing {
                direct.push(Op::Restore { id: row.id, path });
            }
            continue;
        }
        let existing = row.map(|r| r.id);
        match found.kind {
            Kind::Image => jobs.push(Job {
                path,
                size: found.size,
                mtime_ns: found.mtime_ns,
                existing,
            }),
            Kind::Video => direct.push(Op::Store(video(path, &found, existing))),
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
                tx.send(Op::Store(hash(job, donors, thumbnails)))
                    .map_err(drop)
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

struct Slots {
    free: Mutex<usize>,
    freed: Condvar,
}

impl Slots {
    fn new(count: usize) -> Self {
        Self {
            free: Mutex::new(count),
            freed: Condvar::new(),
        }
    }

    fn acquire(&self) -> Slot<'_> {
        let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
        while *free == 0 {
            free = self
                .freed
                .wait(free)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *free -= 1;
        Slot(self)
    }
}

struct Slot<'a>(&'a Slots);

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        *self.0.free.lock().unwrap_or_else(PoisonError::into_inner) += 1;
        self.0.freed.notify_one();
    }
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

/// Image rows whose file is gone, by content: `gone` rows, plus `missing` rows outside `scope` whose path doesn't exist.
fn donors(gone: &[Image], missing: &[Image], scope: &[PathBuf]) -> HashMap<Sha256, Vec<Donor>> {
    let elsewhere = missing.iter().filter(|row| {
        !scope.iter().any(|root| row.path.starts_with(root))
            && fs::symlink_metadata(&row.path).is_err()
    });
    let mut donors: HashMap<Sha256, Vec<Donor>> = HashMap::new();
    for row in gone.iter().chain(elsewhere) {
        if let (Some(sha256), Kind::Image) = (row.sha256, row.kind) {
            donors.entry(sha256).or_default().push(Donor {
                id: row.id,
                path: row.path.clone(),
                meta: Meta {
                    dimensions: row.width.zip(row.height),
                    taken_at: row.taken_at.clone(),
                },
            });
        }
    }
    donors
}

fn hash(job: Job, donors: &HashMap<Sha256, Vec<Donor>>, thumbnails: &Thumbnails) -> Stored {
    let mut stored = Stored {
        path: job.path,
        size: job.size,
        mtime_ns: job.mtime_ns,
        kind: Kind::Image,
        existing: job.existing,
        sha256: None,
        meta: Meta::default(),
        donors: Vec::new(),
        error: None,
        thumbs: Vec::new(),
    };
    let sha256 = match meta::sha256(&stored.path) {
        Ok(sha256) => sha256,
        Err(e) => {
            stored.error = Some(e.to_string());
            return stored;
        }
    };
    stored.sha256 = Some(sha256);
    if let Some(found) = donors.get(&sha256) {
        stored.meta = found[0].meta.clone();
        stored.donors = found.iter().map(|d| (d.id, d.path.clone())).collect();
    } else {
        let (meta, error) = meta::read(&stored.path);
        stored.meta = meta;
        stored.error = error.map(|e| format!("{e:#}"));
    }
    // an unreadable header is already a failure and won't decode either
    if stored.error.is_none() {
        let buckets = thumbnails.claim(sha256);
        if !buckets.is_empty() {
            let decode = |path: &std::path::Path| {
                let _slot = thumbnails.decodes.acquire();
                crate::decode::decode(path)
            };
            match thumbs::generate::generate_with(&stored.path, &buckets, decode) {
                Ok(made) => stored.thumbs = made.into_iter().map(|g| (g.bucket, g.thumb)).collect(),
                Err(e) => stored.error = Some(format!("thumbnail: {e:#}")),
            }
        }
    }
    stored
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::atomic::AtomicUsize;

    use image::RgbImage;

    use super::*;
    use crate::catalog::{self, runs};
    use crate::faces::align::ARCFACE_DST;
    use crate::faces::{Detection, FaceDetector, FaceEmbedder};

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
    fn slots_block_until_one_is_freed() {
        let slots = Slots::new(2);
        let held = [slots.acquire(), slots.acquire()];
        let taken = AtomicBool::new(false);
        thread::scope(|s| {
            s.spawn(|| {
                let _slot = slots.acquire();
                taken.store(true, Ordering::SeqCst);
            });
            thread::sleep(std::time::Duration::from_millis(50));
            assert!(!taken.load(Ordering::SeqCst));
            drop(held);
        });
        assert!(taken.load(Ordering::SeqCst));
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
        let summary = run_with(paths, roots, stage, &AtomicBool::new(false), |e| {
            events.push(e.clone());
        })
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
        let summary = run_with(&paths, &roots, stage, &AtomicBool::new(false), |_| {}).unwrap();
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
}
