use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use assert_cmd::cargo::cargo_bin_cmd;
use rusqlite::{Connection, OptionalExtension};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use vitrine::lock::IndexLock;

const LAYOUT: &[(&str, &str)] = &[
    ("jpeg.jpg", "2024/jpeg.jpg"),
    ("png.png", "2024/png.png"),
    ("webp.webp", "2024/webp.webp"),
    ("gif.gif", "2024/gif.gif"),
    ("tiff.tif", "2024/tiff.tif"),
    ("heic.heic", "2024/heic.heic"),
    ("avif.avif", "2025/avif.avif"),
    ("jxl.jxl", "2025/jxl.jxl"),
    ("dng.dng", "2025/dng.dng"),
    ("dng-orientation-6.dng", "2025/dng-orientation-6.dng"),
    ("orientation-6.jpg", "2025/orientation-6.jpg"),
    ("jpeg.jpg", "2025/.hidden/jpeg.jpg"),
];
const IMAGES: u64 = 11;

struct Env {
    dir: TempDir,
    tree: PathBuf,
    /// Runs with the real `PATH` instead of an empty one.
    tools: bool,
}

#[derive(Debug, PartialEq, Eq)]
struct Row {
    id: i64,
    kind: String,
    status: String,
    sha256: Option<Vec<u8>>,
    width: Option<u32>,
    height: Option<u32>,
}

impl Env {
    /// `with_root` writes a config naming the tree as the only root.
    fn new(with_root: bool) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let tree = dir.path().join("photos");
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        for (fixture, target) in LAYOUT {
            let target = tree.join(target);
            fs::create_dir_all(target.parent().unwrap()).unwrap();
            fs::copy(fixtures.join(fixture), target).unwrap();
        }
        fs::write(tree.join("2025/clip.mp4"), b"not really a video").unwrap();
        fs::write(tree.join("2025/notes.txt"), b"skipped").unwrap();
        let env = Self {
            dir,
            tree,
            tools: false,
        };
        if with_root {
            let config = env.home().join(".config/vitrine/config.toml");
            fs::create_dir_all(config.parent().unwrap()).unwrap();
            fs::write(config, format!("roots = [\"{}\"]\n", env.tree.display())).unwrap();
        }
        env
    }

    /// Like `new(true)` with a real `clip.mp4` and the real `PATH`.
    fn with_ffmpeg() -> Self {
        let mut env = Self::new(true);
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        fs::copy(fixtures.join("clip.mp4"), env.tree.join("2025/clip.mp4")).unwrap();
        env.tools = true;
        env
    }

    fn home(&self) -> PathBuf {
        self.dir.path().join("home")
    }

    fn data_dir(&self) -> PathBuf {
        self.home().join(".local/share/vitrine")
    }

    /// Without `tools`, runs with an empty `PATH`, so `clip.mp4` stays an unhashed placeholder.
    fn vitrine(&self, args: &[&str]) -> assert_cmd::Command {
        let home = self.home();
        let mut cmd = cargo_bin_cmd!("vitrine");
        if !self.tools {
            let bin = self.dir.path().join("bin");
            fs::create_dir_all(&bin).unwrap();
            cmd.env("PATH", bin);
        }
        cmd.args(args)
            .env("HOME", &home)
            .env("XDG_CONFIG_HOME", home.join(".config"))
            .env("XDG_DATA_HOME", home.join(".local/share"))
            .env("XDG_CACHE_HOME", home.join(".cache"));
        cmd
    }

    fn index(&self, args: &[&str]) -> HashMap<String, String> {
        let mut full = vec!["index"];
        full.extend(args);
        let output = self.vitrine(&full).assert().success().get_output().clone();
        fields(&output.stdout)
    }

    fn status(&self) -> HashMap<String, String> {
        let output = self
            .vitrine(&["status"])
            .assert()
            .success()
            .get_output()
            .clone();
        fields(&output.stdout)
    }

    fn row(&self, rel: &str) -> Option<Row> {
        let conn = Connection::open(self.data_dir().join("catalog.db")).unwrap();
        conn.query_row(
            "SELECT id, kind, status, sha256, width, height FROM images WHERE path = ?1",
            [self.tree.join(rel).to_str().unwrap()],
            |r| {
                Ok(Row {
                    id: r.get(0)?,
                    kind: r.get(1)?,
                    status: r.get(2)?,
                    sha256: r.get(3)?,
                    width: r.get(4)?,
                    height: r.get(5)?,
                })
            },
        )
        .optional()
        .unwrap()
    }

    /// `(duration_ms, codec)` of the row at `rel`.
    fn video_meta(&self, rel: &str) -> (Option<u64>, Option<String>) {
        let conn = Connection::open(self.data_dir().join("catalog.db")).unwrap();
        conn.query_row(
            "SELECT duration_ms, codec FROM images WHERE path = ?1",
            [self.tree.join(rel).to_str().unwrap()],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap()
    }

    fn thumbs_db(&self) -> PathBuf {
        self.home().join(".cache/vitrine/thumbs.db")
    }

    /// Asserts each blob decodes at its recorded size.
    fn thumbs(&self) -> BTreeMap<Vec<u8>, Vec<u32>> {
        let conn = Connection::open(self.thumbs_db()).unwrap();
        let mut stmt = conn
            .prepare("SELECT sha256, bucket, jpeg, width, height FROM thumbs ORDER BY bucket")
            .unwrap();
        let mut out: BTreeMap<Vec<u8>, Vec<u32>> = BTreeMap::new();
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, Vec<u8>>(0)?,
                    r.get::<_, u32>(1)?,
                    r.get::<_, Vec<u8>>(2)?,
                    r.get::<_, u32>(3)?,
                    r.get::<_, u32>(4)?,
                ))
            })
            .unwrap();
        for row in rows {
            let (sha256, bucket, jpeg, width, height) = row.unwrap();
            let image = image::load_from_memory(&jpeg).unwrap();
            assert_eq!((image.width(), image.height()), (width, height));
            assert!(width.max(height) <= bucket);
            out.entry(sha256).or_default().push(bucket);
        }
        out
    }

    fn image_hashes(&self) -> Vec<Vec<u8>> {
        let conn = Connection::open(self.data_dir().join("catalog.db")).unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT DISTINCT sha256 FROM images WHERE kind = 'image' AND status = 'active'",
            )
            .unwrap();
        stmt.query_map([], |r| r.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    }

    fn rows(&self) -> i64 {
        let conn = Connection::open(self.data_dir().join("catalog.db")).unwrap();
        conn.query_row("SELECT count(*) FROM images", [], |r| r.get(0))
            .unwrap()
    }

    fn snapshot(&self) -> BTreeMap<PathBuf, Vec<u8>> {
        fn visit(dir: &Path, base: &Path, out: &mut BTreeMap<PathBuf, Vec<u8>>) {
            for entry in fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    visit(&path, base, out);
                } else {
                    let digest = Sha256::digest(fs::read(&path).unwrap()).to_vec();
                    out.insert(path.strip_prefix(base).unwrap().to_owned(), digest);
                }
            }
        }
        let mut out = BTreeMap::new();
        visit(&self.tree, &self.tree, &mut out);
        out
    }

    fn rename(&self, from: &str, to: &str) {
        let to = self.tree.join(to);
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        fs::rename(self.tree.join(from), to).unwrap();
    }
}

fn fields(stdout: &[u8]) -> HashMap<String, String> {
    String::from_utf8_lossy(stdout)
        .lines()
        .filter_map(|line| line.split_once(':'))
        .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
        .collect()
}

/// Whether `ffprobe` and `ffmpeg` run from `PATH`; prints why a gated test is skipped when they don't.
fn ffmpeg_present() -> bool {
    let missing = ["ffprobe", "ffmpeg"].into_iter().find(|program| {
        !Command::new(program)
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    });
    if let Some(program) = missing {
        eprintln!("skipping: {program} not found");
    }
    missing.is_none()
}

fn count(fields: &HashMap<String, String>, key: &str) -> u64 {
    fields
        .get(key)
        .unwrap_or_else(|| panic!("no `{key}` in {fields:?}"))
        .parse()
        .unwrap()
}

#[test]
fn first_index_catalogues_images_and_videos() {
    let env = Env::new(true);
    let before = env.snapshot();

    let run = env.index(&[]);
    assert_eq!(count(&run, "files"), IMAGES + 1);
    assert_eq!(count(&run, "hashed"), IMAGES);
    assert_eq!(count(&run, "added"), IMAGES + 1);
    assert_eq!(count(&run, "failed"), 1);
    assert_eq!(count(&run, "thumbnailed"), IMAGES);

    let thumbs = env.thumbs();
    assert_eq!(thumbs.len() as u64, IMAGES);
    for sha256 in env.image_hashes() {
        assert_eq!(thumbs.get(&sha256), Some(&vec![256, 512]));
    }

    let status = env.status();
    assert_eq!(count(&status, "images"), IMAGES);
    assert_eq!(count(&status, "videos"), 1);
    assert_eq!(count(&status, "missing"), 0);
    assert_eq!(count(&status, "hashed"), IMAGES);
    assert_eq!(count(&status, "unhashed"), 1);
    assert!(status.contains_key("indexed"));

    for (_, rel) in &LAYOUT[..LAYOUT.len() - 1] {
        let row = env.row(rel).unwrap_or_else(|| panic!("{rel} not indexed"));
        assert_eq!(row.kind, "image");
        assert_eq!(row.sha256.as_ref().map(Vec::len), Some(32), "{rel}");
        assert!(row.width.is_some() && row.height.is_some(), "{rel}");
    }
    let rotated = env.row("2025/orientation-6.jpg").unwrap();
    assert_eq!((rotated.width, rotated.height), (Some(48), Some(64)));
    let video = env.row("2025/clip.mp4").unwrap();
    assert_eq!(video.kind, "video");
    assert_eq!((video.sha256, video.width), (None, None));
    assert!(env.row("2025/.hidden/jpeg.jpg").is_none());
    assert!(env.row("2025/notes.txt").is_none());

    assert_eq!(env.snapshot(), before);
}

#[test]
fn unchanged_rerun_hashes_nothing() {
    let env = Env::new(true);
    let before = env.snapshot();
    env.index(&[]);
    let run = env.index(&[]);
    assert_eq!(count(&run, "files"), IMAGES + 1);
    assert_eq!(count(&run, "added"), 0);
    assert_eq!(count(&run, "changed"), 0);

    assert_eq!(count(&run, "thumbnailed"), 0);

    let status = env.status();
    assert_eq!(count(&status, "hashed"), 0);
    assert_eq!(count(&status, "thumbnailed"), 0);
    assert_eq!(count(&status, "images"), IMAGES);
    assert_eq!(env.snapshot(), before);
}

#[test]
fn moved_file_without_thumbnails_gets_them() {
    let env = Env::new(true);
    env.index(&[]);
    let sha256 = env.row("2024/png.png").unwrap().sha256.unwrap();
    fs::remove_file(env.thumbs_db()).unwrap();
    env.rename("2024/png.png", "2026/png.png");

    let run = env.index(&[]);
    assert_eq!((count(&run, "moved"), count(&run, "thumbnailed")), (1, 1));
    let thumbs = env.thumbs();
    assert_eq!(thumbs.len(), 1);
    assert_eq!(thumbs[&sha256], [256, 512]);
}

#[test]
fn video_without_tools_stays_a_placeholder() {
    let env = Env::new(true);
    let output = env
        .vitrine(&["index"])
        .assert()
        .success()
        .get_output()
        .clone();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("clip.mp4: video: ffprobe not found; videos are not thumbnailed"),
        "{stderr}"
    );

    let run = env.index(&[]);
    assert_eq!((count(&run, "changed"), count(&run, "failed")), (0, 1));

    fs::write(env.tree.join("2025/clip.mp4"), b"a longer fake video").unwrap();
    let run = env.index(&[]);
    assert_eq!(count(&run, "changed"), 1);
    assert_eq!(count(&run, "hashed"), 0);
    assert_eq!(count(&run, "failed"), 1);
    let video = env.row("2025/clip.mp4").unwrap();
    assert_eq!((video.kind.as_str(), video.sha256), ("video", None));
}

#[test]
fn first_index_with_ffmpeg_thumbnails_the_video() {
    if !ffmpeg_present() {
        return;
    }
    let env = Env::with_ffmpeg();
    let before = env.snapshot();

    let run = env.index(&[]);
    assert_eq!(count(&run, "files"), IMAGES + 1);
    assert_eq!(count(&run, "hashed"), IMAGES + 1);
    assert_eq!(count(&run, "failed"), 0);
    assert_eq!(count(&run, "thumbnailed"), IMAGES + 1);

    let video = env.row("2025/clip.mp4").unwrap();
    assert_eq!(video.kind, "video");
    assert_eq!((video.width, video.height), (Some(72), Some(56)));
    assert_eq!(
        env.video_meta("2025/clip.mp4"),
        (Some(1000), Some("h264".to_owned()))
    );
    assert_eq!(env.thumbs()[&video.sha256.unwrap()], [256, 512]);
    assert_eq!(count(&env.status(), "unhashed"), 0);
    assert_eq!(env.snapshot(), before);
}

#[test]
fn video_indexed_without_tools_is_upgraded_with_them() {
    if !ffmpeg_present() {
        return;
    }
    let mut env = Env::with_ffmpeg();
    env.tools = false;
    let run = env.index(&[]);
    assert_eq!(count(&run, "failed"), 1);
    assert_eq!(env.row("2025/clip.mp4").unwrap().sha256, None);

    env.tools = true;
    let run = env.index(&[]);
    assert_eq!(count(&run, "changed"), 1);
    assert_eq!(count(&run, "hashed"), 1);
    assert_eq!(count(&run, "thumbnailed"), 1);
    assert_eq!(count(&run, "failed"), 0);
    let video = env.row("2025/clip.mp4").unwrap();
    assert_eq!((video.width, video.height), (Some(72), Some(56)));
    assert_eq!(env.thumbs()[&video.sha256.unwrap()], [256, 512]);
}

#[test]
fn changed_video_that_fails_to_probe_keeps_a_hash_but_no_thumbnails() {
    if !ffmpeg_present() {
        return;
    }
    let env = Env::with_ffmpeg();
    env.index(&[]);
    let original = env.row("2025/clip.mp4").unwrap().sha256.unwrap();

    fs::write(env.tree.join("2025/clip.mp4"), b"no longer a real video").unwrap();
    let run = env.index(&[]);
    assert_eq!(count(&run, "changed"), 1);
    assert_eq!(count(&run, "hashed"), 1);
    assert_eq!(count(&run, "failed"), 1);
    assert_eq!(count(&run, "thumbnailed"), 0);
    let sha256 = env.row("2025/clip.mp4").unwrap().sha256.unwrap();
    assert_ne!(sha256, original);
    assert!(!env.thumbs().contains_key(&sha256));
}

#[test]
fn moved_file_keeps_its_row() {
    let env = Env::new(true);
    env.index(&[]);
    let original = env.row("2024/jpeg.jpg").unwrap();

    let mut expected = env.snapshot();
    env.rename("2024/jpeg.jpg", "2026/renamed.jpg");
    let digest = expected.remove(Path::new("2024/jpeg.jpg")).unwrap();
    expected.insert(PathBuf::from("2026/renamed.jpg"), digest);

    let run = env.index(&[]);
    assert_eq!(count(&run, "moved"), 1);
    assert_eq!(count(&run, "hashed"), 1);
    assert_eq!(count(&run, "thumbnailed"), 0);
    assert_eq!(count(&run, "added"), 0);
    assert_eq!(count(&run, "vanished"), 0);

    assert!(env.row("2024/jpeg.jpg").is_none());
    assert_eq!(env.row("2026/renamed.jpg").unwrap(), original);
    assert_eq!(count(&env.status(), "missing"), 0);

    // a copy of content still in place gets a new row
    fs::copy(
        env.tree.join("2024/png.png"),
        env.tree.join("2026/copy.png"),
    )
    .unwrap();
    let digest = expected[Path::new("2024/png.png")].clone();
    expected.insert(PathBuf::from("2026/copy.png"), digest);
    let run = env.index(&[]);
    assert_eq!((count(&run, "added"), count(&run, "moved")), (1, 0));
    assert_eq!(count(&run, "thumbnailed"), 0);
    assert_ne!(
        env.row("2026/copy.png").unwrap().id,
        env.row("2024/png.png").unwrap().id
    );
    assert_eq!(env.snapshot(), expected);
}

#[test]
fn removed_file_goes_missing_and_comes_back() {
    let env = Env::new(true);
    let outside = env.dir.path().join("outside.webp");
    let before = env.snapshot();
    env.index(&[]);
    let original = env.row("2024/webp.webp").unwrap();

    fs::rename(env.tree.join("2024/webp.webp"), &outside).unwrap();
    let run = env.index(&[]);
    assert_eq!(count(&run, "vanished"), 1);
    let status = env.status();
    assert_eq!(count(&status, "missing"), 1);
    assert_eq!(count(&status, "images"), IMAGES - 1);
    let row = env.row("2024/webp.webp").unwrap();
    assert_eq!(row.status, "missing");

    // a second run leaves it missing without counting it again
    let run = env.index(&[]);
    assert_eq!(count(&run, "vanished"), 0);
    assert_eq!(env.rows(), i64::try_from(IMAGES).unwrap() + 1);

    fs::rename(&outside, env.tree.join("2024/webp.webp")).unwrap();
    let run = env.index(&[]);
    assert_eq!(count(&run, "restored"), 1);
    assert_eq!(count(&run, "hashed"), 0);
    let status = env.status();
    assert_eq!(count(&status, "missing"), 0);
    assert_eq!(count(&status, "images"), IMAGES);
    assert_eq!(env.row("2024/webp.webp").unwrap(), original);
    assert_eq!(env.snapshot(), before);
}

#[test]
fn indexing_a_folder_leaves_the_rest_alone() {
    let env = Env::new(true);
    env.index(&[]);
    fs::remove_file(env.tree.join("2025/jxl.jxl")).unwrap();

    let run = env.index(&[env.tree.join("2024").to_str().unwrap()]);
    assert_eq!(count(&run, "files"), 6);
    assert_eq!(count(&run, "vanished"), 0);
    assert_eq!(count(&env.status(), "missing"), 0);

    let run = env.index(&[]);
    assert_eq!(count(&run, "vanished"), 1);
    assert_eq!(env.row("2025/jxl.jxl").unwrap().status, "missing");
}

#[test]
fn path_without_roots_indexes_ad_hoc() {
    let env = Env::new(false);
    let before = env.snapshot();
    let run = env.index(&[env.tree.to_str().unwrap()]);
    assert_eq!(count(&run, "added"), IMAGES + 1);
    assert_eq!(env.snapshot(), before);
}

#[test]
fn no_path_and_no_roots_is_an_error() {
    let env = Env::new(false);
    let output = env
        .vitrine(&["index"])
        .assert()
        .failure()
        .get_output()
        .clone();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("no roots configured"), "{stderr}");
    assert!(!env.data_dir().join("catalog.db").exists());
}

#[test]
fn held_lock_fails_the_run() {
    let env = Env::new(true);
    let _lock = IndexLock::acquire(&env.data_dir().join("index.lock")).unwrap();
    let output = env
        .vitrine(&["index"])
        .assert()
        .failure()
        .get_output()
        .clone();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("another indexer is running"), "{stderr}");
}

#[test]
fn unreadable_header_is_recorded_once() {
    let env = Env::new(true);
    fs::write(env.tree.join("2024/broken.jpg"), b"not an image").unwrap();
    let output = env
        .vitrine(&["index"])
        .assert()
        .success()
        .get_output()
        .clone();
    let run = fields(&output.stdout);
    assert_eq!(count(&run, "failed"), 2);
    assert_eq!(count(&run, "thumbnailed"), IMAGES);
    assert!(String::from_utf8_lossy(&output.stderr).contains("broken.jpg"));
    let row = env.row("2024/broken.jpg").unwrap();
    assert!(row.sha256.is_some());
    assert_eq!((row.width, row.height), (None, None));

    let run = env.index(&[]);
    assert_eq!((count(&run, "hashed"), count(&run, "failed")), (0, 1));
}

#[test]
fn gps_resolves_against_the_gazetteer() {
    const ADMIN1: &str = "PT.14\tLisbon\tLisbon\t2267056\nPT.17\tPorto\tPorto\t2735941\n";
    const COUNTRIES: &str = "PT\tPRT\t620\tPO\tPortugal\tLisbon\n";
    const CITIES: &str = "\
2267057\tLisbon\tLisbon\tLisboa\t38.71667\t-9.13333\tP\tPPLC\tPT\t\t14\t\t\t\t517802\t\t45\tEurope/Lisbon\t2024-01-01
2735943\tPorto\tPorto\t\t41.14961\t-8.61099\tP\tPPLA\tPT\t\t17\t\t\t\t249633\t\t104\tEurope/Lisbon\t2024-01-01
";
    let env = Env::new(true);
    fs::create_dir_all(env.tree.join("2026")).unwrap();
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gps.jpg"),
        env.tree.join("2026/gps.jpg"),
    )
    .unwrap();
    let status = env.status();
    assert_eq!(
        status["gazetteer"], "not fetched; run vitrine geo fetch",
        "{status:?}"
    );

    fs::create_dir_all(env.data_dir()).unwrap();
    let places = vitrine::geo::build(
        &env.data_dir().join("geo.db"),
        vec![vitrine::geo::Rows::all("cities1000.txt", CITIES.as_bytes())],
        ADMIN1.as_bytes(),
        COUNTRIES.as_bytes(),
        1,
    )
    .unwrap();
    assert_eq!(places, 2);

    let run = env.index(&[]);
    assert_eq!(count(&run, "failed"), 1);
    let status = env.status();
    assert_eq!(count(&status, "located (exif)"), 1);
    assert_eq!(count(&status, "located (manual)"), 0);
    assert_eq!(count(&status, "gps pending"), 0);
    assert_eq!(count(&status, "gps found"), 1);
    assert_eq!(count(&status, "places resolved"), 1);
    let gazetteer = &status["gazetteer"];
    assert!(gazetteer.ends_with(", 2 places"), "{gazetteer}");
    assert_eq!(gazetteer.len(), "2026-10-11 12:00:00, 2 places".len());
}
