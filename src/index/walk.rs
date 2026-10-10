use std::collections::BTreeMap;
use std::fs::Metadata;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::UNIX_EPOCH;

use walkdir::{DirEntry, WalkDir};

use super::Failure;
use crate::catalog::images::Kind;
use crate::decode::RAW_EXTENSIONS;

const IMAGE_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "webp", "gif", "tif", "tiff", "heic", "heif", "avif", "jxl",
];

const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mov", "m4v", "mkv", "avi", "webm", "mts", "m2ts", "3gp",
];

#[derive(Debug, Clone)]
pub(crate) struct Found {
    pub size: u64,
    pub mtime_ns: i64,
    pub kind: Kind,
}

#[derive(Debug, Default)]
pub(crate) struct Walk {
    pub files: BTreeMap<PathBuf, Found>,
    /// Entries that couldn't be read; rows at or below them are left alone.
    pub unreadable: Vec<PathBuf>,
    pub failures: Vec<Failure>,
}

pub(crate) fn lowercase_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
}

pub(crate) fn classify(path: &Path) -> Option<Kind> {
    let extension = lowercase_extension(path)?;
    let extension = extension.as_str();
    if IMAGE_EXTENSIONS.contains(&extension) || RAW_EXTENSIONS.contains(&extension) {
        Some(Kind::Image)
    } else if VIDEO_EXTENSIONS.contains(&extension) {
        Some(Kind::Video)
    } else {
        None
    }
}

fn is_hidden(entry: &DirEntry) -> bool {
    entry.depth() > 0 && entry.file_name().as_encoded_bytes().starts_with(b".")
}

/// Skips symlinks and hidden entries.
pub(crate) fn walk(root: &Path, walk: &mut Walk, cancel: &AtomicBool) {
    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !is_hidden(e))
    {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        let entry = match entry {
            Ok(entry) => entry,
            Err(e) => {
                let path = e.path().unwrap_or(root).to_owned();
                walk.unreadable.push(path.clone());
                walk.failures.push(Failure {
                    path,
                    error: e.to_string(),
                });
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let Some(kind) = classify(entry.path()) else {
            continue;
        };
        if entry.path().to_str().is_none() {
            walk.failures.push(Failure {
                path: entry.into_path(),
                error: "path is not valid UTF-8".to_owned(),
            });
            continue;
        }
        match entry.metadata() {
            Ok(meta) => {
                let found = Found {
                    size: meta.len(),
                    mtime_ns: mtime_ns(&meta),
                    kind,
                };
                walk.files.insert(entry.into_path(), found);
            }
            Err(e) => {
                walk.unreadable.push(entry.path().to_owned());
                walk.failures.push(Failure {
                    path: entry.into_path(),
                    error: e.to_string(),
                });
            }
        }
    }
}

/// Negative before the epoch, 0 when the platform has no mtime.
fn mtime_ns(meta: &Metadata) -> i64 {
    let Ok(modified) = meta.modified() else {
        return 0;
    };
    let nanos = |d: std::time::Duration| i64::try_from(d.as_nanos()).unwrap_or(i64::MAX);
    match modified.duration_since(UNIX_EPOCH) {
        Ok(after) => nanos(after),
        Err(before) => -nanos(before.duration()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn classifies_by_extension() {
        for (name, kind) in [
            ("a.JPG", Some(Kind::Image)),
            ("a.jxl", Some(Kind::Image)),
            ("a.ORF", Some(Kind::Image)),
            ("a.cr3", None),
            ("a.mov", Some(Kind::Video)),
            ("a.m2ts", Some(Kind::Video)),
            ("a.txt", None),
            ("jpg", None),
        ] {
            assert_eq!(classify(Path::new(name)), kind, "{name}");
        }
    }

    #[test]
    fn skips_hidden_symlinks_and_other_files() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("a/.cache")).unwrap();
        fs::write(root.join("a/x.jpg"), b"x").unwrap();
        fs::write(root.join("a/clip.mp4"), b"v").unwrap();
        fs::write(root.join("a/notes.txt"), b"t").unwrap();
        fs::write(root.join("a/.hidden.jpg"), b"h").unwrap();
        fs::write(root.join("a/.cache/y.jpg"), b"y").unwrap();
        std::os::unix::fs::symlink(root.join("a/x.jpg"), root.join("a/link.jpg")).unwrap();
        std::os::unix::fs::symlink(root.join("a"), root.join("b")).unwrap();

        let mut w = Walk::default();
        walk(root, &mut w, &AtomicBool::new(false));
        let names: Vec<_> = w
            .files
            .keys()
            .map(|p| p.strip_prefix(root).unwrap().to_str().unwrap())
            .collect();
        assert_eq!(names, ["a/clip.mp4", "a/x.jpg"]);
        assert_eq!(w.files[&root.join("a/clip.mp4")].kind, Kind::Video);
        assert_eq!(w.files[&root.join("a/x.jpg")].size, 1);
        assert!(w.failures.is_empty());
    }
}
