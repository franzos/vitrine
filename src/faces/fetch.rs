use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use sha2::{Digest, Sha256};

pub const LICENSE_NOTE: &str = "\
The face models are InsightFace buffalo_l (det_10g.onnx, w600k_r50.onnx).
InsightFace's license for them reads:

  \"The training data containing the annotation (and the models trained with
  these data) are available for non-commercial research purposes only.\"

  https://github.com/deepinsight/insightface#license

vitrine downloads them on request and never bundles or redistributes them.";

/// A mirror of the files in InsightFace's `buffalo_l.zip` (release v0.7), pinned to a revision.
pub const BASE_URL: &str =
    "https://huggingface.co/immich-app/buffalo_l/resolve/d09715916a0778919a770c343533641e250b8699";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelFile<'a> {
    pub name: &'a str,
    pub remote: &'a str,
    pub size: u64,
    /// Lowercase hex.
    pub sha256: &'a str,
}

pub const DETECTOR: ModelFile<'static> = ModelFile {
    name: "det_10g.onnx",
    remote: "detection/model.onnx",
    size: 16_923_827,
    sha256: "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
};

pub const EMBEDDER: ModelFile<'static> = ModelFile {
    name: "w600k_r50.onnx",
    remote: "recognition/model.onnx",
    size: 174_383_860,
    sha256: "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
};

/// The `face_models` name; identifies the embedder.
pub const MODEL_NAME: &str = "buffalo_l/w600k_r50";
pub const DETECTOR_NAME: &str = "buffalo_l/det_10g";
pub const EMBEDDING_DIM: usize = 512;

#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("failed to download {url}")]
    Http {
        url: String,
        #[source]
        source: Box<ureq::Error>,
    },
    #[error("failed to write {}", path.display())]
    Io { path: PathBuf, source: io::Error },
    #[error("{name}: sha256 {actual} does not match the pinned {expected}")]
    HashMismatch {
        name: String,
        expected: String,
        actual: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Downloaded,
    Present,
}

pub fn fetch(dir: &Path) -> Result<Vec<(&'static str, Outcome)>, FetchError> {
    fetch_files(BASE_URL, dir, &[DETECTOR, EMBEDDER])
}

/// Skips files already present with the pinned hash.
pub fn fetch_files<'a>(
    base_url: &str,
    dir: &Path,
    files: &[ModelFile<'a>],
) -> Result<Vec<(&'a str, Outcome)>, FetchError> {
    fs::create_dir_all(dir).map_err(|source| FetchError::Io {
        path: dir.to_owned(),
        source,
    })?;
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        // without it a stalled transfer blocks forever
        .timeout_recv_body(Some(Duration::from_secs(1800)))
        .build()
        .into();
    files
        .iter()
        .map(|file| {
            let path = dir.join(file.name);
            if hash_file(&path).is_ok_and(|hash| hash == file.sha256) {
                return Ok((file.name, Outcome::Present));
            }
            download(&agent, &format!("{base_url}/{}", file.remote), file, &path)?;
            Ok((file.name, Outcome::Downloaded))
        })
        .collect()
}

fn download(
    agent: &ureq::Agent,
    url: &str,
    file: &ModelFile<'_>,
    path: &Path,
) -> Result<(), FetchError> {
    let http = |source| FetchError::Http {
        url: url.to_owned(),
        source: Box::new(source),
    };
    let part = path.with_extension("onnx.part");
    let io = |source| FetchError::Io {
        path: part.clone(),
        source,
    };
    let response = agent.get(url).call().map_err(http)?;
    let mut body = response
        .into_body()
        .into_with_config()
        // +1: ureq errors on any read at the limit, even the one that sees EOF
        .limit(file.size + 1)
        .reader();

    let result = (|| {
        let mut out = File::create(&part).map_err(io)?;
        let actual = copy_hashing(&mut body, &mut out).map_err(io)?;
        out.sync_all().map_err(io)?;
        if actual != file.sha256 {
            return Err(FetchError::HashMismatch {
                name: file.name.to_owned(),
                expected: file.sha256.to_owned(),
                actual,
            });
        }
        fs::rename(&part, path).map_err(io)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&part);
    }
    result
}

fn hash_file(path: &Path) -> io::Result<String> {
    copy_hashing(&mut File::open(path)?, &mut io::sink())
}

/// Returns the sha256 of the copied bytes as lowercase hex.
fn copy_hashing(reader: &mut impl Read, writer: &mut impl Write) -> io::Result<String> {
    let mut hasher = Sha256::new();
    let mut buf = vec![0; 1 << 16];
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        hasher.update(&buf[..n]);
        writer.write_all(&buf[..n])?;
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use super::*;

    const BODY: &[u8] = b"not really an onnx model";
    const BODY_SHA256: &str = "cc9b873391468e4c60c91aa103977b986a1e97e69c1abceb3707e844c8dfe8f0";

    fn serve() -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&hits);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap() > 2 {
                    line.clear();
                }
                counter.fetch_add(1, Ordering::SeqCst);
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    BODY.len()
                )
                .unwrap();
                stream.write_all(BODY).unwrap();
            }
        });
        (base, hits)
    }

    fn model(sha256: &str) -> ModelFile<'_> {
        ModelFile {
            name: "m.onnx",
            remote: "dir/model.onnx",
            size: BODY.len() as u64,
            sha256,
        }
    }

    #[test]
    fn body_hash_constant_is_right() {
        assert_eq!(
            copy_hashing(&mut &BODY[..], &mut io::sink()).unwrap(),
            BODY_SHA256
        );
    }

    #[test]
    fn downloads_verifies_and_then_skips() {
        let (base, hits) = serve();
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path().join("models");

        let outcome = fetch_files(&base, &models, &[model(BODY_SHA256)]).unwrap();
        assert_eq!(outcome, [("m.onnx", Outcome::Downloaded)]);
        assert_eq!(fs::read(models.join("m.onnx")).unwrap(), BODY);
        assert!(!models.join("m.onnx.part").exists());

        let outcome = fetch_files(&base, &models, &[model(BODY_SHA256)]).unwrap();
        assert_eq!(outcome, [("m.onnx", Outcome::Present)]);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn hash_mismatch_leaves_nothing_behind() {
        let (base, _) = serve();
        let dir = tempfile::tempdir().unwrap();
        let wrong = "0".repeat(64);

        let err = fetch_files(&base, dir.path(), &[model(&wrong)]).unwrap_err();
        assert!(
            matches!(&err, FetchError::HashMismatch { actual, .. } if actual == BODY_SHA256),
            "{err:?}"
        );
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn corrupt_existing_file_is_replaced() {
        let (base, hits) = serve();
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("m.onnx"), b"truncated").unwrap();

        let outcome = fetch_files(&base, dir.path(), &[model(BODY_SHA256)]).unwrap();
        assert_eq!(outcome, [("m.onnx", Outcome::Downloaded)]);
        assert_eq!(fs::read(dir.path().join("m.onnx")).unwrap(), BODY);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn oversized_body_is_rejected() {
        let (base, _) = serve();
        let dir = tempfile::tempdir().unwrap();
        let small = ModelFile {
            size: 4,
            ..model(BODY_SHA256)
        };

        let err = fetch_files(&base, dir.path(), &[small]).unwrap_err();
        assert!(matches!(err, FetchError::Io { .. }), "{err:?}");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}
