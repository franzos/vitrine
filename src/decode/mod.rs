// Copied from oculante, https://github.com/woelper/oculante (src/utils.rs).
// Copyright (c) 2020 Johann Woelper. MIT license, full text in NOTICE.

mod loader;

use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::thread;

use image::DynamicImage;

pub(crate) use loader::{RAW_EXTENSIONS, check_dimensions};
pub use loader::{open_image, orientation, reorient};

#[derive(Debug)]
pub struct Frame {
    pub path: PathBuf,
    pub image: anyhow::Result<Arc<DynamicImage>>,
}

pub fn decode(path: &Path) -> anyhow::Result<DynamicImage> {
    // quickraw and others index into buffers at offsets read from the file
    panic::catch_unwind(AssertUnwindSafe(|| open_image(path))).unwrap_or_else(|payload| {
        Err(anyhow::anyhow!(
            "decoder panicked on {}: {}",
            path.display(),
            panic_message(payload.as_ref())
        ))
    })
}

pub(crate) fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(ToString::to_string)
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_default()
}

/// Setting `stop` before the decode finishes drops the result and skips `repaint`.
pub fn send_image_threaded(
    img_location: &Path,
    texture_sender: Sender<Frame>,
    stop: Arc<AtomicBool>,
    repaint: impl FnOnce() + Send + 'static,
) {
    let path = img_location.to_path_buf();
    thread::spawn(move || {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        let image = decode(&path).map(Arc::new);
        if stop.load(Ordering::Relaxed) {
            return;
        }
        _ = texture_sender.send(Frame { path, image });
        repaint();
    });
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc::channel;
    use std::time::Duration;

    use image::GenericImageView;
    use sha2::{Digest, Sha256};

    use super::{decode, send_image_threaded};

    const FIXTURES: &[(&str, (u32, u32))] = &[
        ("jpeg.jpg", (64, 48)),
        ("orientation-6.jpg", (48, 64)),
        ("png.png", (60, 44)),
        ("webp.webp", (56, 40)),
        ("gif.gif", (52, 36)),
        ("tiff.tif", (48, 32)),
        ("heic.heic", (68, 52)),
        ("avif.avif", (70, 54)),
        ("jxl.jxl", (74, 58)),
        ("dng.dng", (80, 60)),
        ("dng-orientation-6.dng", (56, 84)),
    ];

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    fn sha256(path: &Path) -> Vec<u8> {
        Sha256::digest(std::fs::read(path).unwrap()).to_vec()
    }

    #[test]
    fn decodes_every_format_at_its_size() {
        for (name, dimensions) in FIXTURES {
            let image = decode(&fixture(name)).unwrap_or_else(|e| panic!("{name}: {e:#}"));
            assert_eq!(image.dimensions(), *dimensions, "{name}");
        }
    }

    #[test]
    fn decoding_leaves_fixtures_unchanged() {
        for (name, _) in FIXTURES {
            let path = fixture(name);
            let before = sha256(&path);
            decode(&path).unwrap();
            assert_eq!(sha256(&path), before, "{name}");
        }
    }

    #[test]
    fn wrong_extension_is_decoded_by_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("png.jpg");
        std::fs::copy(fixture("png.png"), &path).unwrap();
        assert_eq!(decode(&path).unwrap().dimensions(), (60, 44));
    }

    #[test]
    fn unreadable_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.jpg");
        std::fs::write(&path, b"not an image").unwrap();
        assert!(decode(&path).is_err());
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = !0u32;
        for &byte in data {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    fn huge_png() -> Vec<u8> {
        let mut ihdr = b"IHDR".to_vec();
        ihdr.extend(65535u32.to_be_bytes());
        ihdr.extend(65535u32.to_be_bytes());
        ihdr.extend([8, 6, 0, 0, 0]);
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend(13u32.to_be_bytes());
        png.extend(&ihdr);
        png.extend(crc32(&ihdr).to_be_bytes());
        for chunk in [&b"IDAT"[..], b"IEND"] {
            png.extend(0u32.to_be_bytes());
            png.extend(chunk);
            png.extend(crc32(chunk).to_be_bytes());
        }
        png
    }

    fn huge_jpeg() -> Vec<u8> {
        let mut jpeg = std::fs::read(fixture("jpeg.jpg")).unwrap();
        let sof = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        jpeg[sof + 5..sof + 9].copy_from_slice(&[0xFF; 4]);
        jpeg
    }

    fn huge_tiff() -> Vec<u8> {
        let mut tiff = b"II\x2a\x00\x08\x00\x00\x00".to_vec();
        let entries: [(u16, u16, u32); 8] = [
            (256, 4, 65535),
            (257, 4, 65535),
            (258, 3, 8),
            (259, 3, 1),
            (262, 3, 1),
            (273, 4, 8),
            (278, 4, 65535),
            (279, 4, 1),
        ];
        tiff.extend(u16::try_from(entries.len()).unwrap().to_le_bytes());
        for (tag, kind, value) in entries {
            tiff.extend(tag.to_le_bytes());
            tiff.extend(kind.to_le_bytes());
            tiff.extend(1u32.to_le_bytes());
            tiff.extend(value.to_le_bytes());
        }
        tiff.extend(0u32.to_le_bytes());
        tiff
    }

    #[test]
    fn oversized_header_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        for (name, bytes) in [
            ("huge.png", huge_png()),
            ("huge.jpg", huge_jpeg()),
            ("huge.tif", huge_tiff()),
        ] {
            let path = dir.path().join(name);
            std::fs::write(&path, bytes).unwrap();
            let error = decode(&path).expect_err(name);
            assert!(format!("{error:#}").contains("limit"), "{name}: {error:#}");
        }
    }

    #[test]
    fn decoder_panic_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.dng");
        let dng = std::fs::read(fixture("dng.dng")).unwrap();
        std::fs::write(&path, &dng[..dng.len() - 100]).unwrap();
        let error = decode(&path).unwrap_err();
        assert!(format!("{error:#}").contains("panicked"), "{error:#}");
    }

    #[test]
    fn threaded_decode_sends_frame_and_repaints() {
        let (tx, rx) = channel();
        let repaints = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&repaints);
        let path = fixture("heic.heic");
        send_image_threaded(&path, tx, Arc::new(AtomicBool::new(false)), move || {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        let frame = rx.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!(frame.path, path);
        assert_eq!(frame.image.unwrap().dimensions(), (68, 52));
        // the sender is dropped once the thread ends, after the repaint
        assert!(rx.recv_timeout(Duration::from_secs(10)).is_err());
        assert_eq!(repaints.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn threaded_decode_sends_errors() {
        let (tx, rx) = channel();
        send_image_threaded(
            &fixture("missing.png"),
            tx,
            Arc::new(AtomicBool::new(false)),
            || {},
        );
        let frame = rx.recv_timeout(Duration::from_secs(10)).unwrap();
        assert!(frame.image.is_err());
    }

    #[test]
    fn stopped_decode_sends_nothing() {
        let (tx, rx) = channel();
        let repainted = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&repainted);
        send_image_threaded(
            &fixture("jpeg.jpg"),
            tx,
            Arc::new(AtomicBool::new(true)),
            move || flag.store(true, Ordering::SeqCst),
        );
        assert!(rx.recv_timeout(Duration::from_secs(10)).is_err());
        assert!(!repainted.load(Ordering::SeqCst));
    }
}
