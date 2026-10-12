use std::fs::File;
use std::io::{self, BufReader, Cursor, Read};
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, anyhow};
use image::{GenericImageView, ImageReader};
use sha2::Digest;

use super::walk::lowercase_extension;
use crate::db::Sha256;
use crate::decode::RAW_EXTENSIONS;
use crate::video::VideoMeta;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Meta {
    /// Width and height with EXIF orientation applied.
    pub dimensions: Option<(u32, u32)>,
    /// EXIF `DateTimeOriginal` as `%Y-%m-%dT%H:%M:%S`.
    pub taken_at: Option<String>,
    pub duration_ms: Option<u64>,
    pub codec: Option<String>,
}

impl From<VideoMeta> for Meta {
    fn from(video: VideoMeta) -> Self {
        Self {
            dimensions: video.width.zip(video.height),
            taken_at: video.taken_at,
            duration_ms: video.duration_ms,
            codec: video.codec,
        }
    }
}

/// `None` once `cancel` is set, checked between chunks.
pub(crate) fn sha256(path: &Path, cancel: &AtomicBool) -> io::Result<Option<Sha256>> {
    let mut file = File::open(path)?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = vec![0; 1 << 16];
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Ok(None);
        }
        match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => hasher.update(&buf[..n]),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(Some(hasher.finalize().into()))
}

/// The error says why dimensions are unknown; capture time is kept with it, except after a panic.
pub(crate) fn read(path: &Path) -> (Meta, Option<anyhow::Error>) {
    // quickraw and the header readers can panic on offsets read from a malformed file
    panic::catch_unwind(AssertUnwindSafe(|| read_unguarded(path))).unwrap_or_else(|_| {
        (
            Meta::default(),
            Some(anyhow!("metadata reader panicked on {}", path.display())),
        )
    })
}

fn read_unguarded(path: &Path) -> (Meta, Option<anyhow::Error>) {
    let exif = File::open(path).ok().and_then(|f| {
        exif::Reader::new()
            .read_from_container(&mut BufReader::new(f))
            .ok()
    });
    let taken_at = exif.as_ref().and_then(taken_at);
    let orientation = exif.as_ref().and_then(orientation).unwrap_or(1);
    let dimensions = header_dimensions(path, orientation)
        .or_else(|_| crate::decode::decode(path).map(|image| image.dimensions()));
    match dimensions {
        Ok(dimensions) => (
            Meta {
                dimensions: Some(dimensions),
                taken_at,
                ..Meta::default()
            },
            None,
        ),
        Err(e) => (
            Meta {
                dimensions: None,
                taken_at,
                ..Meta::default()
            },
            Some(e),
        ),
    }
}

fn header_dimensions(path: &Path, orientation: u32) -> Result<(u32, u32)> {
    let extension = lowercase_extension(path).unwrap_or_default();
    let (width, height) = match extension.as_str() {
        // libheif and jxl-oxide already apply the orientation
        "heic" | "heif" | "avif" => {
            use libheif_rs::{HeifContext, LibHeif};
            let _lib = LibHeif::new();
            let ctx = HeifContext::read_from_file(&path.to_string_lossy())?;
            let handle = ctx.primary_image_handle()?;
            return Ok((handle.width(), handle.height()));
        }
        "jxl" => {
            let image = jxl_oxide::JxlImage::builder()
                .open(path)
                .map_err(|e| anyhow!("{e}"))?;
            return Ok((image.width(), image.height()));
        }
        e if RAW_EXTENSIONS.contains(&e) => {
            let data = std::fs::read(path)?;
            let (preview, _) = quickraw::Export::export_thumbnail_data(&data)?;
            ImageReader::new(Cursor::new(preview))
                .with_guessed_format()?
                .into_dimensions()
                .context("embedded preview")?
        }
        _ => ImageReader::open(path)?
            .with_guessed_format()?
            .into_dimensions()?,
    };
    Ok(if (5..=8).contains(&orientation) {
        (height, width)
    } else {
        (width, height)
    })
}

fn orientation(exif: &exif::Exif) -> Option<u32> {
    exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)?
        .value
        .get_uint(0)
}

fn taken_at(exif: &exif::Exif) -> Option<String> {
    let field = exif.get_field(exif::Tag::DateTimeOriginal, exif::In::PRIMARY)?;
    let exif::Value::Ascii(values) = &field.value else {
        return None;
    };
    let t = exif::DateTime::from_ascii(values.first()?).ok()?;
    let valid = t.year > 0
        && (1..=12).contains(&t.month)
        && (1..=31).contains(&t.day)
        && t.hour < 24
        && t.minute < 60
        && t.second < 61;
    valid.then(|| {
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            t.year, t.month, t.day, t.hour, t.minute, t.second
        )
    })
}

/// JPEGs with a hand-built little-endian EXIF block, spliced into the `jpeg.jpg` fixture.
#[cfg(test)]
pub(crate) mod exif_jpeg {
    use std::path::Path;

    pub(crate) const EXIF_IFD: u16 = 0x8769;
    pub(crate) const GPS_IFD: u16 = 0x8825;

    pub(crate) enum Value<'a> {
        /// Including the trailing NUL.
        Ascii(&'a [u8]),
        Rational(&'a [(u32, u32)]),
        Ifd(&'a [(u16, Value<'a>)]),
    }

    /// `ifd0` entries, and those of nested IFDs, in ascending tag order.
    pub(crate) fn build(ifd0: &[(u16, Value<'_>)]) -> Vec<u8> {
        let mut tiff = b"II\x2a\x00\x08\x00\x00\x00".to_vec();
        write_ifd(&mut tiff, ifd0);
        let jpeg =
            std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/jpeg.jpg"))
                .unwrap();
        let length = u16::try_from(2 + 6 + tiff.len()).unwrap();
        let mut out = jpeg[..2].to_vec();
        out.extend([0xFF, 0xE1]);
        out.extend(length.to_be_bytes());
        out.extend(b"Exif\0\0");
        out.extend(tiff);
        out.extend(&jpeg[2..]);
        out
    }

    fn write_ifd(tiff: &mut Vec<u8>, entries: &[(u16, Value<'_>)]) -> u32 {
        align(tiff);
        let start = tiff.len();
        tiff.extend(u16::try_from(entries.len()).unwrap().to_le_bytes());
        tiff.resize(start + 2 + 12 * entries.len() + 4, 0);
        for (i, (tag, value)) in entries.iter().enumerate() {
            let (kind, count, field) = match value {
                Value::Ascii(s) => (2u16, s.len(), place(tiff, s)),
                Value::Rational(r) => {
                    let bytes: Vec<u8> = r
                        .iter()
                        .flat_map(|(n, d)| n.to_le_bytes().into_iter().chain(d.to_le_bytes()))
                        .collect();
                    (5, r.len(), place(tiff, &bytes))
                }
                Value::Ifd(sub) => (4, 1, write_ifd(tiff, sub).to_le_bytes()),
            };
            let at = start + 2 + 12 * i;
            tiff[at..at + 2].copy_from_slice(&tag.to_le_bytes());
            tiff[at + 2..at + 4].copy_from_slice(&kind.to_le_bytes());
            tiff[at + 4..at + 8].copy_from_slice(&u32::try_from(count).unwrap().to_le_bytes());
            tiff[at + 8..at + 12].copy_from_slice(&field);
        }
        u32::try_from(start).unwrap()
    }

    /// Up to four bytes go inline in the entry; longer values get an offset.
    fn place(tiff: &mut Vec<u8>, bytes: &[u8]) -> [u8; 4] {
        if bytes.len() <= 4 {
            let mut field = [0; 4];
            field[..bytes.len()].copy_from_slice(bytes);
            return field;
        }
        align(tiff);
        let offset = u32::try_from(tiff.len()).unwrap();
        tiff.extend(bytes);
        offset.to_le_bytes()
    }

    fn align(tiff: &mut Vec<u8>) {
        if tiff.len() % 2 == 1 {
            tiff.push(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::exif_jpeg::{self, Value};
    use super::*;

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

    #[test]
    fn header_dimensions_match_decoded_orientation() {
        for (name, dimensions) in FIXTURES {
            let (meta, error) = read(&fixture(name));
            assert!(error.is_none(), "{name}: {error:?}");
            assert_eq!(meta.dimensions, Some(*dimensions), "{name}");
        }
    }

    #[test]
    fn every_fixture_has_a_readable_header() {
        for (name, _) in FIXTURES {
            let path = fixture(name);
            let exif = File::open(&path).ok().and_then(|f| {
                exif::Reader::new()
                    .read_from_container(&mut BufReader::new(f))
                    .ok()
            });
            let orientation = exif.as_ref().and_then(orientation).unwrap_or(1);
            header_dimensions(&path, orientation).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        }
    }

    fn taken(value: &[u8]) -> Vec<u8> {
        exif_jpeg::build(&[(
            exif_jpeg::EXIF_IFD,
            Value::Ifd(&[(0x9003, Value::Ascii(value))]),
        )])
    }

    #[test]
    fn reads_date_time_original() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("taken.jpg");
        std::fs::write(&path, taken(b"2024:05:01 10:11:12\0")).unwrap();
        let (meta, error) = read(&path);
        assert!(error.is_none(), "{error:?}");
        assert_eq!(meta.taken_at.as_deref(), Some("2024-05-01T10:11:12"));
        assert_eq!(meta.dimensions, Some((64, 48)));

        std::fs::write(&path, taken(b"0000:00:00 00:00:00\0")).unwrap();
        assert_eq!(read(&path).0.taken_at, None);
    }

    #[test]
    fn unreadable_header_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.jpg");
        std::fs::write(&path, b"not an image").unwrap();
        let (meta, error) = read(&path);
        assert_eq!(meta, Meta::default());
        assert!(error.is_some());
    }

    #[test]
    fn sha256_of_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a");
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(
            sha256(&path, &AtomicBool::new(false)).unwrap().unwrap()[..4],
            [0xba, 0x78, 0x16, 0xbf],
            "sha256(\"abc\") starts with ba7816bf"
        );
    }

    #[test]
    fn sha256_stops_once_cancelled() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a");
        std::fs::write(&path, vec![0; 1 << 17]).unwrap();
        assert_eq!(sha256(&path, &AtomicBool::new(true)).unwrap(), None);
    }

    #[test]
    fn meta_from_video_meta() {
        let meta = Meta::from(VideoMeta {
            duration_ms: Some(1000),
            width: Some(56),
            height: Some(72),
            codec: Some("h264".into()),
            taken_at: Some("2024-05-01T13:00:00".into()),
        });
        assert_eq!(meta.dimensions, Some((56, 72)));
        assert_eq!(meta.duration_ms, Some(1000));
        assert_eq!(meta.codec.as_deref(), Some("h264"));
        assert_eq!(meta.taken_at.as_deref(), Some("2024-05-01T13:00:00"));
    }
}
