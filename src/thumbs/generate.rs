//! EXIF thumbnail when big enough, else a DCT-scaled JPEG decode, else a full decode; one decode serves every bucket.

use std::borrow::Cow;
use std::fs::{self, File};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use anyhow::Context;
use image::imageops::FilterType;
use image::metadata::Orientation;
use image::{DynamicImage, GenericImageView, RgbImage};
use rusqlite::Connection;
use turbojpeg::{Decompressor, PixelFormat, ScalingFactor, Subsamp};

use super::Thumb;
use crate::db::Sha256;

pub const BUCKETS: [u32; 2] = [256, 512];

const QUALITY: i32 = 85;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Exif,
    /// JPEG decoded at reduced DCT scale by libjpeg-turbo.
    DctScaled,
    Decoded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generated {
    pub bucket: u32,
    pub thumb: Thumb,
    pub source: Source,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot make a thumbnail of {}", path.display())]
    Generate {
        path: PathBuf,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error(transparent)]
    Db(#[from] crate::db::Error),
}

/// Ordered by bucket.
pub fn generate(path: &Path, buckets: &[u32]) -> anyhow::Result<Vec<Generated>> {
    generate_with(path, buckets, crate::decode::decode)
}

/// Like [`generate`], with `decode` doing the full decode when the image isn't a JPEG served at DCT scale.
pub fn generate_with(
    path: &Path,
    buckets: &[u32],
    decode: impl FnOnce(&Path) -> anyhow::Result<DynamicImage>,
) -> anyhow::Result<Vec<Generated>> {
    let mut pending = buckets.to_vec();
    pending.sort_unstable();
    pending.dedup();
    let mut out = Vec::with_capacity(pending.len());

    let jpeg = if is_jpeg(path)? {
        Some(fs::read(path)?)
    } else {
        None
    };
    let exif = jpeg.as_deref().and_then(|data| {
        exif::Reader::new()
            .read_from_container(&mut Cursor::new(data))
            .ok()
    });
    let orientation = exif
        .as_ref()
        .map_or(Orientation::NoTransforms, crate::decode::orientation);

    if let (Some(data), Some(exif)) = (&jpeg, &exif)
        && let Some((embedded, longest)) = embedded_thumbnail(data, exif)
    {
        let usable: Vec<u32> = pending.iter().copied().filter(|&b| longest >= b).collect();
        if let Some(&target) = usable.last()
            && let Ok(rgb) = decode_jpeg(embedded, target)
        {
            emit(upright(rgb, orientation), &usable, Source::Exif, &mut out)?;
            pending.retain(|&b| longest < b);
        }
    }

    if let Some(&largest) = pending.last() {
        let (image, source) = match jpeg.as_deref().map(|data| decode_jpeg(data, largest)) {
            Some(Ok(rgb)) => (upright(rgb, orientation), Source::DctScaled),
            // CMYK and damaged JPEGs go through the full decoder, which handles both
            _ => (decode(path)?, Source::Decoded),
        };
        emit(image, &pending, source, &mut out)?;
    }
    out.sort_by_key(|g| g.bucket);
    Ok(out)
}

/// On a miss, makes and stores every missing bucket in [`BUCKETS`].
/// Safe to call from several threads, each with its own connection.
pub fn ensure(
    conn: &Connection,
    path: &Path,
    sha256: &Sha256,
    bucket: u32,
) -> Result<Thumb, Error> {
    if let Some(thumb) = super::get(conn, sha256, bucket)? {
        return Ok(thumb);
    }
    let stored = super::buckets(conn, sha256)?;
    let mut wanted = vec![bucket];
    wanted.extend(BUCKETS.iter().filter(|b| !stored.contains(b)));
    let generated = generate(path, &wanted).map_err(|e| Error::Generate {
        path: path.to_owned(),
        source: e.into(),
    })?;

    let tx = conn
        .unchecked_transaction()
        .map_err(crate::db::Error::from)?;
    for g in &generated {
        super::put(&tx, sha256, g.bucket, &g.thumb)?;
    }
    tx.commit().map_err(crate::db::Error::from)?;
    generated
        .into_iter()
        .find(|g| g.bucket == bucket)
        .map(|g| g.thumb)
        .ok_or_else(|| Error::Generate {
            path: path.to_owned(),
            source: format!("no thumbnail for bucket {bucket}").into(),
        })
}

fn is_jpeg(path: &Path) -> std::io::Result<bool> {
    let mut magic = Vec::with_capacity(3);
    File::open(path)?.take(3).read_to_end(&mut magic)?;
    Ok(magic == [0xFF, 0xD8, 0xFF])
}

/// The EXIF thumbnail of `data` and its longest side, when its aspect ratio is within 2% of the main image's.
fn embedded_thumbnail<'e>(data: &[u8], exif: &'e exif::Exif) -> Option<(&'e [u8], u32)> {
    let field = |tag| {
        exif.get_field(tag, exif::In::THUMBNAIL)?
            .value
            .get_uint(0)
            .and_then(|v| usize::try_from(v).ok())
    };
    let offset = field(exif::Tag::JPEGInterchangeFormat)?;
    let length = field(exif::Tag::JPEGInterchangeFormatLength)?;
    let embedded = exif.buf().get(offset..offset.checked_add(length)?)?;

    let mut decompressor = Decompressor::new().ok()?;
    let main = decompressor.read_header(data).ok()?;
    let thumb = decompressor.read_header(embedded).ok()?;
    // cameras pad thumbnails to 4:3 or 16:9 with black bars; those don't match the image
    let (main_w, main_h) = (main.width as u64, main.height as u64);
    let (thumb_w, thumb_h) = (thumb.width as u64, thumb.height as u64);
    let skew = (main_w * thumb_h).abs_diff(thumb_w * main_h);
    if skew * 50 > main_w * thumb_h {
        return None;
    }
    let longest = u32::try_from(thumb.width.max(thumb.height)).ok()?;
    Some((embedded, longest))
}

/// Decodes `data` at the smallest DCT scale whose longest side is still at least `target`.
fn decode_jpeg(data: &[u8], target: u32) -> anyhow::Result<RgbImage> {
    let mut decompressor = Decompressor::new()?;
    let header = decompressor.read_header(data)?;
    let longest = header.width.max(header.height);
    let target = usize::try_from(target)?;
    let factor = if header.is_lossless {
        ScalingFactor::ONE
    } else {
        let mut factors: Vec<ScalingFactor> = Decompressor::supported_scaling_factors()
            .into_iter()
            .filter(|f| f.num() <= f.denom())
            .collect();
        factors.sort_by(|a, b| (a.num() * b.denom()).cmp(&(b.num() * a.denom())));
        factors
            .into_iter()
            .find(|f| f.scale(longest) >= target)
            .unwrap_or(ScalingFactor::ONE)
    };
    decompressor.set_scaling_factor(factor)?;
    let scaled = header.scaled(factor);
    let (width, height) = (u32::try_from(scaled.width)?, u32::try_from(scaled.height)?);
    crate::decode::check_dimensions(width, height)?;
    let mut image = turbojpeg::Image {
        pixels: vec![0; 3 * scaled.width * scaled.height],
        width: scaled.width,
        pitch: 3 * scaled.width,
        height: scaled.height,
        format: PixelFormat::RGB,
    };
    decompressor.decompress(data, image.as_deref_mut())?;
    RgbImage::from_raw(width, height, image.pixels).context("decoded buffer has the wrong size")
}

fn upright(rgb: RgbImage, orientation: Orientation) -> DynamicImage {
    let mut image = DynamicImage::ImageRgb8(rgb);
    crate::decode::reorient(&mut image, orientation);
    image
}

/// Shrinks `image` to each of `buckets`, largest first, each from the previous one.
fn emit(
    mut image: DynamicImage,
    buckets: &[u32],
    source: Source,
    out: &mut Vec<Generated>,
) -> anyhow::Result<()> {
    for &bucket in buckets.iter().rev() {
        image = fit(image, bucket);
        out.push(Generated {
            bucket,
            thumb: encode(&image)?,
            source,
        });
    }
    Ok(())
}

/// Scales `image` down so its longest side is `bucket`; smaller images are left alone.
fn fit(image: DynamicImage, bucket: u32) -> DynamicImage {
    let (width, height) = image.dimensions();
    let longest = width.max(height);
    if longest <= bucket {
        return image;
    }
    let shorter = |side: u32| {
        let scaled =
            (u64::from(side) * u64::from(bucket) + u64::from(longest) / 2) / u64::from(longest);
        u32::try_from(scaled).unwrap_or(bucket).max(1)
    };
    let (width, height) = if width >= height {
        (bucket, shorter(height))
    } else {
        (shorter(width), bucket)
    };
    image.resize_exact(width, height, FilterType::CatmullRom)
}

pub(crate) fn encode(image: &DynamicImage) -> anyhow::Result<Thumb> {
    let rgb = match image {
        DynamicImage::ImageRgb8(rgb) => Cow::Borrowed(rgb),
        other => Cow::Owned(other.to_rgb8()),
    };
    let (width, height) = rgb.dimensions();
    let pixels = turbojpeg::Image {
        pixels: rgb.as_raw().as_slice(),
        width: usize::try_from(width)?,
        pitch: 3 * usize::try_from(width)?,
        height: usize::try_from(height)?,
        format: PixelFormat::RGB,
    };
    let jpeg = turbojpeg::compress(pixels, QUALITY, Subsamp::Sub2x2)?;
    Ok(Thumb {
        jpeg: jpeg.to_vec(),
        width,
        height,
    })
}

#[cfg(test)]
mod tests {
    use std::thread;

    use sha2::Digest;

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
        ("exif-thumb.jpg", (768, 1024)),
    ];

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    fn sha256(path: &Path) -> Sha256 {
        sha2::Sha256::digest(fs::read(path).unwrap()).into()
    }

    fn expected((width, height): (u32, u32), bucket: u32) -> (u32, u32) {
        let longest = width.max(height);
        if longest <= bucket {
            return (width, height);
        }
        let scale = |side: u32| (side * bucket + longest / 2) / longest;
        (scale(width), scale(height))
    }

    fn check(name: &str, generated: &Generated, original: (u32, u32)) {
        let decoded =
            image::load_from_memory_with_format(&generated.thumb.jpeg, image::ImageFormat::Jpeg)
                .unwrap_or_else(|e| panic!("{name}@{}: {e}", generated.bucket));
        let size = expected(original, generated.bucket);
        assert_eq!(decoded.dimensions(), size, "{name}@{}", generated.bucket);
        assert_eq!(
            (generated.thumb.width, generated.thumb.height),
            size,
            "{name}@{}",
            generated.bucket
        );
        assert_eq!(
            size.0.max(size.1),
            generated.bucket.min(original.0.max(original.1)),
            "{name}@{}",
            generated.bucket
        );
    }

    #[test]
    fn every_fixture_yields_a_jpeg_per_bucket() {
        for (name, original) in FIXTURES {
            let path = fixture(name);
            let before = sha256(&path);
            let generated = generate(&path, &BUCKETS).unwrap_or_else(|e| panic!("{name}: {e:#}"));
            assert_eq!(
                generated.iter().map(|g| g.bucket).collect::<Vec<_>>(),
                BUCKETS,
                "{name}"
            );
            for g in &generated {
                check(name, g, *original);
            }
            assert_eq!(sha256(&path), before, "{name}");
        }
    }

    #[test]
    fn jpeg_sources() {
        let sources = |name| {
            generate(&fixture(name), &BUCKETS)
                .unwrap()
                .iter()
                .map(|g| g.source)
                .collect::<Vec<_>>()
        };
        // the 320x240 EXIF thumbnail covers 256, not 512
        assert_eq!(sources("exif-thumb.jpg"), [Source::Exif, Source::DctScaled]);
        assert_eq!(sources("jpeg.jpg"), [Source::DctScaled, Source::DctScaled]);
        assert_eq!(sources("png.png"), [Source::Decoded, Source::Decoded]);
    }

    #[test]
    fn exif_thumbnail_alone_needs_no_decode() {
        let generated = generate(&fixture("exif-thumb.jpg"), &[256]).unwrap();
        assert_eq!(generated.len(), 1);
        assert_eq!(generated[0].source, Source::Exif);
        // turned upright like the image
        assert_eq!(
            (generated[0].thumb.width, generated[0].thumb.height),
            (192, 256)
        );
    }

    #[test]
    fn dct_scaling_stops_at_the_bucket() {
        let data = fs::read(fixture("exif-thumb.jpg")).unwrap();
        assert_eq!(decode_jpeg(&data, 512).unwrap().dimensions(), (512, 384));
        assert_eq!(decode_jpeg(&data, 300).unwrap().dimensions(), (384, 288));
        assert_eq!(decode_jpeg(&data, 2000).unwrap().dimensions(), (1024, 768));
    }

    #[test]
    fn large_image_is_resized_without_dct() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wide.png");
        image::RgbImage::from_pixel(600, 300, image::Rgb([10, 200, 30]))
            .save(&path)
            .unwrap();
        let generated = generate(&path, &BUCKETS).unwrap();
        for g in &generated {
            assert_eq!(g.source, Source::Decoded);
            check("wide.png", g, (600, 300));
        }
    }

    #[test]
    fn undecodable_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.jpg");
        fs::write(&path, b"\xFF\xD8\xFFnot really").unwrap();
        assert!(generate(&path, &BUCKETS).is_err());
    }

    #[test]
    fn ensure_stores_every_bucket_once() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("thumbs.db");
        let path = fixture("exif-thumb.jpg");
        let key = sha256(&path);

        let conn = super::super::open(&db).unwrap();
        let thumb = ensure(&conn, &path, &key, 512).unwrap();
        assert_eq!((thumb.width, thumb.height), (384, 512));
        assert_eq!(super::super::buckets(&conn, &key).unwrap(), BUCKETS);

        // served from the cache: the file isn't read again
        let moved = dir.path().join("gone.jpg");
        let cached = ensure(&conn, &moved, &key, 256).unwrap();
        assert_eq!((cached.width, cached.height), (192, 256));
        assert!(matches!(
            ensure(&conn, &moved, &[0; 32], 256),
            Err(Error::Generate { .. })
        ));
    }

    #[test]
    fn ensure_from_several_threads() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("thumbs.db");
        super::super::open(&db).unwrap();
        thread::scope(|s| {
            for (name, original) in FIXTURES.iter().cycle().take(FIXTURES.len() * 3) {
                let db = &db;
                s.spawn(move || {
                    let conn = super::super::open(db).unwrap();
                    let path = fixture(name);
                    let thumb = ensure(&conn, &path, &sha256(&path), 256).unwrap();
                    assert_eq!(
                        (thumb.width, thumb.height),
                        expected(*original, 256),
                        "{name}"
                    );
                });
            }
        });
        let conn = super::super::open(&db).unwrap();
        let rows: usize = conn
            .query_row("SELECT count(*) FROM thumbs", [], |r| r.get(0))
            .unwrap();
        assert_eq!(rows, FIXTURES.len() * BUCKETS.len());
    }
}
