// Copied from oculante, https://github.com/woelper/oculante (src/image_loader.rs).
// Copyright (c) 2020 Johann Woelper. MIT license, full text in NOTICE.

use anyhow::{Context, Result, anyhow, bail};
use file_format::FileFormat;
use image::{DynamicImage, GrayAlphaImage, ImageDecoder, ImageReader, RgbImage, RgbaImage};
use jxl_oxide::{JxlImage, PixelFormat};
use quickraw::Export;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Seek};
use std::path::Path;
use tiff::decoder::Limits;

/// Largest decode buffer in bytes (2 GiB, about 500 MP of RGBA).
const MAX_ALLOC: u64 = 2 << 30;

/// RAW extensions; quickraw reads their embedded preview (DNG, Nikon, Sony, Panasonic, Olympus, Fujifilm).
pub(crate) const RAW_EXTENSIONS: &[&str] = &["dng", "nef", "nrw", "arw", "rw2", "orf", "raf"];

/// Rejects images whose RGBA pixels exceed [`MAX_ALLOC`], before allocating.
pub(crate) fn check_dimensions(width: u32, height: u32) -> Result<()> {
    check_decode_bytes(width, height, 4)
}

fn check_decode_bytes(width: u32, height: u32, bytes_per_pixel: u64) -> Result<()> {
    let bytes = u64::from(width) * u64::from(height) * bytes_per_pixel;
    if bytes > MAX_ALLOC {
        bail!("{width}x{height} is larger than the decode limit of {MAX_ALLOC} bytes");
    }
    Ok(())
}

fn image_limits() -> image::Limits {
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_ALLOC);
    limits
}

fn tiff_limits() -> Limits {
    let mut limits = Limits::default();
    limits.decoding_buffer_size = usize::try_from(MAX_ALLOC).unwrap_or(usize::MAX);
    limits
}

/// Decoder picked by content, falling back to the extension; a missing or unreadable orientation isn't an error.
pub fn open_image(img_location: &Path) -> Result<DynamicImage> {
    let mut extension = canonical_extension(
        img_location
            .extension()
            .unwrap_or_default()
            .to_str()
            .unwrap_or_default(),
    );

    // DNG content is detected as TIFF
    let unchecked_extensions = ["dng"];

    if let Ok(fmt) = FileFormat::from_file(img_location)
        && canonical_extension(fmt.extension()) != extension
        && !unchecked_extensions.contains(&extension.as_str())
    {
        extension = fmt.extension().into();
    }

    if RAW_EXTENSIONS.contains(&extension.as_str()) {
        return load_raw(img_location);
    }

    let mut image = match extension.as_str() {
        "heif" | "heic" | "avif" => {
            // Built on work in https://github.com/rsuu/rmg - thanks!
            use libheif_rs::{ColorSpace, HeifContext, LibHeif, RgbChroma};

            let lib_heif = LibHeif::new();
            let mut ctx = HeifContext::read_from_file(&img_location.to_string_lossy())?;
            if let Ok(num_threads) = std::thread::available_parallelism() {
                ctx.set_max_decoding_threads(u32::try_from(num_threads.get()).unwrap_or(1));
            }
            let handle = ctx.primary_image_handle()?;
            check_dimensions(handle.width(), handle.height())?;
            let img = lib_heif.decode(&handle, ColorSpace::Rgb(RgbChroma::Rgba), None)?;
            let planes = img.planes();
            let interleaved = planes
                .interleaved
                .context("Can't create interleaved plane")?;

            let data = interleaved.data;
            let width = interleaved.width;
            let height = interleaved.height;
            let stride = interleaved.stride;

            let mut res: Vec<u8> = Vec::with_capacity(width as usize * height as usize * 4);
            for y in 0..height as usize {
                let row = y * stride;
                res.extend_from_slice(&data[row..row + width as usize * 4]);
            }
            let buf = image::ImageBuffer::from_vec(width, height, res)
                .context("Can't create HEIC/HEIF ImageBuffer with given res")?;
            Ok(DynamicImage::ImageRgba8(buf))
        }
        "jxl" => load_jxl(img_location),
        "jpg" | "jpeg" => match load_jpeg_turbojpeg(img_location) {
            Ok(i) => Ok(i),
            // libjpeg-turbo fails on what libjpeg only warns about (e.g. a missing EOI); `image` tolerates it
            Err(turbo_error) => decode_limited(ImageReader::open(img_location)?)
                .map_err(|e| anyhow!("{e} (libjpeg-turbo: {turbo_error})")),
        },
        "tif" | "tiff" => match load_tiff(img_location) {
            Ok(buf) => Ok(buf),
            Err(tiff_error) => match load_raw(img_location) {
                Ok(buf) => return Ok(buf),
                Err(raw_error) => {
                    bail!(
                        "Could not load tiff: {tiff_error}, tried as raw and still got error: {raw_error}"
                    )
                }
            },
        },
        // PNG, WebP and GIF (first frame) are handled by `image`
        _ => decode_limited(ImageReader::open(img_location)?.with_guessed_format()?),
    }?;
    _ = rotate_dynimage(&mut image, img_location);
    Ok(image)
}

/// Embedded preview, rotated by the RAW's orientation.
fn load_raw(img_location: &Path) -> Result<DynamicImage> {
    let raw_data = std::fs::read(img_location)?;
    let (thumbnail_data, _orientation) = Export::export_thumbnail_data(&raw_data)?;
    let mut image =
        decode_limited(ImageReader::new(Cursor::new(thumbnail_data)).with_guessed_format()?)?;
    // quickraw's orientation type is private, so the same IFD0 tag is read here
    reorient(&mut image, exif_orientation(&mut Cursor::new(&raw_data)));
    Ok(image)
}

fn tiff_decoder(path: &Path) -> Result<tiff::decoder::Decoder<BufReader<File>>> {
    Ok(tiff::decoder::Decoder::new(BufReader::new(File::open(path)?))?.with_limits(tiff_limits()))
}

/// Strips are compressed independently, so they decode in parallel, one decoder per thread.
/// 24 MP LZW: 43 ms on 20 threads vs 348 ms serial.
fn read_tiff_image(
    decoder: &mut tiff::decoder::Decoder<BufReader<File>>,
    path: &Path,
) -> Result<tiff::decoder::DecodingResult> {
    use tiff::decoder::{ChunkType, DecodingResult};

    let layout = decoder.image_buffer_layout()?;
    let (width, height) = decoder.dimensions()?;
    let (pixels, height) = (width as usize * height as usize, height as usize);
    if layout.planes > 1 {
        // Planes are stored one after another; interleave in place
        let mut result = DecodingResult::U8(vec![]);
        let layout = decoder.read_image_to_buffer(&mut result)?;
        let sample = layout.len / pixels.max(1);
        let plane_stride = layout
            .plane_stride
            .map_or(layout.len, std::num::NonZero::get);
        let mut buffer = result.as_buffer(0);
        let bytes = buffer.as_bytes_mut();
        let planar_len = plane_stride * (layout.planes - 1) + layout.len;
        if sample == 0 || layout.len != pixels * sample || bytes.len() < planar_len {
            bail!("Planar TIFF of this layout is not supported");
        }
        let planar = bytes[..planar_len].to_vec();
        let interleaved = &mut bytes[..pixels * sample * layout.planes];
        for (pixel, out) in interleaved
            .chunks_exact_mut(sample * layout.planes)
            .enumerate()
        {
            for (plane, out) in out.chunks_exact_mut(sample).enumerate() {
                let at = plane * plane_stride + pixel * sample;
                out.copy_from_slice(&planar[at..at + sample]);
            }
        }
        return Ok(result);
    }
    // Tiled and single-strip images decode serially
    let strips = if decoder.get_chunk_type() == ChunkType::Strip {
        decoder.strip_count()?
    } else {
        0
    };
    if strips < 2 || height == 0 || !layout.len.is_multiple_of(height) {
        return Ok(decoder.read_image()?);
    }
    let row_bytes = layout.len / height;

    let mut result = DecodingResult::U8(vec![]);
    result.resize_to(&layout, &tiff_limits())?;
    let mut buffer = result.as_buffer(0);
    let mut rest = &mut buffer.as_bytes_mut()[..layout.len];
    let mut pieces = Vec::with_capacity(strips as usize);
    for strip in 0..strips {
        let rows = decoder.chunk_data_dimensions(strip).1 as usize;
        if rows * row_bytes > rest.len() {
            return Ok(decoder.read_image()?);
        }
        let (piece, after) = std::mem::take(&mut rest).split_at_mut(rows * row_bytes);
        pieces.push(piece);
        rest = after;
    }
    if !rest.is_empty() {
        return Ok(decoder.read_image()?);
    }
    pieces.into_par_iter().enumerate().try_for_each_init(
        || tiff_decoder(path),
        |decoder, (strip, piece)| -> Result<()> {
            let decoder = decoder.as_mut().map_err(|e| anyhow!("{e}"))?;
            decoder.read_chunk_bytes(u32::try_from(strip)?, piece)?;
            Ok(())
        },
    )?;
    Ok(result)
}

fn load_tiff(img_location: &Path) -> Result<DynamicImage> {
    let mut decoder = tiff_decoder(img_location)?;
    let dim = decoder.dimensions()?;
    check_dimensions(dim.0, dim.1)?;
    let result = read_tiff_image(&mut decoder, img_location)?;
    use tiff::decoder::DecodingResult as D;
    let ldr_img: Vec<u8> = match result {
        D::U8(contents) => contents,
        D::U16(c) => to_ldr(&c, f32::from, (0., f32::from(u16::MAX))),
        D::F16(c) => to_ldr(&c, f32::from, (0., 1.)),
        D::U32(c) => to_ldr(&c, |p| p as f32, (0., u32::MAX as f32)),
        D::U64(c) => to_ldr(&c, |p| p as f32, (0., u64::MAX as f32)),
        D::F32(c) => to_ldr(&c, |p| p, (0., 1.)),
        D::F64(c) => to_ldr(&c, |p| p as f32, (0., 1.)),
        D::I8(c) => to_ldr(&c, f32::from, (f32::from(i8::MIN), f32::from(i8::MAX))),
        D::I16(c) => to_ldr(&c, f32::from, (f32::from(i16::MIN), f32::from(i16::MAX))),
        D::I32(c) => to_ldr(&c, |p| p as f32, (i32::MIN as f32, i32::MAX as f32)),
        D::I64(c) => to_ldr(&c, |p| p as f32, (i64::MIN as f32, i64::MAX as f32)),
    };

    match decoder.colortype()? {
        tiff::ColorType::Gray(_) => {
            let i =
                image::GrayImage::from_raw(dim.0, dim.1, ldr_img).context("Can't load gray img")?;
            Ok(DynamicImage::ImageLuma8(i))
        }
        tiff::ColorType::RGB(_) => {
            let i =
                image::RgbImage::from_raw(dim.0, dim.1, ldr_img).context("Can't load RGB img")?;
            Ok(DynamicImage::ImageRgb8(i))
        }
        tiff::ColorType::RGBA(_) => {
            let i =
                image::RgbaImage::from_raw(dim.0, dim.1, ldr_img).context("Can't load RGBA img")?;
            Ok(image::DynamicImage::ImageRgba8(i))
        }
        tiff::ColorType::GrayA(_) => {
            let i = image::GrayAlphaImage::from_raw(dim.0, dim.1, ldr_img)
                .context("Can't load gray alpha img")?;
            Ok(image::DynamicImage::ImageLumaA8(i))
        }
        other => bail!("This TIFF image type is unsupported: {other:?}"),
    }
}

/// Stretches samples linearly from their min..max to 0..=255.
/// A single-valued image maps against `type_range` instead.
fn to_ldr<T: Copy + Send + Sync>(
    samples: &[T],
    to_f32: impl Fn(T) -> f32 + Send + Sync,
    type_range: (f32, f32),
) -> Vec<u8> {
    let (lowest, highest) = samples
        .par_iter()
        .map(|p| to_f32(*p))
        .fold(
            || (f32::MAX, f32::MIN),
            |(lo, hi), v| (lo.min(v), hi.max(v)),
        )
        .reduce(|| (f32::MAX, f32::MIN), |a, b| (a.0.min(b.0), a.1.max(b.1)));
    let (lowest, highest) = if highest <= lowest {
        type_range
    } else {
        (lowest, highest)
    };
    samples
        .par_iter()
        .map(|p| fit(to_f32(*p), lowest, highest, 0., 255.).clamp(0., 255.) as u8)
        .collect()
}

fn fit(oldvalue: f32, oldmin: f32, oldmax: f32, newmin: f32, newmax: f32) -> f32 {
    (((oldvalue - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin
}

fn load_jxl(img_location: &Path) -> Result<DynamicImage> {
    let mut image = JxlImage::builder()
        .open(img_location)
        .map_err(|e| anyhow!("{e}"))?;
    let colorencoding = jxl_oxide::EnumColourEncoding::srgb(jxl_oxide::RenderingIntent::Perceptual);
    image.request_color_encoding(colorencoding);
    let channels: u64 = match image.pixel_format() {
        PixelFormat::Gray => 1,
        PixelFormat::Graya => 2,
        PixelFormat::Rgb => 3,
        PixelFormat::Rgba => 4,
        other => bail!("JXL: Pixel format: {other:?}"),
    };
    // f32 render buffer plus the u8 copy
    check_decode_bytes(image.width(), image.height(), channels * 5)?;

    let render = image
        .render_frame(0)
        .map_err(|e| anyhow!("{e}"))
        .context("Can't render JXL")?;

    let framebuffer = render.image_all_channels();
    let (width, height) = (
        u32::try_from(framebuffer.width())?,
        u32::try_from(framebuffer.height())?,
    );
    let samples = framebuffer
        .buf()
        .par_iter()
        .map(|x| x * 255. + 0.5)
        .map(|x| x as u8)
        .collect::<Vec<_>>();
    let image_result = match image.pixel_format() {
        PixelFormat::Graya => DynamicImage::ImageLumaA8(
            GrayAlphaImage::from_raw(width, height, samples)
                .context("Can't decode gray alpha buffer")?,
        ),
        PixelFormat::Gray => DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(width, height, samples)
                .context("Can't decode gray buffer")?,
        ),
        PixelFormat::Rgba => DynamicImage::ImageRgba8(
            RgbaImage::from_raw(width, height, samples).context("Can't decode rgba buffer")?,
        ),
        PixelFormat::Rgb => DynamicImage::ImageRgb8(
            RgbImage::from_raw(width, height, samples).context("Can't decode rgb buffer")?,
        ),
        other => bail!("JXL: Pixel format: {other:?}"),
    };
    Ok(image_result)
}

/// One name per format, so a file's extension compares against its detected content.
fn canonical_extension(extension: &str) -> String {
    let extension = extension.to_lowercase();
    match extension.as_str() {
        "tiff" => "tif",
        "jpeg" | "jfif" => "jpg",
        "heic" | "hif" => "heif",
        other => other,
    }
    .to_string()
}

fn rotate_dynimage(di: &mut DynamicImage, path: &Path) -> Result<()> {
    reorient(di, orientation_of(path)?);
    Ok(())
}

fn orientation_of(path: &Path) -> Result<image::metadata::Orientation> {
    let reader = ImageReader::open(path)?.with_guessed_format()?;
    if reader.format() == Some(image::ImageFormat::Jpeg) {
        // The decoder reads the whole file; the EXIF reader stops at the EXIF block
        return Ok(exif_orientation(&mut BufReader::new(File::open(path)?)));
    }
    let mut reader = reader;
    reader.limits(image_limits());
    let mut decoder = reader.into_decoder()?;
    Ok(decoder.orientation()?)
}

fn exif_orientation<R: BufRead + Seek>(container: &mut R) -> image::metadata::Orientation {
    exif::Reader::new()
        .read_from_container(container)
        .map_or(image::metadata::Orientation::NoTransforms, |exif| {
            orientation(&exif)
        })
}

pub fn orientation(exif: &exif::Exif) -> image::metadata::Orientation {
    use image::metadata::Orientation;
    exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)
        .and_then(|field| field.value.get_uint(0))
        .and_then(|value| Orientation::from_exif(u8::try_from(value).ok()?))
        .unwrap_or(Orientation::NoTransforms)
}

/// Decodes with `image` capped at [`MAX_ALLOC`]; its 512 MiB default refuses 200 MP photos.
fn decode_limited<R: BufRead + Seek>(mut reader: ImageReader<R>) -> Result<DynamicImage> {
    reader.limits(image_limits());
    let decoder = reader.into_decoder()?;
    let (width, height) = decoder.dimensions();
    check_dimensions(width, height)?;
    Ok(DynamicImage::from_decoder(decoder)?)
}

/// Like `DynamicImage::apply_orientation`, but quarter turns fill the new image's rows in parallel.
pub fn reorient(image: &mut DynamicImage, orientation: image::metadata::Orientation) {
    use image::metadata::Orientation::{Rotate90, Rotate90FlipH, Rotate270, Rotate270FlipH};
    if matches!(
        orientation,
        Rotate90 | Rotate270 | Rotate90FlipH | Rotate270FlipH
    ) {
        macro_rules! reoriented {
            ($($variant:ident),*) => {
                match image {
                    $(DynamicImage::$variant(buffer) => {
                        *image = DynamicImage::$variant(reoriented(buffer, orientation))
                    })*
                    _ => image.apply_orientation(orientation),
                }
            };
        }
        reoriented!(
            ImageLuma8,
            ImageLumaA8,
            ImageRgb8,
            ImageRgba8,
            ImageLuma16,
            ImageLumaA16,
            ImageRgb16,
            ImageRgba16,
            ImageRgb32F,
            ImageRgba32F
        );
    } else {
        image.apply_orientation(orientation);
    }
}

/// Quarter turns only, optionally flipped.
fn reoriented<P>(
    source: &image::ImageBuffer<P, Vec<P::Subpixel>>,
    orientation: image::metadata::Orientation,
) -> image::ImageBuffer<P, Vec<P::Subpixel>>
where
    P: image::Pixel + Send + Sync,
    P::Subpixel: Send + Sync,
{
    use image::metadata::Orientation::{Rotate90, Rotate90FlipH, Rotate270};
    let (width, height) = (source.width() as usize, source.height() as usize);
    let channels = P::CHANNEL_COUNT as usize;
    let samples = source.as_raw();
    let mut target =
        image::ImageBuffer::<P, Vec<P::Subpixel>>::new(source.height(), source.width());
    target
        .par_chunks_mut(height * channels)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..height {
                // where the pixel at x, y of the new image comes from
                let (from_x, from_y) = match orientation {
                    Rotate90 => (y, height - 1 - x),
                    Rotate270 => (width - 1 - y, x),
                    Rotate90FlipH => (y, x),
                    _ => (width - 1 - y, height - 1 - x),
                };
                let from = (from_y * width + from_x) * channels;
                row[x * channels..(x + 1) * channels]
                    .copy_from_slice(&samples[from..from + channels]);
            }
        });
    target
}

fn load_jpeg_turbojpeg(img_location: &Path) -> Result<DynamicImage> {
    let jpeg_data = std::fs::read(img_location)?;
    let mut decompressor = turbojpeg::Decompressor::new()?;
    let header = decompressor.read_header(&jpeg_data)?;
    let (width, height) = (header.width, header.height);
    check_dimensions(u32::try_from(width)?, u32::try_from(height)?)?;
    let mut image = turbojpeg::Image {
        pixels: vec![0; 3 * width * height],
        width,
        pitch: 3 * width, // no row padding
        height,
        format: turbojpeg::PixelFormat::RGB,
    };
    decompressor.decompress(&jpeg_data, image.as_deref_mut())?;
    let i = RgbImage::from_raw(u32::try_from(width)?, u32::try_from(height)?, image.pixels)
        .context("Can't load RgbImage from decompressed TurboJPEG")?;
    Ok(DynamicImage::ImageRgb8(i))
}

#[cfg(test)]
mod tests {
    use super::{load_tiff, read_tiff_image, tiff_decoder};

    #[test]
    fn tiff_strips_in_parallel() {
        use tiff::encoder::{Compression, TiffEncoder, colortype, compression::DeflateLevel};
        use tiff::tags::Predictor;

        let (w, h) = (101u32, 67u32);
        let n = (w * h) as usize;
        let dir = tempfile::tempdir().unwrap();
        let rgb8: Vec<u8> = (0..n * 3).map(|i| (i * 7 % 251) as u8).collect();
        let rgba8: Vec<u8> = (0..n * 4).map(|i| (i * 11 % 241) as u8).collect();
        let rgb16: Vec<u16> = (0..n * 3).map(|i| (i * 263 % 65521) as u16).collect();
        let gray16: Vec<u16> = (0..n).map(|i| (i * 97 % 65521) as u16).collect();
        let float: Vec<f32> = (0..n).map(|i| i as f32 / 100.0).collect();

        let mut written = vec![];
        macro_rules! write {
            ($name:expr, $color:ty, $data:expr, $compression:expr, $predictor:expr, $rows:expr) => {{
                let path = dir.path().join($name);
                let file = std::fs::File::create(&path).unwrap();
                let mut encoder = TiffEncoder::new(file)
                    .unwrap()
                    .with_compression($compression)
                    .with_predictor($predictor);
                let mut image = encoder.new_image::<$color>(w, h).unwrap();
                image.rows_per_strip($rows).unwrap();
                image.write_data(&$data).unwrap();
                let bytes: Vec<u8> = $data.iter().flat_map(|v| v.to_ne_bytes()).collect();
                written.push((path, $rows, bytes));
            }};
        }
        write!(
            "lzw.tif",
            colortype::RGB8,
            rgb8,
            Compression::Lzw,
            Predictor::Horizontal,
            8
        );
        write!(
            "one_row.tif",
            colortype::RGBA8,
            rgba8,
            Compression::Uncompressed,
            Predictor::None,
            1
        );
        let deflate = Compression::Deflate(DeflateLevel::Balanced);
        write!(
            "deflate.tif",
            colortype::Gray16,
            gray16,
            deflate,
            Predictor::Horizontal,
            5
        );
        write!(
            "float.tif",
            colortype::Gray32Float,
            float,
            Compression::Lzw,
            Predictor::None,
            10
        );
        write!(
            "packbits.tif",
            colortype::RGB16,
            rgb16,
            Compression::Packbits,
            Predictor::None,
            20
        );
        write!(
            "one_strip.tif",
            colortype::RGB8,
            rgb8,
            Compression::Lzw,
            Predictor::None,
            h
        );

        for (path, rows, bytes) in written {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let mut decoder = tiff_decoder(&path).unwrap();
            assert_eq!(decoder.strip_count().unwrap(), h.div_ceil(rows), "{name}");
            let mut serial = tiff_decoder(&path).unwrap().read_image().unwrap();
            assert_eq!(
                serial.as_buffer(0).as_bytes(),
                &bytes[..],
                "{name} as written"
            );
            let mut parallel = read_tiff_image(&mut decoder, &path).unwrap();
            assert!(
                parallel.as_buffer(0).as_bytes() == serial.as_buffer(0).as_bytes(),
                "{name}: the strips decoded in parallel differ"
            );
        }
    }

    /// Written by hand: the encoder can't write planar TIFFs.
    #[test]
    fn planar_tiff() {
        let (w, h) = (7u32, 5u32);
        let n = (w * h) as usize;
        let dir = tempfile::tempdir().unwrap();
        for bits in [8u16, 16] {
            let value = |pixel: usize, plane: usize| ((pixel * 31 + plane * 80) % 256) as u16;
            let sample = |v: u16| {
                if bits == 8 {
                    vec![v as u8]
                } else {
                    (v * 257).to_le_bytes().to_vec()
                }
            };
            let planes: Vec<Vec<u8>> = (0..3)
                .map(|plane| {
                    (0..n)
                        .flat_map(|pixel| sample(value(pixel, plane)))
                        .collect()
                })
                .collect();
            let plane_len = planes[0].len() as u32;
            // header, 10 entries, bits per sample, strip offsets, strip byte counts
            let data_start = 8 + 2 + 10 * 12 + 4 + 6 + 12 + 12;
            let mut tiff = b"II\x2a\x00\x08\x00\x00\x00".to_vec();
            tiff.extend(10u16.to_le_bytes());
            let mut entry = |tag: u16, kind: u16, count: u32, value: u32| {
                tiff.extend(tag.to_le_bytes());
                tiff.extend(kind.to_le_bytes());
                tiff.extend(count.to_le_bytes());
                tiff.extend(value.to_le_bytes());
            };
            let (short, long) = (3, 4);
            entry(256, long, 1, w);
            entry(257, long, 1, h);
            entry(258, short, 3, 134);
            entry(259, short, 1, 1); // uncompressed
            entry(262, short, 1, 2); // RGB
            entry(273, long, 3, 140);
            entry(277, short, 1, 3);
            entry(278, long, 1, h);
            entry(279, long, 3, 152);
            entry(284, short, 1, 2); // planar
            tiff.extend(0u32.to_le_bytes());
            for _ in 0..3 {
                tiff.extend(bits.to_le_bytes());
            }
            for plane in 0..3 {
                tiff.extend((data_start + plane * plane_len).to_le_bytes());
            }
            for _ in 0..3 {
                tiff.extend(plane_len.to_le_bytes());
            }
            assert_eq!(tiff.len() as u32, data_start);
            for plane in &planes {
                tiff.extend(plane);
            }
            let path = dir.path().join(format!("planar{bits}.tif"));
            std::fs::write(&path, tiff).unwrap();

            let values: Vec<u16> = (0..n)
                .flat_map(|pixel| (0..3).map(move |plane| value(pixel, plane)))
                .collect();
            let expected: Vec<u8> = values
                .iter()
                .flat_map(|&v| {
                    if bits == 8 {
                        vec![v as u8]
                    } else {
                        (v * 257).to_ne_bytes().to_vec()
                    }
                })
                .collect();
            let mut result = read_tiff_image(&mut tiff_decoder(&path).unwrap(), &path).unwrap();
            assert!(
                result.as_buffer(0).as_bytes() == &expected[..],
                "{bits} bit"
            );
            // the values span 0 to 255, so the 16-bit image stretches back to the 8-bit one
            let ldr: Vec<u8> = values.iter().map(|&v| v as u8).collect();
            let image = load_tiff(&path).unwrap();
            assert!(image.as_bytes() == &ldr[..], "{bits} bit, loaded");
        }
    }
}
