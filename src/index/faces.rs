//! Face stage over content not yet scanned with the configured model.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Mutex, PoisonError};
use std::thread;

use anyhow::ensure;
use image::imageops::{self, FilterType};
use image::{DynamicImage, RgbImage};

use super::Failure;
use super::writer::Op;
use crate::catalog::faces::ModelId;
use crate::db::Sha256;
use crate::faces::align::align;
use crate::faces::{Detection, FaceDetector, FaceEmbedder};

/// Images decoded at once; inference is serialised by the runtime's session locks.
const WORKERS: usize = 4;
pub const CROP_SIZE: u32 = 128;
/// Crop square side relative to the face box's longer side.
const CROP_MARGIN: f64 = 1.3;

pub(crate) struct Models {
    pub detector: Box<dyn FaceDetector + Send + Sync>,
    pub embedder: Box<dyn FaceEmbedder + Send + Sync>,
}

pub(crate) enum Stage<'a> {
    Off,
    /// Enabled but the models can't be used; reported as a run failure.
    Unavailable(Failure),
    On(Setup<'a>),
}

pub(crate) struct Setup<'a> {
    /// `face_models` name; with `detector` and `dim`, must match what `load` returns.
    pub name: &'a str,
    pub detector: &'a str,
    pub dim: usize,
    /// Path a load failure is reported under.
    pub models_dir: &'a Path,
    /// Called at most once per run, and only when content is pending.
    pub load: &'a dyn Fn() -> anyhow::Result<Models>,
    pub config: &'a crate::config::Faces,
}

#[derive(Debug)]
pub(crate) struct Scan {
    pub sha256: Sha256,
    pub path: PathBuf,
    pub model: ModelId,
    /// `None` leaves the content pending for the next run.
    pub faces: Option<Vec<Found>>,
    pub error: Option<String>,
}

#[derive(Debug)]
pub(crate) struct Found {
    pub detection: Detection,
    pub embedding: Vec<f32>,
    /// [`CROP_SIZE`] square JPEG.
    pub crop: Vec<u8>,
}

/// One [`Op::Faces`] per content; stops once `cancel` is set, the writer hangs up, or a model fails.
pub(crate) fn scan_all(
    pending: Vec<(Sha256, PathBuf)>,
    models: &Models,
    model: ModelId,
    cancel: &AtomicBool,
    tx: &Sender<Op>,
) {
    let queue = Mutex::new(pending.into_iter());
    let stop = AtomicBool::new(false);
    let (queue, stop) = (&queue, &stop);
    thread::scope(|s| {
        for _ in 0..WORKERS {
            let tx = tx.clone();
            s.spawn(move || {
                while !cancel.load(Ordering::Relaxed) && !stop.load(Ordering::Relaxed) {
                    let next = queue.lock().unwrap_or_else(PoisonError::into_inner).next();
                    let Some((sha256, path)) = next else {
                        break;
                    };
                    let (faces, error) = scan(models, &path).unwrap_or_else(|e| {
                        stop.store(true, Ordering::Relaxed);
                        (None, Some(format!("faces: {e:#}")))
                    });
                    let scan = Scan {
                        sha256,
                        path,
                        model,
                        faces,
                        error,
                    };
                    if tx.send(Op::Faces(scan)).is_err() {
                        break;
                    }
                }
            });
        }
    });
}

/// `Err` means a model failed, not the image.
fn scan(models: &Models, path: &Path) -> anyhow::Result<(Option<Vec<Found>>, Option<String>)> {
    // unopenable: retry next run; opens but won't decode: stays that way until the content changes
    if let Err(e) = File::open(path) {
        return Ok((None, Some(format!("faces: {e}"))));
    }
    let image = match crate::decode::decode(path) {
        Ok(image) => image.into_rgb8(),
        Err(e) => return Ok((Some(Vec::new()), Some(format!("faces: {e:#}")))),
    };
    let mut detections = models.detector.detect(&image)?;
    let mut aligned = Vec::with_capacity(detections.len());
    detections.retain(|d| match align(&image, &d.landmarks) {
        Some(crop) => {
            aligned.push(crop);
            true
        }
        None => false,
    });
    let embeddings = models.embedder.embed(&aligned)?;
    ensure!(
        embeddings.len() == detections.len(),
        "embedder returned {} vectors for {} faces",
        embeddings.len(),
        detections.len()
    );
    let faces = detections
        .into_iter()
        .zip(embeddings)
        .map(|(detection, embedding)| {
            Ok(Found {
                crop: crop(&image, &detection.bbox)?,
                detection,
                embedding,
            })
        })
        .collect::<anyhow::Result<_>>()?;
    Ok((Some(faces), None))
}

/// Shifted inside the image rather than padded.
fn crop(image: &RgbImage, bbox: &[f32; 4]) -> anyhow::Result<Vec<u8>> {
    let (width, height) = image.dimensions();
    ensure!(width > 0 && height > 0, "empty image");
    let [x1, y1, x2, y2] = bbox.map(f64::from);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let side = (((x2 - x1).max(y2 - y1) * CROP_MARGIN).round() as u32).clamp(1, width.min(height));
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let start =
        |centre: f64, len: u32| (((centre - f64::from(side) / 2.0).round()) as u32).min(len - side);
    let (x, y) = (
        start((x1 + x2) / 2.0, width),
        start((y1 + y2) / 2.0, height),
    );
    let square = imageops::crop_imm(image, x, y, side, side).to_image();
    let resized = imageops::resize(&square, CROP_SIZE, CROP_SIZE, FilterType::CatmullRom);
    Ok(crate::thumbs::generate::encode(&DynamicImage::ImageRgb8(resized))?.jpeg)
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;

    fn decoded(jpeg: &[u8]) -> RgbImage {
        image::load_from_memory(jpeg).unwrap().to_rgb8()
    }

    #[test]
    fn crop_is_a_square_around_the_box() {
        // left half black, right half white; a box centred on the edge splits the crop down the middle
        let image = RgbImage::from_fn(400, 300, |x, _| Rgb([if x < 200 { 0 } else { 255 }; 3]));
        let crop = decoded(&crop(&image, &[160.0, 100.0, 240.0, 200.0]).unwrap());
        assert_eq!(crop.dimensions(), (CROP_SIZE, CROP_SIZE));
        assert!(crop.get_pixel(10, 64).0[0] < 30);
        assert!(crop.get_pixel(CROP_SIZE - 10, 64).0[0] > 225);
    }

    #[test]
    fn crop_shifts_inside_and_shrinks_to_the_image() {
        // box in the white top-left corner, extending past the edges
        let image = RgbImage::from_fn(200, 100, |x, y| {
            Rgb([if x < 100 && y < 50 { 255 } else { 0 }; 3])
        });
        let corner = decoded(&crop(&image, &[-20.0, -20.0, 60.0, 40.0]).unwrap());
        assert_eq!(corner.dimensions(), (CROP_SIZE, CROP_SIZE));
        assert!(corner.get_pixel(5, 5).0[0] > 225);
        let huge = decoded(&crop(&image, &[-500.0, -500.0, 900.0, 900.0]).unwrap());
        assert_eq!(huge.dimensions(), (CROP_SIZE, CROP_SIZE));
    }
}
