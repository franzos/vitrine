//! SCRFD `det_10g` run the way InsightFace's `model_zoo/scrfd.py` runs it.

use std::sync::Arc;

use anyhow::{Context, bail};
use image::{Rgb, RgbImage};
use ort::value::Tensor;

use super::align::to_u8;
use super::model::{Detection, FaceDetector};
use super::runtime::FaceRuntime;

/// Detections scoring below this are dropped before NMS.
pub const SCORE_THRESHOLD: f32 = 0.5;
/// IoU above which a lower-scoring box is suppressed.
pub const NMS_THRESHOLD: f32 = 0.4;
pub const INPUT_SIZE: u32 = 640;

const STRIDES: [u32; 3] = [8, 16, 32];
const ANCHORS: usize = 2;
const MEAN: f32 = 127.5;
const STD: f32 = 128.0;

pub struct Scrfd {
    runtime: Arc<FaceRuntime>,
}

impl Scrfd {
    #[must_use]
    pub fn new(runtime: Arc<FaceRuntime>) -> Self {
        Self { runtime }
    }
}

impl FaceDetector for Scrfd {
    fn detect(&self, image: &RgbImage) -> anyhow::Result<Vec<Detection>> {
        let Some((width, height, scale)) = letterbox(image.width(), image.height()) else {
            return Ok(Vec::new());
        };
        let input = input_tensor(&resize_linear(image, width, height));
        let side = INPUT_SIZE as usize;
        let tensor = Tensor::from_array(([1, 3, side, side], input))?;

        let mut session = self.runtime.detector();
        let outputs = session
            .run(ort::inputs![tensor])
            .context("detector inference failed")?;
        let values = (0..outputs.len())
            .map(|i| outputs[i].try_extract_tensor::<f32>().map(|(_, data)| data))
            .collect::<Result<Vec<_>, _>>()?;
        let [s8, s16, s32, b8, b16, b32, k8, k16, k32] = values.as_slice() else {
            bail!("detector returned {} outputs, expected 9", values.len());
        };
        let mut detections = Vec::new();
        for (stride, scores, boxes, kps) in [
            (STRIDES[0], s8, b8, k8),
            (STRIDES[1], s16, b16, k16),
            (STRIDES[2], s32, b32, k32),
        ] {
            decode_stride(stride, scores, boxes, kps, &mut detections)?;
        }
        drop(outputs);
        drop(session);

        for detection in &mut detections {
            detection.bbox = detection.bbox.map(|v| v / scale);
            detection.landmarks = detection.landmarks.map(|p| p.map(|v| v / scale));
        }
        Ok(nms(detections, NMS_THRESHOLD))
    }
}

/// `(width, height, scale)` inside the letterbox; scale is the height ratio for both axes, as in InsightFace.
fn letterbox(width: u32, height: u32) -> Option<(u32, u32, f32)> {
    if width == 0 || height == 0 {
        return None;
    }
    let side = f64::from(INPUT_SIZE);
    let ratio = f64::from(height) / f64::from(width);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (new_width, new_height) = if ratio > 1.0 {
        ((side / ratio) as u32, INPUT_SIZE)
    } else {
        (INPUT_SIZE, (side * ratio) as u32)
    };
    if new_width == 0 || new_height == 0 {
        return None;
    }
    #[allow(clippy::cast_possible_truncation)]
    let scale = (f64::from(new_height) / f64::from(height)) as f32;
    Some((new_width, new_height, scale))
}

/// Bilinear resize with half-pixel centres and edge clamping, as `cv2.resize` with `INTER_LINEAR`.
fn resize_linear(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    let xs: Vec<_> = (0..width).map(|d| taps(d, image.width(), width)).collect();
    let ys: Vec<_> = (0..height)
        .map(|d| taps(d, image.height(), height))
        .collect();
    RgbImage::from_fn(width, height, |x, y| {
        let (x0, x1, fx) = xs[x as usize];
        let (y0, y1, fy) = ys[y as usize];
        let [p00, p10, p01, p11] = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
            .map(|(x, y)| image.get_pixel(x, y).0.map(f64::from));
        Rgb(std::array::from_fn(|c| {
            let top = p00[c] + (p10[c] - p00[c]) * fx;
            let bottom = p01[c] + (p11[c] - p01[c]) * fx;
            to_u8(top + (bottom - top) * fy)
        }))
    })
}

fn taps(dst: u32, src_len: u32, dst_len: u32) -> (u32, u32, f64) {
    let position = (f64::from(dst) + 0.5) * f64::from(src_len) / f64::from(dst_len) - 0.5;
    let floor = position.floor();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (start, frac) = if floor < 0.0 {
        (0, 0.0)
    } else if floor >= f64::from(src_len - 1) {
        (src_len - 1, 0.0)
    } else {
        (floor as u32, position - floor)
    };
    (start, (start + 1).min(src_len - 1), frac)
}

/// NCHW RGB input: the image at the top left, zero pixels elsewhere, all as `(px - 127.5) / 128`.
fn input_tensor(resized: &RgbImage) -> Vec<f32> {
    let side = INPUT_SIZE as usize;
    let plane = side * side;
    let mut data = vec![-MEAN / STD; 3 * plane];
    for (x, y, pixel) in resized.enumerate_pixels() {
        let offset = y as usize * side + x as usize;
        for (c, value) in pixel.0.into_iter().enumerate() {
            data[c * plane + offset] = (f32::from(value) - MEAN) / STD;
        }
    }
    data
}

/// Anchor `i` sits at cell `i / 2`, row-major, centred on `(x * stride, y * stride)`; offsets in stride units.
fn decode_stride(
    stride: u32,
    scores: &[f32],
    boxes: &[f32],
    kps: &[f32],
    out: &mut Vec<Detection>,
) -> anyhow::Result<()> {
    let cells = (INPUT_SIZE / stride) as usize;
    let anchors = cells * cells * ANCHORS;
    if scores.len() != anchors || boxes.len() != anchors * 4 || kps.len() != anchors * 10 {
        bail!(
            "stride {stride}: got {}/{}/{} values, expected {anchors} anchors",
            scores.len(),
            boxes.len(),
            kps.len()
        );
    }
    #[allow(clippy::cast_precision_loss)]
    let s = stride as f32;
    for (i, &score) in scores.iter().enumerate() {
        if score < SCORE_THRESHOLD || score.is_nan() {
            continue;
        }
        let cell = i / ANCHORS;
        #[allow(clippy::cast_precision_loss)]
        let (cx, cy) = ((cell % cells) as f32 * s, (cell / cells) as f32 * s);
        let b = &boxes[i * 4..i * 4 + 4];
        let k = &kps[i * 10..i * 10 + 10];
        out.push(Detection {
            bbox: [cx - b[0] * s, cy - b[1] * s, cx + b[2] * s, cy + b[3] * s],
            landmarks: std::array::from_fn(|j| [cx + k[2 * j] * s, cy + k[2 * j + 1] * s]),
            score,
        });
    }
    Ok(())
}

/// Greedy NMS, highest score first, ties in input order.
#[must_use]
pub fn nms(mut detections: Vec<Detection>, threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::new();
    for detection in detections {
        if kept.iter().all(|k| iou(k, &detection) <= threshold) {
            kept.push(detection);
        }
    }
    kept
}

/// IoU with InsightFace's inclusive pixel convention (`x2 - x1 + 1`).
fn iou(a: &Detection, b: &Detection) -> f32 {
    let area = |d: &Detection| (d.width() + 1.0) * (d.height() + 1.0);
    let w = (a.bbox[2].min(b.bbox[2]) - a.bbox[0].max(b.bbox[0]) + 1.0).max(0.0);
    let h = (a.bbox[3].min(b.bbox[3]) - a.bbox[1].max(b.bbox[1]) + 1.0).max(0.0);
    let inter = w * h;
    inter / (area(a) + area(b) - inter)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Outputs {
        scores: Vec<f32>,
        boxes: Vec<f32>,
        kps: Vec<f32>,
    }

    fn outputs(stride: u32) -> Outputs {
        let cells = (INPUT_SIZE / stride) as usize;
        let n = cells * cells * ANCHORS;
        Outputs {
            scores: vec![0.0; n],
            boxes: vec![0.0; n * 4],
            kps: vec![0.0; n * 10],
        }
    }

    fn detection(bbox: [f32; 4], score: f32) -> Detection {
        Detection {
            bbox,
            landmarks: [[0.0; 2]; 5],
            score,
        }
    }

    #[test]
    fn anchor_decoding_uses_cell_centres_and_stride_units() {
        let mut out = outputs(16);
        // cell (x=3, y=5) of a 40-wide grid, second anchor
        let i = (5 * 40 + 3) * 2 + 1;
        out.scores[i] = 0.9;
        out.boxes[i * 4..i * 4 + 4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let kps: Vec<f32> = (0..10)
            .map(|v| f32::from(u8::try_from(v).unwrap()) * 0.5)
            .collect();
        out.kps[i * 10..i * 10 + 10].copy_from_slice(&kps);
        // just below the threshold, ignored
        out.scores[0] = 0.499;

        let mut found = Vec::new();
        decode_stride(16, &out.scores, &out.boxes, &out.kps, &mut found).unwrap();
        assert_eq!(found.len(), 1);
        let d = &found[0];
        let (cx, cy) = (48.0, 80.0);
        assert_eq!(d.bbox, [cx - 16.0, cy - 32.0, cx + 48.0, cy + 64.0]);
        assert_eq!(
            d.landmarks,
            [
                [cx, cy + 8.0],
                [cx + 16.0, cy + 24.0],
                [cx + 32.0, cy + 40.0],
                [cx + 48.0, cy + 56.0],
                [cx + 64.0, cy + 72.0],
            ]
        );
        assert!((d.score - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn anchor_decoding_covers_every_stride_grid() {
        for (stride, cells) in [(8, 80), (16, 40), (32, 20)] {
            let mut out = outputs(stride);
            let last = out.scores.len() - 1;
            out.scores[last] = 0.5;
            let mut found = Vec::new();
            decode_stride(stride, &out.scores, &out.boxes, &out.kps, &mut found).unwrap();
            #[allow(clippy::cast_precision_loss)]
            let edge = ((cells - 1) * stride as usize) as f32;
            assert_eq!(found[0].bbox, [edge, edge, edge, edge], "stride {stride}");
        }
    }

    #[test]
    fn anchor_decoding_rejects_wrong_sizes() {
        let out = outputs(8);
        let mut found = Vec::new();
        assert!(decode_stride(16, &out.scores, &out.boxes, &out.kps, &mut found).is_err());
    }

    #[test]
    fn nms_keeps_best_of_overlapping_boxes() {
        let kept = nms(
            vec![
                detection([0.0, 0.0, 99.0, 99.0], 0.7),
                detection([5.0, 5.0, 104.0, 104.0], 0.9),
                detection([200.0, 200.0, 249.0, 249.0], 0.6),
                detection([0.0, 0.0, 99.0, 99.0], 0.8),
            ],
            NMS_THRESHOLD,
        );
        let scores: Vec<f32> = kept.iter().map(|d| d.score).collect();
        assert_eq!(scores, [0.9, 0.6]);
    }

    #[test]
    fn nms_threshold_is_inclusive_keep() {
        // 100x100 boxes overlapping in 100x40: IoU = 4000 / 16000 = 0.25
        let a = detection([0.0, 0.0, 99.0, 99.0], 0.9);
        let b = detection([0.0, 60.0, 99.0, 159.0], 0.8);
        assert!((iou(&a, &b) - 0.25).abs() < 1e-6);
        assert_eq!(nms(vec![a, b], 0.25).len(), 2);
        assert_eq!(nms(vec![a, b], 0.24).len(), 1);
    }

    #[test]
    fn letterbox_matches_insightface() {
        // landscape: width fills the input, scale from the truncated height
        for ((w, h), expected) in [
            ((1600, 1067), (640, 426, 426.0 / 1067.0)),
            ((1067, 1600), (426, 640, 0.4)),
            ((640, 640), (640, 640, 1.0)),
            ((320, 200), (640, 400, 2.0)),
        ] {
            let (width, height, scale) = letterbox(w, h).unwrap();
            assert_eq!((width, height), (expected.0, expected.1), "{w}x{h}");
            assert!((scale - expected.2).abs() < 1e-6, "{w}x{h}: {scale}");
        }
        assert_eq!(letterbox(0, 10), None);
        assert_eq!(letterbox(10_000, 1), None);
    }

    #[test]
    fn resize_matches_cv2_linear() {
        // cv2.resize(np.array([[0, 100, 200, 250]]), (2, 1)) == [[50, 225]]
        let mut row = RgbImage::new(4, 1);
        for (x, v) in [0, 100, 200, 250].into_iter().enumerate() {
            row.put_pixel(u32::try_from(x).unwrap(), 0, Rgb([v; 3]));
        }
        let half = resize_linear(&row, 2, 1);
        assert_eq!(half.get_pixel(0, 0).0, [50; 3]);
        assert_eq!(half.get_pixel(1, 0).0, [225; 3]);
        // upscaling clamps at the edges: [[0, 25, 75, 100]]
        let pair = RgbImage::from_fn(2, 1, |x, _| Rgb([if x == 0 { 0 } else { 100 }; 3]));
        let up = resize_linear(&pair, 4, 1);
        let values: Vec<u8> = up.pixels().map(|p| p.0[0]).collect();
        assert_eq!(values, [0, 25, 75, 100]);
    }

    #[test]
    fn input_tensor_is_normalised_nchw_with_zero_padding() {
        let image = RgbImage::from_pixel(2, 1, Rgb([255, 127, 0]));
        let data = input_tensor(&image);
        let plane = (INPUT_SIZE * INPUT_SIZE) as usize;
        assert_eq!(data.len(), 3 * plane);
        assert!((data[0] - 127.5 / 128.0).abs() < 1e-6);
        assert!((data[plane + 1] - (-0.5 / 128.0)).abs() < 1e-6);
        assert!((data[2 * plane] - (-127.5 / 128.0)).abs() < 1e-6);
        assert!((data[2] - (-127.5 / 128.0)).abs() < 1e-6);
        assert!((data[INPUT_SIZE as usize] - (-127.5 / 128.0)).abs() < 1e-6);
    }

    #[test]
    #[ignore = "needs ORT_DYLIB_PATH, fetched models and VITRINE_FACE_FIXTURES"]
    fn fixture_face_counts_match_labels() {
        use crate::config::Faces;
        use crate::faces::fixtures;

        let fixtures = fixtures::load();
        let runtime =
            FaceRuntime::new(&fixtures::models_dir(), crate::config::Device::Cpu).unwrap();
        let detector = Scrfd::new(Arc::new(runtime));
        let faces = Faces::default();
        let mut mismatches = Vec::new();
        for label in &fixtures.labels {
            let image = crate::decode::decode(&fixtures.dir.join(&label.file))
                .unwrap()
                .to_rgb8();
            let detections = detector.detect(&image).unwrap();
            let counted: Vec<_> = detections.iter().filter(|d| d.counts(&faces)).collect();
            let ok = label.accepts(counted.len());
            let summary: Vec<String> = detections
                .iter()
                .map(|d| format!("{:.2}/{:.0}", d.score, d.width().min(d.height())))
                .collect();
            eprintln!(
                "{} {:<8} expected {:<4} counted {} of {} [{}] {:?}",
                if ok { "ok  " } else { "FAIL" },
                label.file,
                label.expected(),
                counted.len(),
                detections.len(),
                summary.join(" "),
                label.people,
            );
            if !ok {
                mismatches.push(&label.file);
            }
        }
        assert!(mismatches.is_empty(), "count mismatches: {mismatches:?}");
    }
}
