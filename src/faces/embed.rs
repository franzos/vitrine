//! ArcFace `w600k_r50` run the way InsightFace's `model_zoo/arcface_onnx.py` runs it.

use std::sync::Arc;

use anyhow::{Context, ensure};
use image::RgbImage;
use ort::value::Tensor;

use super::align::ALIGNED_SIZE;
use super::fetch::EMBEDDING_DIM;
use super::model::FaceEmbedder;
use super::runtime::FaceRuntime;

// w600k_r50 has no normalisation nodes; InsightFace feeds it (px - 127.5) / 127.5
const MEAN: f32 = 127.5;
const STD: f32 = 127.5;
/// Crops per inference run, bounding the session's working memory on group photos.
const BATCH: usize = 8;

pub struct ArcFace {
    runtime: Arc<FaceRuntime>,
}

impl ArcFace {
    #[must_use]
    pub fn new(runtime: Arc<FaceRuntime>) -> Self {
        Self { runtime }
    }
}

impl FaceEmbedder for ArcFace {
    /// L2-normalised embeddings, so the dot product of two is their cosine similarity.
    fn embed(&self, aligned: &[RgbImage]) -> anyhow::Result<Vec<Vec<f32>>> {
        let side = ALIGNED_SIZE as usize;
        let mut out = Vec::with_capacity(aligned.len());
        for chunk in aligned.chunks(BATCH) {
            let tensor = Tensor::from_array(([chunk.len(), 3, side, side], input_tensor(chunk)?))?;
            let mut session = self.runtime.embedder();
            let outputs = session
                .run(ort::inputs![tensor])
                .context("embedder inference failed")?;
            let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
            ensure!(
                data.len() == chunk.len() * EMBEDDING_DIM,
                "embedder returned {shape}, expected [{}, {EMBEDDING_DIM}]",
                chunk.len()
            );
            out.extend(
                data.as_chunks::<EMBEDDING_DIM>()
                    .0
                    .iter()
                    .map(|v| normalised(v)),
            );
        }
        Ok(out)
    }
}

/// NCHW RGB, one image after the other.
fn input_tensor(crops: &[RgbImage]) -> anyhow::Result<Vec<f32>> {
    let plane = (ALIGNED_SIZE * ALIGNED_SIZE) as usize;
    let mut data = vec![0.0; crops.len() * 3 * plane];
    for (crop, chunk) in crops.iter().zip(data.chunks_exact_mut(3 * plane)) {
        ensure!(
            crop.dimensions() == (ALIGNED_SIZE, ALIGNED_SIZE),
            "aligned crop is {}x{}, expected {ALIGNED_SIZE}x{ALIGNED_SIZE}",
            crop.width(),
            crop.height()
        );
        for (i, pixel) in crop.pixels().enumerate() {
            for (c, value) in pixel.0.into_iter().enumerate() {
                chunk[c * plane + i] = (f32::from(value) - MEAN) / STD;
            }
        }
    }
    Ok(data)
}

/// A zero vector stays zero.
fn normalised(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;

    #[test]
    fn input_tensor_is_normalised_rgb_nchw() {
        let side = ALIGNED_SIZE;
        let first = RgbImage::from_pixel(side, side, Rgb([255, 127, 0]));
        let mut second = RgbImage::from_pixel(side, side, Rgb([0, 0, 0]));
        second.put_pixel(1, 0, Rgb([0, 0, 255]));
        let data = input_tensor(&[first, second]).unwrap();
        let plane = (side * side) as usize;
        assert_eq!(data.len(), 6 * plane);
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[plane] - (-0.5 / 127.5)).abs() < 1e-6);
        assert!((data[2 * plane] - -1.0).abs() < 1e-6);
        assert!((data[3 * plane] - -1.0).abs() < 1e-6);
        assert!((data[5 * plane + 1] - 1.0).abs() < 1e-6);
        assert!((data[5 * plane] - -1.0).abs() < 1e-6);
    }

    #[test]
    fn input_tensor_rejects_wrong_size() {
        let crop = RgbImage::new(ALIGNED_SIZE, ALIGNED_SIZE - 1);
        assert!(input_tensor(&[crop]).is_err());
    }

    #[test]
    fn normalised_has_unit_length() {
        let v = normalised(&[3.0, -4.0]);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] + 0.8).abs() < 1e-6);
        assert_eq!(normalised(&[0.0, 0.0]), [0.0, 0.0]);
    }

    mod fixtures {
        use std::fmt::Write;

        use super::*;
        use crate::config::{Device, Faces};
        use crate::faces::align::align;
        use crate::faces::detect::Scrfd;
        use crate::faces::fixtures;
        use crate::faces::model::FaceDetector;

        struct Embedded {
            file: String,
            people: Vec<String>,
            accepted: bool,
            embeddings: Vec<Vec<f32>>,
        }

        /// Embeddings of the faces that pass the default count filter, per fixture image.
        fn embed_fixtures() -> (Vec<Embedded>, ArcFace) {
            let fixtures = fixtures::load();
            let runtime = Arc::new(FaceRuntime::new(&fixtures::models_dir(), Device::Cpu).unwrap());
            let detector = Scrfd::new(Arc::clone(&runtime));
            let embedder = ArcFace::new(runtime);
            let faces = Faces::default();
            let embedded = fixtures
                .labels
                .iter()
                .map(|label| {
                    let image = crate::decode::decode(&fixtures.dir.join(&label.file))
                        .unwrap()
                        .to_rgb8();
                    let crops: Vec<RgbImage> = detector
                        .detect(&image)
                        .unwrap()
                        .iter()
                        .filter(|d| d.counts(&faces))
                        .map(|d| align(&image, &d.landmarks).unwrap())
                        .collect();
                    Embedded {
                        file: label.file.clone(),
                        people: label.people.clone(),
                        accepted: label.accepts(crops.len()),
                        embeddings: embedder.embed(&crops).unwrap(),
                    }
                })
                .collect();
            (embedded, embedder)
        }

        fn dot(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b).map(|(x, y)| x * y).sum()
        }

        fn describe(name: &str, pairs: &[(String, f32)]) -> String {
            let mut values: Vec<f32> = pairs.iter().map(|(_, s)| *s).collect();
            values.sort_by(f32::total_cmp);
            let median = if values.len() % 2 == 1 {
                values[values.len() / 2]
            } else {
                f32::midpoint(values[values.len() / 2 - 1], values[values.len() / 2])
            };
            let mut out = format!(
                "{name}: {} pairs, similarity min {:.3} median {median:.3} max {:.3}\n",
                values.len(),
                values[0],
                values[values.len() - 1],
            );
            for (pair, similarity) in pairs {
                writeln!(out, "  {pair} {similarity:.3}").unwrap();
            }
            out
        }

        /// Same person: single-person images with one counted face. Different: across people, plus faces within a correctly counted image.
        #[test]
        #[ignore = "needs ORT_DYLIB_PATH, fetched models and VITRINE_FACE_FIXTURES"]
        fn fixture_pairs_split_at_cluster_threshold() {
            let (embedded, _) = embed_fixtures();
            let singles: Vec<(&str, &str, &[f32])> = embedded
                .iter()
                .filter(|e| e.people.len() == 1 && e.embeddings.len() == 1)
                .map(|e| {
                    (
                        e.people[0].as_str(),
                        e.file.as_str(),
                        e.embeddings[0].as_slice(),
                    )
                })
                .collect();
            let (mut same, mut different) = (Vec::new(), Vec::new());
            for (i, (person_a, file_a, a)) in singles.iter().enumerate() {
                for (person_b, file_b, b) in &singles[i + 1..] {
                    let pair = (
                        format!("{file_a}({person_a}) {file_b}({person_b})"),
                        dot(a, b),
                    );
                    if person_a == person_b {
                        same.push(pair);
                    } else {
                        different.push(pair);
                    }
                }
            }
            for e in embedded.iter().filter(|e| e.accepted) {
                for (i, a) in e.embeddings.iter().enumerate() {
                    for (j, b) in e.embeddings.iter().enumerate().skip(i + 1) {
                        different.push((format!("{}#{i} {}#{j}", e.file, e.file), dot(a, b)));
                    }
                }
            }
            eprint!("{}", describe("same person", &same));
            eprint!("{}", describe("different people", &different));

            let threshold = Faces::default().cluster_threshold;
            let joined: Vec<_> = same.iter().filter(|(_, s)| 1.0 - s > threshold).collect();
            let split: Vec<_> = different
                .iter()
                .filter(|(_, s)| 1.0 - s <= threshold)
                .collect();
            assert!(!same.is_empty() && !different.is_empty());
            assert!(
                joined.is_empty(),
                "same person beyond {threshold}: {joined:?}"
            );
            assert!(
                split.is_empty(),
                "different people within {threshold}: {split:?}"
            );
        }

        #[test]
        #[ignore = "needs ORT_DYLIB_PATH, fetched models and VITRINE_FACE_FIXTURES"]
        fn batched_embeddings_match_single_runs() {
            let fixtures = fixtures::load();
            let runtime = Arc::new(FaceRuntime::new(&fixtures::models_dir(), Device::Cpu).unwrap());
            let detector = Scrfd::new(Arc::clone(&runtime));
            let embedder = ArcFace::new(runtime);
            let label = fixtures
                .labels
                .iter()
                .find(|l| l.faces.is_some_and(|n| n >= 3))
                .expect("a fixture with three or more faces");
            let image = crate::decode::decode(&fixtures.dir.join(&label.file))
                .unwrap()
                .to_rgb8();
            let faces: Vec<RgbImage> = detector
                .detect(&image)
                .unwrap()
                .iter()
                .filter_map(|d| align(&image, &d.landmarks))
                .collect();
            // more crops than one batch holds, so the run splits
            let crops: Vec<RgbImage> = faces.iter().cycle().take(BATCH + 3).cloned().collect();
            let batched = embedder.embed(&crops).unwrap();
            assert_eq!(batched.len(), crops.len());
            for (crop, batched) in crops.iter().zip(&batched) {
                let single = embedder.embed(std::slice::from_ref(crop)).unwrap();
                assert_eq!(single[0].len(), EMBEDDING_DIM);
                assert!((dot(batched, batched) - 1.0).abs() < 1e-4);
                assert!(dot(&single[0], batched) > 0.9999);
            }
        }
    }
}
