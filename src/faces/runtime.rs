use std::path::{Path, PathBuf};
use std::sync::Arc;

use tract_onnx::prelude::*;

use super::align::ALIGNED_SIZE;
use super::detect::INPUT_SIZE;
use super::fetch::{DETECTOR, EMBEDDER};
use crate::config::Device;

pub type Plan = Arc<TypedRunnableModel>;

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("{} is missing; run `vitrine faces fetch`", .0.display())]
    MissingModel(PathBuf),
    #[error("faces.device = \"cuda\" isn't supported; set it to \"cpu\"")]
    UnsupportedDevice,
    #[error("failed to load the face model {}", path.display())]
    Load {
        path: PathBuf,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

pub struct FaceRuntime {
    detector: Plan,
    embedder: Plan,
}

impl FaceRuntime {
    pub fn new(models_dir: &Path, device: Device) -> Result<Self, RuntimeError> {
        if device == Device::Cuda {
            return Err(RuntimeError::UnsupportedDevice);
        }
        let side = INPUT_SIZE as usize;
        let aligned = ALIGNED_SIZE as usize;
        Ok(Self {
            detector: plan(&models_dir.join(DETECTOR.name), |_| {
                tvec![1.into(), 3.into(), side.into(), side.into()]
            })?,
            embedder: plan(&models_dir.join(EMBEDDER.name), |symbols| {
                tvec![
                    symbols.sym("N").into(),
                    3.into(),
                    aligned.into(),
                    aligned.into()
                ]
            })?,
        })
    }

    #[must_use]
    pub fn detector(&self) -> &Plan {
        &self.detector
    }

    #[must_use]
    pub fn embedder(&self) -> &Plan {
        &self.embedder
    }
}

/// Checked without loading anything.
#[must_use]
pub fn missing_model(models_dir: &Path) -> Option<PathBuf> {
    [DETECTOR, EMBEDDER]
        .into_iter()
        .map(|file| models_dir.join(file.name))
        .find(|path| !path.is_file())
}

/// Optimised for an NCHW f32 input of `shape`, so the plan is built once and shared across threads.
fn plan(path: &Path, shape: impl FnOnce(&SymbolScope) -> TVec<TDim>) -> Result<Plan, RuntimeError> {
    if !path.is_file() {
        return Err(RuntimeError::MissingModel(path.to_owned()));
    }
    let build = || -> TractResult<Plan> {
        let model = tract_onnx::onnx().model_for_path(path)?;
        let shape = shape(&model.symbols);
        model
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), shape))?
            .into_optimized()?
            .into_runnable()
    };
    build().map_err(|source| RuntimeError::Load {
        path: path.to_owned(),
        source: source.into(),
    })
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::faces::fixtures::models_dir;

    type Shapes = Vec<Vec<usize>>;

    fn forward(plan: &Plan, shape: [usize; 4]) -> TractResult<Shapes> {
        let zeros = Tensor::zero::<f32>(&shape)?;
        let outputs = plan.run(tvec![zeros.into()])?;
        Ok(outputs
            .iter()
            .map(|value| {
                eprintln!("  ran    {:?} {:?}", value.datum_type(), value.shape());
                value.shape().to_vec()
            })
            .collect())
    }

    fn spike(dir: &Path) -> anyhow::Result<(Shapes, Shapes)> {
        let start = Instant::now();
        let runtime = FaceRuntime::new(dir, Device::Cpu)?;
        eprintln!("plans built in {:?}", start.elapsed());
        eprintln!("{}:", DETECTOR.name);
        let det = forward(runtime.detector(), [1, 3, 640, 640])?;
        eprintln!("{}:", EMBEDDER.name);
        let emb = forward(runtime.embedder(), [1, 3, 112, 112])?;
        eprintln!("forward passes done after {:?}", start.elapsed());
        Ok((det, emb))
    }

    #[test]
    #[ignore = "needs fetched models (VITRINE_MODELS_DIR or the XDG models dir)"]
    fn plans_run_a_zero_forward_pass() {
        let (det, emb) = spike(&models_dir()).unwrap();

        let mut rows: Vec<_> = det
            .iter()
            .map(|shape| match shape.as_slice() {
                [1, rows, cols] | [rows, cols] => (*rows, *cols),
                other => panic!("unexpected detector output shape {other:?}"),
            })
            .collect();
        rows.sort_unstable();
        let mut expected: Vec<_> = [12800, 3200, 800]
            .into_iter()
            .flat_map(|n| [(n, 1), (n, 4), (n, 10)])
            .collect();
        expected.sort_unstable();
        assert_eq!(rows, expected);
        assert_eq!(emb.len(), 1);
        assert_eq!(emb[0], [1, 512]);
    }

    #[test]
    fn missing_model_is_reported_before_loading() {
        let dir = tempfile::tempdir().unwrap();
        let err = plan(&dir.path().join(DETECTOR.name), |_| tvec![]).unwrap_err();
        assert!(matches!(err, RuntimeError::MissingModel(_)), "{err:?}");
        assert_eq!(
            missing_model(dir.path()),
            Some(dir.path().join(DETECTOR.name))
        );
        std::fs::write(dir.path().join(DETECTOR.name), b"").unwrap();
        assert_eq!(
            missing_model(dir.path()),
            Some(dir.path().join(EMBEDDER.name))
        );
        std::fs::write(dir.path().join(EMBEDDER.name), b"").unwrap();
        assert_eq!(missing_model(dir.path()), None);
    }
}
