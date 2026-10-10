use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use ort::ep;
use ort::session::Session;

use super::fetch::{DETECTOR, EMBEDDER};
use crate::config::Device;

/// Intra-op threads per session; the index pipeline already runs images in parallel.
pub const INTRA_THREADS: usize = 4;

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("ORT_DYLIB_PATH is not set; point it at libonnxruntime.so")]
    NoDylibPath,
    #[error("failed to load ONNX Runtime from {}: {message}", path.display())]
    Load { path: PathBuf, message: String },
    #[error("{} is missing; run `vitrine faces fetch`", .0.display())]
    MissingModel(PathBuf),
    #[error("failed to create an ONNX Runtime session for {}", path.display())]
    Session { path: PathBuf, source: ort::Error },
}

pub struct FaceRuntime {
    detector: Mutex<Session>,
    embedder: Mutex<Session>,
}

impl FaceRuntime {
    /// Loads ONNX Runtime from `ORT_DYLIB_PATH` on first use.
    pub fn new(models_dir: &Path, device: Device) -> Result<Self, RuntimeError> {
        init()?;
        Ok(Self {
            detector: Mutex::new(session(&models_dir.join(DETECTOR.name), device)?),
            embedder: Mutex::new(session(&models_dir.join(EMBEDDER.name), device)?),
        })
    }

    pub fn detector(&self) -> MutexGuard<'_, Session> {
        self.detector.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn embedder(&self) -> MutexGuard<'_, Session> {
        self.embedder.lock().unwrap_or_else(PoisonError::into_inner)
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

fn init() -> Result<(), RuntimeError> {
    static INIT: OnceLock<Result<(), (PathBuf, String)>> = OnceLock::new();
    let path = env::var_os("ORT_DYLIB_PATH")
        .map(PathBuf::from)
        .ok_or(RuntimeError::NoDylibPath)?;
    INIT.get_or_init(|| {
        ort::init_from(&path)
            .map(|env| {
                env.with_name("vitrine").commit();
            })
            .map_err(|e| (path, e.to_string()))
    })
    .clone()
    .map_err(|(path, message)| RuntimeError::Load { path, message })
}

fn session(path: &Path, device: Device) -> Result<Session, RuntimeError> {
    if !path.is_file() {
        return Err(RuntimeError::MissingModel(path.to_owned()));
    }
    let build = || -> ort::Result<Session> {
        let mut builder = Session::builder()?.with_intra_threads(INTRA_THREADS)?;
        if device == Device::Cuda {
            builder = builder
                .with_execution_providers([ep::CUDA::default().build().error_on_failure()])?;
        }
        builder.commit_from_file(path)
    };
    build().map_err(|source| RuntimeError::Session {
        path: path.to_owned(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::{self, RecvTimeoutError};
    use std::thread;
    use std::time::{Duration, Instant};

    use ort::value::Tensor;

    use super::*;
    use crate::faces::fixtures::models_dir;

    type Shapes = Vec<(String, Vec<i64>)>;

    fn forward(session: &mut Session, shape: [usize; 4]) -> ort::Result<Shapes> {
        for input in session.inputs() {
            eprintln!("  input  {} {}", input.name(), input.dtype());
        }
        for output in session.outputs() {
            eprintln!("  output {} {}", output.name(), output.dtype());
        }
        let zeros = Tensor::from_array((shape, vec![0.0_f32; shape.iter().product()]))?;
        let outputs = session.run(ort::inputs![zeros])?;
        outputs
            .iter()
            .map(|(name, value)| {
                let (shape, _) = value.try_extract_tensor::<f32>()?;
                eprintln!("  ran    {name} {shape}");
                Ok((name.to_owned(), shape.to_vec()))
            })
            .collect()
    }

    fn spike(dir: &Path) -> anyhow::Result<(Shapes, Shapes)> {
        let start = Instant::now();
        let runtime = FaceRuntime::new(dir, Device::Cpu)?;
        eprintln!("sessions created in {:?}", start.elapsed());
        eprintln!("{}:", DETECTOR.name);
        let det = forward(&mut runtime.detector(), [1, 3, 640, 640])?;
        eprintln!("{}:", EMBEDDER.name);
        let emb = forward(&mut runtime.embedder(), [1, 3, 112, 112])?;
        eprintln!("forward passes done after {:?}", start.elapsed());
        Ok((det, emb))
    }

    #[test]
    #[ignore = "needs ORT_DYLIB_PATH and fetched models (VITRINE_MODELS_DIR or the XDG models dir)"]
    fn sessions_run_a_zero_forward_pass() {
        let dir = models_dir();
        let (tx, rx) = mpsc::channel();
        // a mismatched ONNX Runtime dylib can deadlock in session creation
        thread::spawn(move || tx.send(spike(&dir)));
        let (det, emb) = match rx.recv_timeout(Duration::from_secs(30)) {
            Ok(result) => result.unwrap(),
            Err(RecvTimeoutError::Timeout) => panic!("no result within 30 s"),
            Err(RecvTimeoutError::Disconnected) => panic!("spike thread panicked"),
        };

        let mut rows: Vec<_> = det
            .iter()
            .map(|(_, shape)| match shape.as_slice() {
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
        assert_eq!(emb[0].1, [1, 512]);
    }

    #[test]
    fn missing_model_is_reported_before_loading() {
        let dir = tempfile::tempdir().unwrap();
        let err = session(&dir.path().join(DETECTOR.name), Device::Cpu).unwrap_err();
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
