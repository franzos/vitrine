//! The local photo folder named by `VITRINE_FACE_FIXTURES`, described by its `labels.toml`.

use std::env;
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

use crate::config::Paths;

pub struct Fixtures {
    pub dir: PathBuf,
    pub labels: Vec<Label>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LabelsFile {
    image: Vec<Label>,
}

/// One `[[image]]` entry: an exact `faces` count or a `faces_min`/`faces_max` range, and opaque labels for the people in it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Label {
    pub file: String,
    pub faces: Option<usize>,
    pub faces_min: Option<usize>,
    pub faces_max: Option<usize>,
    #[serde(default)]
    pub people: Vec<String>,
}

impl Label {
    pub fn accepts(&self, count: usize) -> bool {
        match self.faces {
            Some(faces) => count == faces,
            None => {
                self.faces_min.is_none_or(|min| count >= min)
                    && self.faces_max.is_none_or(|max| count <= max)
            }
        }
    }

    pub fn expected(&self) -> String {
        match (self.faces, self.faces_min, self.faces_max) {
            (Some(n), _, _) => n.to_string(),
            (None, min, max) => format!(
                "{}-{}",
                min.unwrap_or(0),
                max.map_or_else(String::new, |m| m.to_string())
            ),
        }
    }
}

pub fn load() -> Fixtures {
    let dir = PathBuf::from(
        env::var_os("VITRINE_FACE_FIXTURES")
            .expect("set VITRINE_FACE_FIXTURES to a folder with labels.toml"),
    );
    let text = fs::read_to_string(dir.join("labels.toml")).expect("labels.toml is readable");
    let labels = toml::from_str::<LabelsFile>(&text)
        .expect("labels.toml parses")
        .image;
    Fixtures { dir, labels }
}

pub fn models_dir() -> PathBuf {
    env::var_os("VITRINE_MODELS_DIR")
        .map_or_else(|| Paths::from_env().unwrap().models_dir(), PathBuf::from)
}
