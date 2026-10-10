use std::path::PathBuf;

use directories::{BaseDirs, ProjectDirs};

use super::ConfigError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    pub home_dir: PathBuf,
    pub config_file: PathBuf,
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
}

impl Paths {
    pub fn from_env() -> Result<Self, ConfigError> {
        let base = BaseDirs::new().ok_or(ConfigError::NoHome)?;
        let project = ProjectDirs::from("", "", "vitrine").ok_or(ConfigError::NoHome)?;
        Ok(Self {
            home_dir: base.home_dir().to_owned(),
            config_file: project.config_dir().join("config.toml"),
            data_dir: project.data_dir().to_owned(),
            cache_dir: project.cache_dir().to_owned(),
        })
    }

    #[must_use]
    pub fn catalog_db(&self) -> PathBuf {
        self.data_dir.join("catalog.db")
    }

    #[must_use]
    pub fn index_lock(&self) -> PathBuf {
        self.data_dir.join("index.lock")
    }

    #[must_use]
    pub fn models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }

    #[must_use]
    pub fn thumbs_db(&self) -> PathBuf {
        self.cache_dir.join("thumbs.db")
    }
}
