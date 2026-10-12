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
    pub fn geo_db(&self) -> PathBuf {
        self.data_dir.join("geo.db")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geo_db_sits_in_data_dir() {
        let paths = Paths {
            home_dir: PathBuf::from("/home/test"),
            config_file: PathBuf::from("/home/test/.config/vitrine/config.toml"),
            data_dir: PathBuf::from("/home/test/.local/share/vitrine"),
            cache_dir: PathBuf::from("/home/test/.cache/vitrine"),
        };
        assert_eq!(
            paths.geo_db(),
            PathBuf::from("/home/test/.local/share/vitrine/geo.db")
        );
    }
}
