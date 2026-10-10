use std::fs::{self, File, TryLockError};
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum LockError {
    #[error("another indexer is running (lock held on {})", path.display())]
    Held { path: PathBuf },
    #[error("failed to lock {}", path.display())]
    Io { path: PathBuf, source: io::Error },
}

/// Exclusive indexer lock, an OS advisory lock on a file, released on drop.
#[derive(Debug)]
pub struct IndexLock {
    _file: File,
}

impl IndexLock {
    /// Non-blocking; [`LockError::Held`] when any handle holds it, in this process or another.
    pub fn acquire(path: &Path) -> Result<Self, LockError> {
        let io_err = |source| LockError::Io {
            path: path.to_owned(),
            source,
        };
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(io_err)?;
        }
        let file = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)
            .map_err(io_err)?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file }),
            Err(TryLockError::WouldBlock) => Err(LockError::Held {
                path: path.to_owned(),
            }),
            Err(TryLockError::Error(e)) => Err(io_err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_acquire_fails_until_first_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data/index.lock");
        let first = IndexLock::acquire(&path).unwrap();
        let err = IndexLock::acquire(&path).unwrap_err();
        assert!(matches!(err, LockError::Held { .. }), "{err:?}");
        assert!(err.to_string().contains("another indexer is running"));
        drop(first);
        IndexLock::acquire(&path).unwrap();
    }
}
