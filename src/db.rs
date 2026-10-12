use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, ErrorCode, OpenFlags};
use rusqlite_migration::Migrations;

const BUSY_TIMEOUT: Duration = Duration::from_secs(5);
const BUSY_RETRY: Duration = Duration::from_millis(10);

pub type Sha256 = [u8; 32];

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to create {}", path.display())]
    CreateDir { path: PathBuf, source: io::Error },
    #[error("failed to open {}", path.display())]
    Open {
        path: PathBuf,
        source: rusqlite::Error,
    },
    #[error("failed to migrate {}", path.display())]
    Migrate {
        path: PathBuf,
        source: rusqlite_migration::Error,
    },
    #[error("{} is not valid UTF-8", .0.display())]
    NonUtf8Path(PathBuf),
    #[error("face model {model} expects {expected}-d embeddings, got {actual}")]
    EmbeddingDim {
        model: i64,
        expected: usize,
        actual: usize,
    },
    #[error("people {0} and {1} belong to different face models")]
    ModelMismatch(i64, i64),
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
}

pub(crate) fn open(path: &Path, migrations: &Migrations) -> Result<Connection> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|source| Error::CreateDir {
            path: parent.to_owned(),
            source,
        })?;
    }
    let deadline = Instant::now() + BUSY_TIMEOUT;
    loop {
        match open_and_migrate(path, migrations) {
            // switching a new file to WAL returns BUSY at once, bypassing the busy handler
            Err(Error::Open {
                source: rusqlite::Error::SqliteFailure(e, _),
                ..
            }) if e.code == ErrorCode::DatabaseBusy && Instant::now() < deadline => {
                thread::sleep(BUSY_RETRY);
            }
            result => return result,
        }
    }
}

fn open_and_migrate(path: &Path, migrations: &Migrations) -> Result<Connection> {
    let open_err = |source| Error::Open {
        path: path.to_owned(),
        source,
    };
    let mut conn = Connection::open(path).map_err(open_err)?;
    configure(&conn).map_err(open_err)?;
    if let Err(source) = migrations.to_latest(&mut conn) {
        // another connection can migrate between the version check and the migration transaction
        if migrations.pending_migrations(&conn).ok() != Some(0) {
            return Err(Error::Migrate {
                path: path.to_owned(),
                source,
            });
        }
    }
    Ok(conn)
}

/// Milliseconds since the Unix epoch, the unit of `set_at` columns.
pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

pub(crate) fn open_read_only(path: &Path) -> Result<Connection> {
    let open_err = |source| Error::Open {
        path: path.to_owned(),
        source,
    };
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(open_err)?;
    conn.busy_timeout(BUSY_TIMEOUT).map_err(open_err)?;
    Ok(conn)
}

fn configure(conn: &Connection) -> rusqlite::Result<()> {
    conn.busy_timeout(BUSY_TIMEOUT)?;
    conn.pragma_update_and_check(None, "journal_mode", "WAL", |row| row.get::<_, String>(0))?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "foreign_keys", true)?;
    Ok(())
}

pub(crate) fn path_str(path: &Path) -> Result<&str> {
    path.to_str()
        .ok_or_else(|| Error::NonUtf8Path(path.to_owned()))
}

#[cfg(test)]
pub(crate) fn sha256_hex(s: &str) -> String {
    use sha2::Digest;
    use std::fmt::Write;
    sha2::Sha256::digest(s.as_bytes())
        .iter()
        .fold(String::with_capacity(64), |mut out, b| {
            let _ = write!(out, "{b:02x}");
            out
        })
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;
    use std::thread;

    #[test]
    fn concurrent_opens_of_a_new_database_all_succeed() {
        for _ in 0..20 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("catalog.db");
            let barrier = Barrier::new(4);
            thread::scope(|s| {
                let handles: Vec<_> = (0..4)
                    .map(|_| {
                        s.spawn(|| {
                            barrier.wait();
                            crate::catalog::open(&path)
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().unwrap().unwrap();
                }
            });
        }
    }
}
