//! The catalog database: images by path, and face and location data by content hash.

pub mod faces;
pub mod filter;
pub mod geo;
pub mod images;
pub mod people;
pub mod runs;
mod schema;

use std::path::Path;

use rusqlite::Connection;
use rusqlite::types::{FromSqlError, FromSqlResult, ValueRef};

use crate::db::Result;

pub fn open(path: &Path) -> Result<Connection> {
    crate::db::open(path, &schema::migrations())
}

pub fn open_read_only(path: &Path) -> Result<Connection> {
    crate::db::open_read_only(path)
}

fn enum_from_sql<T>(
    value: ValueRef<'_>,
    parse: impl FnOnce(&str) -> Option<T>,
) -> FromSqlResult<T> {
    let s = value.as_str()?;
    parse(s).ok_or_else(|| FromSqlError::Other(format!("unexpected value `{s}`").into()))
}

#[cfg(test)]
pub(crate) fn edit(
    conn: &mut Connection,
    f: impl FnOnce(&rusqlite::Transaction<'_>) -> Result<()>,
) {
    let tx = conn.transaction().unwrap();
    f(&tx).unwrap();
    tx.commit().unwrap();
}

#[cfg(test)]
pub(crate) fn temp() -> (tempfile::TempDir, Connection) {
    let dir = tempfile::tempdir().unwrap();
    let conn = open(&dir.path().join("catalog.db")).unwrap();
    (dir, conn)
}

/// A catalog migrated only up to schema `version`.
#[cfg(test)]
pub(crate) fn open_at_version(path: &Path, version: usize) -> Connection {
    let mut conn = Connection::open(path).unwrap();
    schema::migrations().to_version(&mut conn, version).unwrap();
    conn
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_configures_connection() {
        let (dir, conn) = temp();
        let mode: String = conn
            .pragma_query_value(None, "journal_mode", |r| r.get(0))
            .unwrap();
        assert_eq!(mode, "wal");
        let fk: bool = conn
            .pragma_query_value(None, "foreign_keys", |r| r.get(0))
            .unwrap();
        assert!(fk);
        drop(conn);
        let again = open(&dir.path().join("catalog.db")).unwrap();
        let version: i64 = again
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(version, 10);
    }

    #[test]
    fn read_only_rejects_writes_and_sees_committed_rows() {
        let (dir, conn) = temp();
        let ro = open_read_only(&dir.path().join("catalog.db")).unwrap();
        conn.execute(
            "INSERT INTO face_models (name, detector, dim) VALUES ('m', 'd', 4)",
            [],
        )
        .unwrap();
        let n: i64 = ro
            .query_row("SELECT count(*) FROM face_models", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
        assert!(ro.execute("DELETE FROM face_models", []).is_err());
    }
}
