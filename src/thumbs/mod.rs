//! Append new migrations to [`migrations`]; never edit a shipped one.

pub mod generate;

use std::collections::HashSet;
use std::path::Path;

use rusqlite::{Connection, OptionalExtension, params};
use rusqlite_migration::{M, Migrations};

use crate::db::{Result, Sha256};

const M0001_INITIAL: &str = r"
CREATE TABLE thumbs (
    sha256 BLOB    NOT NULL CHECK (length(sha256) = 32),
    bucket INTEGER NOT NULL,
    jpeg   BLOB    NOT NULL,
    width  INTEGER NOT NULL,
    height INTEGER NOT NULL,
    PRIMARY KEY (sha256, bucket)
);
";

fn migrations() -> Migrations<'static> {
    Migrations::new(vec![M::up(M0001_INITIAL)])
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Thumb {
    pub jpeg: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

pub fn open(path: &Path) -> Result<Connection> {
    crate::db::open(path, &migrations())
}

pub fn get(conn: &Connection, sha256: &Sha256, bucket: u32) -> Result<Option<Thumb>> {
    Ok(conn
        .query_row(
            "SELECT jpeg, width, height FROM thumbs WHERE sha256 = ?1 AND bucket = ?2",
            params![sha256, bucket],
            |row| {
                Ok(Thumb {
                    jpeg: row.get(0)?,
                    width: row.get(1)?,
                    height: row.get(2)?,
                })
            },
        )
        .optional()?)
}

/// The buckets stored for `sha256`, smallest first.
pub fn buckets(conn: &Connection, sha256: &Sha256) -> Result<Vec<u32>> {
    let mut stmt = conn.prepare("SELECT bucket FROM thumbs WHERE sha256 = ?1 ORDER BY bucket")?;
    let rows = stmt.query_map([sha256], |row| row.get(0))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

pub fn keys(conn: &Connection) -> Result<HashSet<(Sha256, u32)>> {
    let mut stmt = conn.prepare("SELECT sha256, bucket FROM thumbs")?;
    let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// Stores `thumb`, replacing any existing one for the same key.
pub fn put(conn: &Connection, sha256: &Sha256, bucket: u32, thumb: &Thumb) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO thumbs (sha256, bucket, jpeg, width, height)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![sha256, bucket, thumb.jpeg, thumb.width, thumb.height],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::sha256_hex;

    #[test]
    fn migrations_are_valid() {
        migrations().validate().unwrap();
    }

    /// Shipped migrations must not change; ship a new migration and append a pin instead.
    #[test]
    fn migration_fingerprints_are_pinned() {
        let pins: &[(&str, &str)] = &[(
            "M0001_INITIAL",
            "4517c7c1a069534e08650f0144eb11a28f98aec9e0f8697c838ea04148e96399",
        )];
        let sources: &[(&str, &str)] = &[("M0001_INITIAL", M0001_INITIAL)];
        assert_eq!(
            sources.len(),
            pins.len(),
            "add a new pin when shipping a new migration"
        );
        for ((name, src), (pin_name, pin_hash)) in sources.iter().zip(pins) {
            assert_eq!(name, pin_name, "pin order must match source order");
            assert_eq!(
                &sha256_hex(src),
                pin_hash,
                "migration `{name}` changed; ship a new migration instead"
            );
        }
    }

    #[test]
    fn put_get_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let conn = open(&dir.path().join("cache/thumbs.db")).unwrap();
        let thumb = Thumb {
            jpeg: vec![0xff, 0xd8, 1, 2],
            width: 256,
            height: 192,
        };
        put(&conn, &[1; 32], 256, &thumb).unwrap();
        assert_eq!(get(&conn, &[1; 32], 256).unwrap(), Some(thumb.clone()));
        assert_eq!(get(&conn, &[1; 32], 512).unwrap(), None);
        assert_eq!(get(&conn, &[2; 32], 256).unwrap(), None);

        let replaced = Thumb {
            width: 255,
            ..thumb
        };
        put(&conn, &[1; 32], 256, &replaced).unwrap();
        assert_eq!(get(&conn, &[1; 32], 256).unwrap(), Some(replaced.clone()));

        put(&conn, &[1; 32], 512, &replaced).unwrap();
        put(&conn, &[2; 32], 256, &replaced).unwrap();
        assert_eq!(buckets(&conn, &[1; 32]).unwrap(), [256, 512]);
        assert_eq!(buckets(&conn, &[3; 32]).unwrap(), Vec::<u32>::new());
        assert_eq!(
            keys(&conn).unwrap(),
            HashSet::from([([1; 32], 256), ([1; 32], 512), ([2; 32], 256)])
        );
    }
}
