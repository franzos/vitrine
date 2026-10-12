use rusqlite::{Connection, OptionalExtension, params};

use crate::db::Result;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunStats {
    pub files: u64,
    pub hashed: u64,
    pub added: u64,
    /// Known paths with a new size or mtime.
    pub changed: u64,
    /// Rows relinked to a new path holding their content.
    pub moved: u64,
    /// Missing rows whose file is back unchanged.
    pub restored: u64,
    /// Rows marked missing by this run.
    pub vanished: u64,
    pub failed: u64,
    pub thumbnailed: u64,
    pub faces_scanned: u64,
    /// Before the face count filter.
    pub faces_found: u64,
    pub people_created: u64,
    pub faces_assigned: u64,
    /// Scanned content with EXIF coordinates.
    pub gps_found: u64,
    /// EXIF locations written.
    pub places_resolved: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Run {
    /// Local time, `%Y-%m-%d %H:%M:%S`.
    pub finished_at: String,
    pub stats: RunStats,
}

pub fn record(conn: &Connection, stats: &RunStats) -> Result<()> {
    conn.execute(
        "INSERT INTO index_runs
             (files, hashed, added, changed, moved, restored, vanished, failed, thumbnailed,
              faces_scanned, faces_found, people_created, faces_assigned, gps_found,
              places_resolved)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
        params![
            stats.files,
            stats.hashed,
            stats.added,
            stats.changed,
            stats.moved,
            stats.restored,
            stats.vanished,
            stats.failed,
            stats.thumbnailed,
            stats.faces_scanned,
            stats.faces_found,
            stats.people_created,
            stats.faces_assigned,
            stats.gps_found,
            stats.places_resolved,
        ],
    )?;
    Ok(())
}

pub fn last(conn: &Connection) -> Result<Option<Run>> {
    Ok(conn
        .query_row(
            "SELECT datetime(finished_at, 'unixepoch', 'localtime'),
                    files, hashed, added, changed, moved, restored, vanished, failed, thumbnailed,
                    faces_scanned, faces_found, people_created, faces_assigned, gps_found,
                    places_resolved
             FROM index_runs ORDER BY id DESC LIMIT 1",
            [],
            |row| {
                Ok(Run {
                    finished_at: row.get(0)?,
                    stats: RunStats {
                        files: row.get(1)?,
                        hashed: row.get(2)?,
                        added: row.get(3)?,
                        changed: row.get(4)?,
                        moved: row.get(5)?,
                        restored: row.get(6)?,
                        vanished: row.get(7)?,
                        failed: row.get(8)?,
                        thumbnailed: row.get(9)?,
                        faces_scanned: row.get(10)?,
                        faces_found: row.get(11)?,
                        people_created: row.get(12)?,
                        faces_assigned: row.get(13)?,
                        gps_found: row.get(14)?,
                        places_resolved: row.get(15)?,
                    },
                })
            },
        )
        .optional()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::temp;

    #[test]
    fn last_returns_latest_run() {
        let (_dir, conn) = temp();
        assert_eq!(last(&conn).unwrap(), None);
        let first = RunStats {
            files: 3,
            hashed: 3,
            added: 3,
            thumbnailed: 2,
            faces_scanned: 3,
            faces_found: 5,
            people_created: 1,
            faces_assigned: 4,
            gps_found: 2,
            places_resolved: 1,
            ..RunStats::default()
        };
        let second = RunStats {
            files: 3,
            ..RunStats::default()
        };
        record(&conn, &first).unwrap();
        assert_eq!(last(&conn).unwrap().unwrap().stats, first);
        record(&conn, &second).unwrap();
        let run = last(&conn).unwrap().unwrap();
        assert_eq!(run.stats, second);
        assert_eq!(run.finished_at.len(), "2026-10-08 12:00:00".len());
    }
}
