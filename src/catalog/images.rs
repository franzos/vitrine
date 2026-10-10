use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rusqlite::types::{FromSql, FromSqlResult, ToSqlOutput, ValueRef};
use rusqlite::{Connection, OptionalExtension, Row, ToSql, params};

use crate::db::{Result, Sha256, path_str};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ImageId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Image,
    Video,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Active,
    Missing,
}

#[derive(Debug, Clone)]
pub struct NewImage<'a> {
    pub path: &'a Path,
    pub size: u64,
    pub mtime_ns: i64,
    pub sha256: Option<Sha256>,
    pub kind: Kind,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub taken_at: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub id: ImageId,
    pub path: PathBuf,
    pub size: u64,
    pub mtime_ns: i64,
    pub sha256: Option<Sha256>,
    pub kind: Kind,
    pub status: Status,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub taken_at: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    pub images: u64,
    pub videos: u64,
    pub missing: u64,
    pub unhashed: u64,
}

const COLUMNS: &str = "id, path, size, mtime_ns, sha256, kind, status, width, height, taken_at";

fn from_row(row: &Row<'_>) -> rusqlite::Result<Image> {
    Ok(Image {
        id: ImageId(row.get(0)?),
        path: PathBuf::from(row.get::<_, String>(1)?),
        size: row.get(2)?,
        mtime_ns: row.get(3)?,
        sha256: row.get(4)?,
        kind: row.get(5)?,
        status: row.get(6)?,
        width: row.get(7)?,
        height: row.get(8)?,
        taken_at: row.get(9)?,
    })
}

pub fn upsert(conn: &Connection, image: &NewImage<'_>) -> Result<ImageId> {
    let id = conn.query_row(
        "INSERT INTO images (path, size, mtime_ns, sha256, kind, status, width, height, taken_at)
         VALUES (?1, ?2, ?3, ?4, ?5, 'active', ?6, ?7, ?8)
         ON CONFLICT (path) DO UPDATE SET
             size = excluded.size, mtime_ns = excluded.mtime_ns, sha256 = excluded.sha256,
             kind = excluded.kind, status = 'active', width = excluded.width,
             height = excluded.height, taken_at = excluded.taken_at
         RETURNING id",
        params![
            path_str(image.path)?,
            image.size,
            image.mtime_ns,
            image.sha256,
            image.kind,
            image.width,
            image.height,
            image.taken_at,
        ],
        |row| row.get(0),
    )?;
    Ok(ImageId(id))
}

pub fn get_by_path(conn: &Connection, path: &Path) -> Result<Option<Image>> {
    let sql = format!("SELECT {COLUMNS} FROM images WHERE path = ?1");
    Ok(conn
        .query_row(&sql, [path_str(path)?], from_row)
        .optional()?)
}

#[cfg(test)]
pub fn find_by_sha256(conn: &Connection, sha256: &Sha256) -> Result<Vec<Image>> {
    let sql = format!("SELECT {COLUMNS} FROM images WHERE sha256 = ?1 ORDER BY path");
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([sha256], from_row)?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

pub fn set_status(conn: &Connection, id: ImageId, status: Status) -> Result<()> {
    conn.execute(
        "UPDATE images SET status = ?2 WHERE id = ?1",
        params![id.0, status],
    )?;
    Ok(())
}

pub fn relink(conn: &Connection, id: ImageId, path: &Path, size: u64, mtime_ns: i64) -> Result<()> {
    conn.execute(
        "UPDATE images SET path = ?2, size = ?3, mtime_ns = ?4, status = 'active' WHERE id = ?1",
        params![id.0, path_str(path)?, size, mtime_ns],
    )?;
    Ok(())
}

pub fn missing(conn: &Connection) -> Result<Vec<Image>> {
    let sql = format!("SELECT {COLUMNS} FROM images WHERE status = 'missing' ORDER BY path");
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], from_row)?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

pub fn under(conn: &Connection, folder: &Path) -> Result<Vec<Image>> {
    let (lower, upper) = prefix_range(path_str(folder)?);
    let sql = format!("SELECT {COLUMNS} FROM images WHERE path >= ?1 AND path < ?2 ORDER BY path");
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([lower, upper], from_row)?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub id: ImageId,
    pub path: PathBuf,
    pub sha256: Option<Sha256>,
    pub kind: Kind,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

pub fn entries_under(conn: &Connection, folder: &Path) -> Result<Vec<Entry>> {
    let (lower, upper) = prefix_range(path_str(folder)?);
    let mut stmt = conn.prepare(
        "SELECT id, path, sha256, kind, width, height FROM images
         WHERE path >= ?1 AND path < ?2 AND status = 'active' ORDER BY path",
    )?;
    let rows = stmt.query_map([lower, upper], entry_from_row)?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// Expects columns `id, path, sha256, kind, width, height` in that order.
pub(crate) fn entry_from_row(row: &Row<'_>) -> rusqlite::Result<Entry> {
    Ok(Entry {
        id: ImageId(row.get(0)?),
        path: PathBuf::from(row.get::<_, String>(1)?),
        sha256: row.get(2)?,
        kind: row.get(3)?,
        width: row.get(4)?,
        height: row.get(5)?,
    })
}

/// Counts only direct children per folder; folders without active rows are absent.
pub fn folder_counts_under(conn: &Connection, folder: &Path) -> Result<Vec<(PathBuf, u64)>> {
    let (lower, upper) = prefix_range(path_str(folder)?);
    // range scan on the path index; grouping by parent in SQL costs string functions per row
    let mut stmt = conn
        .prepare("SELECT path FROM images WHERE path >= ?1 AND path < ?2 AND status = 'active'")?;
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut rows = stmt.query([lower, upper])?;
    while let Some(row) = rows.next()? {
        let path = row.get_ref(0)?.as_str().map_err(rusqlite::Error::from)?;
        let dir = path.rsplit_once('/').map_or("", |(dir, _)| dir);
        match counts.get_mut(dir) {
            Some(n) => *n += 1,
            None => {
                counts.insert(dir.to_owned(), 1);
            }
        }
    }
    Ok(counts
        .into_iter()
        .map(|(dir, n)| (PathBuf::from(dir), n))
        .collect())
}

/// Half-open range `[folder/, folder0)`, since `0` is the byte after `/`.
pub(crate) fn prefix_range(folder: &str) -> (String, String) {
    let base = folder.trim_end_matches('/');
    (format!("{base}/"), format!("{base}0"))
}

pub fn counts(conn: &Connection) -> Result<Counts> {
    Ok(conn.query_row(
        "SELECT
             count(*) FILTER (WHERE status = 'active' AND kind = 'image'),
             count(*) FILTER (WHERE status = 'active' AND kind = 'video'),
             count(*) FILTER (WHERE status = 'missing'),
             count(*) FILTER (WHERE sha256 IS NULL AND kind = 'image')
         FROM images",
        [],
        |row| {
            Ok(Counts {
                images: row.get(0)?,
                videos: row.get(1)?,
                missing: row.get(2)?,
                unhashed: row.get(3)?,
            })
        },
    )?)
}

impl Kind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Video => "video",
        }
    }
}

impl ToSql for Kind {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(self.as_str().into())
    }
}

impl FromSql for Kind {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        super::enum_from_sql(value, |s| match s {
            "image" => Some(Self::Image),
            "video" => Some(Self::Video),
            _ => None,
        })
    }
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Missing => "missing",
        }
    }
}

impl ToSql for Status {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(self.as_str().into())
    }
}

impl FromSql for Status {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        super::enum_from_sql(value, |s| match s {
            "active" => Some(Self::Active),
            "missing" => Some(Self::Missing),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::temp;

    fn new(path: &str) -> NewImage<'_> {
        NewImage {
            path: Path::new(path),
            size: 100,
            mtime_ns: 1_700_000_000_000_000_000,
            sha256: None,
            kind: Kind::Image,
            width: None,
            height: None,
            taken_at: None,
        }
    }

    fn paths(images: &[Image]) -> Vec<&str> {
        images.iter().map(|i| i.path.to_str().unwrap()).collect()
    }

    #[test]
    fn upsert_round_trips_and_updates_in_place() {
        let (_dir, conn) = temp();
        let id = upsert(&conn, &new("/p/a.jpg")).unwrap();
        let mut img = new("/p/a.jpg");
        img.sha256 = Some([7; 32]);
        img.width = Some(4000);
        img.height = Some(3000);
        img.taken_at = Some("2024-05-01T10:11:12");
        set_status(&conn, id, Status::Missing).unwrap();
        assert_eq!(upsert(&conn, &img).unwrap(), id);

        let got = get_by_path(&conn, Path::new("/p/a.jpg")).unwrap().unwrap();
        assert_eq!(
            got,
            Image {
                id,
                path: PathBuf::from("/p/a.jpg"),
                size: 100,
                mtime_ns: 1_700_000_000_000_000_000,
                sha256: Some([7; 32]),
                kind: Kind::Image,
                status: Status::Active,
                width: Some(4000),
                height: Some(3000),
                taken_at: Some("2024-05-01T10:11:12".to_owned()),
            }
        );
        assert!(get_by_path(&conn, Path::new("/p/b.jpg")).unwrap().is_none());
    }

    #[test]
    fn find_by_sha256_returns_all_copies() {
        let (_dir, conn) = temp();
        for path in ["/p/b.jpg", "/p/a.jpg", "/p/c.jpg"] {
            let mut img = new(path);
            img.sha256 = Some(if path == "/p/c.jpg" { [2; 32] } else { [1; 32] });
            upsert(&conn, &img).unwrap();
        }
        let found = find_by_sha256(&conn, &[1; 32]).unwrap();
        assert_eq!(paths(&found), ["/p/a.jpg", "/p/b.jpg"]);
    }

    #[test]
    fn relink_moves_row_and_keeps_content() {
        let (_dir, conn) = temp();
        let mut img = new("/p/a.jpg");
        img.sha256 = Some([3; 32]);
        img.width = Some(10);
        img.taken_at = Some("2024-05-01T10:11:12");
        let id = upsert(&conn, &img).unwrap();
        set_status(&conn, id, Status::Missing).unwrap();
        assert_eq!(paths(&missing(&conn).unwrap()), ["/p/a.jpg"]);

        relink(&conn, id, Path::new("/q/a.jpg"), 101, 5).unwrap();
        assert!(get_by_path(&conn, Path::new("/p/a.jpg")).unwrap().is_none());
        let got = get_by_path(&conn, Path::new("/q/a.jpg")).unwrap().unwrap();
        assert_eq!(
            (got.id, got.size, got.mtime_ns, got.status),
            (id, 101, 5, Status::Active)
        );
        assert_eq!(got.sha256, Some([3; 32]));
        assert_eq!(got.width, Some(10));
        assert_eq!(got.taken_at.as_deref(), Some("2024-05-01T10:11:12"));
        assert!(missing(&conn).unwrap().is_empty());
    }

    #[test]
    fn under_lists_descendants_in_order() {
        let (_dir, conn) = temp();
        for path in [
            "/p/2024/b.jpg",
            "/p/2024/01/a.jpg",
            "/p/2024-old/x.jpg",
            "/p/20240/y.jpg",
            "/p/2023/z.jpg",
            "/q/a.jpg",
        ] {
            upsert(&conn, &new(path)).unwrap();
        }
        let got = under(&conn, Path::new("/p/2024")).unwrap();
        assert_eq!(paths(&got), ["/p/2024/01/a.jpg", "/p/2024/b.jpg"]);
        let got = under(&conn, Path::new("/p/2024/")).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(under(&conn, Path::new("/")).unwrap().len(), 6);
    }

    #[test]
    fn entries_and_folder_counts_skip_missing_rows() {
        let (_dir, conn) = temp();
        for path in [
            "/p/b.jpg",
            "/p/2024/01/a.jpg",
            "/p/2024/01/b.jpg",
            "/p/2024/c.mp4",
            "/p/gone/x.jpg",
            "/q/a.jpg",
        ] {
            let mut img = new(path);
            if path.ends_with(".mp4") {
                img.kind = Kind::Video;
            }
            let id = upsert(&conn, &img).unwrap();
            if path.starts_with("/p/gone") {
                set_status(&conn, id, Status::Missing).unwrap();
            }
        }
        let entries = entries_under(&conn, Path::new("/p")).unwrap();
        let got: Vec<_> = entries.iter().map(|e| e.path.to_str().unwrap()).collect();
        assert_eq!(
            got,
            [
                "/p/2024/01/a.jpg",
                "/p/2024/01/b.jpg",
                "/p/2024/c.mp4",
                "/p/b.jpg"
            ]
        );
        assert_eq!(entries[2].kind, Kind::Video);
        assert_eq!(
            folder_counts_under(&conn, Path::new("/p")).unwrap(),
            [
                (PathBuf::from("/p"), 1),
                (PathBuf::from("/p/2024"), 1),
                (PathBuf::from("/p/2024/01"), 2),
            ]
        );
    }

    #[test]
    fn under_uses_path_index() {
        let (_dir, conn) = temp();
        let plan: String = conn
            .query_row(
                "EXPLAIN QUERY PLAN SELECT id FROM images WHERE path >= ?1 AND path < ?2 ORDER BY path",
                ["/a/", "/a0"],
                |r| r.get(3),
            )
            .unwrap();
        assert!(plan.contains("INDEX sqlite_autoindex_images_1"), "{plan}");
    }

    #[test]
    fn counts_by_kind_and_status() {
        let (_dir, conn) = temp();
        let a = upsert(&conn, &new("/p/a.jpg")).unwrap();
        let mut hashed = new("/p/b.jpg");
        hashed.sha256 = Some([1; 32]);
        upsert(&conn, &hashed).unwrap();
        let mut video = new("/p/c.mp4");
        video.kind = Kind::Video;
        upsert(&conn, &video).unwrap();
        set_status(&conn, a, Status::Missing).unwrap();
        assert_eq!(
            counts(&conn).unwrap(),
            Counts {
                images: 1,
                videos: 1,
                missing: 1,
                unhashed: 1,
            }
        );
    }
}
