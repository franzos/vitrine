use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use rusqlite::types::{FromSql, FromSqlResult, ToSqlOutput, ValueRef};
use rusqlite::{Connection, OptionalExtension, Row, ToSql, Transaction, params};

use super::images::{ImageId, id_array, prefix_range};
use crate::db::{Result, Sha256, path_str};
use crate::geo::Place;
use crate::geo::events::{Item, When};

const RECENT: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Exif,
    Manual,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Location {
    /// `None` is a manual "no location" marker.
    pub place: Option<Place>,
    pub source: Source,
}

/// Active image content with no `geo_scans` row, as [`super::faces::pending`] scopes and orders it.
pub fn unscanned(conn: &Connection, roots: &[PathBuf]) -> Result<Vec<(Sha256, PathBuf)>> {
    let mut stmt = conn.prepare(
        "SELECT i.sha256, min(i.path) FROM images i
         WHERE i.status = 'active' AND i.kind = 'image' AND i.sha256 IS NOT NULL
           AND (i.path = ?1 OR (i.path >= ?2 AND i.path < ?3))
           AND NOT EXISTS (SELECT 1 FROM geo_scans s WHERE s.sha256 = i.sha256)
         GROUP BY i.sha256
         ORDER BY 2",
    )?;
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for root in roots {
        let root = path_str(root)?;
        let (lower, upper) = prefix_range(root);
        let rows = stmt.query_map(params![root, lower, upper], |row| {
            Ok((row.get(0)?, PathBuf::from(row.get::<_, String>(1)?)))
        })?;
        for row in rows {
            let (sha256, path): (Sha256, PathBuf) = row?;
            if seen.insert(sha256) {
                out.push((sha256, path));
            }
        }
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    pub exif: u64,
    pub manual: u64,
    /// Coordinates without a location, waiting for a gazetteer.
    pub pending: u64,
}

/// Active images by location source, leaving out "no location" markers.
pub fn counts(conn: &Connection) -> Result<Counts> {
    Ok(conn.query_row(
        "SELECT
             count(*) FILTER (WHERE l.source = 'exif' AND l.place_id IS NOT NULL),
             count(*) FILTER (WHERE l.source = 'manual' AND l.place_id IS NOT NULL),
             count(*) FILTER (WHERE l.sha256 IS NULL AND s.lat IS NOT NULL)
         FROM images i
         LEFT JOIN locations l ON l.sha256 = i.sha256
         LEFT JOIN geo_scans s ON s.sha256 = i.sha256
         WHERE i.status = 'active' AND i.kind = 'image' AND i.sha256 IS NOT NULL",
        [],
        |row| {
            Ok(Counts {
                exif: row.get(0)?,
                manual: row.get(1)?,
                pending: row.get(2)?,
            })
        },
    )?)
}

/// Up to `limit` scans with coordinates and no location, ordered by `sha256` and starting after `after`.
pub fn unresolved(
    conn: &Connection,
    after: Option<&Sha256>,
    limit: usize,
) -> Result<Vec<(Sha256, f64, f64)>> {
    let mut stmt = conn.prepare_cached(
        "SELECT s.sha256, s.lat, s.lon FROM geo_scans s
         WHERE s.lat IS NOT NULL AND (?1 IS NULL OR s.sha256 > ?1)
           AND NOT EXISTS (SELECT 1 FROM locations l WHERE l.sha256 = s.sha256)
         ORDER BY s.sha256
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![after, limit], |row| {
        Ok((row.get(0)?, row.get(1)?, row.get(2)?))
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// `None` records content without usable GPS so it isn't read again.
pub fn insert_scan(tx: &Transaction<'_>, sha256: &Sha256, gps: Option<(f64, f64)>) -> Result<()> {
    let (lat, lon) = gps.unzip();
    tx.execute(
        "INSERT INTO geo_scans (sha256, lat, lon) VALUES (?1, ?2, ?3)",
        params![sha256, lat, lon],
    )?;
    Ok(())
}

/// Refreshes the names of a known place; its id stays, and `None` coordinates keep stored ones.
pub fn upsert_place(tx: &Transaction<'_>, place: &Place) -> Result<()> {
    let (lat, lon) = place.coords.unzip();
    tx.execute(
        "INSERT INTO places (geoname_id, name, region_code, region, country_code, country, lat, lon)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
         ON CONFLICT (geoname_id) DO UPDATE SET
             name = excluded.name, region_code = excluded.region_code, region = excluded.region,
             country_code = excluded.country_code, country = excluded.country,
             lat = COALESCE(excluded.lat, lat), lon = COALESCE(excluded.lon, lon)",
        params![
            place.geoname_id,
            place.name,
            place.region_code,
            place.region,
            place.country_code,
            place.country,
            lat,
            lon,
        ],
    )?;
    Ok(())
}

/// Ids of places stored without coordinates, ascending.
pub fn places_without_coords(conn: &Connection) -> Result<Vec<i64>> {
    let mut stmt =
        conn.prepare_cached("SELECT geoname_id FROM places WHERE lat IS NULL ORDER BY geoname_id")?;
    let rows = stmt.query_map([], |row| row.get(0))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// `false` when the place is unknown or already has coordinates.
pub fn set_place_coords(tx: &Transaction<'_>, geoname_id: i64, lat: f64, lon: f64) -> Result<bool> {
    let changed = tx.execute(
        "UPDATE places SET lat = ?2, lon = ?3 WHERE geoname_id = ?1 AND lat IS NULL",
        params![geoname_id, lat, lon],
    )?;
    Ok(changed == 1)
}

/// `false` when the content already has a location of any source, a marker included.
pub fn insert_exif(
    tx: &Transaction<'_>,
    sha256: &Sha256,
    place: &Place,
    set_at: i64,
) -> Result<bool> {
    upsert_place(tx, place)?;
    let changed = tx.execute(
        "INSERT INTO locations (sha256, place_id, source, set_at) VALUES (?1, ?2, 'exif', ?3)
         ON CONFLICT (sha256) DO NOTHING",
        params![sha256, place.geoname_id, set_at],
    )?;
    Ok(changed == 1)
}

pub fn set_manual(tx: &Transaction<'_>, sha256: &Sha256, place: &Place, set_at: i64) -> Result<()> {
    upsert_place(tx, place)?;
    upsert_manual(tx, sha256, Some(place.geoname_id), set_at)
}

/// Leaves a marker so a later index run doesn't bring an EXIF location back.
pub fn clear(tx: &Transaction<'_>, sha256: &Sha256, set_at: i64) -> Result<()> {
    upsert_manual(tx, sha256, None, set_at)
}

fn upsert_manual(
    tx: &Transaction<'_>,
    sha256: &Sha256,
    place_id: Option<i64>,
    set_at: i64,
) -> Result<()> {
    tx.execute(
        "INSERT INTO locations (sha256, place_id, source, set_at) VALUES (?1, ?2, 'manual', ?3)
         ON CONFLICT (sha256) DO UPDATE SET
             place_id = excluded.place_id, source = 'manual', set_at = excluded.set_at",
        params![sha256, place_id, set_at],
    )?;
    Ok(())
}

/// Writes manual rows only for content with no location; returns how many.
pub fn apply_event(
    tx: &Transaction<'_>,
    shas: &[Sha256],
    place: &Place,
    set_at: i64,
) -> Result<usize> {
    upsert_place(tx, place)?;
    // `WHERE true` keeps SQLite from parsing ON CONFLICT as a join constraint
    Ok(tx.execute(
        "INSERT INTO locations (sha256, place_id, source, set_at)
         SELECT unhex(value), ?2, 'manual', ?3 FROM json_each(?1) WHERE true
         ON CONFLICT (sha256) DO NOTHING",
        params![hex_array(shas), place.geoname_id, set_at],
    )?)
}

fn hex_array(shas: &[Sha256]) -> String {
    let mut out = String::with_capacity(2 + shas.len() * 67);
    out.push('[');
    for (i, sha) in shas.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        for byte in sha {
            // writing to a String can't fail
            _ = write!(out, "{byte:02x}");
        }
        out.push('"');
    }
    out.push(']');
    out
}

pub fn location_of(conn: &Connection, sha256: &Sha256) -> Result<Option<Location>> {
    Ok(conn
        .query_row(
            "SELECT l.source, p.geoname_id, p.name, p.region_code, p.region, p.country_code,
                    p.country, p.lat, p.lon
             FROM locations l LEFT JOIN places p ON p.geoname_id = l.place_id
             WHERE l.sha256 = ?1",
            [sha256],
            |row| {
                let place = match row.get::<_, Option<i64>>(1)? {
                    Some(_) => Some(place_from(row, 1)?),
                    None => None,
                };
                Ok(Location {
                    place,
                    source: row.get(0)?,
                })
            },
        )
        .optional()?)
}

#[derive(Debug, Clone, PartialEq)]
pub enum LocationSummary {
    None,
    /// `source` is `None` when the content mixes EXIF and manual locations.
    One {
        place: Place,
        source: Option<Source>,
    },
    /// Distinct places, with unlocated content counting as one.
    Mixed(usize),
}

/// Effective location of distinct content; a "no location" marker counts as unlocated.
pub fn summarize(conn: &Connection, shas: &[Sha256]) -> Result<LocationSummary> {
    let mut stmt = conn.prepare_cached(
        "WITH c AS (
             SELECT l.place_id, l.source
             FROM (SELECT DISTINCT unhex(value) AS sha256 FROM json_each(?1)) j
             LEFT JOIN locations l ON l.sha256 = j.sha256
         ), a AS (
             SELECT count(DISTINCT place_id) AS places,
                    count(*) FILTER (WHERE place_id IS NULL) AS unlocated,
                    count(*) FILTER (WHERE place_id IS NOT NULL AND source = 'exif') AS exif,
                    count(*) FILTER (WHERE place_id IS NOT NULL AND source = 'manual') AS manual,
                    max(place_id) AS place_id
             FROM c
         )
         SELECT a.places, a.unlocated, a.exif, a.manual,
                p.geoname_id, p.name, p.region_code, p.region, p.country_code, p.country,
                p.lat, p.lon
         FROM a LEFT JOIN places p ON p.geoname_id = a.place_id",
    )?;
    Ok(stmt.query_row([hex_array(shas)], |row| {
        let places: usize = row.get(0)?;
        let unlocated: u64 = row.get(1)?;
        Ok(match places {
            0 => LocationSummary::None,
            1 if unlocated == 0 => {
                let (exif, manual): (u64, u64) = (row.get(2)?, row.get(3)?);
                LocationSummary::One {
                    place: place_from(row, 4)?,
                    source: match (exif, manual) {
                        (_, 0) => Some(Source::Exif),
                        (0, _) => Some(Source::Manual),
                        _ => None,
                    },
                }
            }
            n => LocationSummary::Mixed(n + usize::from(unlocated > 0)),
        })
    })?)
}

/// Places set by hand, most recently used first.
pub fn recent(conn: &Connection) -> Result<Vec<Place>> {
    let mut stmt = conn.prepare(
        "SELECT p.geoname_id, p.name, p.region_code, p.region, p.country_code, p.country,
                p.lat, p.lon
         FROM locations l JOIN places p ON p.geoname_id = l.place_id
         WHERE l.source = 'manual'
         GROUP BY l.place_id
         ORDER BY max(l.set_at) DESC, l.place_id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map([RECENT], |row| place_from(row, 0))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// Active hashed images with a capture time or a dated file name; videos never get a location.
pub fn event_items(conn: &Connection) -> Result<Vec<Item>> {
    let mut stmt = conn.prepare(
        "SELECT id, taken_at, path FROM images
         WHERE status = 'active' AND kind = 'image' AND sha256 IS NOT NULL",
    )?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let taken_at: Option<String> = row.get(1)?;
        let path: String = row.get(2)?;
        let stem = Path::new(&path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if let Some(when) = When::of(taken_at.as_deref(), stem) {
            out.push(Item {
                id: ImageId(row.get(0)?),
                when,
            });
        }
    }
    Ok(out)
}

/// Distinct image content of `ids` with no location; a "no location" marker counts as located.
pub fn unlocated(conn: &Connection, ids: &[ImageId]) -> Result<Vec<Sha256>> {
    let mut stmt = conn.prepare(
        "SELECT DISTINCT i.sha256 FROM json_each(?1) j JOIN images i ON i.id = j.value
         WHERE i.sha256 IS NOT NULL AND i.kind = 'image'
           AND NOT EXISTS (SELECT 1 FROM locations l WHERE l.sha256 = i.sha256)
         ORDER BY i.sha256",
    )?;
    let rows = stmt.query_map([id_array(ids)], |row| row.get(0))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// `(country_code, country, images)` for countries with located active images, by name.
pub fn countries(conn: &Connection) -> Result<Vec<(String, String, u64)>> {
    let mut stmt = conn.prepare(
        "SELECT p.country_code, max(p.country), count(*) FROM images i
         JOIN locations l ON l.sha256 = i.sha256
         JOIN places p ON p.geoname_id = l.place_id
         WHERE i.status = 'active' AND i.kind = 'image'
         GROUP BY p.country_code
         ORDER BY 2, 1",
    )?;
    let rows = stmt.query_map([], counted)?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// `(region_code, region, images)` within `country_code`, by name; a region name may be empty.
pub fn regions(conn: &Connection, country_code: &str) -> Result<Vec<(String, String, u64)>> {
    let mut stmt = conn.prepare(
        "SELECT p.region_code, max(p.region), count(*) FROM images i
         JOIN locations l ON l.sha256 = i.sha256
         JOIN places p ON p.geoname_id = l.place_id
         WHERE i.status = 'active' AND i.kind = 'image' AND p.country_code = ?1
         GROUP BY p.region_code
         ORDER BY 2, 1",
    )?;
    let rows = stmt.query_map([country_code], counted)?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// `(geoname_id, name, images)` within a region, by name.
pub fn cities(
    conn: &Connection,
    country_code: &str,
    region_code: &str,
) -> Result<Vec<(i64, String, u64)>> {
    let mut stmt = conn.prepare(
        "SELECT p.geoname_id, p.name, count(*) FROM images i
         JOIN locations l ON l.sha256 = i.sha256
         JOIN places p ON p.geoname_id = l.place_id
         WHERE i.status = 'active' AND i.kind = 'image'
           AND p.country_code = ?1 AND p.region_code = ?2
         GROUP BY p.geoname_id
         ORDER BY 2, 1",
    )?;
    let rows = stmt.query_map([country_code, region_code], counted)?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

fn counted<K: FromSql>(row: &Row<'_>) -> rusqlite::Result<(K, String, u64)> {
    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
}

fn place_from(row: &Row<'_>, first: usize) -> rusqlite::Result<Place> {
    Ok(Place {
        geoname_id: row.get(first)?,
        name: row.get(first + 1)?,
        region_code: row.get(first + 2)?,
        region: row.get(first + 3)?,
        country_code: row.get(first + 4)?,
        country: row.get(first + 5)?,
        coords: row
            .get::<_, Option<f64>>(first + 6)?
            .zip(row.get(first + 7)?),
    })
}

impl Source {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exif => "exif",
            Self::Manual => "manual",
        }
    }
}

impl ToSql for Source {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(self.as_str().into())
    }
}

impl FromSql for Source {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        super::enum_from_sql(value, |s| match s {
            "exif" => Some(Self::Exif),
            "manual" => Some(Self::Manual),
            _ => None,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::catalog::faces::tests::image;
    use crate::catalog::images::{self, Kind, NewImage, Status};
    use crate::catalog::{edit, temp};
    use crate::geo::events::events_of;

    pub(crate) fn place(geoname_id: i64, name: &str) -> Place {
        Place {
            geoname_id,
            name: name.to_owned(),
            region_code: "14".to_owned(),
            region: "Lisbon".to_owned(),
            country_code: "PT".to_owned(),
            country: "Portugal".to_owned(),
            coords: None,
        }
    }

    fn located(conn: &Connection, sha256: &Sha256) -> Option<(Option<i64>, Source)> {
        location_of(conn, sha256)
            .unwrap()
            .map(|l| (l.place.map(|p| p.geoname_id), l.source))
    }

    fn set_at_of(conn: &Connection, sha256: &Sha256) -> i64 {
        conn.query_row(
            "SELECT set_at FROM locations WHERE sha256 = ?1",
            [sha256],
            |row| row.get(0),
        )
        .unwrap()
    }

    fn scanned(conn: &Connection, sha256: &Sha256) -> Option<(Option<f64>, Option<f64>)> {
        conn.query_row(
            "SELECT lat, lon FROM geo_scans WHERE sha256 = ?1",
            [sha256],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .unwrap()
    }

    #[test]
    fn unscanned_follows_scans_and_dedupes_content() {
        let (_dir, mut conn) = temp();
        image(&conn, "/p/x2.jpg", [1; 32]);
        image(&conn, "/p/x1.jpg", [1; 32]);
        let gone = image(&conn, "/p/y.jpg", [2; 32]);
        images::set_status(&conn, gone, Status::Missing).unwrap();
        image(&conn, "/p/z.jpg", [3; 32]);
        image(&conn, "/q/w.jpg", [4; 32]);

        let all = [PathBuf::from("/p")];
        assert_eq!(
            unscanned(&conn, &all).unwrap(),
            [
                ([1; 32], PathBuf::from("/p/x1.jpg")),
                ([3; 32], PathBuf::from("/p/z.jpg")),
            ]
        );
        assert_eq!(
            unscanned(&conn, &[PathBuf::from("/q/w.jpg"), PathBuf::from("/p")]).unwrap()[0],
            ([4; 32], PathBuf::from("/q/w.jpg"))
        );
        edit(&mut conn, |tx| {
            insert_scan(tx, &[1; 32], Some((38.7223, -9.1393)))?;
            insert_scan(tx, &[3; 32], None)
        });
        assert!(unscanned(&conn, &all).unwrap().is_empty());
        assert_eq!(
            scanned(&conn, &[1; 32]),
            Some((Some(38.7223), Some(-9.1393)))
        );
        assert_eq!(scanned(&conn, &[3; 32]), Some((None, None)));
    }

    #[test]
    fn exif_never_overwrites_a_location() {
        let (_dir, mut conn) = temp();
        let (a, b) = (place(1, "A"), place(2, "B"));
        let (exif, manual, marker) = ([1; 32], [2; 32], [3; 32]);
        edit(&mut conn, |tx| {
            assert!(insert_exif(tx, &exif, &a, 10)?);
            set_manual(tx, &manual, &a, 10)?;
            clear(tx, &marker, 10)
        });
        edit(&mut conn, |tx| {
            for sha in [exif, manual, marker] {
                assert!(!insert_exif(tx, &sha, &b, 20)?);
            }
            Ok(())
        });
        assert_eq!(located(&conn, &exif), Some((Some(1), Source::Exif)));
        assert_eq!(located(&conn, &manual), Some((Some(1), Source::Manual)));
        assert_eq!(located(&conn, &marker), Some((None, Source::Manual)));
        assert_eq!(set_at_of(&conn, &exif), 10);
        assert_eq!(located(&conn, &[4; 32]), None);
    }

    #[test]
    fn manual_overwrites_exif() {
        let (_dir, mut conn) = temp();
        let sha = [1; 32];
        edit(&mut conn, |tx| {
            insert_exif(tx, &sha, &place(1, "A"), 10).map(drop)
        });
        edit(&mut conn, |tx| set_manual(tx, &sha, &place(2, "B"), 20));
        let got = location_of(&conn, &sha).unwrap().unwrap();
        assert_eq!(got.place, Some(place(2, "B")));
        assert_eq!(got.source, Source::Manual);
        assert_eq!(set_at_of(&conn, &sha), 20);
    }

    #[test]
    fn clear_leaves_a_marker_that_survives_exif() {
        let (_dir, mut conn) = temp();
        let sha = [1; 32];
        edit(&mut conn, |tx| {
            insert_exif(tx, &sha, &place(1, "A"), 10).map(drop)
        });
        edit(&mut conn, |tx| clear(tx, &sha, 20));
        edit(&mut conn, |tx| {
            assert!(!insert_exif(tx, &sha, &place(1, "A"), 30)?);
            Ok(())
        });
        assert_eq!(located(&conn, &sha), Some((None, Source::Manual)));
        assert_eq!(set_at_of(&conn, &sha), 20);
    }

    #[test]
    fn apply_event_only_fills_content_without_a_row() {
        let (_dir, mut conn) = temp();
        let (exif, manual, marker, free, other) = ([1; 32], [2; 32], [3; 32], [4; 32], [5; 32]);
        edit(&mut conn, |tx| {
            insert_exif(tx, &exif, &place(1, "A"), 10)?;
            set_manual(tx, &manual, &place(1, "A"), 10)?;
            clear(tx, &marker, 10)
        });
        let porto = place(2, "Porto");
        let tx = conn.transaction().unwrap();
        let shas = [exif, manual, marker, free, free, other];
        assert_eq!(apply_event(&tx, &shas, &porto, 20).unwrap(), 2);
        tx.commit().unwrap();

        assert_eq!(located(&conn, &exif), Some((Some(1), Source::Exif)));
        assert_eq!(located(&conn, &manual), Some((Some(1), Source::Manual)));
        assert_eq!(located(&conn, &marker), Some((None, Source::Manual)));
        assert_eq!(located(&conn, &free), Some((Some(2), Source::Manual)));
        assert_eq!(located(&conn, &other), Some((Some(2), Source::Manual)));
        assert_eq!(set_at_of(&conn, &free), 20);

        let tx = conn.transaction().unwrap();
        assert_eq!(apply_event(&tx, &[], &porto, 30).unwrap(), 0);
    }

    #[test]
    fn place_upsert_keeps_the_id_and_refreshes_names() {
        let (_dir, mut conn) = temp();
        let sha = [1; 32];
        edit(&mut conn, |tx| {
            set_manual(tx, &sha, &place(7, "Lisbon"), 10)
        });
        let renamed = Place {
            region: "Lisboa".to_owned(),
            ..place(7, "Lisboa")
        };
        edit(&mut conn, |tx| upsert_place(tx, &renamed));
        let got = location_of(&conn, &sha).unwrap().unwrap();
        assert_eq!(got.place, Some(renamed));
        let n: i64 = conn
            .query_row("SELECT count(*) FROM places", [], |row| row.get(0))
            .unwrap();
        assert_eq!(n, 1);
    }

    #[test]
    fn place_upsert_without_coords_keeps_stored_ones() {
        let (_dir, mut conn) = temp();
        let sha = [1; 32];
        let lisbon = Place {
            coords: Some((38.72, -9.14)),
            ..place(7, "Lisbon")
        };
        edit(&mut conn, |tx| set_manual(tx, &sha, &lisbon, 10));
        edit(&mut conn, |tx| upsert_place(tx, &place(7, "Lisboa")));
        let got = location_of(&conn, &sha).unwrap().unwrap().place.unwrap();
        assert_eq!(got.name, "Lisboa");
        assert_eq!(got.coords, Some((38.72, -9.14)));
    }

    #[test]
    fn coords_round_trip_through_location_of_and_recent() {
        let (_dir, mut conn) = temp();
        let lisbon = Place {
            coords: Some((38.72, -9.14)),
            ..place(7, "Lisbon")
        };
        edit(&mut conn, |tx| {
            set_manual(tx, &[1; 32], &lisbon, 10)?;
            set_manual(tx, &[2; 32], &place(8, "Sintra"), 20)
        });
        let got = location_of(&conn, &[1; 32]).unwrap().unwrap();
        assert_eq!(got.place, Some(lisbon.clone()));
        let sintra = location_of(&conn, &[2; 32]).unwrap().unwrap();
        assert_eq!(sintra.place.unwrap().coords, None);
        assert_eq!(recent(&conn).unwrap(), [place(8, "Sintra"), lisbon]);
    }

    #[test]
    fn unresolved_pages_past_located_and_empty_scans() {
        let (_dir, mut conn) = temp();
        edit(&mut conn, |tx| {
            insert_scan(tx, &[1; 32], Some((1.0, 2.0)))?;
            insert_scan(tx, &[2; 32], None)?;
            insert_scan(tx, &[3; 32], Some((3.0, 4.0)))?;
            insert_scan(tx, &[4; 32], Some((5.0, 6.0)))?;
            clear(tx, &[3; 32], 10)
        });
        assert_eq!(unresolved(&conn, None, 1).unwrap(), [([1; 32], 1.0, 2.0)]);
        assert_eq!(
            unresolved(&conn, Some(&[1; 32]), 10).unwrap(),
            [([4; 32], 5.0, 6.0)]
        );
        assert!(unresolved(&conn, Some(&[4; 32]), 10).unwrap().is_empty());
    }

    #[test]
    fn counts_cover_active_images_by_source() {
        let (_dir, mut conn) = temp();
        image(&conn, "/p/exif1.jpg", [1; 32]);
        image(&conn, "/p/exif2.jpg", [1; 32]);
        image(&conn, "/p/manual.jpg", [2; 32]);
        image(&conn, "/p/marker.jpg", [3; 32]);
        image(&conn, "/p/pending.jpg", [4; 32]);
        let gone = image(&conn, "/p/gone.jpg", [5; 32]);
        images::set_status(&conn, gone, Status::Missing).unwrap();
        image(&conn, "/p/no-gps.jpg", [6; 32]);
        edit(&mut conn, |tx| {
            for sha in [[1; 32], [3; 32], [4; 32], [5; 32]] {
                insert_scan(tx, &sha, Some((1.0, 2.0)))?;
            }
            insert_scan(tx, &[6; 32], None)?;
            insert_exif(tx, &[1; 32], &place(1, "A"), 10)?;
            set_manual(tx, &[2; 32], &place(1, "A"), 10)?;
            clear(tx, &[3; 32], 10)
        });
        assert_eq!(
            counts(&conn).unwrap(),
            Counts {
                exif: 2,
                manual: 1,
                pending: 1
            }
        );
    }

    #[test]
    fn unknown_place_id_fails() {
        let (_dir, conn) = temp();
        let err = conn.execute(
            "INSERT INTO locations (sha256, place_id, source, set_at) VALUES (?1, 9, 'manual', 0)",
            [[1_u8; 32]],
        );
        assert!(err.is_err());
    }

    #[test]
    fn recent_orders_by_last_manual_use() {
        let (_dir, mut conn) = temp();
        edit(&mut conn, |tx| {
            set_manual(tx, &[1; 32], &place(1, "A"), 10)?;
            set_manual(tx, &[2; 32], &place(2, "B"), 30)?;
            set_manual(tx, &[3; 32], &place(1, "A"), 40)?;
            set_manual(tx, &[4; 32], &place(3, "C"), 40)?;
            insert_exif(tx, &[5; 32], &place(4, "D"), 50)?;
            clear(tx, &[6; 32], 60)
        });
        let ids = |conn: &Connection| -> Vec<i64> {
            recent(conn).unwrap().iter().map(|p| p.geoname_id).collect()
        };
        assert_eq!(ids(&conn), [3, 1, 2]);

        edit(&mut conn, |tx| {
            (10..20).try_for_each(|i| set_manual(tx, &[i; 32], &place(i64::from(i), "X"), 100))
        });
        assert_eq!(ids(&conn), [19, 18, 17, 16, 15, 14, 13, 12]);
    }
    fn dated(
        conn: &Connection,
        path: &str,
        sha: Option<Sha256>,
        taken_at: Option<&str>,
        kind: Kind,
    ) -> ImageId {
        images::upsert(
            conn,
            &NewImage {
                path: Path::new(path),
                size: 1,
                mtime_ns: 1,
                sha256: sha,
                kind,
                width: None,
                height: None,
                taken_at,
                duration_ms: None,
                codec: None,
            },
        )
        .unwrap()
    }

    #[test]
    fn event_items_cover_active_hashed_images_with_a_date() {
        let (_dir, conn) = temp();
        let at = Some("2024-05-01T10:11:12");
        let timed = dated(
            &conn,
            "/a/2020-05-07_00001.jpg",
            Some([1; 32]),
            at,
            Kind::Image,
        );
        let day = dated(
            &conn,
            "/b/2020-05-07_00002.jpg",
            Some([2; 32]),
            None,
            Kind::Image,
        );
        let bad = dated(
            &conn,
            "/b/2020-05-07_00003.jpg",
            Some([3; 32]),
            Some("2024-02-30T00:00:00"),
            Kind::Image,
        );
        dated(&conn, "/b/IMG_1.jpg", Some([4; 32]), None, Kind::Image);
        dated(
            &conn,
            "/b/2020-05-07_00004.mp4",
            Some([5; 32]),
            at,
            Kind::Video,
        );
        dated(&conn, "/b/2020-05-07_00005.jpg", None, at, Kind::Image);
        let gone = dated(
            &conn,
            "/b/2020-05-07_00006.jpg",
            Some([6; 32]),
            at,
            Kind::Image,
        );
        images::set_status(&conn, gone, Status::Missing).unwrap();

        let mut got = event_items(&conn).unwrap();
        got.sort_by_key(|item| item.id);
        let may_7 = When::of(None, "2020-05-07").unwrap();
        assert_eq!(
            got,
            [
                Item {
                    id: timed,
                    when: When::of(at, "").unwrap()
                },
                Item {
                    id: day,
                    when: may_7
                },
                Item {
                    id: bad,
                    when: may_7
                },
            ]
        );
    }

    #[test]
    fn events_span_folders() {
        let (_dir, conn) = temp();
        let a = dated(
            &conn,
            "/a/x.jpg",
            Some([1; 32]),
            Some("2024-05-01T10:00:00"),
            Kind::Image,
        );
        let b = dated(
            &conn,
            "/b/c/y.jpg",
            Some([2; 32]),
            Some("2024-05-01T15:00:00"),
            Kind::Image,
        );
        dated(
            &conn,
            "/a/z.jpg",
            Some([3; 32]),
            Some("2024-05-02T10:00:00"),
            Kind::Image,
        );
        let items = event_items(&conn).unwrap();
        assert_eq!(events_of(&items, 6 * 3600, &[a]), [a, b]);
    }

    #[test]
    fn unlocated_counts_distinct_content_and_treats_markers_as_located() {
        let (_dir, mut conn) = temp();
        let ids = [
            image(&conn, "/p/free1.jpg", [1; 32]),
            image(&conn, "/q/free2.jpg", [1; 32]),
            image(&conn, "/p/other.jpg", [2; 32]),
            image(&conn, "/p/exif.jpg", [3; 32]),
            image(&conn, "/p/manual.jpg", [4; 32]),
            image(&conn, "/p/marker.jpg", [5; 32]),
            dated(&conn, "/p/clip.mp4", Some([7; 32]), None, Kind::Video),
        ];
        image(&conn, "/p/outside.jpg", [6; 32]);
        edit(&mut conn, |tx| {
            insert_exif(tx, &[3; 32], &place(1, "A"), 10)?;
            set_manual(tx, &[4; 32], &place(1, "A"), 10)?;
            clear(tx, &[5; 32], 10)
        });
        assert_eq!(unlocated(&conn, &ids).unwrap(), [[1; 32], [2; 32]]);
        assert!(unlocated(&conn, &[]).unwrap().is_empty());
    }

    #[test]
    fn summarize_resolves_the_effective_location() {
        let (_dir, mut conn) = temp();
        let lisbon = Place {
            coords: Some((38.72, -9.14)),
            ..place(1, "Lisbon")
        };
        let (exif, manual, porto, marker, free) = ([1; 32], [2; 32], [3; 32], [4; 32], [5; 32]);
        let exif2 = [6; 32];
        edit(&mut conn, |tx| {
            insert_exif(tx, &exif, &lisbon, 10)?;
            insert_exif(tx, &exif2, &lisbon, 10)?;
            set_manual(tx, &manual, &lisbon, 10)?;
            set_manual(tx, &porto, &place(2, "Porto"), 10)?;
            insert_exif(tx, &marker, &lisbon, 10)?;
            clear(tx, &marker, 20)
        });
        let one = |source| LocationSummary::One {
            place: lisbon.clone(),
            source,
        };
        let summary = |shas: &[Sha256]| summarize(&conn, shas).unwrap();

        assert_eq!(summary(&[]), LocationSummary::None);
        assert_eq!(summary(&[free]), LocationSummary::None);
        assert_eq!(summary(&[marker]), LocationSummary::None);
        assert_eq!(summary(&[marker, free]), LocationSummary::None);
        assert_eq!(summary(&[exif, exif2, exif]), one(Some(Source::Exif)));
        assert_eq!(summary(&[manual]), one(Some(Source::Manual)));
        assert_eq!(summary(&[exif, manual]), one(None));
        assert_eq!(summary(&[exif, porto]), LocationSummary::Mixed(2));
        assert_eq!(summary(&[exif, free]), LocationSummary::Mixed(2));
        assert_eq!(summary(&[exif, marker]), LocationSummary::Mixed(2));
        assert_eq!(
            summary(&[exif, porto, marker, free]),
            LocationSummary::Mixed(3)
        );
    }

    #[test]
    fn place_lists_count_active_located_images_in_use() {
        let (_dir, mut conn) = temp();
        let at = |id, name: &str, region_code: &str, region: &str, cc: &str, country: &str| Place {
            geoname_id: id,
            name: name.to_owned(),
            region_code: region_code.to_owned(),
            region: region.to_owned(),
            country_code: cc.to_owned(),
            country: country.to_owned(),
            coords: None,
        };
        let sintra = at(2, "Sintra", "14", "Lisbon", "PT", "Portugal");
        let porto = at(3, "Porto", "17", "Porto", "PT", "Portugal");
        let paris = at(4, "Paris", "11", "Ile-de-France", "FR", "France");
        let unnamed = at(5, "Aldeia", "99", "", "PT", "Portugal");
        let unused = at(6, "Bern", "BE", "Bern", "CH", "Switzerland");
        image(&conn, "/p/lisbon1.jpg", [1; 32]);
        image(&conn, "/p/lisbon2.jpg", [1; 32]);
        image(&conn, "/p/sintra.jpg", [2; 32]);
        image(&conn, "/p/porto.jpg", [3; 32]);
        image(&conn, "/p/paris.jpg", [4; 32]);
        image(&conn, "/p/aldeia.jpg", [5; 32]);
        image(&conn, "/p/marker.jpg", [6; 32]);
        let gone = image(&conn, "/p/bern.jpg", [7; 32]);
        images::set_status(&conn, gone, Status::Missing).unwrap();
        dated(&conn, "/p/clip.mp4", Some([8; 32]), None, Kind::Video);
        edit(&mut conn, |tx| {
            insert_exif(tx, &[1; 32], &place(1, "Lisbon"), 10)?;
            set_manual(tx, &[2; 32], &sintra, 10)?;
            set_manual(tx, &[3; 32], &porto, 10)?;
            set_manual(tx, &[4; 32], &paris, 10)?;
            set_manual(tx, &[5; 32], &unnamed, 10)?;
            clear(tx, &[6; 32], 10)?;
            set_manual(tx, &[7; 32], &unused, 10)?;
            set_manual(tx, &[8; 32], &paris, 10)
        });
        let s = str::to_owned;

        assert_eq!(
            countries(&conn).unwrap(),
            [(s("FR"), s("France"), 1), (s("PT"), s("Portugal"), 5)]
        );
        assert_eq!(
            regions(&conn, "PT").unwrap(),
            [
                (s("99"), s(""), 1),
                (s("14"), s("Lisbon"), 3),
                (s("17"), s("Porto"), 1)
            ]
        );
        assert_eq!(
            regions(&conn, "FR").unwrap(),
            [(s("11"), s("Ile-de-France"), 1)]
        );
        assert_eq!(
            cities(&conn, "PT", "14").unwrap(),
            [(1, s("Lisbon"), 2), (2, s("Sintra"), 1)]
        );
        assert!(cities(&conn, "FR", "14").unwrap().is_empty());
        assert!(regions(&conn, "CH").unwrap().is_empty());
    }
}
