use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;

use rusqlite::{Connection, params};

use super::GeoError;

const SCHEMA: &str = r"
CREATE TABLE meta (
    key   TEXT    PRIMARY KEY,
    value INTEGER NOT NULL
) WITHOUT ROWID;

CREATE TABLE places (
    geoname_id   INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    region_code  TEXT    NOT NULL,
    region       TEXT    NOT NULL,
    country_code TEXT    NOT NULL,
    country      TEXT    NOT NULL,
    lat          REAL    NOT NULL,
    lon          REAL    NOT NULL,
    population   INTEGER NOT NULL
);

CREATE VIRTUAL TABLE place_idx USING rtree(id, minlat, maxlat, minlon, maxlon);

CREATE VIRTUAL TABLE places_fts USING fts5(
    name, asciiname, alternatenames,
    content = '',
    tokenize = 'unicode61 remove_diacritics 2',
    prefix = '2 3'
);
";

struct City<'a> {
    id: i64,
    name: &'a str,
    ascii: &'a str,
    alternates: &'a str,
    lat: f64,
    lon: f64,
    country: &'a str,
    admin1: &'a str,
    population: i64,
}

/// A GeoNames place table, such as `cities1000.txt`, and which of its rows to keep.
pub struct Rows<'a> {
    pub file: &'static str,
    pub reader: Box<dyn BufRead + 'a>,
    pub keep: fn(&[&str]) -> bool,
}

impl<'a> Rows<'a> {
    /// Keeps every row.
    pub fn all(file: &'static str, reader: impl BufRead + 'a) -> Self {
        Self {
            file,
            reader: Box::new(reader),
            keep: |_| true,
        }
    }
}

/// Builds a new gazetteer at `dest` from GeoNames place tables, `admin1CodesASCII.txt` and
/// `countryInfo.txt`, returning the number of places. Comment and malformed rows are skipped.
/// On error `dest` is left incomplete and should be removed.
pub fn build(
    dest: &Path,
    sources: Vec<Rows<'_>>,
    admin1: impl BufRead,
    countries: impl BufRead,
    min_rows: u64,
) -> Result<u64, GeoError> {
    let regions = names(admin1, "admin1CodesASCII.txt", 1)?;
    let countries = names(countries, "countryInfo.txt", 4)?;

    let mut conn = Connection::open(dest).map_err(|source| GeoError::Open {
        path: dest.to_owned(),
        source,
    })?;
    conn.pragma_update_and_check(None, "journal_mode", "OFF", |row| row.get::<_, String>(0))?;
    let tx = conn.transaction()?;
    tx.execute_batch(SCHEMA)?;
    let mut rows = 0;
    {
        let mut place = tx.prepare(
            "INSERT INTO places
                 (geoname_id, name, region_code, region, country_code, country, lat, lon, population)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        )?;
        let mut point = tx.prepare("INSERT INTO place_idx VALUES (?1, ?2, ?2, ?3, ?3)")?;
        let mut text = tx.prepare(
            "INSERT INTO places_fts (rowid, name, asciiname, alternatenames)
             VALUES (?1, ?2, ?3, ?4)",
        )?;
        for source in sources {
            each_row(source.reader, source.file, |fields| {
                if !(source.keep)(fields) {
                    return Ok(());
                }
                let Some(c) = parse_city(fields) else {
                    return Ok(());
                };
                let region = regions
                    .get(&format!("{}.{}", c.country, c.admin1))
                    .map_or("", String::as_str);
                let country = countries.get(c.country).map_or("", String::as_str);
                place.execute(params![
                    c.id,
                    c.name,
                    c.admin1,
                    region,
                    c.country,
                    country,
                    c.lat,
                    c.lon,
                    c.population
                ])?;
                point.execute(params![c.id, c.lat, c.lon])?;
                text.execute(params![c.id, c.name, c.ascii, c.alternates])?;
                rows += 1;
                Ok(())
            })?;
        }
    }
    if rows < min_rows {
        return Err(GeoError::TooFewRows {
            rows,
            min: min_rows,
        });
    }
    tx.execute(
        "INSERT INTO places_fts (places_fts) VALUES ('optimize')",
        [],
    )?;
    tx.execute(
        "INSERT INTO meta (key, value) VALUES ('fetched_at', unixepoch()), ('rows', ?1)",
        [rows],
    )?;
    tx.commit()?;
    Ok(rows)
}

fn parse_city<'a>(f: &[&'a str]) -> Option<City<'a>> {
    if f.len() < 15 || f[1].is_empty() {
        return None;
    }
    let lat = f[4]
        .parse()
        .ok()
        .filter(|v: &f64| (-90.0..=90.0).contains(v))?;
    let lon = f[5]
        .parse()
        .ok()
        .filter(|v: &f64| (-180.0..=180.0).contains(v))?;
    Some(City {
        id: f[0].parse().ok()?,
        name: f[1],
        ascii: f[2],
        alternates: f[3],
        lat,
        lon,
        country: f[8],
        admin1: f[10],
        population: f[14].parse().ok()?,
    })
}

/// Column 0 to column `value` of each row with both.
fn names(
    reader: impl BufRead,
    file: &'static str,
    value: usize,
) -> Result<HashMap<String, String>, GeoError> {
    let mut map = HashMap::new();
    each_row(reader, file, |fields| {
        if let (Some(key), Some(name)) = (fields.first(), fields.get(value)) {
            map.insert((*key).to_owned(), (*name).to_owned());
        }
        Ok(())
    })?;
    Ok(map)
}

/// Tab-split fields of each non-empty line not starting with `#`.
fn each_row(
    mut reader: impl BufRead,
    file: &'static str,
    mut f: impl FnMut(&[&str]) -> Result<(), GeoError>,
) -> Result<(), GeoError> {
    let mut line = String::new();
    loop {
        line.clear();
        if reader
            .read_line(&mut line)
            .map_err(|source| GeoError::Read { file, source })?
            == 0
        {
            return Ok(());
        }
        let row = line.trim_end_matches(['\n', '\r']);
        if row.is_empty() || row.starts_with('#') {
            continue;
        }
        f(&row.split('\t').collect::<Vec<_>>())?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::fixtures::{ADMIN1, COUNTRIES, city};

    fn cities() -> String {
        [
            "# a comment\n".to_owned(),
            city(
                1, "Lisbon", "Lisbon", "Lisboa", 38.7, -9.1, "PT", "14", 500_000,
            ),
            city(2, "Porto", "Porto", "", 41.1, -8.6, "PT", "17", 230_000),
            "3\tShort\trow\n".to_owned(),
            city(4, "Nowhere", "Nowhere", "", 95.0, 0.0, "PT", "17", 1),
            "x".repeat(3) + &city(5, "BadId", "BadId", "", 1.0, 1.0, "PT", "17", 1),
            city(6, "Somewhere", "Somewhere", "", 1.0, 1.0, "XX", "99", 1),
            "\n".to_owned(),
        ]
        .concat()
    }

    fn build_into(dir: &Path, min_rows: u64) -> Result<u64, GeoError> {
        build(
            &dir.join("geo.db"),
            vec![Rows::all("cities1000.txt", cities().as_bytes())],
            ADMIN1.as_bytes(),
            COUNTRIES.as_bytes(),
            min_rows,
        )
    }

    #[test]
    fn bundled_sqlite_has_fts5_and_rtree() {
        let conn = Connection::open_in_memory().unwrap();
        let options: Vec<String> = conn
            .prepare("PRAGMA compile_options")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert!(options.iter().any(|o| o == "ENABLE_FTS5"), "{options:?}");
        assert!(options.iter().any(|o| o == "ENABLE_RTREE"), "{options:?}");
    }

    #[test]
    fn skips_comment_and_malformed_rows() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(build_into(dir.path(), 1).unwrap(), 3);
        let conn = Connection::open(dir.path().join("geo.db")).unwrap();
        let ids: Vec<i64> = conn
            .prepare("SELECT geoname_id FROM places ORDER BY geoname_id")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(ids, [1, 2, 6]);
    }

    #[test]
    fn joins_region_and_country_names() {
        let dir = tempfile::tempdir().unwrap();
        build_into(dir.path(), 1).unwrap();
        let conn = Connection::open(dir.path().join("geo.db")).unwrap();
        let row = |id: i64| -> (String, String, String, String) {
            conn.query_row(
                "SELECT region_code, region, country_code, country FROM places WHERE geoname_id = ?1",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .unwrap()
        };
        assert_eq!(
            row(2),
            ("17".into(), "Porto".into(), "PT".into(), "Portugal".into())
        );
        assert_eq!(
            row(6),
            ("99".into(), String::new(), "XX".into(), "Nowhere".into())
        );
    }

    #[test]
    fn fewer_than_min_rows_errors() {
        let dir = tempfile::tempdir().unwrap();
        let err = build_into(dir.path(), 4).unwrap_err();
        assert!(
            matches!(err, GeoError::TooFewRows { rows: 3, min: 4 }),
            "{err:?}"
        );
    }

    #[test]
    fn records_fetch_time_and_rows() {
        let dir = tempfile::tempdir().unwrap();
        build_into(dir.path(), 1).unwrap();
        let conn = Connection::open(dir.path().join("geo.db")).unwrap();
        let (fetched_at, rows): (i64, i64) = conn
            .query_row(
                "SELECT (SELECT value FROM meta WHERE key = 'fetched_at'),
                        (SELECT value FROM meta WHERE key = 'rows')",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert!(fetched_at > 1_700_000_000);
        assert_eq!(rows, 3);
    }
}
