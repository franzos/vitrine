use std::path::Path;

use rusqlite::{Connection, OpenFlags, OptionalExtension, Row, params};

use super::{GeoError, Place};
use crate::map::labels::City;

const SEARCH_LIMIT: u32 = 8;
const MIN_QUERY_CHARS: usize = 2;
/// Half-widths in degrees of the boxes `nearest` tries in turn; the last covers the globe.
const BOXES: [f64; 6] = [0.25, 1.0, 4.0, 16.0, 64.0, 180.0];

const BOX_QUERY: &str = "SELECT p.geoname_id, p.lat, p.lon
     FROM place_idx i JOIN places p ON p.geoname_id = i.id
     WHERE i.maxlat >= ?1 AND i.minlat <= ?2 AND i.maxlon >= ?3 AND i.minlon <= ?4";

const BAND_QUERY: &str = "SELECT p.geoname_id, p.lat, p.lon
     FROM place_idx i JOIN places p ON p.geoname_id = i.id
     WHERE i.maxlat >= ?1 AND i.minlat <= ?2";

const CITIES_IN_QUERY: &str = "SELECT p.name, p.lat, p.lon, p.population
     FROM place_idx i JOIN places p ON p.geoname_id = i.id
     WHERE i.maxlat >= ?1 AND i.minlat <= ?2 AND i.maxlon >= ?3 AND i.minlon <= ?4
       AND p.population >= ?5
     ORDER BY p.population DESC, p.geoname_id
     LIMIT ?6";

/// A read-only `geo.db`.
pub struct GeoDb {
    conn: Connection,
    fetched_at: String,
    count: u64,
}

impl GeoDb {
    /// `None` when `path` doesn't exist.
    pub fn open(path: &Path) -> Result<Option<Self>, GeoError> {
        let exists = path.try_exists().map_err(|source| GeoError::Io {
            path: path.to_owned(),
            source,
        })?;
        if !exists {
            return Ok(None);
        }
        let open_err = |source| GeoError::Open {
            path: path.to_owned(),
            source,
        };
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(open_err)?;
        let (fetched_at, count) = conn
            .query_row(
                "SELECT (SELECT datetime(value, 'unixepoch', 'localtime')
                         FROM meta WHERE key = 'fetched_at'),
                        (SELECT value FROM meta WHERE key = 'rows')",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(open_err)?;
        Ok(Some(Self {
            conn,
            fetched_at,
            count,
        }))
    }

    /// Local time, `%Y-%m-%d %H:%M:%S`.
    pub fn fetched_at(&self) -> &str {
        &self.fetched_at
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    /// Places matching every word of `query` as a prefix of a name, ascii name or alternate name,
    /// most populous first.
    pub fn search(&self, query: &str) -> Result<Vec<Place>, GeoError> {
        let Some(expr) = match_expr(query) else {
            return Ok(Vec::new());
        };
        let mut stmt = self.conn.prepare_cached(
            "SELECT p.geoname_id, p.name, p.region_code, p.region, p.country_code, p.country,
                    p.lat, p.lon
             FROM places_fts f JOIN places p ON p.geoname_id = f.rowid
             WHERE places_fts MATCH ?1
             ORDER BY p.population DESC, p.geoname_id
             LIMIT ?2",
        )?;
        let places = stmt
            .query_map(params![expr, SEARCH_LIMIT], place_from_row)?
            .collect::<Result<_, _>>()?;
        Ok(places)
    }

    /// The place closest to `lat`, `lon` by great-circle distance; `None` for invalid
    /// coordinates or an empty gazetteer.
    pub fn nearest(&self, lat: f64, lon: f64) -> Result<Option<Place>, GeoError> {
        if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
            return Ok(None);
        }
        let mut first = None;
        for h in BOXES {
            first = self.closest(
                lat,
                lon,
                BOX_QUERY,
                params![
                    (lat - h).max(-90.0),
                    (lat + h).min(90.0),
                    (lon - h).max(-180.0),
                    (lon + h).min(180.0)
                ],
            )?;
            if first.is_some() {
                break;
            }
        }
        let Some(first) = first else {
            return Ok(None);
        };
        // No closer place can differ in latitude by more than the distance to this one.
        let d = first.angle.to_degrees();
        let best = self
            .closest(
                lat,
                lon,
                BAND_QUERY,
                params![(lat - d).max(-90.0), (lat + d).min(90.0)],
            )?
            .unwrap_or(first);
        let place = self
            .conn
            .prepare_cached(
                "SELECT geoname_id, name, region_code, region, country_code, country, lat, lon
             FROM places WHERE geoname_id = ?1",
            )?
            .query_row([best.id], place_from_row)?;
        Ok(Some(place))
    }

    /// `(lat, lon)` of the place with `geoname_id`; `None` when the gazetteer doesn't have it.
    pub fn coords(&self, geoname_id: i64) -> Result<Option<(f64, f64)>, GeoError> {
        let coords = self
            .conn
            .prepare_cached("SELECT lat, lon FROM places WHERE geoname_id = ?1")?
            .query_row([geoname_id], |row| Ok((row.get(0)?, row.get(1)?)))
            .optional()?;
        Ok(coords)
    }

    /// Places inside the box with at least `min_pop` inhabitants, most populous first.
    pub fn cities_in(
        &self,
        min_lat: f64,
        max_lat: f64,
        min_lon: f64,
        max_lon: f64,
        min_pop: u32,
        limit: u32,
    ) -> Result<Vec<City>, GeoError> {
        let cities = self
            .conn
            .prepare_cached(CITIES_IN_QUERY)?
            .query_map(
                params![min_lat, max_lat, min_lon, max_lon, min_pop, limit],
                city_from_row,
            )?
            .collect::<Result<_, _>>()?;
        Ok(cities)
    }

    /// Places anywhere with at least `min_pop` inhabitants, most populous first.
    pub fn major_cities(&self, min_pop: u32) -> Result<Vec<City>, GeoError> {
        let cities = self
            .conn
            .prepare_cached(
                "SELECT name, lat, lon, population FROM places
                 WHERE population >= ?1
                 ORDER BY population DESC, geoname_id",
            )?
            .query_map([min_pop], city_from_row)?
            .collect::<Result<_, _>>()?;
        Ok(cities)
    }

    fn closest(
        &self,
        lat: f64,
        lon: f64,
        sql: &str,
        params: impl rusqlite::Params,
    ) -> Result<Option<Candidate>, GeoError> {
        let mut stmt = self.conn.prepare_cached(sql)?;
        let mut rows = stmt.query(params)?;
        let mut best: Option<Candidate> = None;
        while let Some(row) = rows.next()? {
            let candidate = Candidate {
                id: row.get(0)?,
                angle: central_angle(lat, lon, row.get(1)?, row.get(2)?),
            };
            if best.as_ref().is_none_or(|b| {
                candidate
                    .angle
                    .total_cmp(&b.angle)
                    .then(candidate.id.cmp(&b.id))
                    .is_lt()
            }) {
                best = Some(candidate);
            }
        }
        Ok(best)
    }
}

struct Candidate {
    id: i64,
    /// Radians.
    angle: f64,
}

fn place_from_row(row: &Row<'_>) -> rusqlite::Result<Place> {
    Ok(Place {
        geoname_id: row.get(0)?,
        name: row.get(1)?,
        region_code: row.get(2)?,
        region: row.get(3)?,
        country_code: row.get(4)?,
        country: row.get(5)?,
        coords: Some((row.get(6)?, row.get(7)?)),
    })
}

fn city_from_row(row: &Row<'_>) -> rusqlite::Result<City> {
    Ok(City {
        name: row.get(0)?,
        lat: row.get(1)?,
        lon: row.get(2)?,
        population: row.get(3)?,
    })
}

/// An FTS5 expression ANDing each word as a quoted prefix; words with fewer than
/// `MIN_QUERY_CHARS` letters or digits are left out, `None` when nothing is left.
fn match_expr(query: &str) -> Option<String> {
    let terms: Vec<String> = query
        .split_whitespace()
        .filter(|word| word.chars().filter(|c| c.is_alphanumeric()).count() >= MIN_QUERY_CHARS)
        .map(|word| format!("\"{}\"*", word.replace('"', "\"\"")))
        .collect();
    (!terms.is_empty()).then(|| terms.join(" AND "))
}

/// Haversine, in radians.
fn central_angle(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let dp = p2 - p1;
    let dl = (lon2 - lon1).to_radians();
    let h = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    2.0 * h.sqrt().min(1.0).asin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::fixtures::{self, city};
    use crate::map::labels::{LIST_MIN_POPULATION, min_population};

    fn search_db(dir: &Path) -> GeoDb {
        let cities = [
            city(
                10,
                "Lisbon",
                "Lisbon",
                "Lisboa,Lissabon",
                38.72,
                -9.14,
                "PT",
                "14",
                517_802,
            ),
            city(
                11, "Porto", "Porto", "Oporto", 41.15, -8.61, "PT", "17", 249_633,
            ),
            city(
                12,
                "München",
                "Muenchen",
                "Munich,Monaco di Baviera",
                48.14,
                11.58,
                "DE",
                "02",
                1_260_391,
            ),
            city(20, "Paris", "Paris", "", 33.66, -95.56, "US", "TX", 24_171),
            city(
                21, "Paris", "Paris", "Lutetia", 48.85, 2.35, "FR", "11", 2_138_551,
            ),
        ]
        .concat();
        fixtures::build(dir, &cities)
    }

    fn ids(places: &[Place]) -> Vec<i64> {
        places.iter().map(|p| p.geoname_id).collect()
    }

    fn nearest_id(db: &GeoDb, lat: f64, lon: f64) -> i64 {
        db.nearest(lat, lon).unwrap().unwrap().geoname_id
    }

    #[test]
    fn open_returns_none_when_absent() {
        let dir = tempfile::tempdir().unwrap();
        assert!(GeoDb::open(&dir.path().join("geo.db")).unwrap().is_none());
    }

    #[test]
    fn open_rejects_a_file_that_is_not_a_gazetteer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("geo.db");
        std::fs::write(&path, "not a database").unwrap();
        assert!(matches!(GeoDb::open(&path), Err(GeoError::Open { .. })));
    }

    #[test]
    fn open_reads_fetch_time_and_count() {
        let dir = tempfile::tempdir().unwrap();
        let db = search_db(dir.path());
        assert_eq!(db.count(), 5);
        assert_eq!(db.fetched_at().len(), "2026-10-11 12:00:00".len());
    }

    #[test]
    fn search_matches_alternate_names() {
        let dir = tempfile::tempdir().unwrap();
        let db = search_db(dir.path());
        assert_eq!(ids(&db.search("Lisboa").unwrap()), [10]);
        assert_eq!(ids(&db.search("Munich").unwrap()), [12]);
    }

    #[test]
    fn search_ignores_diacritics() {
        let dir = tempfile::tempdir().unwrap();
        let db = search_db(dir.path());
        let found = db.search("Munchen").unwrap();
        assert_eq!(ids(&found), [12]);
        assert_eq!(found[0].name, "München");
        assert_eq!(ids(&db.search("münchen").unwrap()), [12]);
    }

    #[test]
    fn search_matches_prefixes_of_every_word() {
        let dir = tempfile::tempdir().unwrap();
        let db = search_db(dir.path());
        assert_eq!(ids(&db.search("mun").unwrap()), [12]);
        assert_eq!(ids(&db.search("mon di bav").unwrap()), [12]);
        assert!(db.search("mun lis").unwrap().is_empty());
    }

    #[test]
    fn search_ranks_by_population() {
        let dir = tempfile::tempdir().unwrap();
        let db = search_db(dir.path());
        let found = db.search("Paris").unwrap();
        assert_eq!(ids(&found), [21, 20]);
        assert_eq!(found[0].label(), "Paris, Île-de-France, France");
        assert_eq!(found[1].label(), "Paris, Texas, United States");
        assert_eq!(found[0].coords, Some((48.85, 2.35)));
    }

    #[test]
    fn coords_by_id() {
        let dir = tempfile::tempdir().unwrap();
        let db = search_db(dir.path());
        assert_eq!(db.coords(21).unwrap(), Some((48.85, 2.35)));
        assert_eq!(db.coords(99).unwrap(), None);
    }

    #[test]
    fn search_needs_two_characters() {
        let dir = tempfile::tempdir().unwrap();
        let db = search_db(dir.path());
        assert!(db.search("p").unwrap().is_empty());
        assert!(db.search("  ").unwrap().is_empty());
        assert!(db.search("-p").unwrap().is_empty());
        assert!(db.search("a b").unwrap().is_empty());
        assert_eq!(ids(&db.search("po").unwrap()), [11]);
        assert_eq!(ids(&db.search("Paris F").unwrap()), [21, 20]);
    }

    #[test]
    fn search_accepts_fts_syntax_as_text() {
        let dir = tempfile::tempdir().unwrap();
        let db = search_db(dir.path());
        for query in [
            "\"",
            "\"\"",
            "Pa\"ris",
            "\"Paris",
            "-",
            "- -",
            "-Paris",
            "Saint-Denis",
            "NOT",
            "Paris NOT",
            "a OR b",
            "*",
            "Par*",
            "(",
            "NEAR(a b)",
            "^Paris",
            "name:Paris",
            "{name}",
        ] {
            db.search(query)
                .unwrap_or_else(|e| panic!("{query}: {e:?}"));
        }
        assert_eq!(ids(&db.search("-Paris").unwrap()), [21, 20]);
        assert_eq!(ids(&db.search("\"Paris\"").unwrap()), [21, 20]);
    }

    #[test]
    fn nearest_from_sea_is_the_closest_town() {
        let dir = tempfile::tempdir().unwrap();
        let db = search_db(dir.path());
        assert_eq!(nearest_id(&db, 38.6, -10.5), 10);
        assert_eq!(nearest_id(&db, 42.0, -12.0), 11);
        let porto = db.nearest(42.0, -12.0).unwrap().unwrap();
        assert_eq!(porto.coords, Some((41.15, -8.61)));
    }

    #[test]
    fn nearest_corrects_the_first_box() {
        let dir = tempfile::tempdir().unwrap();
        let h = BOXES[0];
        // At 60 degrees a degree of longitude is half a degree of latitude.
        let cities = [
            city(
                1,
                "Inside",
                "Inside",
                "",
                60.0 + h * 0.9,
                10.0,
                "XX",
                "",
                1000,
            ),
            city(
                2,
                "Outside",
                "Outside",
                "",
                60.0,
                10.0 + h * 1.2,
                "XX",
                "",
                1000,
            ),
        ]
        .concat();
        let db = fixtures::build(dir.path(), &cities);
        let first = db
            .closest(
                60.0,
                10.0,
                BOX_QUERY,
                params![60.0 - h, 60.0 + h, 10.0 - h, 10.0 + h],
            )
            .unwrap()
            .unwrap();
        assert_eq!(first.id, 1);
        assert_eq!(nearest_id(&db, 60.0, 10.0), 2);
    }

    #[test]
    fn nearest_crosses_the_antimeridian() {
        let dir = tempfile::tempdir().unwrap();
        let cities = [
            city(1, "East", "East", "", -17.0, -179.95, "XX", "", 1000),
            city(2, "West", "West", "", -17.0, 179.0, "XX", "", 1000),
        ]
        .concat();
        let db = fixtures::build(dir.path(), &cities);
        assert_eq!(nearest_id(&db, -17.0, 179.9), 1);
        assert_eq!(nearest_id(&db, -17.0, -179.9), 1);
        assert_eq!(nearest_id(&db, -17.0, 179.2), 2);
    }

    #[test]
    fn nearest_crosses_the_pole() {
        let dir = tempfile::tempdir().unwrap();
        let cities = [
            city(
                1,
                "Same meridian",
                "Same meridian",
                "",
                84.5,
                0.0,
                "XX",
                "",
                1000,
            ),
            city(
                2,
                "Across the pole",
                "Across the pole",
                "",
                89.0,
                180.0,
                "XX",
                "",
                1000,
            ),
        ]
        .concat();
        let db = fixtures::build(dir.path(), &cities);
        assert_eq!(nearest_id(&db, 88.0, 0.0), 2);
        assert_eq!(nearest_id(&db, 85.0, 0.0), 1);
    }

    fn names(cities: &[City]) -> Vec<&str> {
        cities.iter().map(|c| c.name.as_str()).collect()
    }

    #[test]
    fn cities_in_keeps_the_box_and_threshold_in_population_order() {
        let dir = tempfile::tempdir().unwrap();
        let db = search_db(dir.path());
        let europe = db.cities_in(35.0, 50.0, -10.0, 12.0, 0, 10).unwrap();
        assert_eq!(names(&europe), ["Paris", "München", "Lisbon", "Porto"]);
        assert_eq!(
            europe[0],
            City {
                name: "Paris".into(),
                lat: 48.85,
                lon: 2.35,
                population: 2_138_551,
            }
        );
        let big = db.cities_in(35.0, 50.0, -10.0, 12.0, 517_802, 10).unwrap();
        assert_eq!(names(&big), ["Paris", "München", "Lisbon"]);
        let limited = db.cities_in(35.0, 50.0, -10.0, 12.0, 0, 2).unwrap();
        assert_eq!(names(&limited), ["Paris", "München"]);
        let iberia = db.cities_in(38.72, 41.15, -9.14, -8.61, 0, 10).unwrap();
        assert_eq!(names(&iberia), ["Lisbon", "Porto"]);
        assert!(
            db.cities_in(38.73, 41.14, -9.14, -8.61, 0, 10)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn major_cities_keeps_the_threshold_in_population_order() {
        let dir = tempfile::tempdir().unwrap();
        let db = search_db(dir.path());
        assert_eq!(
            names(&db.major_cities(0).unwrap()),
            ["Paris", "München", "Lisbon", "Porto", "Paris"]
        );
        assert_eq!(
            names(&db.major_cities(249_633).unwrap()),
            ["Paris", "München", "Lisbon", "Porto"]
        );
        assert_eq!(
            names(&db.major_cities(1_000_000).unwrap()),
            ["Paris", "München"]
        );
    }

    /// Run with `VITRINE_GEO_DB=<path> cargo test --release -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs a real geo.db"]
    fn city_queries_are_fast_on_a_real_gazetteer() {
        let Some(path) = std::env::var_os("VITRINE_GEO_DB") else {
            println!("VITRINE_GEO_DB is unset, skipping");
            return;
        };
        let db = GeoDb::open(Path::new(&path)).unwrap().unwrap();
        let start = std::time::Instant::now();
        let list = db.major_cities(LIST_MIN_POPULATION).unwrap();
        let list_time = start.elapsed();
        // About a 1000 x 700 px window at level 11 over Lisbon.
        let start = std::time::Instant::now();
        let local = db
            .cities_in(38.53, 38.91, -9.48, -8.80, min_population(11).unwrap(), 200)
            .unwrap();
        let local_time = start.elapsed();
        println!(
            "major_cities({LIST_MIN_POPULATION}): {} rows in {list_time:?}; cities_in at level 11: {} rows in {local_time:?}",
            list.len(),
            local.len()
        );
        assert!(local_time.as_millis() < 50, "{local_time:?}");
    }

    #[test]
    fn nearest_handles_empty_gazetteers_and_invalid_coordinates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("geo.db");
        crate::geo::build(
            &path,
            vec![crate::geo::Rows::all("cities1000.txt", &b""[..])],
            fixtures::ADMIN1.as_bytes(),
            fixtures::COUNTRIES.as_bytes(),
            0,
        )
        .unwrap();
        let empty = GeoDb::open(&path).unwrap().unwrap();
        assert_eq!(empty.nearest(0.0, 0.0).unwrap(), None);

        let other = tempfile::tempdir().unwrap();
        let db = search_db(other.path());
        assert_eq!(db.nearest(f64::NAN, 0.0).unwrap(), None);
        assert_eq!(db.nearest(91.0, 0.0).unwrap(), None);
        assert_eq!(db.nearest(0.0, 181.0).unwrap(), None);
    }
}
