//! Entries and map points below a folder, narrowed by faces, places and area.

use std::path::Path;

use rusqlite::types::Value;
use rusqlite::{Connection, OptionalExtension};

use super::faces::ModelId;
use super::images::{ENTRY_ORDER, Entry, ImageId, entry_from_row, prefix_range};
use super::people::PersonId;
use crate::db::{Result, Sha256, path_str};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FaceCount {
    #[default]
    Any,
    Zero,
    One,
    Two,
    ThreeOrMore,
}

impl FaceCount {
    pub const ALL: [Self; 5] = [
        Self::Any,
        Self::Zero,
        Self::One,
        Self::Two,
        Self::ThreeOrMore,
    ];

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::Zero => "0",
            Self::One => "1",
            Self::Two => "2",
            Self::ThreeOrMore => "3+",
        }
    }
}

/// Content located at the selected place, through `locations` and the catalog's `places`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlaceSel {
    /// ISO 3166-1 alpha-2.
    pub country: Option<String>,
    /// GeoNames admin1 code; ignored without `country`, since codes repeat across countries.
    pub region: Option<String>,
    /// GeoNames id.
    pub city: Option<i64>,
    /// Images, not videos, with no place, "no location" markers included; combined with a place it matches nothing.
    pub none: bool,
}

/// A face counts when it isn't marked not-a-face, scores at least `min_score`, and both box sides are at least `min_face_size`.
#[derive(Debug, Clone, PartialEq)]
pub struct Filter {
    /// Face criteria apply only with a model.
    pub model: Option<ModelId>,
    pub min_score: f32,
    pub min_face_size: u32,
    /// Content scanned with `model` with this many counted faces; [`FaceCount::Any`] also keeps unscanned content and videos.
    pub count: FaceCount,
    /// Content with faces of all of these people.
    pub people: Vec<PersonId>,
    /// Content with a counted face that has no person, or an unnamed, non-ignored one.
    pub unknown: bool,
    pub place: PlaceSel,
    /// Images whose point lies inside, bounds included.
    pub area: Option<Area>,
}

/// Latitudes `south..=north`, longitudes `west..=east`; never crosses the antimeridian.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Area {
    pub south: f64,
    pub north: f64,
    pub west: f64,
    pub east: f64,
}

const COORD_JOINS: &str = " LEFT JOIN locations ml ON ml.sha256 = i.sha256
         LEFT JOIN places mp ON mp.geoname_id = ml.place_id
         LEFT JOIN geo_scans gs ON gs.sha256 = i.sha256";
// A manual place without coordinates yields NULL rather than falling back to EXIF.
const LAT: &str = "CASE WHEN ml.source = 'manual' THEN mp.lat ELSE gs.lat END";
const LON: &str = "CASE WHEN ml.source = 'manual' THEN mp.lon ELSE gs.lon END";

/// Any depth below `folder`, in [`ENTRY_ORDER`].
pub fn entries(conn: &Connection, folder: &Path, filter: &Filter) -> Result<Vec<Entry>> {
    let (sql, params) = entries_sql(folder, filter)?;
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params), entry_from_row)?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

fn entries_sql(folder: &Path, filter: &Filter) -> Result<(String, Vec<Value>)> {
    let (from, params) = base(folder, filter, false)?;
    Ok((
        format!(
            "SELECT i.id, i.path, i.sha256, i.kind, i.width, i.height, i.duration_ms {from} ORDER BY {ENTRY_ORDER}"
        ),
        params,
    ))
}

/// Located images below `folder` as `(id, lat, lon)`, path order; a manual place wins over EXIF.
pub fn points(
    conn: &Connection,
    folder: &Path,
    filter: &Filter,
) -> Result<Vec<(ImageId, f64, f64)>> {
    let (from, params) = base(folder, filter, true)?;
    let sql = format!(
        "SELECT i.id, {LAT}, {LON} {from}
         AND i.kind = 'image' AND {LAT} IS NOT NULL AND {LON} IS NOT NULL ORDER BY i.path"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params), |row| {
        Ok((ImageId(row.get(0)?), row.get(1)?, row.get(2)?))
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// The `(lat, lon)` [`points`] gives active images of this content.
pub fn point_of(conn: &Connection, sha256: &Sha256) -> Result<Option<(f64, f64)>> {
    Ok(conn
        .query_row(
            &format!(
                "SELECT {LAT}, {LON} FROM images i{COORD_JOINS}
                 WHERE i.sha256 = ?1 AND i.status = 'active' AND i.kind = 'image'
                   AND {LAT} IS NOT NULL AND {LON} IS NOT NULL
                 LIMIT 1"
            ),
            [sha256],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?)
}

/// `FROM ... WHERE ...` for `filter` below `folder`; `coords` adds the joins `LAT` and `LON` read.
fn base(folder: &Path, filter: &Filter, coords: bool) -> Result<(String, Vec<Value>)> {
    let (lower, upper) = prefix_range(path_str(folder)?);
    let mut sql = String::from("FROM images i");
    if coords || filter.area.is_some() {
        sql.push_str(COORD_JOINS);
    }
    sql.push_str("\n         WHERE i.path >= ? AND i.path < ? AND i.status = 'active'");
    let mut params = vec![Value::Text(lower), Value::Text(upper)];
    if let Some(model) = filter.model {
        faces_sql(&mut sql, &mut params, model, filter);
    }
    place_sql(&mut sql, &mut params, &filter.place);
    if let Some(area) = filter.area {
        sql.push_str(&format!(
            " AND {LAT} BETWEEN ? AND ? AND {LON} BETWEEN ? AND ?"
        ));
        params.extend([area.south, area.north, area.west, area.east].map(Value::Real));
    }
    Ok((sql, params))
}

fn faces_sql(sql: &mut String, params: &mut Vec<Value>, model: ModelId, filter: &Filter) {
    let model = Value::Integer(model.0);
    let counted = |params: &mut Vec<Value>| {
        params.extend([
            model.clone(),
            Value::Real(f64::from(filter.min_score)),
            Value::Integer(i64::from(filter.min_face_size)),
        ]);
        "f.model_id = ? AND f.not_a_face = 0 AND f.score >= ? AND min(f.w, f.h) >= ?"
    };

    let compare = match filter.count {
        FaceCount::Any => None,
        FaceCount::Zero => {
            sql.push_str(
                " AND EXISTS (SELECT 1 FROM face_scans s WHERE s.sha256 = i.sha256 AND s.model_id = ?)",
            );
            params.push(model.clone());
            Some("= 0")
        }
        FaceCount::One => Some("= 1"),
        FaceCount::Two => Some("= 2"),
        FaceCount::ThreeOrMore => Some(">= 3"),
    };
    if let Some(compare) = compare {
        let counted = counted(params);
        sql.push_str(&format!(
            " AND (SELECT count(*) FROM faces f WHERE f.sha256 = i.sha256 AND {counted}) {compare}"
        ));
    }

    let mut people = filter.people.clone();
    people.sort_unstable();
    people.dedup();
    if !people.is_empty() {
        let marks = vec!["?"; people.len()].join(", ");
        sql.push_str(&format!(
            " AND i.sha256 IN (SELECT sha256 FROM faces WHERE person_id IN ({marks})
              GROUP BY sha256 HAVING count(DISTINCT person_id) = {})",
            people.len()
        ));
        params.extend(people.iter().map(|p| Value::Integer(p.0)));
    }

    if filter.unknown {
        let counted = counted(params);
        sql.push_str(&format!(
            " AND i.sha256 IN (SELECT f.sha256 FROM faces f LEFT JOIN people p ON p.id = f.person_id
              WHERE {counted} AND (f.person_id IS NULL OR (p.name IS NULL AND p.ignored = 0)))"
        ));
    }
}

fn place_sql(sql: &mut String, params: &mut Vec<Value>, place: &PlaceSel) {
    let mut at = String::new();
    if let Some(country) = &place.country {
        at.push_str(" AND p.country_code = ?");
        params.push(Value::Text(country.clone()));
        if let Some(region) = &place.region {
            at.push_str(" AND p.region_code = ?");
            params.push(Value::Text(region.clone()));
        }
    }
    if let Some(city) = place.city {
        at.push_str(" AND l.place_id = ?");
        params.push(Value::Integer(city));
    }
    if !at.is_empty() {
        sql.push_str(&format!(
            " AND EXISTS (SELECT 1 FROM locations l JOIN places p ON p.geoname_id = l.place_id
              WHERE l.sha256 = i.sha256{at})"
        ));
    }
    if place.none {
        sql.push_str(
            " AND i.kind = 'image' AND NOT EXISTS
              (SELECT 1 FROM locations l WHERE l.sha256 = i.sha256 AND l.place_id IS NOT NULL)",
        );
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::catalog::faces::{self, AssignedBy, FaceId, NewFace, Rect, register_model};
    use crate::catalog::images::{self, Kind, NewImage};
    use crate::catalog::{edit, geo, people, temp};
    use crate::db::Sha256;
    use crate::geo::Place;

    fn image(conn: &Connection, path: &str, sha: Option<Sha256>, kind: Kind) {
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
                taken_at: None,
                duration_ms: None,
                codec: None,
            },
        )
        .unwrap();
    }

    /// Stores a scan of `sha`; `faces` are `(side, score)` pairs.
    fn scan(
        conn: &mut Connection,
        sha: Sha256,
        model: ModelId,
        faces: &[(f32, f32)],
    ) -> Vec<FaceId> {
        let e = [0.5_f32, -0.5];
        let faces: Vec<_> = faces
            .iter()
            .map(|&(side, score)| NewFace {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: side,
                    h: side,
                },
                score,
                ..faces::tests::face(&e)
            })
            .collect();
        let tx = conn.transaction().unwrap();
        let ids = faces::insert_scan(&tx, &sha, model, &faces).unwrap();
        tx.commit().unwrap();
        ids
    }

    fn filter(model: ModelId) -> Filter {
        Filter {
            model: Some(model),
            min_score: 0.6,
            min_face_size: 40,
            count: FaceCount::Any,
            people: Vec::new(),
            unknown: false,
            place: PlaceSel::default(),
            area: None,
        }
    }

    #[test]
    fn entries_sort_by_capture_time_then_path_with_undated_last() {
        let (_dir, conn) = temp();
        for (path, kind, taken_at) in [
            ("/p/IMG_1.jpg", Kind::Image, Some("2024-05-01T12:00:00")),
            ("/p/IMG_2.jpg", Kind::Image, None),
            ("/p/IMG_3.jpg", Kind::Image, Some("2024-05-01T08:00:00")),
            ("/p/VID_1.mp4", Kind::Video, Some("2024-05-01T10:00:00")),
            ("/p/VID_2.mp4", Kind::Video, None),
        ] {
            image(&conn, path, None, kind);
            conn.execute(
                "UPDATE images SET taken_at = ?1 WHERE path = ?2",
                (taken_at, path),
            )
            .unwrap();
        }
        let all = Filter {
            model: None,
            ..filter(ModelId(0))
        };
        assert_eq!(
            names(&conn, "/p", &all),
            [
                "IMG_3.jpg",
                "VID_1.mp4",
                "IMG_1.jpg",
                "IMG_2.jpg",
                "VID_2.mp4"
            ]
        );
    }

    fn names(conn: &Connection, folder: &str, filter: &Filter) -> Vec<String> {
        entries(conn, Path::new(folder), filter)
            .unwrap()
            .into_iter()
            .map(|e| e.path.strip_prefix("/p/").unwrap().display().to_string())
            .collect()
    }

    const BIG: (f32, f32) = (80.0, 0.9);
    const SMALL: (f32, f32) = (20.0, 0.9);
    const WEAK: (f32, f32) = (80.0, 0.55);

    struct Library {
        _dir: tempfile::TempDir,
        conn: Connection,
        model: ModelId,
        other: ModelId,
        faces: Vec<Vec<FaceId>>,
    }

    /// `a/N.jpg`: N counted faces plus uncounted ones; `b/0.jpg`: 1; plus an unscanned image, a video, and `other`-only content.
    fn library() -> Library {
        let (dir, mut conn) = temp();
        let model = register_model(&conn, "m", "d", 2).unwrap();
        let other = register_model(&conn, "o", "d", 2).unwrap();
        let contents: [(&str, &[(f32, f32)]); 6] = [
            ("/p/a/0.jpg", &[SMALL, WEAK]),
            ("/p/a/1.jpg", &[BIG, SMALL]),
            ("/p/a/2.jpg", &[BIG, BIG, WEAK]),
            ("/p/a/3.jpg", &[BIG, BIG, BIG]),
            ("/p/a/4.jpg", &[BIG, BIG, BIG, BIG]),
            ("/p/b/0.jpg", &[BIG]),
        ];
        let mut faces = Vec::new();
        for (n, (path, found)) in contents.iter().enumerate() {
            let sha = [u8::try_from(n).unwrap() + 1; 32];
            image(&conn, path, Some(sha), Kind::Image);
            faces.push(scan(&mut conn, sha, model, found));
        }
        image(&conn, "/p/a/unscanned.jpg", Some([50; 32]), Kind::Image);
        image(&conn, "/p/a/video.mp4", None, Kind::Video);
        image(&conn, "/p/a/other.jpg", Some([60; 32]), Kind::Image);
        scan(&mut conn, [60; 32], other, &[BIG]);
        Library {
            _dir: dir,
            conn,
            model,
            other,
            faces,
        }
    }

    #[test]
    fn count_buckets_use_counted_faces_of_scanned_content() {
        let lib = library();
        let at = |count| {
            names(
                &lib.conn,
                "/p",
                &Filter {
                    count,
                    ..filter(lib.model)
                },
            )
        };
        assert_eq!(at(FaceCount::Any).len(), 9);
        assert_eq!(at(FaceCount::Zero), ["a/0.jpg"]);
        assert_eq!(at(FaceCount::One), ["a/1.jpg", "b/0.jpg"]);
        assert_eq!(at(FaceCount::Two), ["a/2.jpg"]);
        assert_eq!(at(FaceCount::ThreeOrMore), ["a/3.jpg", "a/4.jpg"]);

        faces::set_not_a_face(&lib.conn, lib.faces[1][0]).unwrap();
        assert_eq!(at(FaceCount::Zero), ["a/0.jpg", "a/1.jpg"]);
    }

    #[test]
    fn people_filter_keeps_content_with_all_selected_people() {
        let mut lib = library();
        let ada = people::create(&lib.conn, lib.model).unwrap();
        let bob = people::create(&lib.conn, lib.model).unwrap();
        let assign = |face, person| {
            assert!(faces::assign(&lib.conn, face, person, AssignedBy::Manual).unwrap());
        };
        assign(lib.faces[2][0], ada);
        assign(lib.faces[2][1], bob);
        assign(lib.faces[3][0], ada);
        assign(lib.faces[3][1], ada);
        assign(lib.faces[5][0], bob);
        edit(&mut lib.conn, |tx| people::rename(tx, ada, Some("Ada")));
        let with = |people: &[PersonId]| {
            names(
                &lib.conn,
                "/p",
                &Filter {
                    people: people.to_vec(),
                    ..filter(lib.model)
                },
            )
        };
        assert_eq!(with(&[ada]), ["a/2.jpg", "a/3.jpg"]);
        assert_eq!(with(&[bob]), ["a/2.jpg", "b/0.jpg"]);
        assert_eq!(with(&[ada, bob]), ["a/2.jpg"]);
        assert_eq!(with(&[bob, ada, bob]), ["a/2.jpg"]);
        assert_eq!(
            names(
                &lib.conn,
                "/p",
                &Filter {
                    people: vec![ada],
                    count: FaceCount::Two,
                    ..filter(lib.model)
                }
            ),
            ["a/2.jpg"]
        );
    }

    #[test]
    fn unknown_keeps_counted_faces_without_a_named_or_ignored_person() {
        let mut lib = library();
        let unknown = |lib: &Library| {
            names(
                &lib.conn,
                "/p",
                &Filter {
                    unknown: true,
                    ..filter(lib.model)
                },
            )
        };
        assert_eq!(
            unknown(&lib),
            ["a/1.jpg", "a/2.jpg", "a/3.jpg", "a/4.jpg", "b/0.jpg"]
        );

        let ada = people::create(&lib.conn, lib.model).unwrap();
        let unnamed = people::create(&lib.conn, lib.model).unwrap();
        let ignored = people::create(&lib.conn, lib.model).unwrap();
        for (face, person) in [
            (lib.faces[1][0], ada),
            (lib.faces[2][0], ada),
            (lib.faces[2][1], ignored),
            (lib.faces[3][0], unnamed),
            (lib.faces[3][1], ada),
            (lib.faces[3][2], ada),
            (lib.faces[5][0], ignored),
        ] {
            faces::assign(&lib.conn, face, person, AssignedBy::Auto).unwrap();
        }
        edit(&mut lib.conn, |tx| people::rename(tx, ada, Some("Ada")));
        edit(&mut lib.conn, |tx| people::set_ignored(tx, ignored, true));
        // a/1, a/2: their unassigned faces (small, weak) don't count
        assert_eq!(unknown(&lib), ["a/3.jpg", "a/4.jpg"]);
        faces::set_not_a_face(&lib.conn, lib.faces[3][0]).unwrap();
        assert_eq!(unknown(&lib), ["a/4.jpg"]);
    }

    #[test]
    fn filters_combine_with_the_folder() {
        let lib = library();
        let one = Filter {
            count: FaceCount::One,
            ..filter(lib.model)
        };
        assert_eq!(names(&lib.conn, "/p/a", &one), ["a/1.jpg"]);
        assert_eq!(names(&lib.conn, "/p/b", &one), ["b/0.jpg"]);
        assert!(names(&lib.conn, "/p/c", &one).is_empty());
        let unknown = Filter {
            unknown: true,
            ..one
        };
        assert_eq!(names(&lib.conn, "/p/b", &unknown), ["b/0.jpg"]);
    }

    #[test]
    fn faces_of_other_models_are_invisible() {
        let mut lib = library();
        let p = people::create(&lib.conn, lib.other).unwrap();
        let theirs: Vec<FaceId> = lib
            .conn
            .prepare("SELECT id FROM faces WHERE model_id = ?1")
            .unwrap()
            .query_map([lib.other.0], |r| r.get(0).map(FaceId))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        faces::assign(&lib.conn, theirs[0], p, AssignedBy::Manual).unwrap();
        edit(&mut lib.conn, |tx| people::rename(tx, p, Some("Other")));
        let with = |f: Filter| names(&lib.conn, "/p/a", &f);
        assert!(
            !with(Filter {
                count: FaceCount::One,
                ..filter(lib.model)
            })
            .contains(&"a/other.jpg".to_owned())
        );
        assert!(
            !with(Filter {
                count: FaceCount::Zero,
                ..filter(lib.model)
            })
            .contains(&"a/other.jpg".to_owned())
        );
        assert!(
            !with(Filter {
                unknown: true,
                ..filter(lib.model)
            })
            .contains(&"a/other.jpg".to_owned())
        );
        assert_eq!(
            with(Filter {
                count: FaceCount::One,
                ..filter(lib.other)
            }),
            ["a/other.jpg"]
        );
        assert!(people::list(&lib.conn, lib.model).unwrap().is_empty());
    }

    fn place(geoname_id: i64, name: &str, region_code: &str, country_code: &str) -> Place {
        Place {
            geoname_id,
            name: name.to_owned(),
            region_code: region_code.to_owned(),
            region: format!("region {region_code}"),
            country_code: country_code.to_owned(),
            country: format!("country {country_code}"),
            coords: None,
        }
    }

    fn sel(country: Option<&str>, region: Option<&str>, city: Option<i64>, none: bool) -> PlaceSel {
        PlaceSel {
            country: country.map(str::to_owned),
            region: region.map(str::to_owned),
            city,
            none,
        }
    }

    /// Located images in PT and FR, where region `11` exists in both, plus a marker, an unlocated and an unhashed image, a video and a missing image.
    fn places() -> (tempfile::TempDir, Connection) {
        let (dir, mut conn) = temp();
        let located = [
            ("/p/lisbon.jpg", place(1, "Lisbon", "14", "PT")),
            ("/p/q/lisbon-copy.jpg", place(1, "Lisbon", "14", "PT")),
            ("/p/sintra.jpg", place(2, "Sintra", "14", "PT")),
            ("/p/porto.jpg", place(3, "Porto", "17", "PT")),
            ("/p/guarda.jpg", place(4, "Guarda", "11", "PT")),
            ("/p/paris.jpg", place(5, "Paris", "11", "FR")),
        ];
        for (n, (path, _)) in located.iter().enumerate() {
            image(
                &conn,
                path,
                Some([u8::try_from(n).unwrap(); 32]),
                Kind::Image,
            );
        }
        image(&conn, "/p/marker.jpg", Some([100; 32]), Kind::Image);
        image(&conn, "/p/unlocated.jpg", Some([101; 32]), Kind::Image);
        image(&conn, "/p/unhashed.jpg", None, Kind::Image);
        image(&conn, "/p/video.mp4", None, Kind::Video);
        image(&conn, "/p/gone.jpg", Some([102; 32]), Kind::Image);
        conn.execute(
            "UPDATE images SET status = 'missing' WHERE path = '/p/gone.jpg'",
            [],
        )
        .unwrap();
        edit(&mut conn, |tx| {
            for (n, (_, at)) in located.iter().enumerate() {
                geo::set_manual(tx, &[u8::try_from(n).unwrap(); 32], at, 10)?;
            }
            geo::insert_exif(tx, &[102; 32], &place(1, "Lisbon", "14", "PT"), 10)?;
            geo::clear(tx, &[100; 32], 10)
        });
        (dir, conn)
    }

    fn placed(conn: &Connection, place: PlaceSel) -> Vec<String> {
        names(
            conn,
            "/p",
            &Filter {
                model: None,
                place,
                ..filter(ModelId(0))
            },
        )
    }

    #[test]
    fn place_filter_narrows_by_country_region_and_city() {
        let (_dir, conn) = places();
        assert_eq!(
            placed(&conn, sel(Some("PT"), None, None, false)),
            [
                "guarda.jpg",
                "lisbon.jpg",
                "porto.jpg",
                "q/lisbon-copy.jpg",
                "sintra.jpg"
            ]
        );
        assert_eq!(
            placed(&conn, sel(Some("PT"), Some("14"), None, false)),
            ["lisbon.jpg", "q/lisbon-copy.jpg", "sintra.jpg"]
        );
        assert_eq!(
            placed(&conn, sel(Some("PT"), Some("14"), Some(2), false)),
            ["sintra.jpg"]
        );
        assert_eq!(
            placed(&conn, sel(None, None, Some(1), false)),
            ["lisbon.jpg", "q/lisbon-copy.jpg"]
        );
        assert!(placed(&conn, sel(Some("FR"), None, Some(1), false)).is_empty());
        assert!(placed(&conn, sel(Some("ES"), None, None, false)).is_empty());
        assert_eq!(
            names(
                &conn,
                "/p/q",
                &Filter {
                    model: None,
                    place: sel(Some("PT"), None, None, false),
                    ..filter(ModelId(0))
                }
            ),
            ["q/lisbon-copy.jpg"]
        );
        assert_eq!(placed(&conn, PlaceSel::default()).len(), 10);
        assert_eq!(placed(&conn, sel(None, Some("14"), None, false)).len(), 10);
    }

    #[test]
    fn region_codes_are_scoped_to_their_country() {
        let (_dir, conn) = places();
        assert_eq!(
            placed(&conn, sel(Some("PT"), Some("11"), None, false)),
            ["guarda.jpg"]
        );
        assert_eq!(
            placed(&conn, sel(Some("FR"), Some("11"), None, false)),
            ["paris.jpg"]
        );
    }

    #[test]
    fn no_location_keeps_images_without_a_place() {
        let (_dir, conn) = places();
        assert_eq!(
            placed(&conn, sel(None, None, None, true)),
            ["marker.jpg", "unhashed.jpg", "unlocated.jpg"]
        );
        assert!(placed(&conn, sel(Some("PT"), None, None, true)).is_empty());
        assert!(placed(&conn, sel(None, None, Some(1), true)).is_empty());
    }

    #[test]
    fn place_and_face_filters_combine() {
        let mut lib = library();
        edit(&mut lib.conn, |tx| {
            geo::set_manual(tx, &[2; 32], &place(1, "Lisbon", "14", "PT"), 10)?;
            geo::set_manual(tx, &[3; 32], &place(5, "Paris", "11", "FR"), 10)
        });
        let with = |count, place, model| {
            names(
                &lib.conn,
                "/p",
                &Filter {
                    count,
                    place,
                    model,
                    ..filter(lib.model)
                },
            )
        };
        let pt = || sel(Some("PT"), None, None, false);
        assert_eq!(with(FaceCount::Any, pt(), Some(lib.model)), ["a/1.jpg"]);
        assert_eq!(with(FaceCount::One, pt(), Some(lib.model)), ["a/1.jpg"]);
        assert!(with(FaceCount::Two, pt(), Some(lib.model)).is_empty());
        assert_eq!(
            with(FaceCount::One, sel(None, None, None, true), Some(lib.model)),
            ["b/0.jpg"]
        );
        assert_eq!(with(FaceCount::Two, pt(), None), ["a/1.jpg"]);
        assert_eq!(with(FaceCount::One, PlaceSel::default(), None).len(), 9);
    }

    #[test]
    fn entries_sql_matches_the_single_query_builder() {
        let old = |folder: &str, filter: &Filter| {
            let (lower, upper) = prefix_range(folder);
            let mut sql = String::from(
                "SELECT i.id, i.path, i.sha256, i.kind, i.width, i.height, i.duration_ms FROM images i
         WHERE i.path >= ? AND i.path < ? AND i.status = 'active'",
            );
            let mut params = vec![Value::Text(lower), Value::Text(upper)];
            if let Some(model) = filter.model {
                faces_sql(&mut sql, &mut params, model, filter);
            }
            place_sql(&mut sql, &mut params, &filter.place);
            sql.push_str(&format!(" ORDER BY {ENTRY_ORDER}"));
            (sql, params)
        };
        let full = Filter {
            count: FaceCount::Zero,
            people: vec![PersonId(2), PersonId(1)],
            unknown: true,
            place: sel(Some("PT"), Some("14"), Some(1), true),
            ..filter(ModelId(3))
        };
        for f in [
            filter(ModelId(3)),
            Filter {
                model: None,
                ..full.clone()
            },
            full,
        ] {
            assert_eq!(entries_sql(Path::new("/p"), &f).unwrap(), old("/p", &f));
        }
    }

    const LISBON: (f64, f64) = (38.7223, -9.1393);
    const SCAN: (f64, f64) = (38.71, -9.15);
    const PORTO: (f64, f64) = (41.1579, -8.6291);
    const PARIS: (f64, f64) = (48.8566, 2.3522);

    /// EXIF-located, manual, marker, manual without coordinates, unresolved, no GPS, video and missing content, plus a copy; `a/exif.jpg` and its copy have one face.
    fn located() -> (tempfile::TempDir, Connection, ModelId) {
        let (dir, mut conn) = temp();
        let model = register_model(&conn, "m", "d", 2).unwrap();
        type Row<'a> = (&'a str, u8, Kind, Option<(f64, f64)>);
        let rows: [Row; 9] = [
            ("/p/a/exif.jpg", 1, Kind::Image, Some(SCAN)),
            ("/p/a/manual.jpg", 2, Kind::Image, Some(SCAN)),
            ("/p/a/marker.jpg", 3, Kind::Image, Some(SCAN)),
            ("/p/a/nocoords.jpg", 4, Kind::Image, Some(SCAN)),
            ("/p/b/unresolved.jpg", 5, Kind::Image, Some(PARIS)),
            ("/p/b/nogps.jpg", 6, Kind::Image, None),
            ("/p/b/video.mp4", 7, Kind::Video, Some(PARIS)),
            ("/p/b/gone.jpg", 8, Kind::Image, Some(PARIS)),
            ("/p/c/copy.jpg", 1, Kind::Image, Some(SCAN)),
        ];
        for (path, n, kind, _) in rows {
            image(&conn, path, Some([n; 32]), kind);
        }
        conn.execute(
            "UPDATE images SET status = 'missing' WHERE path = '/p/b/gone.jpg'",
            [],
        )
        .unwrap();
        let coords = |p: Place, at| Place {
            coords: Some(at),
            ..p
        };
        edit(&mut conn, |tx| {
            for (path, n, _, gps) in rows {
                if path != "/p/c/copy.jpg" {
                    geo::insert_scan(tx, &[n; 32], gps)?;
                }
            }
            geo::insert_exif(
                tx,
                &[1; 32],
                &coords(place(1, "Lisbon", "14", "PT"), LISBON),
                10,
            )?;
            geo::insert_exif(
                tx,
                &[2; 32],
                &coords(place(1, "Lisbon", "14", "PT"), LISBON),
                10,
            )?;
            geo::set_manual(
                tx,
                &[2; 32],
                &coords(place(3, "Porto", "17", "PT"), PORTO),
                10,
            )?;
            geo::insert_exif(
                tx,
                &[3; 32],
                &coords(place(1, "Lisbon", "14", "PT"), LISBON),
                10,
            )?;
            geo::clear(tx, &[3; 32], 10)?;
            geo::set_manual(tx, &[4; 32], &place(9, "Nowhere", "01", "PT"), 10)?;
            geo::set_manual(
                tx,
                &[8; 32],
                &coords(place(5, "Paris", "11", "FR"), PARIS),
                10,
            )
        });
        scan(&mut conn, [1; 32], model, &[BIG]);
        scan(&mut conn, [2; 32], model, &[]);
        (dir, conn, model)
    }

    fn pointed(conn: &Connection, folder: &str, filter: &Filter) -> Vec<(String, f64, f64)> {
        points(conn, Path::new(folder), filter)
            .unwrap()
            .into_iter()
            .map(|(id, lat, lon)| {
                let path: String = conn
                    .query_row("SELECT path FROM images WHERE id = ?1", [id.0], |r| {
                        r.get(0)
                    })
                    .unwrap();
                (path.strip_prefix("/p/").unwrap().to_owned(), lat, lon)
            })
            .collect()
    }

    fn unfiltered() -> Filter {
        Filter {
            model: None,
            ..filter(ModelId(0))
        }
    }

    #[test]
    fn points_take_manual_places_over_exif_and_skip_unlocated() {
        let (_dir, conn, _) = located();
        let p = |name: &str, (lat, lon): (f64, f64)| (name.to_owned(), lat, lon);
        assert_eq!(
            pointed(&conn, "/p", &unfiltered()),
            [
                p("a/exif.jpg", SCAN),
                p("a/manual.jpg", PORTO),
                p("b/unresolved.jpg", PARIS),
                p("c/copy.jpg", SCAN),
            ]
        );
        assert_eq!(
            pointed(&conn, "/p/b", &unfiltered()),
            [p("b/unresolved.jpg", PARIS)]
        );
    }

    #[test]
    fn point_of_agrees_with_points() {
        let (_dir, conn, _) = located();
        assert_eq!(point_of(&conn, &[1; 32]).unwrap(), Some(SCAN));
        assert_eq!(point_of(&conn, &[2; 32]).unwrap(), Some(PORTO));
        assert_eq!(point_of(&conn, &[5; 32]).unwrap(), Some(PARIS));
        for n in [3, 4, 6, 7, 8, 99] {
            assert_eq!(point_of(&conn, &[n; 32]).unwrap(), None, "content {n}");
        }
        for (id, lat, lon) in points(&conn, Path::new("/p"), &unfiltered()).unwrap() {
            let sha: Sha256 = conn
                .query_row("SELECT sha256 FROM images WHERE id = ?1", [id.0], |r| {
                    r.get(0)
                })
                .unwrap();
            assert_eq!(point_of(&conn, &sha).unwrap(), Some((lat, lon)));
        }
    }

    #[test]
    fn area_bounds_are_inclusive_and_combine_with_other_filters() {
        let (_dir, conn, model) = located();
        let around = |(lat, lon): (f64, f64)| Area {
            south: lat,
            north: lat,
            west: lon,
            east: lon,
        };
        let portugal = Area {
            south: 37.0,
            north: 42.0,
            west: -10.0,
            east: -6.0,
        };
        let within = |area, f: Filter| {
            names(
                &conn,
                "/p",
                &Filter {
                    area: Some(area),
                    ..f
                },
            )
        };
        assert_eq!(within(around(PORTO), unfiltered()), ["a/manual.jpg"]);
        assert_eq!(
            within(around(SCAN), unfiltered()),
            ["a/exif.jpg", "c/copy.jpg"]
        );
        assert_eq!(
            within(portugal, unfiltered()),
            ["a/exif.jpg", "a/manual.jpg", "c/copy.jpg"]
        );
        assert_eq!(
            within(
                portugal,
                Filter {
                    place: sel(Some("PT"), Some("17"), None, false),
                    ..unfiltered()
                }
            ),
            ["a/manual.jpg"]
        );
        assert_eq!(
            within(
                portugal,
                Filter {
                    count: FaceCount::One,
                    ..filter(model)
                }
            ),
            ["a/exif.jpg", "c/copy.jpg"]
        );
        assert!(within(around(LISBON), unfiltered()).is_empty());
        assert_eq!(
            pointed(
                &conn,
                "/p/a",
                &Filter {
                    area: Some(portugal),
                    count: FaceCount::Zero,
                    ..filter(model)
                }
            ),
            [("a/manual.jpg".to_owned(), PORTO.0, PORTO.1)]
        );
    }

    /// 50k synthetic images; run with `--ignored --nocapture`.
    #[test]
    #[ignore = "benchmark"]
    fn filter_timing_on_50k_images() {
        use std::time::Instant;

        const IMAGES: usize = 50_000;
        const PEOPLE: i64 = 1500;
        let (_dir, mut conn) = temp();
        let model = register_model(&conn, "m", "d", 512).unwrap();
        let embedding = faces::encode_f32(&[0.1; 512]);
        let crop = vec![0xa5_u8; 4300];
        let started = Instant::now();
        let tx = conn.transaction().unwrap();
        for p in 0..PEOPLE {
            tx.execute(
                "INSERT INTO people (model_id, name) VALUES (?1, ?2)",
                rusqlite::params![model.0, (p < 200).then(|| format!("person {p}"))],
            )
            .unwrap();
        }
        for id in 0..60_i64 {
            tx.execute(
                "INSERT INTO places (geoname_id, name, region_code, region, country_code, country, lat, lon)
                 VALUES (?1, ?2, ?3, ?3, ?4, ?4, ?5, ?6)",
                rusqlite::params![
                    id + 1,
                    format!("city {id}"),
                    format!("{:02}", id / 5 % 4),
                    ["PT", "FR", "DE"][usize::try_from(id / 20).unwrap()],
                    f64::from(u8::try_from(id).unwrap()).mul_add(0.3, 35.0),
                    f64::from(u8::try_from(id).unwrap()).mul_add(0.4, -10.0),
                ],
            )
            .unwrap();
        }
        let mut face = 0_usize;
        for n in 0..IMAGES {
            let mut sha = [0_u8; 32];
            sha[..8].copy_from_slice(&n.to_le_bytes());
            let path = format!("/lib/{}/{:02}/{n}.jpg", 2000 + n / 2500, n % 12);
            tx.execute(
                "INSERT INTO images (path, size, mtime_ns, sha256, kind) VALUES (?1, 1, 1, ?2, 'image')",
                rusqlite::params![path, sha],
            )
            .unwrap();
            tx.execute(
                "INSERT INTO face_scans (sha256, model_id) VALUES (?1, ?2)",
                rusqlite::params![sha, model.0],
            )
            .unwrap();
            if n % 5 != 0 {
                let k = f64::from(u32::try_from(n % 1000).unwrap());
                tx.execute(
                    "INSERT INTO geo_scans (sha256, lat, lon) VALUES (?1, ?2, ?3)",
                    rusqlite::params![sha, k.mul_add(0.02, 35.0), k.mul_add(0.03, -10.0)],
                )
                .unwrap();
                tx.execute(
                    "INSERT INTO locations (sha256, place_id, source, set_at) VALUES (?1, ?2, ?3, 0)",
                    rusqlite::params![
                        sha,
                        i64::try_from(n % 60).unwrap() + 1,
                        if n % 7 == 0 { "manual" } else { "exif" },
                    ],
                )
                .unwrap();
            }
            for _ in 0..[0, 0, 0, 1, 1, 1, 2, 2, 3, 5][n % 10] {
                face += 1;
                let person =
                    (face % 10 < 7).then(|| i64::try_from(face * 7919).unwrap() % PEOPLE + 1);
                tx.execute(
                    "INSERT INTO faces (sha256, model_id, x, y, w, h, landmarks, score, embedding, crop, person_id, assigned_by)
                     VALUES (?1, ?2, 0, 0, ?3, ?3, zeroblob(40), ?4, ?5, ?6, ?7, ?8)",
                    rusqlite::params![
                        sha,
                        model.0,
                        f64::from(u32::try_from(20 + face % 100).unwrap()),
                        0.5 + f64::from(u32::try_from(face % 50).unwrap()) / 100.0,
                        embedding,
                        crop,
                        person,
                        person.map(|_| "auto"),
                    ],
                )
                .unwrap();
            }
        }
        tx.commit().unwrap();
        println!(
            "built {IMAGES} images, {face} faces in {:?}",
            started.elapsed()
        );

        let base = filter(model);
        let cases = [
            ("no filter", base.clone()),
            (
                "count 0",
                Filter {
                    count: FaceCount::Zero,
                    ..base.clone()
                },
            ),
            (
                "count 1",
                Filter {
                    count: FaceCount::One,
                    ..base.clone()
                },
            ),
            (
                "count 3+",
                Filter {
                    count: FaceCount::ThreeOrMore,
                    ..base.clone()
                },
            ),
            (
                "one person",
                Filter {
                    people: vec![PersonId(1)],
                    ..base.clone()
                },
            ),
            (
                "two people",
                Filter {
                    people: vec![PersonId(1), PersonId(2)],
                    ..base.clone()
                },
            ),
            (
                "unknown",
                Filter {
                    unknown: true,
                    ..base.clone()
                },
            ),
            (
                "all",
                Filter {
                    count: FaceCount::Two,
                    people: vec![PersonId(1)],
                    unknown: true,
                    ..base.clone()
                },
            ),
            (
                "country",
                Filter {
                    place: sel(Some("FR"), None, None, false),
                    ..base.clone()
                },
            ),
            (
                "region",
                Filter {
                    place: sel(Some("FR"), Some("02"), None, false),
                    ..base.clone()
                },
            ),
            (
                "city",
                Filter {
                    place: sel(Some("FR"), Some("02"), Some(32), false),
                    ..base.clone()
                },
            ),
            (
                "no location",
                Filter {
                    place: sel(None, None, None, true),
                    ..base.clone()
                },
            ),
            (
                "area",
                Filter {
                    area: Some(Area {
                        south: 40.0,
                        north: 50.0,
                        west: 0.0,
                        east: 20.0,
                    }),
                    ..base.clone()
                },
            ),
            (
                "place+person",
                Filter {
                    people: vec![PersonId(1)],
                    place: sel(Some("PT"), None, None, false),
                    ..base.clone()
                },
            ),
        ];
        let root = PathBuf::from("/lib");
        for (name, f) in &cases {
            entries(&conn, &root, f).unwrap();
            let started = Instant::now();
            let n = entries(&conn, &root, f).unwrap().len();
            let took = started.elapsed();
            points(&conn, &root, f).unwrap();
            let started = Instant::now();
            let p = points(&conn, &root, f).unwrap().len();
            println!(
                "{name:<12} {n:>6} entries {took:?}, {p:>6} points {:?}",
                started.elapsed()
            );
        }
        let started = Instant::now();
        let n = people::list(&conn, model).unwrap().len();
        println!("people list  {n:>6} people  {:?}", started.elapsed());
    }
}
