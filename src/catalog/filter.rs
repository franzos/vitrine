//! Entries below a folder, narrowed by their faces.

use std::path::Path;

use rusqlite::Connection;
use rusqlite::types::Value;

use super::faces::ModelId;
use super::images::{Entry, entry_from_row, prefix_range};
use super::people::PersonId;
use crate::db::{Result, path_str};

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

/// A face counts when it isn't marked not-a-face, scores at least `min_score`, and both box sides are at least `min_face_size`.
#[derive(Debug, Clone, PartialEq)]
pub struct Filter {
    pub model: ModelId,
    pub min_score: f32,
    pub min_face_size: u32,
    /// Content scanned with `model` with this many counted faces; [`FaceCount::Any`] also keeps unscanned content and videos.
    pub count: FaceCount,
    /// Content with faces of all of these people.
    pub people: Vec<PersonId>,
    /// Content with a counted face that has no person, or an unnamed, non-ignored one.
    pub unknown: bool,
}

/// Any depth below `folder`, path order.
pub fn entries(conn: &Connection, folder: &Path, filter: &Filter) -> Result<Vec<Entry>> {
    let (lower, upper) = prefix_range(path_str(folder)?);
    let mut sql = String::from(
        "SELECT i.id, i.path, i.sha256, i.kind, i.width, i.height FROM images i
         WHERE i.path >= ? AND i.path < ? AND i.status = 'active'",
    );
    let mut params = vec![Value::Text(lower), Value::Text(upper)];
    let model = Value::Integer(filter.model.0);
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
        let counted = counted(&mut params);
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
        let counted = counted(&mut params);
        sql.push_str(&format!(
            " AND i.sha256 IN (SELECT f.sha256 FROM faces f LEFT JOIN people p ON p.id = f.person_id
              WHERE {counted} AND (f.person_id IS NULL OR (p.name IS NULL AND p.ignored = 0)))"
        ));
    }
    sql.push_str(" ORDER BY i.path");

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params), entry_from_row)?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::catalog::faces::{self, AssignedBy, FaceId, NewFace, Rect, register_model};
    use crate::catalog::images::{self, Kind, NewImage};
    use crate::catalog::{edit, people, temp};
    use crate::db::Sha256;

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
            model,
            min_score: 0.6,
            min_face_size: 40,
            count: FaceCount::Any,
            people: Vec::new(),
            unknown: false,
        }
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
        ];
        let root = PathBuf::from("/lib");
        for (name, f) in &cases {
            entries(&conn, &root, f).unwrap();
            let started = Instant::now();
            let n = entries(&conn, &root, f).unwrap().len();
            println!("{name:<12} {n:>6} entries {:?}", started.elapsed());
        }
        let started = Instant::now();
        let n = people::list(&conn, model).unwrap().len();
        println!("people list  {n:>6} people  {:?}", started.elapsed());
    }
}
