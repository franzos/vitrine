#[cfg(test)]
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::PathBuf;

use rusqlite::types::{FromSql, FromSqlResult, ToSqlOutput, Type, ValueRef};
use rusqlite::{Connection, OptionalExtension, ToSql, Transaction, params};

use super::images::prefix_range;
use super::people::{self, Person, PersonId};
use crate::db::{Error, Result, Sha256, path_str};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FaceId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignedBy {
    Auto,
    Manual,
}

/// Face box in source image pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Debug, Clone)]
pub struct NewFace<'a> {
    pub rect: Rect,
    pub landmarks: [[f32; 2]; 5],
    pub score: f32,
    pub embedding: &'a [f32],
    /// JPEG.
    pub crop: &'a [u8],
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterFace {
    pub id: FaceId,
    pub person_id: Option<PersonId>,
    pub assigned_by: Option<AssignedBy>,
    /// Belongs to an ignored person.
    pub ignored: bool,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageFace {
    pub id: FaceId,
    pub rect: Rect,
    pub person: Option<Person>,
}

pub fn register_model(
    conn: &Connection,
    name: &str,
    detector: &str,
    dim: usize,
) -> Result<ModelId> {
    let id = conn.query_row(
        "INSERT INTO face_models (name, detector, dim) VALUES (?1, ?2, ?3)
         ON CONFLICT (name) DO UPDATE SET name = name
         RETURNING id",
        params![name, detector, dim],
        |row| row.get(0),
    )?;
    Ok(ModelId(id))
}

pub fn model_by_name(conn: &Connection, name: &str) -> Result<Option<ModelId>> {
    Ok(conn
        .query_row(
            "SELECT id FROM face_models WHERE name = ?1",
            [name],
            |row| row.get(0).map(ModelId),
        )
        .optional()?)
}

/// One entry per content, with its lowest path under the first root that has it; root order, then path order.
pub fn pending(
    conn: &Connection,
    model: ModelId,
    roots: &[PathBuf],
) -> Result<Vec<(Sha256, PathBuf)>> {
    let mut stmt = conn.prepare(
        "SELECT i.sha256, min(i.path) FROM images i
         WHERE i.status = 'active' AND i.kind = 'image' AND i.sha256 IS NOT NULL
           AND (i.path = ?2 OR (i.path >= ?3 AND i.path < ?4))
           AND NOT EXISTS (
               SELECT 1 FROM face_scans s WHERE s.sha256 = i.sha256 AND s.model_id = ?1
           )
         GROUP BY i.sha256
         ORDER BY 2",
    )?;
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for root in roots {
        let root = path_str(root)?;
        let (lower, upper) = prefix_range(root);
        let rows = stmt.query_map(params![model.0, root, lower, upper], |row| {
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

#[cfg(test)]
const COUNTS: &str = "SELECT sha256, count(*) FROM faces
     WHERE model_id = ?1 AND not_a_face = 0 AND score >= ?2 AND min(w, h) >= ?3
     GROUP BY sha256";

/// Content without counted faces is absent; `face_scans` says whether it was scanned.
#[cfg(test)]
pub fn counts(
    conn: &Connection,
    model: ModelId,
    min_score: f32,
    min_face_size: u32,
) -> Result<HashMap<Sha256, u32>> {
    let mut stmt = conn.prepare(COUNTS)?;
    let rows = stmt.query_map(
        params![model.0, f64::from(min_score), min_face_size],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// `faces` may be empty; the scan is still recorded.
pub fn insert_scan(
    tx: &Transaction<'_>,
    sha256: &Sha256,
    model: ModelId,
    faces: &[NewFace<'_>],
) -> Result<Vec<FaceId>> {
    let dim: usize = tx.query_row(
        "SELECT dim FROM face_models WHERE id = ?1",
        [model.0],
        |row| row.get(0),
    )?;
    if let Some(face) = faces.iter().find(|f| f.embedding.len() != dim) {
        return Err(Error::EmbeddingDim {
            model: model.0,
            expected: dim,
            actual: face.embedding.len(),
        });
    }
    tx.execute(
        "INSERT INTO face_scans (sha256, model_id) VALUES (?1, ?2)",
        params![sha256, model.0],
    )?;
    let mut stmt = tx.prepare(
        "INSERT INTO faces (sha256, model_id, x, y, w, h, landmarks, score, embedding, crop)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
         RETURNING id",
    )?;
    faces
        .iter()
        .map(|f| {
            let id = stmt.query_row(
                params![
                    sha256,
                    model.0,
                    f.rect.x,
                    f.rect.y,
                    f.rect.w,
                    f.rect.h,
                    encode_f32(f.landmarks.as_flattened()),
                    f.score,
                    encode_f32(f.embedding),
                    f.crop,
                ],
                |row| row.get(0),
            )?;
            Ok(FaceId(id))
        })
        .collect()
}

/// `false` when the face is gone or not-a-face, the person is gone or of another model, or `by` is auto and the face is manual or the person ignored.
pub fn assign(conn: &Connection, face: FaceId, person: PersonId, by: AssignedBy) -> Result<bool> {
    let changed = conn.execute(
        "UPDATE faces SET person_id = ?2, assigned_by = ?3
         WHERE id = ?1 AND not_a_face = 0
           AND (?3 = 'manual' OR (assigned_by IS NOT 'manual'
                AND (SELECT ignored FROM people WHERE id = ?2) = 0))
           AND model_id = (SELECT model_id FROM people WHERE id = ?2)",
        params![face.0, person.0, by],
    )?;
    Ok(changed == 1)
}

/// Pins the face as manual so clustering leaves it alone.
pub fn unassign(conn: &Connection, face: FaceId) -> Result<()> {
    conn.execute(
        "UPDATE faces SET person_id = NULL, assigned_by = 'manual' WHERE id = ?1",
        [face.0],
    )?;
    Ok(())
}

pub fn set_not_a_face(conn: &Connection, face: FaceId) -> Result<()> {
    conn.execute(
        "UPDATE faces SET not_a_face = 1, person_id = NULL, assigned_by = 'manual' WHERE id = ?1",
        [face.0],
    )?;
    Ok(())
}

pub fn reset_auto(conn: &Connection, model: ModelId) -> Result<usize> {
    Ok(conn.execute(
        "UPDATE faces SET person_id = NULL, assigned_by = NULL
         WHERE model_id = ?1 AND assigned_by = 'auto'",
        [model.0],
    )?)
}

pub fn for_clustering(
    conn: &Connection,
    model: ModelId,
    min_score: f32,
    min_face_size: u32,
) -> Result<Vec<ClusterFace>> {
    let mut stmt = conn.prepare(
        "SELECT f.id, f.person_id, f.assigned_by, coalesce(p.ignored, 0), f.embedding FROM faces f
         LEFT JOIN people p ON p.id = f.person_id
         WHERE f.model_id = ?1 AND f.not_a_face = 0 AND f.score >= ?2 AND min(f.w, f.h) >= ?3
         ORDER BY f.id",
    )?;
    let rows = stmt.query_map(
        params![model.0, f64::from(min_score), min_face_size],
        |row| {
            let blob = row.get_ref(4)?.as_blob()?;
            let embedding = decode_f32(blob).ok_or_else(|| {
                rusqlite::Error::FromSqlConversionFailure(
                    4,
                    Type::Blob,
                    format!("embedding blob of {} bytes", blob.len()).into(),
                )
            })?;
            Ok(ClusterFace {
                id: FaceId(row.get(0)?),
                person_id: row.get::<_, Option<i64>>(1)?.map(PersonId),
                assigned_by: row.get(2)?,
                ignored: row.get(3)?,
                embedding,
            })
        },
    )?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// Faces the count filter counts, minus those of ignored people; left to right.
pub fn of_image(
    conn: &Connection,
    sha256: &Sha256,
    model: ModelId,
    min_score: f32,
    min_face_size: u32,
) -> Result<Vec<ImageFace>> {
    let sql = format!(
        "SELECT fc.id, fc.x, fc.y, fc.w, fc.h, {} FROM faces fc
         LEFT JOIN people p ON p.id = fc.person_id
         WHERE fc.sha256 = ?1 AND fc.model_id = ?2 AND fc.not_a_face = 0
           AND fc.score >= ?3 AND min(fc.w, fc.h) >= ?4 AND p.ignored IS NOT 1
         ORDER BY fc.x, fc.id",
        people::COLUMNS
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(
        params![sha256, model.0, f64::from(min_score), min_face_size],
        |row| {
            let person = row
                .get::<_, Option<i64>>(5)?
                .map(|_| people::from_row(row, 5))
                .transpose()?;
            Ok(ImageFace {
                id: FaceId(row.get(0)?),
                rect: Rect {
                    x: row.get(1)?,
                    y: row.get(2)?,
                    w: row.get(3)?,
                    h: row.get(4)?,
                },
                person,
            })
        },
    )?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

pub fn unclustered(conn: &Connection, model: ModelId) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS (
             SELECT 1 FROM faces
             WHERE model_id = ?1 AND id > (SELECT clustered_to FROM face_models WHERE id = ?1)
         )",
        [model.0],
        |row| row.get(0),
    )?)
}

pub fn mark_clustered(conn: &Connection, model: ModelId) -> Result<()> {
    conn.execute(
        "UPDATE face_models
         SET clustered_to = (SELECT coalesce(max(id), 0) FROM faces WHERE model_id = ?1)
         WHERE id = ?1",
        [model.0],
    )?;
    Ok(())
}

pub fn of_person(conn: &Connection, person: PersonId) -> Result<Vec<FaceId>> {
    let mut stmt =
        conn.prepare("SELECT id FROM faces WHERE person_id = ?1 ORDER BY score DESC, id")?;
    let rows = stmt.query_map([person.0], |row| row.get(0).map(FaceId))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

pub fn crop(conn: &Connection, face: FaceId) -> Result<Option<Vec<u8>>> {
    Ok(conn
        .query_row("SELECT crop FROM faces WHERE id = ?1", [face.0], |row| {
            row.get(0)
        })
        .optional()?)
}

/// Little-endian; the storage format for embeddings and landmarks.
#[must_use]
pub fn encode_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[must_use]
pub fn decode_f32(bytes: &[u8]) -> Option<Vec<f32>> {
    let (chunks, rest) = bytes.as_chunks::<4>();
    rest.is_empty()
        .then(|| chunks.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

impl ToSql for AssignedBy {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(match self {
            Self::Auto => "auto",
            Self::Manual => "manual",
        }
        .into())
    }
}

impl FromSql for AssignedBy {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        super::enum_from_sql(value, |s| match s {
            "auto" => Some(Self::Auto),
            "manual" => Some(Self::Manual),
            _ => None,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::Path;

    use super::*;
    use crate::catalog::images::{self, Kind, NewImage, Status};
    use crate::catalog::{people, temp};

    pub(crate) fn face(embedding: &[f32]) -> NewFace<'_> {
        NewFace {
            rect: Rect {
                x: 1.0,
                y: 2.0,
                w: 30.0,
                h: 40.0,
            },
            landmarks: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            score: 0.9,
            embedding,
            crop: b"jpeg",
        }
    }

    pub(crate) fn scan(
        conn: &mut Connection,
        sha: Sha256,
        model: ModelId,
        n: usize,
    ) -> Vec<FaceId> {
        let embeddings = vec![[0.5_f32, -0.5]; n];
        let faces: Vec<_> = embeddings.iter().map(|e| face(e)).collect();
        let tx = conn.transaction().unwrap();
        let ids = insert_scan(&tx, &sha, model, &faces).unwrap();
        tx.commit().unwrap();
        ids
    }

    pub(crate) fn image(conn: &Connection, path: &str, sha: Sha256) -> images::ImageId {
        images::upsert(
            conn,
            &NewImage {
                path: Path::new(path),
                size: 1,
                mtime_ns: 1,
                sha256: Some(sha),
                kind: Kind::Image,
                width: None,
                height: None,
                taken_at: None,
                duration_ms: None,
                codec: None,
            },
        )
        .unwrap()
    }

    #[test]
    fn f32_blob_round_trip() {
        let v = [0.0, -1.5, f32::MIN_POSITIVE, 3.25];
        let bytes = encode_f32(&v);
        assert_eq!(bytes.len(), 16);
        assert_eq!(decode_f32(&bytes).unwrap(), v);
        assert!(decode_f32(&bytes[..15]).is_none());
    }

    #[test]
    fn register_model_is_idempotent() {
        let (_dir, conn) = temp();
        let a = register_model(&conn, "buffalo_l/w600k_r50", "scrfd_10g", 512).unwrap();
        let b = register_model(&conn, "buffalo_l/w600k_r50", "scrfd_10g", 512).unwrap();
        assert_eq!(a, b);
        assert_eq!(
            model_by_name(&conn, "buffalo_l/w600k_r50").unwrap(),
            Some(a)
        );
        assert_eq!(model_by_name(&conn, "other").unwrap(), None);
    }

    #[test]
    fn pending_follows_scans_per_model_and_dedupes_content() {
        let (_dir, mut conn) = temp();
        let a = register_model(&conn, "a", "d", 2).unwrap();
        let b = register_model(&conn, "b", "d", 2).unwrap();
        image(&conn, "/p/x2.jpg", [1; 32]);
        image(&conn, "/p/x1.jpg", [1; 32]);
        let gone = image(&conn, "/p/y.jpg", [2; 32]);
        images::set_status(&conn, gone, Status::Missing).unwrap();
        image(&conn, "/p/z.jpg", [3; 32]);

        let all = [PathBuf::from("/p")];
        assert_eq!(
            pending(&conn, a, &all).unwrap(),
            [
                ([1; 32], PathBuf::from("/p/x1.jpg")),
                ([3; 32], PathBuf::from("/p/z.jpg")),
            ]
        );
        scan(&mut conn, [1; 32], a, 2);
        scan(&mut conn, [3; 32], a, 0);
        assert!(pending(&conn, a, &all).unwrap().is_empty());
        assert_eq!(pending(&conn, b, &all).unwrap().len(), 2);
    }

    #[test]
    fn pending_is_limited_to_roots() {
        let (_dir, conn) = temp();
        let model = register_model(&conn, "m", "d", 2).unwrap();
        image(&conn, "/p/a/x.jpg", [1; 32]);
        image(&conn, "/p/b/x.jpg", [1; 32]);
        image(&conn, "/p/ab/y.jpg", [2; 32]);
        image(&conn, "/p/c.jpg", [3; 32]);
        let roots = |roots: &[&str]| -> Vec<PathBuf> { roots.iter().map(PathBuf::from).collect() };

        assert_eq!(
            pending(&conn, model, &roots(&["/p/b"])).unwrap(),
            [([1; 32], PathBuf::from("/p/b/x.jpg"))]
        );
        assert_eq!(
            pending(&conn, model, &roots(&["/p/c.jpg", "/p/a", "/p"])).unwrap(),
            [
                ([3; 32], PathBuf::from("/p/c.jpg")),
                ([1; 32], PathBuf::from("/p/a/x.jpg")),
                ([2; 32], PathBuf::from("/p/ab/y.jpg")),
            ]
        );
    }

    #[test]
    fn counts_read_only_the_covering_index() {
        let (_dir, conn) = temp();
        let plan: Vec<String> = conn
            .prepare(&format!("EXPLAIN QUERY PLAN {COUNTS}"))
            .unwrap()
            .query_map(params![1, 0.5, 40], |r| r.get(3))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(
            plan,
            ["SEARCH faces USING COVERING INDEX faces_for_counts (model_id=?)"]
        );
    }

    #[test]
    fn counts_apply_score_size_and_not_a_face() {
        let (_dir, mut conn) = temp();
        let model = register_model(&conn, "m", "d", 2).unwrap();
        let other = register_model(&conn, "o", "d", 2).unwrap();
        let e = [0.5_f32, -0.5];
        let sized = |w: f32, h: f32, score: f32| NewFace {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                w,
                h,
            },
            score,
            ..face(&e)
        };
        let tx = conn.transaction().unwrap();
        let ids = insert_scan(
            &tx,
            &[1; 32],
            model,
            &[
                sized(40.0, 50.0, 0.6),
                sized(40.0, 39.9, 0.9),
                sized(80.0, 80.0, 0.59),
                sized(80.0, 80.0, 0.95),
            ],
        )
        .unwrap();
        insert_scan(&tx, &[2; 32], model, &[sized(10.0, 10.0, 0.9)]).unwrap();
        insert_scan(&tx, &[1; 32], other, &[sized(80.0, 80.0, 0.9)]).unwrap();
        tx.commit().unwrap();

        assert_eq!(
            counts(&conn, model, 0.6, 40).unwrap(),
            HashMap::from([([1; 32], 2)])
        );
        set_not_a_face(&conn, ids[3]).unwrap();
        assert_eq!(
            counts(&conn, model, 0.6, 40).unwrap(),
            HashMap::from([([1; 32], 1)])
        );
        assert_eq!(
            counts(&conn, model, 0.5, 10).unwrap(),
            HashMap::from([([1; 32], 3), ([2; 32], 1)])
        );
    }

    #[test]
    fn insert_scan_round_trips_and_rejects_rescan_and_wrong_dim() {
        let (_dir, mut conn) = temp();
        let model = register_model(&conn, "m", "d", 2).unwrap();
        let ids = scan(&mut conn, [1; 32], model, 2);
        assert_eq!(ids.len(), 2);

        let (rect, landmarks, score): (Vec<f32>, Vec<u8>, f32) = conn
            .query_row(
                "SELECT x, y, w, h, landmarks, score FROM faces WHERE id = ?1",
                [ids[0].0],
                |r| {
                    Ok((
                        vec![r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?],
                        r.get(4)?,
                        r.get(5)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(rect, [1.0, 2.0, 30.0, 40.0]);
        assert_eq!(
            decode_f32(&landmarks).unwrap(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        );
        assert!((score - 0.9).abs() < f32::EPSILON);
        assert_eq!(crop(&conn, ids[0]).unwrap().unwrap(), b"jpeg");
        let faces = for_clustering(&conn, model, 0.0, 0).unwrap();
        assert_eq!(faces[1].embedding, [0.5, -0.5]);

        let tx = conn.transaction().unwrap();
        assert!(insert_scan(&tx, &[1; 32], model, &[]).is_err());
        let wrong = [1.0_f32; 3];
        assert!(matches!(
            insert_scan(&tx, &[2; 32], model, &[face(&wrong)]),
            Err(Error::EmbeddingDim {
                expected: 2,
                actual: 3,
                ..
            })
        ));
    }

    #[test]
    fn manual_assignment_wins_over_auto() {
        let (_dir, mut conn) = temp();
        let model = register_model(&conn, "m", "d", 2).unwrap();
        let ids = scan(&mut conn, [1; 32], model, 2);
        let p1 = people::create(&conn, model).unwrap();
        let p2 = people::create(&conn, model).unwrap();

        assert!(assign(&conn, ids[0], p1, AssignedBy::Auto).unwrap());
        assert!(assign(&conn, ids[0], p2, AssignedBy::Manual).unwrap());
        assert!(!assign(&conn, ids[0], p1, AssignedBy::Auto).unwrap());
        unassign(&conn, ids[1]).unwrap();
        assert!(!assign(&conn, ids[1], p1, AssignedBy::Auto).unwrap());

        let faces = for_clustering(&conn, model, 0.0, 0).unwrap();
        assert_eq!(faces[0].person_id, Some(p2));
        assert_eq!(faces[0].assigned_by, Some(AssignedBy::Manual));
        assert_eq!(faces[1].person_id, None);
        assert_eq!(faces[1].assigned_by, Some(AssignedBy::Manual));
    }

    #[test]
    fn only_manual_assignment_reaches_ignored_people() {
        let (_dir, mut conn) = temp();
        let model = register_model(&conn, "m", "d", 2).unwrap();
        let ids = scan(&mut conn, [1; 32], model, 2);
        let ignored = people::create(&conn, model).unwrap();
        crate::catalog::edit(&mut conn, |tx| people::set_ignored(tx, ignored, true));
        assert!(!assign(&conn, ids[0], ignored, AssignedBy::Auto).unwrap());
        assert!(assign(&conn, ids[1], ignored, AssignedBy::Manual).unwrap());
        let faces = for_clustering(&conn, model, 0.0, 0).unwrap();
        assert_eq!(faces[0].person_id, None);
        assert_eq!(faces[1].person_id, Some(ignored));
    }

    #[test]
    fn assign_rejects_person_of_other_model() {
        let (_dir, mut conn) = temp();
        let a = register_model(&conn, "a", "d", 2).unwrap();
        let b = register_model(&conn, "b", "d", 2).unwrap();
        let ids = scan(&mut conn, [1; 32], a, 1);
        let other = people::create(&conn, b).unwrap();
        assert!(!assign(&conn, ids[0], other, AssignedBy::Manual).unwrap());
    }

    #[test]
    fn reset_auto_keeps_manual_and_clustering_skips_excluded() {
        let (_dir, mut conn) = temp();
        let model = register_model(&conn, "m", "d", 2).unwrap();
        let ids = scan(&mut conn, [1; 32], model, 4);
        let p = people::create(&conn, model).unwrap();
        let ignored = people::create(&conn, model).unwrap();
        crate::catalog::edit(&mut conn, |tx| people::set_ignored(tx, ignored, true));

        assign(&conn, ids[0], p, AssignedBy::Auto).unwrap();
        assign(&conn, ids[1], p, AssignedBy::Manual).unwrap();
        assign(&conn, ids[2], ignored, AssignedBy::Manual).unwrap();
        set_not_a_face(&conn, ids[3]).unwrap();
        assert!(!assign(&conn, ids[3], p, AssignedBy::Manual).unwrap());

        assert_eq!(reset_auto(&conn, model).unwrap(), 1);
        let faces = for_clustering(&conn, model, 0.0, 0).unwrap();
        assert_eq!(
            faces
                .iter()
                .map(|f| (f.id, f.person_id, f.assigned_by, f.ignored))
                .collect::<Vec<_>>(),
            [
                (ids[0], None, None, false),
                (ids[1], Some(p), Some(AssignedBy::Manual), false),
                (ids[2], Some(ignored), Some(AssignedBy::Manual), true),
            ]
        );
    }

    #[test]
    fn clustering_applies_the_count_filter() {
        let (_dir, mut conn) = temp();
        let model = register_model(&conn, "m", "d", 2).unwrap();
        let e = [0.5_f32, -0.5];
        let sized = |side: f32, score: f32| NewFace {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                w: side,
                h: side * 2.0,
            },
            score,
            ..face(&e)
        };
        let tx = conn.transaction().unwrap();
        let ids = insert_scan(
            &tx,
            &[1; 32],
            model,
            &[sized(40.0, 0.6), sized(39.0, 0.9), sized(80.0, 0.59)],
        )
        .unwrap();
        tx.commit().unwrap();
        let kept: Vec<_> = for_clustering(&conn, model, 0.6, 40)
            .unwrap()
            .iter()
            .map(|f| f.id)
            .collect();
        assert_eq!(kept, [ids[0]]);
    }

    #[test]
    fn of_image_applies_the_count_filter_and_skips_ignored_people() {
        let (_dir, mut conn) = temp();
        let model = register_model(&conn, "m", "d", 2).unwrap();
        let other = register_model(&conn, "o", "d", 2).unwrap();
        let e = [0.5_f32, -0.5];
        let at = |x: f32, side: f32, score: f32| NewFace {
            rect: Rect {
                x,
                y: 5.0,
                w: side,
                h: side * 2.0,
            },
            score,
            ..face(&e)
        };
        let tx = conn.transaction().unwrap();
        let ids = insert_scan(
            &tx,
            &[1; 32],
            model,
            &[
                at(30.0, 40.0, 0.6),
                at(10.0, 40.0, 0.9),
                at(20.0, 39.9, 0.9),
                at(25.0, 80.0, 0.59),
                at(30.0, 80.0, 0.9),
                at(5.0, 80.0, 0.9),
                at(0.0, 80.0, 0.9),
            ],
        )
        .unwrap();
        insert_scan(&tx, &[2; 32], model, &[at(0.0, 80.0, 0.9)]).unwrap();
        insert_scan(&tx, &[1; 32], other, &[at(0.0, 80.0, 0.9)]).unwrap();
        tx.commit().unwrap();
        set_not_a_face(&conn, ids[4]).unwrap();
        let ada = people::create(&conn, model).unwrap();
        crate::catalog::edit(&mut conn, |tx| people::rename(tx, ada, Some("Ada")));
        assign(&conn, ids[1], ada, AssignedBy::Manual).unwrap();
        assign(&conn, ids[6], ada, AssignedBy::Manual).unwrap();

        let got = of_image(&conn, &[1; 32], model, 0.6, 40).unwrap();
        assert_eq!(
            got.iter().map(|f| f.id).collect::<Vec<_>>(),
            [ids[6], ids[5], ids[1], ids[0]]
        );
        assert_eq!(
            counts(&conn, model, 0.6, 40).unwrap()[&[1; 32]] as usize,
            got.len()
        );
        assert_eq!(
            got[2].rect,
            Rect {
                x: 10.0,
                y: 5.0,
                w: 40.0,
                h: 80.0,
            }
        );
        let ada_row = people::list(&conn, model).unwrap().remove(0);
        assert_eq!(got[2].person.as_ref(), Some(&ada_row));
        assert_eq!(got[3].person, None);

        crate::catalog::edit(&mut conn, |tx| people::set_ignored(tx, ada, true));
        let got = of_image(&conn, &[1; 32], model, 0.6, 40).unwrap();
        assert_eq!(
            got.iter().map(|f| f.id).collect::<Vec<_>>(),
            [ids[5], ids[0]]
        );
    }

    #[test]
    fn clustering_watermark_follows_new_faces_per_model() {
        let (_dir, mut conn) = temp();
        let a = register_model(&conn, "a", "d", 2).unwrap();
        let b = register_model(&conn, "b", "d", 2).unwrap();
        assert!(!unclustered(&conn, a).unwrap());
        scan(&mut conn, [1; 32], a, 2);
        assert!(unclustered(&conn, a).unwrap());
        mark_clustered(&conn, a).unwrap();
        assert!(!unclustered(&conn, a).unwrap());
        scan(&mut conn, [1; 32], b, 1);
        assert!(!unclustered(&conn, a).unwrap());
        assert!(unclustered(&conn, b).unwrap());
        scan(&mut conn, [2; 32], a, 1);
        assert!(unclustered(&conn, a).unwrap());
    }
}
