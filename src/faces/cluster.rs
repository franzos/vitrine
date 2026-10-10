//! Immich-style face clustering over an in-memory matrix of one model's faces.
//!
//! - Neighbours: up to [`TOP_K`] nearest faces within `cluster_threshold`, nearest first, ties to the lower face id.
//! - Faces with `min_neighbours` go first, by id: join the plurality person of assigned neighbours (ties to nearest), else found a person their unassigned neighbours may join.
//! - Other faces only join a person, else stay unassigned. A plurality on an ignored person leaves the face unassigned.
//! - Not-a-face and below-count-filter faces take no part; manual rows are never written.
//! - Merge: closest mean pair within `merge_threshold` first; ignored never, two named never. Survivor: named, else more faces, else lower id.

use std::collections::{HashMap, HashSet};

use ndarray::{Array2, ArrayView1, Axis};
use rayon::prelude::*;
use rusqlite::Connection;

use crate::catalog::faces::{self, AssignedBy, ClusterFace, ModelId};
use crate::catalog::people::{self, PersonId};
use crate::config;

/// Nearest neighbours considered per face, unless `faces.min_neighbours` asks for more.
pub const TOP_K: usize = 10;
/// Faces per similarity block; each search thread holds a `BLOCK` x faces buffer.
const BLOCK: usize = 64;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("embeddings of face models {0} and {1} can't be compared")]
    ModelMismatch(i64, i64),
    #[error("embeddings of {0} and {1} dimensions can't be compared")]
    Dim(usize, usize),
    #[error(transparent)]
    Db(#[from] crate::db::Error),
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub struct Embedding<'a> {
    pub model: ModelId,
    pub values: &'a [f32],
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Outcome {
    pub people_created: u64,
    pub faces_assigned: u64,
    pub people_merged: u64,
}

/// Cosine similarity of two embeddings; 0 when either is a zero vector.
#[cfg(test)]
pub fn similarity(a: Embedding<'_>, b: Embedding<'_>) -> Result<f32, Error> {
    if a.model != b.model {
        return Err(Error::ModelMismatch(a.model.0, b.model.0));
    }
    if a.values.len() != b.values.len() {
        return Err(Error::Dim(a.values.len(), b.values.len()));
    }
    let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(x, y)| x * y).sum::<f32>();
    let norms = (dot(a.values, a.values) * dot(b.values, b.values)).sqrt();
    Ok(if norms > 0.0 {
        dot(a.values, b.values) / norms
    } else {
        0.0
    })
}

/// No-op unless faces were stored since the last pass.
pub fn incremental(
    conn: &mut Connection,
    model: ModelId,
    config: &config::Faces,
) -> Result<Outcome, Error> {
    if !faces::unclustered(conn, model)? {
        return Ok(Outcome::default());
    }
    run(conn, model, config, false)
}

/// Names, ignored people and manual rows stay; unnamed people left without faces are deleted.
pub fn recluster(
    conn: &mut Connection,
    model: ModelId,
    config: &config::Faces,
) -> Result<Outcome, Error> {
    run(conn, model, config, true)
}

/// Plans in memory, then writes in one short transaction; `faces::assign` skips faces a manual edit claimed meanwhile.
fn run(
    conn: &mut Connection,
    model: ModelId,
    config: &config::Faces,
    reset: bool,
) -> Result<Outcome, Error> {
    let (faces, plan) = snapshot_plan(conn, model, config, reset)?;
    let mut outcome = write(conn, model, &faces, plan, reset)?;
    outcome.people_merged = merge_close(conn, model, config)?;
    Ok(outcome)
}

struct Group {
    id: PersonId,
    named: bool,
    faces: usize,
    sum: Vec<f32>,
}

/// Returns how many people were absorbed.
fn merge_close(
    conn: &mut Connection,
    model: ModelId,
    config: &config::Faces,
) -> Result<u64, Error> {
    let mut faces = faces::for_clustering(conn, model, config.min_score, config.min_face_size)?;
    faces.retain(|f| f.person_id.is_some() && !f.ignored);
    if faces.is_empty() {
        return Ok(0);
    }
    let named: HashSet<PersonId> = people::list(conn, model)?
        .into_iter()
        .filter(|p| p.name.is_some())
        .map(|p| p.id)
        .collect();
    let matrix = matrix(&mut faces)?;
    let mut groups: Vec<Group> = Vec::new();
    let mut index: HashMap<PersonId, usize> = HashMap::new();
    for (row, face) in faces.iter().enumerate() {
        let person = face.person_id.expect("unassigned faces were dropped");
        let g = *index.entry(person).or_insert_with(|| {
            groups.push(Group {
                id: person,
                named: named.contains(&person),
                faces: 0,
                sum: vec![0.0; matrix.ncols()],
            });
            groups.len() - 1
        });
        groups[g].faces += 1;
        for (s, x) in groups[g].sum.iter_mut().zip(matrix.row(row)) {
            *s += x;
        }
    }
    let merges = plan_merges(&mut groups, 1.0 - config.merge_threshold);
    if merges.is_empty() {
        return Ok(0);
    }
    let tx = conn.transaction()?;
    let mut merged = 0;
    for (from, into) in merges {
        if people::absorb(&tx, from, into)? {
            merged += 1;
        }
    }
    tx.commit()?;
    Ok(merged)
}

/// `(absorbed, survivor)` in apply order.
fn plan_merges(groups: &mut [Group], min_similarity: f32) -> Vec<(PersonId, PersonId)> {
    let n = groups.len();
    let centroid = |g: &Group| -> Vec<f32> {
        let norm = g.sum.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scale = if norm > 0.0 { norm.recip() } else { 0.0 };
        g.sum.iter().map(|x| x * scale).collect()
    };
    let mut centroids: Vec<Vec<f32>> = groups.iter().map(centroid).collect();
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let dim = groups.first().map_or(0, |g| g.sum.len());
    let rows =
        Array2::from_shape_vec((n, dim), centroids.concat()).expect("one centroid row per group");
    let mut similarity = rows.dot(&rows.t()).into_raw_vec_and_offset().0;
    let mut alive = vec![true; n];
    let mut merges = Vec::new();
    loop {
        let mut best: Option<(f32, usize, usize)> = None;
        for i in 0..n {
            if !alive[i] {
                continue;
            }
            for j in i + 1..n {
                if !alive[j] || (groups[i].named && groups[j].named) {
                    continue;
                }
                let s = similarity[i * n + j];
                if s >= min_similarity && best.is_none_or(|(b, ..)| s > b) {
                    best = Some((s, i, j));
                }
            }
        }
        let Some((_, i, j)) = best else {
            break;
        };
        let (survivor, absorbed) = {
            let (a, b) = (&groups[i], &groups[j]);
            if a.named != b.named {
                if a.named { (i, j) } else { (j, i) }
            } else if a.faces != b.faces {
                if a.faces > b.faces { (i, j) } else { (j, i) }
            } else if a.id < b.id {
                (i, j)
            } else {
                (j, i)
            }
        };
        let sum = std::mem::take(&mut groups[absorbed].sum);
        let (faces, named) = (groups[absorbed].faces, groups[absorbed].named);
        let s = &mut groups[survivor];
        for (acc, x) in s.sum.iter_mut().zip(&sum) {
            *acc += x;
        }
        s.faces += faces;
        s.named |= named;
        alive[absorbed] = false;
        centroids[survivor] = centroid(&groups[survivor]);
        for k in (0..n).filter(|&k| k != survivor && alive[k]) {
            let s = dot(&centroids[survivor], &centroids[k]);
            similarity[survivor * n + k] = s;
            similarity[k * n + survivor] = s;
        }
        merges.push((groups[absorbed].id, groups[survivor].id));
    }
    merges
}

fn snapshot_plan(
    conn: &Connection,
    model: ModelId,
    config: &config::Faces,
    reset: bool,
) -> Result<(Vec<ClusterFace>, Plan), Error> {
    let mut faces = faces::for_clustering(conn, model, config.min_score, config.min_face_size)?;
    if reset {
        for face in faces
            .iter_mut()
            .filter(|f| f.assigned_by == Some(AssignedBy::Auto))
        {
            face.person_id = None;
            face.assigned_by = None;
            face.ignored = false;
        }
    }
    let plan = plan(&mut faces, config)?;
    Ok((faces, plan))
}

fn write(
    conn: &mut Connection,
    model: ModelId,
    faces: &[ClusterFace],
    plan: Plan,
    reset: bool,
) -> Result<Outcome, Error> {
    let tx = conn.transaction()?;
    if reset {
        faces::reset_auto(&tx, model)?;
    }
    let created = (0..plan.new_people)
        .map(|_| people::create(&tx, model))
        .collect::<crate::db::Result<Vec<_>>>()?;
    let mut filled = vec![false; created.len()];
    let mut outcome = Outcome::default();
    for (row, owner) in plan.assignments {
        let person = match owner {
            Owner::Existing(person) => person,
            Owner::New(n) => created[n],
        };
        if faces::assign(&tx, faces[row].id, person, AssignedBy::Auto)? {
            outcome.faces_assigned += 1;
            if let Owner::New(n) = owner {
                filled[n] = true;
            }
        }
    }
    // a manual edit since the snapshot can leave a new person without faces
    for (person, filled) in created.into_iter().zip(filled) {
        if filled {
            outcome.people_created += 1;
        } else {
            people::delete(&tx, person)?;
        }
    }
    if reset {
        people::delete_empty_unnamed(&tx, model)?;
    }
    faces::mark_clustered(&tx, model)?;
    tx.commit()?;
    Ok(outcome)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Owner {
    Existing(PersonId),
    /// The n-th person this pass creates.
    New(usize),
}

struct Plan {
    new_people: usize,
    /// Row in the face list and its person, in decision order.
    assignments: Vec<(usize, Owner)>,
}

#[derive(Debug, Clone, Copy)]
struct Neighbour {
    row: usize,
    similarity: f32,
}

/// Takes the embeddings out of `faces`.
fn plan(faces: &mut [ClusterFace], config: &config::Faces) -> Result<Plan, Error> {
    let candidates: Vec<usize> = (0..faces.len())
        .filter(|&i| faces[i].assigned_by.is_none())
        .collect();
    let mut plan = Plan {
        new_people: 0,
        assignments: Vec::new(),
    };
    if candidates.is_empty() {
        return Ok(plan);
    }
    let matrix = matrix(faces)?;
    let k = TOP_K.max(config.min_neighbours);
    let neighbours = neighbours(&matrix, &candidates, 1.0 - config.cluster_threshold, k);

    let mut owner: Vec<Option<Owner>> = faces
        .iter()
        .map(|f| f.person_id.map(Owner::Existing))
        .collect();
    let ignored: HashSet<Owner> = faces
        .iter()
        .filter(|f| f.ignored)
        .filter_map(|f| f.person_id.map(Owner::Existing))
        .collect();
    let mut candidate_of = vec![None; faces.len()];
    for (c, &row) in candidates.iter().enumerate() {
        candidate_of[row] = Some(c);
    }

    let (core, rest): (Vec<usize>, Vec<usize>) =
        (0..candidates.len()).partition(|&c| neighbours[c].len() >= config.min_neighbours);
    for c in core {
        let row = candidates[c];
        if owner[row].is_some() {
            continue;
        }
        match vote(&neighbours[c], &owner) {
            Some(winner) if ignored.contains(&winner) => {}
            Some(winner) => {
                owner[row] = Some(winner);
                plan.assignments.push((row, winner));
            }
            None => {
                let new = Owner::New(plan.new_people);
                plan.new_people += 1;
                owner[row] = Some(new);
                plan.assignments.push((row, new));
                for n in &neighbours[c] {
                    let Some(g) = candidate_of[n.row] else {
                        continue;
                    };
                    if owner[n.row].is_none()
                        && vote(&neighbours[g], &owner).is_none_or(|w| w == new)
                    {
                        owner[n.row] = Some(new);
                        plan.assignments.push((n.row, new));
                    }
                }
            }
        }
    }
    for c in rest {
        let row = candidates[c];
        if owner[row].is_some() {
            continue;
        }
        if let Some(winner) = vote(&neighbours[c], &owner)
            && !ignored.contains(&winner)
        {
            owner[row] = Some(winner);
            plan.assignments.push((row, winner));
        }
    }
    Ok(plan)
}

/// The plurality owner of `neighbours`; ties to the one seen first, i.e. the nearest.
fn vote(neighbours: &[Neighbour], owner: &[Option<Owner>]) -> Option<Owner> {
    let mut tally: HashMap<Owner, (usize, usize)> = HashMap::new();
    for (position, n) in neighbours.iter().enumerate() {
        if let Some(o) = owner[n.row] {
            tally.entry(o).or_insert((0, position)).0 += 1;
        }
    }
    tally
        .into_iter()
        .max_by(|(_, (count_a, first_a)), (_, (count_b, first_b))| {
            count_a.cmp(count_b).then(first_b.cmp(first_a))
        })
        .map(|(o, _)| o)
}

/// L2-normalised rows.
fn matrix(faces: &mut [ClusterFace]) -> Result<Array2<f32>, Error> {
    let dim = faces[0].embedding.len();
    let mut data = Vec::with_capacity(faces.len() * dim);
    for face in faces.iter_mut() {
        let embedding = std::mem::take(&mut face.embedding);
        if embedding.len() != dim {
            return Err(Error::Dim(dim, embedding.len()));
        }
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scale = if norm > 0.0 { norm.recip() } else { 0.0 };
        data.extend(embedding.iter().map(|x| x * scale));
    }
    Ok(Array2::from_shape_vec((faces.len(), dim), data).expect("data holds faces x dim values"))
}

/// Blocks depend only on the query list and each runs on one thread, so the result is independent of the thread count.
fn neighbours(
    matrix: &Array2<f32>,
    queries: &[usize],
    min_similarity: f32,
    k: usize,
) -> Vec<Vec<Neighbour>> {
    let blocks: Vec<Vec<Vec<Neighbour>>> = queries
        .par_chunks(BLOCK)
        .map(|block| {
            let similarities = matrix.select(Axis(0), block).dot(&matrix.t());
            block
                .iter()
                .zip(similarities.outer_iter())
                .map(|(&own, row)| nearest(own, &row, min_similarity, k))
                .collect()
        })
        .collect();
    blocks.into_iter().flatten().collect()
}

/// Up to `k` rows other than `own` with at least `min_similarity`, most similar first, ties to the lower row.
fn nearest(
    own: usize,
    similarities: &ArrayView1<f32>,
    min_similarity: f32,
    k: usize,
) -> Vec<Neighbour> {
    let mut top: Vec<Neighbour> = Vec::with_capacity(k + 1);
    for (row, &similarity) in similarities.iter().enumerate() {
        if row == own || similarity.is_nan() || similarity < min_similarity {
            continue;
        }
        if top.len() == k && top.last().is_some_and(|last| similarity <= last.similarity) {
            continue;
        }
        let at = top.partition_point(|n| n.similarity >= similarity);
        top.insert(at, Neighbour { row, similarity });
        top.truncate(k);
    }
    top
}

#[cfg(test)]
mod tests {
    use rusqlite::params;

    use super::*;
    use crate::catalog::faces::{FaceId, NewFace, Rect, insert_scan, register_model};
    use crate::catalog::temp;

    const DIM: usize = 16;

    fn config() -> config::Faces {
        config::Faces {
            enabled: true,
            ..config::Faces::default()
        }
    }

    /// Face `i` of synthetic person `p`: axis `p` plus 0.2 on noise axis `8 + i % 8`.
    /// Same person: cosine distance at most 0.04. Different people: beyond 0.96.
    fn person(p: usize, i: usize) -> Vec<f32> {
        let mut v = vec![0.0; DIM];
        v[p] = 1.0;
        v[8 + i % 8] += 0.2;
        v
    }

    /// One face per content, big and confident enough to count.
    fn add(conn: &mut Connection, model: ModelId, embeddings: &[Vec<f32>]) -> Vec<FaceId> {
        add_sized(conn, model, embeddings, 100.0)
    }

    fn add_sized(
        conn: &mut Connection,
        model: ModelId,
        embeddings: &[Vec<f32>],
        side: f32,
    ) -> Vec<FaceId> {
        let next: usize = conn
            .query_row("SELECT count(*) FROM face_scans", [], |r| r.get(0))
            .unwrap();
        let tx = conn.transaction().unwrap();
        let mut ids = Vec::new();
        for (n, embedding) in embeddings.iter().enumerate() {
            let mut sha = [0; 32];
            let key = (next + n).to_le_bytes();
            sha[..key.len()].copy_from_slice(&key);
            let face = NewFace {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: side,
                    h: side,
                },
                landmarks: [[0.0; 2]; 5],
                score: 0.9,
                embedding,
                crop: b"jpeg",
            };
            ids.extend(insert_scan(&tx, &sha, model, &[face]).unwrap());
        }
        tx.commit().unwrap();
        ids
    }

    fn model(conn: &Connection, name: &str) -> ModelId {
        register_model(conn, name, "d", DIM).unwrap()
    }

    fn row(conn: &Connection, face: FaceId) -> (Option<PersonId>, Option<AssignedBy>) {
        conn.query_row(
            "SELECT person_id, assigned_by FROM faces WHERE id = ?1",
            [face.0],
            |r| Ok((r.get::<_, Option<i64>>(0)?.map(PersonId), r.get(1)?)),
        )
        .unwrap()
    }

    fn person_of(conn: &Connection, face: FaceId) -> Option<PersonId> {
        row(conn, face).0
    }

    fn people_of(conn: &Connection, model: ModelId) -> Vec<PersonId> {
        people::list(conn, model)
            .unwrap()
            .iter()
            .map(|p| p.id)
            .collect()
    }

    fn snapshot(conn: &Connection, model: ModelId) -> Vec<(i64, Option<i64>, Option<String>)> {
        conn.prepare("SELECT id, person_id, assigned_by FROM faces WHERE model_id = ?1 ORDER BY id")
            .unwrap()
            .query_map([model.0], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
    }

    #[test]
    fn similarity_is_cosine_and_rejects_other_models_and_lengths() {
        let (_dir, conn) = temp();
        let a = model(&conn, "a");
        let b = model(&conn, "b");
        let x = Embedding {
            model: a,
            values: &[3.0, 4.0],
        };
        let y = Embedding {
            model: a,
            values: &[4.0, 3.0],
        };
        assert!((similarity(x, y).unwrap() - 0.96).abs() < 1e-6);
        assert!(matches!(
            similarity(x, Embedding { model: b, ..y }),
            Err(Error::ModelMismatch(..))
        ));
        assert!(matches!(
            similarity(
                x,
                Embedding {
                    model: a,
                    values: &[1.0, 0.0, 0.0]
                }
            ),
            Err(Error::Dim(2, 3))
        ));
        let zero = Embedding {
            model: a,
            values: &[0.0, 0.0],
        };
        assert!(similarity(x, zero).unwrap().abs() < f32::EPSILON);
    }

    #[test]
    fn nearest_keeps_k_within_threshold_with_ties_to_the_lower_row() {
        let similarities = ndarray::arr1(&[1.0, 0.5, 0.9, 0.5, 0.3, f32::NAN, 0.7]);
        let rows = |k| -> Vec<usize> {
            nearest(0, &similarities.view(), 0.4, k)
                .iter()
                .map(|n| n.row)
                .collect()
        };
        assert_eq!(rows(10), [2, 6, 1, 3]);
        assert_eq!(rows(3), [2, 6, 1]);
    }

    #[test]
    fn a_fresh_set_of_one_person_forms_one_person() {
        let (_dir, mut conn) = temp();
        let m = model(&conn, "m");
        let ids = add(
            &mut conn,
            m,
            &(0..6).map(|i| person(0, i)).collect::<Vec<_>>(),
        );
        let outcome = incremental(&mut conn, m, &config()).unwrap();
        assert_eq!(
            outcome,
            Outcome {
                people_created: 1,
                faces_assigned: 6,
                people_merged: 0
            }
        );
        let p = person_of(&conn, ids[0]).unwrap();
        for id in &ids {
            assert_eq!(row(&conn, *id), (Some(p), Some(AssignedBy::Auto)));
        }
    }

    #[test]
    fn a_new_person_needs_enough_neighbours_and_others_wait_for_a_later_pass() {
        let (_dir, mut conn) = temp();
        let m = model(&conn, "m");
        let a = add(&mut conn, m, &[person(0, 0), person(0, 1), person(0, 2)]);
        let b = add(&mut conn, m, &[person(1, 0), person(1, 1)]);
        let c = add(&mut conn, m, &[person(2, 0)]);
        let outcome = incremental(&mut conn, m, &config()).unwrap();
        assert_eq!((outcome.people_created, outcome.faces_assigned), (1, 3));
        assert!(person_of(&conn, a[0]).is_some());
        assert!(
            a.iter()
                .all(|f| person_of(&conn, *f) == person_of(&conn, a[0]))
        );
        for f in b.iter().chain(&c) {
            assert_eq!(row(&conn, *f), (None, None));
        }

        let more = add(&mut conn, m, &[person(1, 2)]);
        let outcome = incremental(&mut conn, m, &config()).unwrap();
        assert_eq!((outcome.people_created, outcome.faces_assigned), (1, 3));
        let pb = person_of(&conn, b[0]).unwrap();
        assert_ne!(Some(pb), person_of(&conn, a[0]));
        assert_eq!(person_of(&conn, b[1]), Some(pb));
        assert_eq!(person_of(&conn, more[0]), Some(pb));
        assert_eq!(row(&conn, c[0]), (None, None));
    }

    #[test]
    fn faces_join_the_plurality_of_assigned_neighbours() {
        let (_dir, mut conn) = temp();
        let m = model(&conn, "m");
        let named = people::create(&conn, m).unwrap();
        crate::catalog::edit(&mut conn, |tx| people::rename(tx, named, Some("Ada")));
        let known = add(&mut conn, m, &[person(0, 0), person(0, 1), person(0, 2)]);
        for f in &known {
            faces::assign(&conn, *f, named, AssignedBy::Manual).unwrap();
        }
        // one face of another person, nearer to the new face than any of Ada's
        let other = people::create(&conn, m).unwrap();
        let mut near = person(0, 3);
        near[8 + 3] = 0.05;
        let near = add(&mut conn, m, &[near]);
        faces::assign(&conn, near[0], other, AssignedBy::Manual).unwrap();

        let new = add(&mut conn, m, &[person(0, 3), person(0, 4)]);
        let outcome = incremental(&mut conn, m, &config()).unwrap();
        assert_eq!((outcome.people_created, outcome.faces_assigned), (0, 2));
        for f in &new {
            assert_eq!(row(&conn, *f), (Some(named), Some(AssignedBy::Auto)));
        }
    }

    #[test]
    fn plurality_ties_go_to_the_nearest_and_few_neighbours_still_join() {
        let (_dir, mut conn) = temp();
        let m = model(&conn, "m");
        let far = people::create(&conn, m).unwrap();
        let close = people::create(&conn, m).unwrap();
        let mut a = person(0, 0);
        a[8] = 0.5;
        let b = person(0, 1);
        let known = add(&mut conn, m, &[a, b]);
        faces::assign(&conn, known[0], far, AssignedBy::Manual).unwrap();
        faces::assign(&conn, known[1], close, AssignedBy::Manual).unwrap();

        let new = add(&mut conn, m, &[person(0, 1)]);
        incremental(&mut conn, m, &config()).unwrap();
        assert_eq!(person_of(&conn, new[0]), Some(close));

        // a single neighbour is below min_neighbours, and still enough to join
        let lone = people::create(&conn, m).unwrap();
        let single = add(&mut conn, m, &[person(3, 0)]);
        faces::assign(&conn, single[0], lone, AssignedBy::Manual).unwrap();
        let joiner = add(&mut conn, m, &[person(3, 1)]);
        incremental(&mut conn, m, &config()).unwrap();
        assert_eq!(person_of(&conn, joiner[0]), Some(lone));
    }

    #[test]
    fn manual_rows_are_never_moved() {
        let (_dir, mut conn) = temp();
        let m = model(&conn, "m");
        let ada = people::create(&conn, m).unwrap();
        let misfiled = add(&mut conn, m, &[person(1, 0)]);
        faces::assign(&conn, misfiled[0], ada, AssignedBy::Manual).unwrap();
        let removed = add(&mut conn, m, &[person(1, 1)]);
        faces::unassign(&conn, removed[0]).unwrap();
        let others = add(&mut conn, m, &[person(1, 2), person(1, 3), person(1, 4)]);

        incremental(&mut conn, m, &config()).unwrap();
        let check = |conn: &Connection| {
            assert_eq!(
                row(conn, misfiled[0]),
                (Some(ada), Some(AssignedBy::Manual))
            );
            assert_eq!(row(conn, removed[0]), (None, Some(AssignedBy::Manual)));
            // the misfiled face makes Ada the others' plurality
            for f in &others {
                assert_eq!(row(conn, *f), (Some(ada), Some(AssignedBy::Auto)));
            }
        };
        check(&conn);
        recluster(&mut conn, m, &config()).unwrap();
        check(&conn);
    }

    #[test]
    fn a_manually_removed_face_is_never_assigned_again() {
        let (_dir, mut conn) = temp();
        let m = model(&conn, "m");
        let ids = add(
            &mut conn,
            m,
            &(0..4).map(|i| person(0, i)).collect::<Vec<_>>(),
        );
        faces::unassign(&conn, ids[0]).unwrap();
        incremental(&mut conn, m, &config()).unwrap();
        let p = person_of(&conn, ids[1]).unwrap();
        assert_eq!(person_of(&conn, ids[3]), Some(p));
        assert_eq!(row(&conn, ids[0]), (None, Some(AssignedBy::Manual)));
        recluster(&mut conn, m, &config()).unwrap();
        assert_eq!(row(&conn, ids[0]), (None, Some(AssignedBy::Manual)));
    }

    #[test]
    fn ignored_people_take_no_faces_and_block_new_people() {
        let (_dir, mut conn) = temp();
        let m = model(&conn, "m");
        let ignored = people::create(&conn, m).unwrap();
        crate::catalog::edit(&mut conn, |tx| people::set_ignored(tx, ignored, true));
        let known = add(&mut conn, m, &[person(0, 0), person(0, 1)]);
        for f in &known {
            faces::assign(&conn, *f, ignored, AssignedBy::Manual).unwrap();
        }
        let new = add(&mut conn, m, &[person(0, 2), person(0, 3), person(0, 4)]);
        let outcome = incremental(&mut conn, m, &config()).unwrap();
        assert_eq!(outcome, Outcome::default());
        for f in &new {
            assert_eq!(row(&conn, *f), (None, None));
        }
        assert_eq!(people_of(&conn, m), [ignored]);
        recluster(&mut conn, m, &config()).unwrap();
        assert_eq!(people_of(&conn, m), [ignored]);
        assert!(new.iter().all(|f| person_of(&conn, *f).is_none()));
    }

    #[test]
    fn not_a_face_and_filtered_faces_take_no_part() {
        let (_dir, mut conn) = temp();
        let m = model(&conn, "m");
        let ids = add(&mut conn, m, &[person(0, 0), person(0, 1), person(0, 2)]);
        faces::set_not_a_face(&conn, ids[2]).unwrap();
        let small = add_sized(&mut conn, m, &[person(0, 3)], 20.0);
        // without them, each remaining face has one neighbour, fewer than min_neighbours
        let outcome = incremental(&mut conn, m, &config()).unwrap();
        assert_eq!(outcome, Outcome::default());
        assert_eq!(row(&conn, ids[2]), (None, Some(AssignedBy::Manual)));
        assert_eq!(row(&conn, small[0]), (None, None));

        let p = people::create(&conn, m).unwrap();
        faces::assign(&conn, ids[0], p, AssignedBy::Manual).unwrap();
        recluster(&mut conn, m, &config()).unwrap();
        assert_eq!(person_of(&conn, ids[1]), Some(p));
        assert_eq!(row(&conn, ids[2]), (None, Some(AssignedBy::Manual)));
        assert_eq!(row(&conn, small[0]), (None, None));
    }

    #[test]
    fn recluster_keeps_names_and_manual_rows_and_drops_emptied_people() {
        let (_dir, mut conn) = temp();
        let m = model(&conn, "m");
        let ada = people::create(&conn, m).unwrap();
        crate::catalog::edit(&mut conn, |tx| people::rename(tx, ada, Some("Ada")));
        let unnamed = people::create(&conn, m).unwrap();
        let ada_manual = add(&mut conn, m, &[person(0, 0), person(0, 1)]);
        let ada_auto = add(&mut conn, m, &[person(0, 2), person(0, 3)]);
        let wrong = add(&mut conn, m, &[person(1, 0)]);
        let bob = add(&mut conn, m, &[person(1, 1), person(1, 2)]);
        for f in &ada_manual {
            faces::assign(&conn, *f, ada, AssignedBy::Manual).unwrap();
        }
        for f in ada_auto.iter().chain(&wrong) {
            faces::assign(&conn, *f, ada, AssignedBy::Auto).unwrap();
        }
        for f in &bob {
            faces::assign(&conn, *f, unnamed, AssignedBy::Auto).unwrap();
        }

        let outcome = recluster(&mut conn, m, &config()).unwrap();
        assert_eq!(
            outcome,
            Outcome {
                people_created: 1,
                faces_assigned: 5,
                people_merged: 0
            }
        );
        let list = people::list(&conn, m).unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!((list[0].id, list[0].name.as_deref()), (ada, Some("Ada")));
        assert!(!people_of(&conn, m).contains(&unnamed));
        for f in &ada_manual {
            assert_eq!(row(&conn, *f), (Some(ada), Some(AssignedBy::Manual)));
        }
        for f in &ada_auto {
            assert_eq!(row(&conn, *f), (Some(ada), Some(AssignedBy::Auto)));
        }
        let new = list[1].id;
        for f in wrong.iter().chain(&bob) {
            assert_eq!(row(&conn, *f), (Some(new), Some(AssignedBy::Auto)));
        }
    }

    /// Faces pinned by hand to `person`, so the pass can't move them and only the merge step can.
    fn pin(
        conn: &mut Connection,
        m: ModelId,
        person: PersonId,
        embeddings: &[Vec<f32>],
    ) -> Vec<FaceId> {
        let ids = add(conn, m, embeddings);
        for f in &ids {
            faces::assign(conn, *f, person, AssignedBy::Manual).unwrap();
        }
        ids
    }

    #[test]
    fn close_people_merge_into_the_larger_and_faces_keep_their_assignment() {
        let (_dir, mut conn) = temp();
        let m = model(&conn, "m");
        let small = people::create(&conn, m).unwrap();
        let big = people::create(&conn, m).unwrap();
        let other = people::create(&conn, m).unwrap();
        let small_faces = pin(&mut conn, m, small, &[person(0, 0), person(0, 1)]);
        let big_faces = pin(
            &mut conn,
            m,
            big,
            &[person(0, 2), person(0, 3), person(0, 4)],
        );
        let other_faces = pin(&mut conn, m, other, &[person(1, 0), person(1, 1)]);

        let outcome = recluster(&mut conn, m, &config()).unwrap();
        assert_eq!(outcome.people_merged, 1);
        assert_eq!(people_of(&conn, m), [big, other]);
        for f in small_faces.iter().chain(&big_faces) {
            assert_eq!(row(&conn, *f), (Some(big), Some(AssignedBy::Manual)));
        }
        for f in &other_faces {
            assert_eq!(row(&conn, *f), (Some(other), Some(AssignedBy::Manual)));
        }
        assert_eq!(recluster(&mut conn, m, &config()).unwrap().people_merged, 0);
    }

    #[test]
    fn a_named_person_absorbs_an_unnamed_one_and_two_named_never_merge() {
        let (_dir, mut conn) = temp();
        let m = model(&conn, "m");
        let ada = people::create(&conn, m).unwrap();
        let unnamed = people::create(&conn, m).unwrap();
        let bob = people::create(&conn, m).unwrap();
        let bob_twin = people::create(&conn, m).unwrap();
        for (p, name) in [(ada, "Ada"), (bob, "Bob"), (bob_twin, "Bob again")] {
            crate::catalog::edit(&mut conn, |tx| people::rename(tx, p, Some(name)));
        }
        let ada_faces = pin(&mut conn, m, ada, &[person(0, 0)]);
        let unnamed_faces = pin(
            &mut conn,
            m,
            unnamed,
            &[person(0, 1), person(0, 2), person(0, 3)],
        );
        let bob_faces = pin(&mut conn, m, bob, &[person(1, 0)]);
        let twin_faces = pin(&mut conn, m, bob_twin, &[person(1, 1)]);

        let outcome = recluster(&mut conn, m, &config()).unwrap();
        assert_eq!(outcome.people_merged, 1);
        assert_eq!(people_of(&conn, m), [ada, bob, bob_twin]);
        for f in ada_faces.iter().chain(&unnamed_faces) {
            assert_eq!(person_of(&conn, *f), Some(ada));
        }
        for f in &bob_faces {
            assert_eq!(person_of(&conn, *f), Some(bob));
        }
        for f in &twin_faces {
            assert_eq!(person_of(&conn, *f), Some(bob_twin));
        }
    }

    #[test]
    fn ignored_people_and_distant_people_are_not_merged() {
        let (_dir, mut conn) = temp();
        let m = model(&conn, "m");
        let ignored = people::create(&conn, m).unwrap();
        let near = people::create(&conn, m).unwrap();
        let far = people::create(&conn, m).unwrap();
        crate::catalog::edit(&mut conn, |tx| people::set_ignored(tx, ignored, true));
        pin(&mut conn, m, ignored, &[person(0, 0), person(0, 1)]);
        pin(&mut conn, m, near, &[person(0, 2), person(0, 3)]);
        pin(&mut conn, m, far, &[person(1, 0), person(1, 1)]);

        let outcome = recluster(&mut conn, m, &config()).unwrap();
        assert_eq!(outcome.people_merged, 0);
        assert_eq!(people_of(&conn, m).len(), 3);
    }

    #[test]
    fn merges_go_closest_first_with_the_mean_recomputed() {
        let group = |id: i64, named: bool, sum: Vec<f32>, faces: usize| Group {
            id: PersonId(id),
            named,
            faces,
            sum,
        };
        // a 0 deg, b 24, c 49: a-b merge first; b-c alone is within 0.9, the a+b mean at 12 deg isn't
        let mut groups = vec![
            group(1, false, vec![1.0, 0.0], 1),
            group(2, false, vec![0.9135, 0.4067], 2),
            group(3, false, vec![0.6561, 0.7547], 2),
        ];
        assert_eq!(plan_merges(&mut groups, 0.9), [(PersonId(1), PersonId(2))]);

        let mut named = vec![
            group(1, true, vec![1.0, 0.0], 1),
            group(2, true, vec![1.0, 0.1], 5),
            group(3, false, vec![1.0, 0.2], 9),
        ];
        assert_eq!(plan_merges(&mut named, 0.9), [(PersonId(3), PersonId(2))]);
    }

    #[test]
    fn recluster_keeps_people_named_or_ignored_after_clustering_and_manual_fixes() {
        let (_dir, mut conn) = temp();
        let m = model(&conn, "m");
        let of = |p| (0..4).map(|i| person(p, i)).collect::<Vec<_>>();
        let ada_faces = add(&mut conn, m, &of(0));
        let ignored_faces = add(&mut conn, m, &of(1));
        let other_faces = add(&mut conn, m, &of(2));
        incremental(&mut conn, m, &config()).unwrap();
        let ada = person_of(&conn, ada_faces[0]).unwrap();
        let ignored = person_of(&conn, ignored_faces[0]).unwrap();
        let other = person_of(&conn, other_faces[0]).unwrap();

        crate::catalog::edit(&mut conn, |tx| people::rename(tx, ada, Some("Ada")));
        crate::catalog::edit(&mut conn, |tx| people::set_ignored(tx, ignored, true));
        faces::unassign(&conn, ada_faces[3]).unwrap();
        faces::set_not_a_face(&conn, other_faces[3]).unwrap();
        let kept = |conn: &Connection| {
            let rows: Vec<_> = ada_faces
                .iter()
                .chain(&ignored_faces)
                .chain(&other_faces[3..])
                .map(|f| row(conn, *f))
                .collect();
            let people: Vec<_> = people::list(conn, m)
                .unwrap()
                .into_iter()
                .filter(|p| p.id == ada || p.id == ignored)
                .map(|p| (p.id, p.name, p.ignored, p.faces))
                .collect();
            (rows, people)
        };
        let before = kept(&conn);
        assert_eq!(
            before.1,
            [
                (ada, Some("Ada".to_owned()), false, 3),
                (ignored, None, true, 4)
            ]
        );

        recluster(&mut conn, m, &config()).unwrap();
        assert_eq!(kept(&conn), before);
        assert_eq!(row(&conn, ada_faces[3]), (None, Some(AssignedBy::Manual)));
        assert_eq!(row(&conn, other_faces[3]), (None, Some(AssignedBy::Manual)));
        // the unnamed person's faces were automatic, so they form a new person
        let list = people::list(&conn, m).unwrap();
        assert_eq!(list.len(), 3);
        assert!(!list.iter().any(|p| p.id == other));
        let regrouped = person_of(&conn, other_faces[0]).unwrap();
        assert!(
            other_faces[..3]
                .iter()
                .all(|f| person_of(&conn, *f) == Some(regrouped))
        );
    }

    #[test]
    fn a_new_person_left_empty_by_a_manual_edit_is_removed() {
        let (_dir, mut conn) = temp();
        let m = model(&conn, "m");
        let ids = add(
            &mut conn,
            m,
            &(0..3).map(|i| person(0, i)).collect::<Vec<_>>(),
        );
        let (faces, plan) = snapshot_plan(&conn, m, &config(), false).unwrap();
        assert_eq!(plan.new_people, 1);
        for f in &ids {
            faces::unassign(&conn, *f).unwrap();
        }
        let outcome = write(&mut conn, m, &faces, plan, false).unwrap();
        assert_eq!(outcome, Outcome::default());
        assert!(people_of(&conn, m).is_empty());
        for f in &ids {
            assert_eq!(row(&conn, *f), (None, Some(AssignedBy::Manual)));
        }
    }

    #[test]
    fn a_person_ignored_after_the_snapshot_takes_no_faces() {
        let (_dir, mut conn) = temp();
        let m = model(&conn, "m");
        let p = people::create(&conn, m).unwrap();
        let known = add(&mut conn, m, &[person(0, 0), person(0, 1)]);
        for f in &known {
            faces::assign(&conn, *f, p, AssignedBy::Manual).unwrap();
        }
        let new = add(&mut conn, m, &[person(0, 2), person(0, 3)]);
        let (faces, plan) = snapshot_plan(&conn, m, &config(), false).unwrap();
        assert_eq!(plan.assignments.len(), 2);
        crate::catalog::edit(&mut conn, |tx| people::set_ignored(tx, p, true));
        let outcome = write(&mut conn, m, &faces, plan, false).unwrap();
        assert_eq!(outcome, Outcome::default());
        for f in &new {
            assert_eq!(row(&conn, *f), (None, None));
        }
    }

    #[test]
    fn incremental_runs_only_when_faces_were_added() {
        let (_dir, mut conn) = temp();
        let m = model(&conn, "m");
        let ids = add(&mut conn, m, &[person(0, 0), person(0, 1)]);
        assert_eq!(
            incremental(&mut conn, m, &config()).unwrap(),
            Outcome::default()
        );
        let p = people::create(&conn, m).unwrap();
        faces::assign(&conn, ids[0], p, AssignedBy::Manual).unwrap();
        incremental(&mut conn, m, &config()).unwrap();
        assert_eq!(person_of(&conn, ids[1]), None);

        add(&mut conn, m, &[person(5, 0)]);
        let outcome = incremental(&mut conn, m, &config()).unwrap();
        assert_eq!(outcome.faces_assigned, 1);
        assert_eq!(person_of(&conn, ids[1]), Some(p));
    }

    #[test]
    fn models_are_clustered_separately() {
        let (_dir, mut conn) = temp();
        let a = model(&conn, "a");
        let b = model(&conn, "b");
        let on_a = add(
            &mut conn,
            a,
            &(0..3).map(|i| person(0, i)).collect::<Vec<_>>(),
        );
        incremental(&mut conn, a, &config()).unwrap();
        let pa = person_of(&conn, on_a[0]).unwrap();
        crate::catalog::edit(&mut conn, |tx| people::rename(tx, pa, Some("Ada")));
        let before = (snapshot(&conn, a), people::list(&conn, a).unwrap());

        let on_b = add(
            &mut conn,
            b,
            &(0..3).map(|i| person(0, i)).collect::<Vec<_>>(),
        );
        let outcome = incremental(&mut conn, b, &config()).unwrap();
        assert_eq!((outcome.people_created, outcome.faces_assigned), (1, 3));
        recluster(&mut conn, b, &config()).unwrap();
        assert_eq!(
            (snapshot(&conn, a), people::list(&conn, a).unwrap()),
            before
        );
        let pb = person_of(&conn, on_b[0]).unwrap();
        assert_ne!(pa, pb);
        let model_of: i64 = conn
            .query_row(
                "SELECT model_id FROM people WHERE id = ?1",
                params![pb.0],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(model_of, b.0);
        assert!(on_b.iter().all(|f| person_of(&conn, *f) == Some(pb)));

        let x = person(0, 0);
        assert!(matches!(
            similarity(
                Embedding {
                    model: a,
                    values: &x
                },
                Embedding {
                    model: b,
                    values: &x
                }
            ),
            Err(Error::ModelMismatch(..))
        ));
    }

    struct Noise(u64);

    impl Noise {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            #[allow(clippy::cast_precision_loss)]
            let unit = (self.0 >> 40) as f32 / (1_u64 << 24) as f32;
            unit - 0.5
        }

        fn vector(&mut self, dim: usize) -> Vec<f32> {
            (0..dim).map(|_| self.next()).collect()
        }
    }

    /// `per_person` faces around each of `people` random centres (centre + `spread` x noise), people interleaved.
    fn synthetic(dim: usize, people: usize, per_person: usize, spread: f32) -> Vec<Vec<f32>> {
        let mut noise = Noise(0x9e37_79b9_7f4a_7c15);
        let centres: Vec<Vec<f32>> = (0..people).map(|_| noise.vector(dim)).collect();
        (0..per_person)
            .flat_map(|_| centres.clone())
            .map(|centre| {
                let offset = noise.vector(dim);
                centre
                    .iter()
                    .zip(offset)
                    .map(|(c, o)| c + spread * o)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn results_do_not_depend_on_threads_or_runs() {
        let embeddings = synthetic(64, 30, 12, 1.1);
        let cluster_with = |threads: usize| {
            let (dir, mut conn) = temp();
            let m = register_model(&conn, "m", "d", 64).unwrap();
            add(&mut conn, m, &embeddings);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let outcome = pool.install(|| recluster(&mut conn, m, &config()).unwrap());
            (dir, conn, m, outcome)
        };
        let (_one, conn_one, m, outcome) = cluster_with(1);
        let (_many, conn_many, _, outcome_many) = cluster_with(8);
        assert!(outcome.people_created > 1 && outcome.faces_assigned > 0);
        assert_eq!(outcome, outcome_many);
        let first = snapshot(&conn_one, m);
        assert_eq!(first, snapshot(&conn_many, m));

        let mut conn = conn_one;
        recluster(&mut conn, m, &config()).unwrap();
        // people are recreated, so compare which faces share a person
        let groups = |snapshot: &[(i64, Option<i64>, Option<String>)]| {
            let mut by_person: HashMap<i64, Vec<i64>> = HashMap::new();
            for (face, person, _) in snapshot {
                if let Some(p) = person {
                    by_person.entry(*p).or_default().push(*face);
                }
            }
            let mut groups: Vec<Vec<i64>> = by_person.into_values().collect();
            groups.sort();
            groups
        };
        assert_eq!(groups(&snapshot(&conn, m)), groups(&first));
    }

    /// Times a recluster over 50k synthetic 512-d faces; run with `--release`.
    #[test]
    #[ignore = "slow; measures clustering at 50k faces"]
    fn fifty_thousand_faces() {
        use std::time::Instant;

        let (_dir, mut conn) = temp();
        let m = register_model(&conn, "m", "d", 512).unwrap();
        let embeddings = synthetic(512, 2_000, 25, 1.0);
        let (existing, new) = embeddings.split_at(49_500);
        let start = Instant::now();
        store(&mut conn, m, existing, 0);
        eprintln!("insert 49.5k faces: {:?}", start.elapsed());

        let start = Instant::now();
        let outcome = recluster(&mut conn, m, &config()).unwrap();
        eprintln!("recluster: {:?}, {outcome:?}", start.elapsed());
        store(&mut conn, m, new, existing.len());
        let start = Instant::now();
        let outcome = incremental(&mut conn, m, &config()).unwrap();
        eprintln!("incremental, 500 new: {:?}, {outcome:?}", start.elapsed());
        let start = Instant::now();
        let outcome = recluster(&mut conn, m, &config()).unwrap();
        eprintln!("recluster 50k: {:?}, {outcome:?}", start.elapsed());
        let start = Instant::now();
        let outcome = incremental(&mut conn, m, &config()).unwrap();
        eprintln!(
            "incremental, nothing new: {:?}, {outcome:?}",
            start.elapsed()
        );
        let unassigned: i64 = conn
            .query_row(
                "SELECT count(*) FROM faces WHERE person_id IS NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        eprintln!("unassigned: {unassigned}");
        assert_eq!(outcome, Outcome::default());
    }

    fn store(conn: &mut Connection, m: ModelId, embeddings: &[Vec<f32>], first: usize) {
        let tx = conn.transaction().unwrap();
        for (n, chunk) in embeddings.chunks(5).enumerate() {
            let mut sha = [0; 32];
            let key = (first + n).to_le_bytes();
            sha[..key.len()].copy_from_slice(&key);
            let faces: Vec<NewFace<'_>> = chunk
                .iter()
                .map(|e| NewFace {
                    rect: Rect {
                        x: 0.0,
                        y: 0.0,
                        w: 100.0,
                        h: 100.0,
                    },
                    landmarks: [[0.0; 2]; 5],
                    score: 0.9,
                    embedding: e,
                    crop: b"jpeg",
                })
                .collect();
            insert_scan(&tx, &sha, m, &faces).unwrap();
        }
        tx.commit().unwrap();
    }

    mod fixtures {
        use std::collections::BTreeMap;
        use std::sync::atomic::AtomicBool;

        use super::*;
        use crate::catalog::images;
        use crate::config::Paths;
        use crate::faces::fetch::MODEL_NAME;
        use crate::faces::fixtures;

        /// Each image with one labelled person and one counted face must land in that person's single cluster.
        #[test]
        #[ignore = "needs ORT_DYLIB_PATH, fetched models and VITRINE_FACE_FIXTURES"]
        fn fixture_people_form_one_cluster_each() {
            let fixtures = fixtures::load();
            let dir = tempfile::tempdir().unwrap();
            let paths = Paths {
                home_dir: dir.path().to_owned(),
                config_file: dir.path().join("config.toml"),
                data_dir: dir.path().join("data"),
                cache_dir: dir.path().join("cache"),
            };
            std::fs::create_dir_all(&paths.data_dir).unwrap();
            std::os::unix::fs::symlink(fixtures::models_dir(), paths.models_dir()).unwrap();
            let config = config();
            let root = fixtures.dir.canonicalize().unwrap();
            let summary = crate::index::run(
                &paths,
                std::slice::from_ref(&root),
                &config,
                &AtomicBool::new(false),
                |_| {},
            )
            .unwrap();
            eprintln!(
                "people created {}, faces assigned {}",
                summary.stats.people_created, summary.stats.faces_assigned
            );

            let conn = crate::catalog::open(&paths.catalog_db()).unwrap();
            let m = faces::model_by_name(&conn, MODEL_NAME).unwrap().unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT person_id FROM faces
                     WHERE sha256 = ?1 AND model_id = ?2 AND not_a_face = 0
                       AND score >= ?3 AND min(w, h) >= ?4
                     ORDER BY x",
                )
                .unwrap();
            let mut clusters: BTreeMap<String, Vec<Option<i64>>> = BTreeMap::new();
            for label in &fixtures.labels {
                let image = images::get_by_path(&conn, &root.join(&label.file))
                    .unwrap()
                    .unwrap();
                let persons: Vec<Option<i64>> = stmt
                    .query_map(
                        params![
                            image.sha256.unwrap(),
                            m.0,
                            f64::from(config.min_score),
                            config.min_face_size
                        ],
                        |r| r.get(0),
                    )
                    .unwrap()
                    .collect::<rusqlite::Result<_>>()
                    .unwrap();
                eprintln!("{} {:?}: {persons:?}", label.file, label.people);
                if let ([who], [person]) = (label.people.as_slice(), persons.as_slice()) {
                    clusters.entry(who.clone()).or_default().push(*person);
                }
            }
            eprintln!("single-face clusters by label: {clusters:?}");
            let one = |who: &str| -> i64 {
                let found = &clusters[who];
                assert!(found.len() >= 2, "{who}: {found:?}");
                assert!(
                    found.iter().all(|p| p.is_some() && *p == found[0]),
                    "{who} is split or unassigned: {found:?}"
                );
                found[0].unwrap()
            };
            assert_ne!(one("a"), one("b"));
        }
    }
}
