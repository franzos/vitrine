use rusqlite::{Connection, Row, Transaction, params};

use super::faces::{FaceId, ModelId};
use crate::db::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PersonId(pub i64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Person {
    pub id: PersonId,
    pub name: Option<String>,
    pub ignored: bool,
    pub faces: u64,
    /// Highest-scoring face, lowest id on ties.
    pub cover: Option<FaceId>,
}

/// Selects from `people p`; the face count is column 4 of the five.
pub(crate) const COLUMNS: &str = "p.id, p.name, p.ignored,
     (SELECT count(*) FROM faces f WHERE f.person_id = p.id),
     (SELECT f.id FROM faces f WHERE f.person_id = p.id ORDER BY f.score DESC, f.id LIMIT 1)";

/// Reads [`COLUMNS`] starting at column `start`.
pub(crate) fn from_row(row: &Row<'_>, start: usize) -> rusqlite::Result<Person> {
    Ok(Person {
        id: PersonId(row.get(start)?),
        name: row.get(start + 1)?,
        ignored: row.get(start + 2)?,
        faces: row.get(start + 3)?,
        cover: row.get::<_, Option<i64>>(start + 4)?.map(FaceId),
    })
}

pub fn create(conn: &Connection, model: ModelId) -> Result<PersonId> {
    let id = conn.query_row(
        "INSERT INTO people (model_id) VALUES (?1) RETURNING id",
        [model.0],
        |row| row.get(0),
    )?;
    Ok(PersonId(id))
}

/// Also pins the person's auto faces as manual.
pub fn rename(tx: &Transaction<'_>, person: PersonId, name: Option<&str>) -> Result<()> {
    tx.execute(
        "UPDATE people SET name = ?2 WHERE id = ?1",
        params![person.0, name],
    )?;
    pin_faces(tx, person)
}

/// Also pins the person's auto faces as manual.
pub fn set_ignored(tx: &Transaction<'_>, person: PersonId, ignored: bool) -> Result<()> {
    tx.execute(
        "UPDATE people SET ignored = ?2 WHERE id = ?1",
        params![person.0, ignored],
    )?;
    pin_faces(tx, person)
}

/// A recluster resets auto rows, which would empty a named person or revive an ignored one as new.
fn pin_faces(tx: &Transaction<'_>, person: PersonId) -> Result<()> {
    tx.execute(
        "UPDATE faces SET assigned_by = 'manual' WHERE person_id = ?1 AND assigned_by = 'auto'",
        [person.0],
    )?;
    Ok(())
}

/// Faces move as manual; `into` takes the name of `from` when it has none.
pub fn merge(tx: &Transaction<'_>, from: PersonId, into: PersonId) -> Result<()> {
    if from == into {
        return Ok(());
    }
    let model_of = |p: PersonId| -> rusqlite::Result<i64> {
        tx.query_row("SELECT model_id FROM people WHERE id = ?1", [p.0], |row| {
            row.get(0)
        })
    };
    if model_of(from)? != model_of(into)? {
        return Err(Error::ModelMismatch(from.0, into.0));
    }
    tx.execute(
        "UPDATE faces SET person_id = ?2, assigned_by = 'manual' WHERE person_id = ?1",
        [from.0, into.0],
    )?;
    tx.execute(
        "UPDATE people SET name = (SELECT name FROM people WHERE id = ?1)
         WHERE id = ?2 AND name IS NULL",
        [from.0, into.0],
    )?;
    tx.execute("DELETE FROM people WHERE id = ?1", [from.0])?;
    Ok(())
}

/// Keeps each face's `assigned_by`. `false` when `from` is named, `into` is ignored, or either is gone: a snapshot-planned merge mustn't swallow a person named since or feed an ignored one.
pub fn absorb(tx: &Transaction<'_>, from: PersonId, into: PersonId) -> Result<bool> {
    if from == into {
        return Ok(false);
    }
    let movable: bool = tx.query_row(
        "SELECT EXISTS (SELECT 1 FROM people WHERE id = ?1 AND name IS NULL)
            AND EXISTS (SELECT 1 FROM people WHERE id = ?2 AND ignored = 0)",
        [from.0, into.0],
        |row| row.get(0),
    )?;
    if !movable {
        return Ok(false);
    }
    tx.execute(
        "UPDATE faces SET person_id = ?2 WHERE person_id = ?1",
        [from.0, into.0],
    )?;
    tx.execute("DELETE FROM people WHERE id = ?1", [from.0])?;
    Ok(true)
}

/// Faces lose the person but keep `assigned_by`.
pub fn delete(conn: &Connection, person: PersonId) -> Result<()> {
    conn.execute("DELETE FROM people WHERE id = ?1", [person.0])?;
    Ok(())
}

pub fn delete_empty_unnamed(conn: &Connection, model: ModelId) -> Result<usize> {
    Ok(conn.execute(
        "DELETE FROM people
         WHERE model_id = ?1 AND name IS NULL AND ignored = 0
           AND NOT EXISTS (SELECT 1 FROM faces f WHERE f.person_id = people.id)",
        [model.0],
    )?)
}

/// Named by name, then unnamed by face count (most first), ties by id.
pub fn list(conn: &Connection, model: ModelId) -> Result<Vec<Person>> {
    let sql = format!(
        "SELECT {COLUMNS} FROM people p
         WHERE p.model_id = ?1
         ORDER BY p.name IS NULL, p.name, 4 DESC, p.id"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([model.0], |row| from_row(row, 0))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::faces::tests::scan;
    use crate::catalog::faces::{self, AssignedBy, register_model};
    use crate::catalog::{edit, temp};

    #[test]
    fn create_rename_ignore_list() {
        let (_dir, mut conn) = temp();
        let model = register_model(&conn, "m", "d", 2).unwrap();
        let other = register_model(&conn, "o", "d", 2).unwrap();
        let unnamed = create(&conn, model).unwrap();
        let bob = create(&conn, model).unwrap();
        let ada = create(&conn, model).unwrap();
        create(&conn, other).unwrap();
        edit(&mut conn, |tx| rename(tx, bob, Some("Bob")));
        edit(&mut conn, |tx| rename(tx, ada, Some("Ada")));
        edit(&mut conn, |tx| set_ignored(tx, unnamed, true));

        let got = list(&conn, model).unwrap();
        let summary: Vec<_> = got
            .iter()
            .map(|p| (p.id, p.name.as_deref(), p.ignored))
            .collect();
        assert_eq!(
            summary,
            [
                (ada, Some("Ada"), false),
                (bob, Some("Bob"), false),
                (unnamed, None, true),
            ]
        );
        edit(&mut conn, |tx| rename(tx, bob, None));
        assert_eq!(list(&conn, model).unwrap()[1].name, None);
    }

    #[test]
    fn merge_moves_faces_as_manual_and_keeps_a_name() {
        let (_dir, mut conn) = temp();
        let model = register_model(&conn, "m", "d", 2).unwrap();
        let ids = scan(&mut conn, [1; 32], model, 3);
        let from = create(&conn, model).unwrap();
        let into = create(&conn, model).unwrap();
        edit(&mut conn, |tx| rename(tx, from, Some("Ada")));
        faces::assign(&conn, ids[0], from, AssignedBy::Auto).unwrap();
        faces::assign(&conn, ids[1], from, AssignedBy::Auto).unwrap();
        faces::assign(&conn, ids[2], into, AssignedBy::Auto).unwrap();

        let tx = conn.transaction().unwrap();
        merge(&tx, from, into).unwrap();
        tx.commit().unwrap();

        assert_eq!(
            list(&conn, model).unwrap(),
            [Person {
                id: into,
                name: Some("Ada".to_owned()),
                ignored: false,
                faces: 3,
                cover: Some(ids[0]),
            }]
        );
        let faces = faces::for_clustering(&conn, model, 0.0, 0).unwrap();
        assert_eq!(faces[0].assigned_by, Some(AssignedBy::Manual));
        assert_eq!(faces[2].assigned_by, Some(AssignedBy::Auto));
    }

    #[test]
    fn absorb_skips_named_sources_and_ignored_targets() {
        let (_dir, mut conn) = temp();
        let model = register_model(&conn, "m", "d", 2).unwrap();
        let ids = scan(&mut conn, [1; 32], model, 3);
        let from = create(&conn, model).unwrap();
        let named = create(&conn, model).unwrap();
        let ignored = create(&conn, model).unwrap();
        let into = create(&conn, model).unwrap();
        edit(&mut conn, |tx| rename(tx, named, Some("Ada")));
        edit(&mut conn, |tx| set_ignored(tx, ignored, true));
        faces::assign(&conn, ids[0], from, AssignedBy::Auto).unwrap();
        faces::assign(&conn, ids[1], named, AssignedBy::Auto).unwrap();
        faces::assign(&conn, ids[2], into, AssignedBy::Manual).unwrap();

        let tx = conn.transaction().unwrap();
        assert!(!absorb(&tx, named, into).unwrap());
        assert!(!absorb(&tx, from, ignored).unwrap());
        assert!(!absorb(&tx, from, PersonId(999)).unwrap());
        assert!(absorb(&tx, from, into).unwrap());
        tx.commit().unwrap();

        let rows: Vec<_> = faces::for_clustering(&conn, model, 0.0, 0)
            .unwrap()
            .iter()
            .map(|f| (f.person_id, f.assigned_by))
            .collect();
        assert_eq!(
            rows,
            [
                (Some(into), Some(AssignedBy::Auto)),
                (Some(named), Some(AssignedBy::Auto)),
                (Some(into), Some(AssignedBy::Manual)),
            ]
        );
        let left: Vec<_> = list(&conn, model).unwrap().iter().map(|p| p.id).collect();
        assert_eq!(left, [named, into, ignored]);
    }

    #[test]
    fn delete_empty_unnamed_keeps_named_ignored_and_used() {
        let (_dir, mut conn) = temp();
        let model = register_model(&conn, "m", "d", 2).unwrap();
        let other = register_model(&conn, "o", "d", 2).unwrap();
        let ids = scan(&mut conn, [1; 32], model, 1);
        let empty = create(&conn, model).unwrap();
        let named = create(&conn, model).unwrap();
        let ignored = create(&conn, model).unwrap();
        let used = create(&conn, model).unwrap();
        let elsewhere = create(&conn, other).unwrap();
        edit(&mut conn, |tx| rename(tx, named, Some("Ada")));
        edit(&mut conn, |tx| set_ignored(tx, ignored, true));
        faces::assign(&conn, ids[0], used, AssignedBy::Auto).unwrap();

        assert_eq!(delete_empty_unnamed(&conn, model).unwrap(), 1);
        let left: Vec<_> = list(&conn, model).unwrap().iter().map(|p| p.id).collect();
        assert!(!left.contains(&empty));
        assert_eq!(left.len(), 3);
        assert_eq!(list(&conn, other).unwrap()[0].id, elsewhere);
    }

    #[test]
    fn merge_rejects_other_model() {
        let (_dir, mut conn) = temp();
        let a = register_model(&conn, "a", "d", 2).unwrap();
        let b = register_model(&conn, "b", "d", 2).unwrap();
        let pa = create(&conn, a).unwrap();
        let pb = create(&conn, b).unwrap();
        let tx = conn.transaction().unwrap();
        assert!(matches!(merge(&tx, pa, pb), Err(Error::ModelMismatch(..))));
    }

    #[test]
    fn naming_and_ignoring_turn_auto_faces_manual() {
        let (_dir, mut conn) = temp();
        let model = register_model(&conn, "m", "d", 2).unwrap();
        let ids = scan(&mut conn, [1; 32], model, 3);
        let named = create(&conn, model).unwrap();
        let ignored = create(&conn, model).unwrap();
        faces::assign(&conn, ids[0], named, AssignedBy::Auto).unwrap();
        faces::assign(&conn, ids[1], ignored, AssignedBy::Auto).unwrap();
        faces::assign(&conn, ids[2], ignored, AssignedBy::Auto).unwrap();

        edit(&mut conn, |tx| rename(tx, named, Some("Ada")));
        edit(&mut conn, |tx| set_ignored(tx, ignored, true));
        let rows: Vec<_> = faces::for_clustering(&conn, model, 0.0, 0)
            .unwrap()
            .iter()
            .map(|f| (f.person_id, f.assigned_by))
            .collect();
        assert_eq!(
            rows,
            [
                (Some(named), Some(AssignedBy::Manual)),
                (Some(ignored), Some(AssignedBy::Manual)),
                (Some(ignored), Some(AssignedBy::Manual)),
            ]
        );
    }

    #[test]
    fn list_orders_named_by_name_then_unnamed_by_count_with_the_best_face() {
        let (_dir, mut conn) = temp();
        let model = register_model(&conn, "m", "d", 2).unwrap();
        let e = [0.5_f32, -0.5];
        let scored = |score| faces::NewFace {
            score,
            ..faces::tests::face(&e)
        };
        let tx = conn.transaction().unwrap();
        let ids = faces::insert_scan(
            &tx,
            &[1; 32],
            model,
            &[scored(0.6), scored(0.9), scored(0.7)],
        )
        .unwrap();
        tx.commit().unwrap();
        let few = create(&conn, model).unwrap();
        let many = create(&conn, model).unwrap();
        let zed = create(&conn, model).unwrap();
        faces::assign(&conn, ids[0], few, AssignedBy::Auto).unwrap();
        faces::assign(&conn, ids[1], many, AssignedBy::Auto).unwrap();
        faces::assign(&conn, ids[2], many, AssignedBy::Auto).unwrap();
        edit(&mut conn, |tx| rename(tx, zed, Some("Zed")));

        let got: Vec<_> = list(&conn, model)
            .unwrap()
            .into_iter()
            .map(|p| (p.id, p.faces, p.cover))
            .collect();
        assert_eq!(
            got,
            [
                (zed, 0, None),
                (many, 2, Some(ids[1])),
                (few, 1, Some(ids[0])),
            ]
        );
    }
}
