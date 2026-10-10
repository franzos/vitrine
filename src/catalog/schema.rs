//! Catalog schema migrations. Append new ones to [`migrations`]; never edit a shipped one.
//!
//! - `images.path`: its unique index also serves folder prefix range scans.
//! - `images.mtime_ns`: nanoseconds since the Unix epoch.
//! - `images.taken_at`: naive local time, `%Y-%m-%dT%H:%M:%S`, no timezone.
//! - Face data is keyed by `sha256`, not image row, so it survives moves and duplicates share it; no foreign key since `images.sha256` isn't unique.
//! - `face_scans`: includes content with no faces or that failed to decode.
//! - `faces`: every detection; the face count's score and size filter applies at query time.
//! - `faces.assigned_by`: `auto` is set by clustering and reset on recluster; `manual` (also with no person) is never touched by clustering.
//! - `faces_for_counts`: covers face counts without reading blob columns, the unknown-faces filter, and lookups by content.
//! - `faces_by_person`: covers per-person counts, each person's best face, and the people filter.
//! - `faces.landmarks`: 5 `(x, y)` pairs. `faces.embedding`: the model's vector. Both little-endian `f32`.
//! - `people.created_at`, `index_runs.finished_at`: seconds since the Unix epoch.
//! - `face_models.clustered_to`: highest face id of that model the last clustering pass saw; a pass runs when a newer face exists.

use rusqlite_migration::{M, Migrations};

const M0001_INITIAL: &str = r"
CREATE TABLE face_models (
    id       INTEGER PRIMARY KEY,
    name     TEXT    NOT NULL UNIQUE,
    detector TEXT    NOT NULL,
    dim      INTEGER NOT NULL CHECK (dim > 0)
);

CREATE TABLE images (
    id       INTEGER PRIMARY KEY,
    path     TEXT    NOT NULL UNIQUE,
    size     INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    sha256   BLOB    CHECK (length(sha256) = 32),
    kind     TEXT    NOT NULL CHECK (kind IN ('image', 'video')),
    status   TEXT    NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'missing')),
    width    INTEGER,
    height   INTEGER,
    taken_at TEXT
);

CREATE INDEX images_by_sha256 ON images(sha256);

CREATE TABLE face_scans (
    sha256   BLOB    NOT NULL CHECK (length(sha256) = 32),
    model_id INTEGER NOT NULL REFERENCES face_models(id),
    PRIMARY KEY (sha256, model_id)
) WITHOUT ROWID;

CREATE TABLE people (
    id         INTEGER PRIMARY KEY,
    model_id   INTEGER NOT NULL REFERENCES face_models(id),
    name       TEXT,
    ignored    INTEGER NOT NULL DEFAULT 0 CHECK (ignored IN (0, 1)),
    created_at INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX people_by_model_id ON people(model_id);

CREATE TABLE faces (
    id          INTEGER PRIMARY KEY,
    sha256      BLOB    NOT NULL CHECK (length(sha256) = 32),
    model_id    INTEGER NOT NULL REFERENCES face_models(id),
    x           REAL    NOT NULL,
    y           REAL    NOT NULL,
    w           REAL    NOT NULL,
    h           REAL    NOT NULL,
    landmarks   BLOB    NOT NULL CHECK (length(landmarks) = 40),
    score       REAL    NOT NULL,
    embedding   BLOB    NOT NULL,
    crop        BLOB    NOT NULL,
    person_id   INTEGER REFERENCES people(id) ON DELETE SET NULL,
    assigned_by TEXT    CHECK (assigned_by IN ('auto', 'manual')),
    not_a_face  INTEGER NOT NULL DEFAULT 0 CHECK (not_a_face IN (0, 1)),
    CHECK (person_id IS NULL OR assigned_by IS NOT NULL)
);

CREATE INDEX faces_by_sha256 ON faces(sha256);
CREATE INDEX faces_by_model_id ON faces(model_id);
CREATE INDEX faces_by_person_id ON faces(person_id);
";

const M0002_INDEX_RUNS: &str = r"
CREATE TABLE index_runs (
    id          INTEGER PRIMARY KEY,
    finished_at INTEGER NOT NULL DEFAULT (unixepoch()),
    files       INTEGER NOT NULL,
    hashed      INTEGER NOT NULL,
    added       INTEGER NOT NULL,
    changed     INTEGER NOT NULL,
    moved       INTEGER NOT NULL,
    restored    INTEGER NOT NULL,
    vanished    INTEGER NOT NULL,
    failed      INTEGER NOT NULL
);
";

const M0003_RUN_THUMBNAILED: &str = r"
ALTER TABLE index_runs ADD COLUMN thumbnailed INTEGER NOT NULL DEFAULT 0;
";

const M0004_FACE_STAGE: &str = r"
ALTER TABLE index_runs ADD COLUMN faces_scanned INTEGER NOT NULL DEFAULT 0;
ALTER TABLE index_runs ADD COLUMN faces_found INTEGER NOT NULL DEFAULT 0;

DROP INDEX faces_by_model_id;
CREATE INDEX faces_for_counts ON faces(model_id, sha256, not_a_face, score, w, h);
";

const M0005_CLUSTERING: &str = r"
ALTER TABLE face_models ADD COLUMN clustered_to INTEGER NOT NULL DEFAULT 0;
ALTER TABLE index_runs ADD COLUMN people_created INTEGER NOT NULL DEFAULT 0;
ALTER TABLE index_runs ADD COLUMN faces_assigned INTEGER NOT NULL DEFAULT 0;
";

const M0006_FACE_FILTERS: &str = r"
DROP INDEX faces_by_person_id;
CREATE INDEX faces_by_person ON faces(person_id, score, sha256);

DROP INDEX faces_for_counts;
CREATE INDEX faces_for_counts ON faces(model_id, sha256, not_a_face, score, w, h, person_id);
";

const M0007_DROP_FACES_BY_SHA256: &str = r"
DROP INDEX faces_by_sha256;
";

pub(crate) fn migrations() -> Migrations<'static> {
    Migrations::new(vec![
        M::up(M0001_INITIAL),
        M::up(M0002_INDEX_RUNS),
        M::up(M0003_RUN_THUMBNAILED),
        M::up(M0004_FACE_STAGE),
        M::up(M0005_CLUSTERING),
        M::up(M0006_FACE_FILTERS),
        M::up(M0007_DROP_FACES_BY_SHA256),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::sha256_hex;

    #[test]
    fn migrations_are_valid() {
        migrations().validate().unwrap();
    }

    #[test]
    fn migration_fingerprints_are_pinned() {
        let pins: &[(&str, &str)] = &[
            (
                "M0001_INITIAL",
                "b2f7713dfb0eadce8b452266463dd5a6a09d3e1e80a02d5767853fe3775ae351",
            ),
            (
                "M0002_INDEX_RUNS",
                "70515d316d0a93c80c93ee78a627eec4dbd1f80049c61d0131d9298c6c6d4ad6",
            ),
            (
                "M0003_RUN_THUMBNAILED",
                "45b1cd8c579050f5429628aca9ba7458848c34013042948daf887a45e1f0c032",
            ),
            (
                "M0004_FACE_STAGE",
                "426cbaa23457169b52a46621188dddc167fdf93581312b19c0320ef83494a558",
            ),
            (
                "M0005_CLUSTERING",
                "5e8a4c1ad300af121bdab4b178f475261d3e22b8f22b6a3b7e4daf16ed9759a8",
            ),
            (
                "M0006_FACE_FILTERS",
                "8268141ff261dfca9f7674420fdb04c02b2868122a2abc701e8bb209663520ef",
            ),
            (
                "M0007_DROP_FACES_BY_SHA256",
                "b8eee8c3bda08047bf4ad886a9f2f01b22a0dc77a75bac99087707b39524ed3a",
            ),
        ];
        let sources: &[(&str, &str)] = &[
            ("M0001_INITIAL", M0001_INITIAL),
            ("M0002_INDEX_RUNS", M0002_INDEX_RUNS),
            ("M0003_RUN_THUMBNAILED", M0003_RUN_THUMBNAILED),
            ("M0004_FACE_STAGE", M0004_FACE_STAGE),
            ("M0005_CLUSTERING", M0005_CLUSTERING),
            ("M0006_FACE_FILTERS", M0006_FACE_FILTERS),
            ("M0007_DROP_FACES_BY_SHA256", M0007_DROP_FACES_BY_SHA256),
        ];
        assert_eq!(
            sources.len(),
            pins.len(),
            "add a new pin when shipping a new migration"
        );
        for ((name, src), (pin_name, pin_hash)) in sources.iter().zip(pins) {
            assert_eq!(name, pin_name, "pin order must match source order");
            assert_eq!(
                &sha256_hex(src),
                pin_hash,
                "migration `{name}` changed; ship a new migration instead"
            );
        }
    }
}
