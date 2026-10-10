use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use rusqlite::{Connection, Transaction};

use super::faces::Scan;
use super::meta::Meta;
use super::{Failure, IndexEvent};
use crate::catalog::faces::{self, NewFace, Rect};
use crate::catalog::images::{self, Image, ImageId, Kind, NewImage, Status};
use crate::catalog::runs::{self, RunStats};
use crate::db::{Result, Sha256};
use crate::thumbs::{self, Thumb};

const BATCH_ROWS: usize = 500;
const BATCH_TIME: Duration = Duration::from_secs(1);

#[derive(Debug)]
pub(crate) enum Op {
    Restore { id: ImageId, path: PathBuf },
    Store(Stored),
    Faces(Scan),
}

#[derive(Debug)]
pub(crate) struct Stored {
    pub path: PathBuf,
    pub size: u64,
    pub mtime_ns: i64,
    pub kind: Kind,
    pub existing: Option<ImageId>,
    pub sha256: Option<Sha256>,
    pub meta: Meta,
    /// Rows with the same content whose file is gone; a new path takes over the first unclaimed one.
    pub donors: Vec<(ImageId, PathBuf)>,
    pub error: Option<String>,
    pub thumbs: Vec<(u32, Thumb)>,
}

pub(crate) struct Writer<'c, F> {
    pub conn: &'c mut Connection,
    pub on_event: F,
    pub stats: RunStats,
    pub failures: Vec<Failure>,
    /// Active rows in scope not found on disk; marked missing at the end unless claimed.
    pub vanished: Vec<Image>,
    /// Thumbnail cache; open only when there are files to hash.
    pub thumbs: Option<Connection>,
    claimed: HashSet<ImageId>,
    cancel: &'c AtomicBool,
}

impl<'c, F: FnMut(&IndexEvent)> Writer<'c, F> {
    pub(crate) fn new(
        conn: &'c mut Connection,
        on_event: F,
        stats: RunStats,
        cancel: &'c AtomicBool,
    ) -> Self {
        Self {
            conn,
            on_event,
            stats,
            failures: Vec::new(),
            vanished: Vec::new(),
            thumbs: None,
            claimed: HashSet::new(),
            cancel,
        }
    }

    /// `false` when cancelled; what arrived is committed either way.
    pub(crate) fn drain(&mut self, initial: Vec<Op>, rx: &Receiver<Op>) -> Result<bool> {
        let mut batch = initial;
        let mut deadline = Instant::now() + BATCH_TIME;
        loop {
            match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(op) => {
                    batch.push(op);
                    if batch.len() >= BATCH_ROWS {
                        self.flush(&mut batch)?;
                        deadline = Instant::now() + BATCH_TIME;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    if !batch.is_empty() {
                        self.flush(&mut batch)?;
                    }
                    deadline = Instant::now() + BATCH_TIME;
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        self.flush(&mut batch)?;
        // senders stop early on cancel, so a hang-up doesn't mean everything was sent
        Ok(!self.cancel.load(Ordering::Relaxed))
    }

    pub(crate) fn mark_vanished(&mut self) -> Result<()> {
        let mut events = Vec::new();
        let tx = self.conn.transaction()?;
        for row in self.vanished.drain(..) {
            if !self.claimed.contains(&row.id) {
                images::set_status(&tx, row.id, Status::Missing)?;
                self.stats.vanished += 1;
                events.push(IndexEvent::Missing {
                    id: row.id,
                    path: row.path,
                });
            }
        }
        tx.commit()?;
        for event in &events {
            (self.on_event)(event);
        }
        Ok(())
    }

    pub(crate) fn fail(&mut self, failure: Failure) {
        (self.on_event)(&IndexEvent::Failed {
            path: failure.path.clone(),
            error: failure.error.clone(),
        });
        self.failures.push(failure);
    }

    pub(crate) fn finish(mut self) -> Result<(RunStats, Vec<Failure>)> {
        self.stats.failed = self.failures.len() as u64;
        runs::record(self.conn, &self.stats)?;
        Ok((self.stats, self.failures))
    }

    fn flush(&mut self, batch: &mut Vec<Op>) -> Result<()> {
        // thumbnails first: a crash in between leaves unused thumbnails, not rows without them
        if let Some(conn) = &mut self.thumbs {
            let tx = conn.transaction()?;
            for op in batch.iter() {
                if let Op::Store(Stored {
                    sha256: Some(sha256),
                    thumbs: made,
                    ..
                }) = op
                {
                    for (bucket, thumb) in made {
                        thumbs::put(&tx, sha256, *bucket, thumb)?;
                    }
                }
            }
            tx.commit()?;
        }
        let mut events = Vec::with_capacity(batch.len());
        let tx = self.conn.transaction()?;
        for op in batch.drain(..) {
            match op {
                Op::Restore { id, path } => {
                    images::set_status(&tx, id, Status::Active)?;
                    self.stats.restored += 1;
                    events.push(IndexEvent::Restored { id, path });
                }
                Op::Store(stored) => {
                    apply_stored(
                        &tx,
                        stored,
                        &mut self.claimed,
                        &mut self.stats,
                        &mut self.failures,
                        &mut events,
                    )?;
                }
                Op::Faces(scan) => {
                    apply_scan(&tx, scan, &mut self.stats, &mut self.failures, &mut events)?;
                }
            }
        }
        tx.commit()?;
        for event in &events {
            (self.on_event)(event);
        }
        Ok(())
    }
}

fn apply_stored(
    conn: &Connection,
    stored: Stored,
    claimed: &mut HashSet<ImageId>,
    stats: &mut RunStats,
    failures: &mut Vec<Failure>,
    events: &mut Vec<IndexEvent>,
) -> Result<()> {
    if stored.kind == Kind::Image {
        stats.hashed += 1;
    }
    if !stored.thumbs.is_empty() {
        stats.thumbnailed += 1;
    }
    if let Some(error) = stored.error {
        failures.push(Failure {
            path: stored.path.clone(),
            error: error.clone(),
        });
        events.push(IndexEvent::Failed {
            path: stored.path.clone(),
            error,
        });
    }
    if stored.existing.is_none()
        && let Some((id, from)) = stored
            .donors
            .into_iter()
            .find(|(id, _)| !claimed.contains(id))
    {
        images::relink(conn, id, &stored.path, stored.size, stored.mtime_ns)?;
        claimed.insert(id);
        stats.moved += 1;
        events.push(IndexEvent::Moved {
            id,
            from,
            to: stored.path,
        });
        return Ok(());
    }
    let (width, height) = stored.meta.dimensions.unzip();
    let id = images::upsert(
        conn,
        &NewImage {
            path: &stored.path,
            size: stored.size,
            mtime_ns: stored.mtime_ns,
            sha256: stored.sha256,
            kind: stored.kind,
            width,
            height,
            taken_at: stored.meta.taken_at.as_deref(),
        },
    )?;
    let path = stored.path;
    if stored.existing.is_some() {
        stats.changed += 1;
        events.push(IndexEvent::Changed { id, path });
    } else {
        stats.added += 1;
        events.push(IndexEvent::Added { id, path });
    }
    Ok(())
}

fn apply_scan(
    tx: &Transaction<'_>,
    scan: Scan,
    stats: &mut RunStats,
    failures: &mut Vec<Failure>,
    events: &mut Vec<IndexEvent>,
) -> Result<()> {
    if let Some(found) = &scan.faces {
        let new: Vec<NewFace<'_>> = found
            .iter()
            .map(|f| {
                let [x1, y1, x2, y2] = f.detection.bbox;
                NewFace {
                    rect: Rect {
                        x: x1,
                        y: y1,
                        w: x2 - x1,
                        h: y2 - y1,
                    },
                    landmarks: f.detection.landmarks,
                    score: f.detection.score,
                    embedding: &f.embedding,
                    crop: &f.crop,
                }
            })
            .collect();
        faces::insert_scan(tx, &scan.sha256, scan.model, &new)?;
        stats.faces_scanned += 1;
        stats.faces_found += new.len() as u64;
    }
    if let Some(error) = scan.error {
        failures.push(Failure {
            path: scan.path.clone(),
            error: error.clone(),
        });
        events.push(IndexEvent::Failed {
            path: scan.path.clone(),
            error,
        });
    }
    events.push(IndexEvent::FacesScanned {
        sha256: scan.sha256,
        path: scan.path,
        faces: scan.faces.map_or(0, |f| f.len()),
    });
    Ok(())
}
