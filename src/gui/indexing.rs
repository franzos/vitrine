use std::io;
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use eframe::egui;

use crate::config::{Faces, Paths};
use crate::index::{self, IndexEvent, Summary};
use crate::lock::LockError;

pub const LOCK_POLL: Duration = Duration::from_secs(2);
const CANCEL_POLL: Duration = Duration::from_millis(50);

#[derive(Debug)]
pub enum Update {
    /// Sent on every retry.
    Held,
    Event(IndexEvent),
    Finished(Result<Summary, String>),
}

/// Cancelled and joined on drop.
pub struct Indexing {
    rx: Receiver<Update>,
    cancel: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Indexing {
    /// Requests a repaint of `ctx` after every update.
    pub fn start(
        paths: Paths,
        roots: Vec<PathBuf>,
        faces: Faces,
        ctx: egui::Context,
    ) -> io::Result<Self> {
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let thread = thread::Builder::new()
            .name("vitrine-index".to_owned())
            .spawn({
                let cancel = Arc::clone(&cancel);
                move || work(&paths, &roots, &faces, &cancel, &tx, &ctx)
            })?;
        Ok(Self {
            rx,
            cancel,
            thread: Some(thread),
        })
    }

    pub fn drain(&self) -> impl Iterator<Item = Update> + '_ {
        self.rx.try_iter()
    }
}

impl Drop for Indexing {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            _ = thread.join();
        }
    }
}

/// Retries every [`LOCK_POLL`] while the lock is held; sends nothing once cancelled.
fn work(
    paths: &Paths,
    roots: &[PathBuf],
    faces: &Faces,
    cancel: &AtomicBool,
    tx: &Sender<Update>,
    ctx: &egui::Context,
) {
    let send = |update| {
        _ = tx.send(update);
        ctx.request_repaint();
    };
    loop {
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            index::run(paths, roots, faces, cancel, |event| {
                send(Update::Event(event.clone()));
            })
        }));
        let result = match result {
            Ok(result) => result,
            Err(payload) => {
                let message = crate::decode::panic_message(payload.as_ref());
                send(Update::Finished(Err(format!(
                    "indexing panicked: {message}"
                ))));
                return;
            }
        };
        match result {
            Err(index::Error::Lock(LockError::Held { .. })) => {
                send(Update::Held);
                if !sleep_unless_cancelled(cancel, LOCK_POLL) {
                    return;
                }
            }
            Err(index::Error::Cancelled) => return,
            result => {
                send(Update::Finished(result.map_err(|e| e.to_string())));
                return;
            }
        }
    }
}

/// Sleeps for `duration`; returns `false` as soon as `cancel` is set.
fn sleep_unless_cancelled(cancel: &AtomicBool, duration: Duration) -> bool {
    let mut left = duration;
    while !left.is_zero() {
        if cancel.load(Ordering::Relaxed) {
            return false;
        }
        let step = left.min(CANCEL_POLL);
        thread::sleep(step);
        left -= step;
    }
    !cancel.load(Ordering::Relaxed)
}
