use std::collections::{HashSet, VecDeque};
use std::io;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;

use eframe::egui::{ColorImage, Context};
use turbojpeg::PixelFormat;

use crate::db::Sha256;
use crate::thumbs;

pub type Key = (Sha256, u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub key: Key,
    pub path: PathBuf,
}

pub struct Loaded {
    pub key: Key,
    pub image: Result<ColorImage, String>,
}

pub struct Loader {
    shared: Arc<Shared>,
    results: Receiver<Loaded>,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
}

#[derive(Default)]
struct State {
    queue: VecDeque<Request>,
    working: HashSet<Key>,
    ctx: Option<Context>,
    shutdown: bool,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Loader {
    /// Starts `workers` threads, each with its own connection to the thumbnail cache at `thumbs_db`.
    pub fn spawn(thumbs_db: &Path, workers: usize) -> io::Result<Self> {
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
        });
        let (tx, results) = mpsc::channel();
        for i in 0..workers.max(1) {
            let shared = Arc::clone(&shared);
            let tx = tx.clone();
            let db = thumbs_db.to_owned();
            thread::Builder::new()
                .name(format!("thumbs-{i}"))
                .spawn(move || work(&shared, &db, &tx))?;
        }
        Ok(Self { shared, results })
    }

    /// Replaces the queue with `wanted` in order, skipping keys a worker is already on.
    pub fn request(&self, ctx: &Context, wanted: Vec<Request>) {
        let mut state = self.shared.lock();
        if state.ctx.is_none() {
            state.ctx = Some(ctx.clone());
        }
        let working = &state.working;
        let queue: VecDeque<Request> = wanted
            .into_iter()
            .filter(|r| !working.contains(&r.key))
            .collect();
        let wake = !queue.is_empty();
        state.queue = queue;
        drop(state);
        if wake {
            self.shared.wake.notify_all();
        }
    }

    pub fn drain(&self) -> impl Iterator<Item = Loaded> + '_ {
        self.results.try_iter()
    }

    /// Requests queued or being loaded.
    #[must_use]
    pub fn pending(&self) -> usize {
        let state = self.shared.lock();
        state.queue.len() + state.working.len()
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.shared.lock().shutdown = true;
        self.shared.wake.notify_all();
    }
}

fn work(shared: &Shared, thumbs_db: &Path, tx: &Sender<Loaded>) {
    let conn = thumbs::open(thumbs_db).map_err(|e| e.to_string());
    loop {
        let request = {
            let mut state = shared.lock();
            loop {
                if state.shutdown {
                    return;
                }
                if let Some(request) = state.queue.pop_front() {
                    state.working.insert(request.key);
                    break request;
                }
                state = shared
                    .wake
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        };
        let image = panic::catch_unwind(AssertUnwindSafe(|| {
            conn.as_ref().map_err(Clone::clone).and_then(|conn| {
                let (sha256, bucket) = &request.key;
                let thumb = thumbs::generate::ensure(conn, &request.path, sha256, *bucket)
                    .map_err(|e| error_chain(&e))?;
                decode(&thumb.jpeg).map_err(|e| e.to_string())
            })
        }))
        .unwrap_or_else(|payload| {
            Err(format!(
                "thumbnail worker panicked: {}",
                crate::decode::panic_message(payload.as_ref())
            ))
        });
        let sent = tx.send(Loaded {
            key: request.key,
            image,
        });
        let ctx = {
            let mut state = shared.lock();
            state.working.remove(&request.key);
            state.ctx.clone()
        };
        if sent.is_err() {
            return;
        }
        if let Some(ctx) = ctx {
            ctx.request_repaint();
        }
    }
}

pub(super) fn decode(jpeg: &[u8]) -> turbojpeg::Result<ColorImage> {
    let image = turbojpeg::decompress(jpeg, PixelFormat::RGB)?;
    Ok(ColorImage::from_rgb(
        [image.width, image.height],
        &image.pixels,
    ))
}

fn error_chain(e: &dyn std::error::Error) -> String {
    let mut out = e.to_string();
    let mut source = e.source();
    while let Some(e) = source {
        out.push_str(": ");
        out.push_str(&e.to_string());
        source = e.source();
    }
    out
}
