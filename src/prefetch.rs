//! Decodes the image under the [`NavSet`] cursor and its neighbours ahead of time, on a small pool of threads.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;

use image::DynamicImage;

use crate::catalog::images::{ImageId, Kind};
use crate::nav::NavSet;

pub type Decode = Arc<dyn Fn(&Path) -> anyhow::Result<DynamicImage> + Send + Sync>;

pub type Decoded = Result<Arc<DynamicImage>, String>;

/// Neighbours decoded on each side of the cursor.
pub const RADIUS: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Decoded images kept, including the window around the cursor.
    pub images: usize,
    /// Bytes of decoded pixels kept; neighbours that don't fit aren't decoded, the current image always is.
    pub bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            images: 7,
            bytes: 1 << 30,
        }
    }
}

pub struct Prefetcher {
    shared: Arc<Shared>,
    results: Receiver<Done>,
    limits: Limits,
    cache: HashMap<ImageId, Cached>,
    /// Ids around the cursor, in decode priority order.
    window: Vec<ImageId>,
    tick: u64,
}

struct Cached {
    image: Decoded,
    bytes: u64,
    used: u64,
}

struct Job {
    id: ImageId,
    path: PathBuf,
}

struct Done {
    id: ImageId,
    image: Decoded,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
}

#[derive(Default)]
struct State {
    queue: VecDeque<Job>,
    running: HashMap<ImageId, Arc<AtomicBool>>,
    shutdown: bool,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Prefetcher {
    /// `notify` is called after each finished decode.
    pub fn new(
        decode: Decode,
        workers: usize,
        limits: Limits,
        notify: impl Fn() + Send + Sync + 'static,
    ) -> io::Result<Self> {
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
        });
        let (tx, results) = mpsc::channel();
        let notify: Arc<dyn Fn() + Send + Sync> = Arc::new(notify);
        for i in 0..workers.max(1) {
            let shared = Arc::clone(&shared);
            let tx = tx.clone();
            let decode = Arc::clone(&decode);
            let notify = Arc::clone(&notify);
            thread::Builder::new()
                .name(format!("prefetch-{i}"))
                .spawn(move || work(&shared, decode.as_ref(), &tx, notify.as_ref()))?;
        }
        Ok(Self {
            shared,
            results,
            limits: Limits {
                images: limits.images.max(1),
                bytes: limits.bytes,
            },
            cache: HashMap::new(),
            window: Vec::new(),
            tick: 0,
        })
    }

    /// Moves the window to the cursor of `nav`: queues the current image, then neighbours within [`RADIUS`],
    /// nearest and forward first; cancels decodes outside the window. Videos are skipped.
    pub fn update(&mut self, nav: &NavSet) {
        self.tick += 1;
        let shared = Arc::clone(&self.shared);
        let mut state = shared.lock();
        // workers send under the lock, so every finished decode is already in the channel
        self.receive();

        self.window.clear();
        let mut wanted = Vec::new();
        let mut estimate = 0u64;
        for index in window(nav.cursor(), nav.len()) {
            let Some(entry) = nav.get(index) else {
                continue;
            };
            if entry.kind != Kind::Image || self.window.len() >= self.limits.images {
                continue;
            }
            estimate += self.cache.get(&entry.id).map_or_else(
                || u64::from(entry.width.unwrap_or(0)) * u64::from(entry.height.unwrap_or(0)) * 4,
                |c| c.bytes,
            );
            if !self.window.is_empty() && estimate > self.limits.bytes {
                break;
            }
            self.window.push(entry.id);
            if let Some(cached) = self.cache.get_mut(&entry.id) {
                cached.used = self.tick;
            } else {
                wanted.push(Job {
                    id: entry.id,
                    path: entry.path.clone(),
                });
            }
        }

        for (id, stop) in &state.running {
            stop.store(!self.window.contains(id), Ordering::Relaxed);
        }
        let running = &state.running;
        state.queue = wanted
            .into_iter()
            .filter(|job| !running.contains_key(&job.id))
            .collect();
        let wake = !state.queue.is_empty();
        drop(state);
        if wake {
            self.shared.wake.notify_all();
        }
        self.evict();
    }

    pub fn poll(&mut self) -> bool {
        let arrived = self.receive();
        if arrived {
            self.evict();
        }
        arrived
    }

    #[must_use]
    pub fn get(&self, id: ImageId) -> Option<&Decoded> {
        self.cache.get(&id).map(|c| &c.image)
    }

    pub fn clear(&mut self) {
        let mut state = self.shared.lock();
        state.queue.clear();
        for stop in state.running.values() {
            stop.store(true, Ordering::Relaxed);
        }
        drop(state);
        self.results.try_iter().for_each(drop);
        self.cache.clear();
        self.window.clear();
    }

    #[must_use]
    pub fn pending(&self) -> usize {
        let state = self.shared.lock();
        state.queue.len() + state.running.len()
    }

    fn receive(&mut self) -> bool {
        let mut arrived = false;
        for done in self.results.try_iter() {
            let bytes = done
                .image
                .as_ref()
                .map_or(0, |image| image.as_bytes().len() as u64);
            self.cache.insert(
                done.id,
                Cached {
                    image: done.image,
                    bytes,
                    used: self.tick,
                },
            );
            arrived = true;
        }
        arrived
    }

    /// Drops the least recently wanted images outside the window until the cache is within its limits.
    fn evict(&mut self) {
        loop {
            let bytes: u64 = self.cache.values().map(|c| c.bytes).sum();
            if self.cache.len() <= self.limits.images && bytes <= self.limits.bytes {
                return;
            }
            let victim = self
                .cache
                .iter()
                .filter(|(id, _)| !self.window.contains(id))
                .min_by_key(|(_, c)| c.used)
                .map(|(id, _)| *id);
            match victim {
                Some(id) => self.cache.remove(&id),
                None => return,
            };
        }
    }

    #[cfg(test)]
    fn cached(&self) -> Vec<ImageId> {
        let mut ids: Vec<_> = self.cache.keys().copied().collect();
        ids.sort_unstable_by_key(|id| id.0);
        ids
    }

    #[cfg(test)]
    fn cached_bytes(&self) -> u64 {
        self.cache.values().map(|c| c.bytes).sum()
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        let mut state = self.shared.lock();
        state.shutdown = true;
        for stop in state.running.values() {
            stop.store(true, Ordering::Relaxed);
        }
        drop(state);
        self.shared.wake.notify_all();
    }
}

/// Indices around `cursor` in priority order: the cursor, then forward before backward, nearest first.
fn window(cursor: Option<usize>, len: usize) -> impl Iterator<Item = usize> {
    let offsets = std::iter::once(0).chain((1..=RADIUS as isize).flat_map(|d| [d, -d]));
    offsets.filter_map(move |d| {
        let index = cursor?.checked_add_signed(d)?;
        (index < len).then_some(index)
    })
}

fn work(
    shared: &Shared,
    decode: &(dyn Fn(&Path) -> anyhow::Result<DynamicImage> + Send + Sync),
    tx: &Sender<Done>,
    notify: &(dyn Fn() + Send + Sync),
) {
    loop {
        let (job, stop) = {
            let mut state = shared.lock();
            loop {
                if state.shutdown {
                    return;
                }
                if let Some(job) = state.queue.pop_front() {
                    let stop = Arc::new(AtomicBool::new(false));
                    state.running.insert(job.id, Arc::clone(&stop));
                    break (job, stop);
                }
                state = shared
                    .wake
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        };
        let image = decode(&job.path)
            .map(Arc::new)
            .map_err(|e| format!("{e:#}"));
        let sent = {
            let mut state = shared.lock();
            state.running.remove(&job.id);
            !stop.load(Ordering::Relaxed) && tx.send(Done { id: job.id, image }).is_ok()
        };
        if sent {
            notify();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::time::{Duration, Instant};

    use image::GenericImageView;

    use super::*;
    use crate::catalog::images::Entry;

    fn entry(id: i64, path: PathBuf, size: Option<(u32, u32)>) -> Entry {
        Entry {
            id: ImageId(id),
            path,
            sha256: None,
            kind: Kind::Image,
            width: size.map(|s| s.0),
            height: size.map(|s| s.1),
            duration_ms: None,
        }
    }

    /// Entries `0..n` named `{id}.jpg`, each 10x10.
    fn nav(n: i64) -> NavSet {
        NavSet::new(
            (0..n)
                .map(|id| entry(id, PathBuf::from(format!("{id}.jpg")), Some((10, 10))))
                .collect(),
        )
    }

    fn id_of(path: &Path) -> i64 {
        path.file_stem().unwrap().to_str().unwrap().parse().unwrap()
    }

    /// A decoder of 10x10 RGBA images that records each call and waits while `gate` is closed.
    #[derive(Clone, Default)]
    struct Fake {
        calls: Arc<Mutex<Vec<i64>>>,
        gate: Arc<(Mutex<bool>, Condvar)>,
    }

    impl Fake {
        fn open() -> Self {
            let fake = Self::default();
            fake.release();
            fake
        }

        fn decode(&self) -> Decode {
            let fake = self.clone();
            Arc::new(move |path| {
                fake.calls.lock().unwrap().push(id_of(path));
                let (open, wake) = &*fake.gate;
                let _open = wake
                    .wait_while(open.lock().unwrap(), |open| !*open)
                    .unwrap();
                if path.extension().is_some_and(|e| e == "bad") {
                    anyhow::bail!("not an image");
                }
                Ok(DynamicImage::new_rgba8(10, 10))
            })
        }

        fn release(&self) {
            *self.gate.0.lock().unwrap() = true;
            self.gate.1.notify_all();
        }

        fn calls(&self) -> Vec<i64> {
            self.calls.lock().unwrap().clone()
        }

        fn wait_for_calls(&self, n: usize) {
            let deadline = Instant::now() + Duration::from_secs(10);
            while self.calls.lock().unwrap().len() < n {
                assert!(Instant::now() < deadline, "{:?}", self.calls());
                thread::sleep(Duration::from_millis(2));
            }
        }
    }

    fn settle(prefetcher: &mut Prefetcher) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while prefetcher.pending() > 0 {
            assert!(Instant::now() < deadline, "decodes never finished");
            thread::sleep(Duration::from_millis(2));
        }
        prefetcher.poll();
    }

    fn ids(ids: &[i64]) -> Vec<ImageId> {
        ids.iter().copied().map(ImageId).collect()
    }

    #[test]
    fn window_order_is_current_then_nearest_forward_first() {
        assert_eq!(window(Some(5), 10).collect::<Vec<_>>(), [5, 6, 4, 7, 3]);
        assert_eq!(window(Some(0), 10).collect::<Vec<_>>(), [0, 1, 2]);
        assert_eq!(window(Some(9), 10).collect::<Vec<_>>(), [9, 8, 7]);
        assert_eq!(window(None, 10).count(), 0);
    }

    #[test]
    fn decodes_current_and_neighbours_from_fixtures() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let files = [
            ("jpeg.jpg", (64, 48)),
            ("png.png", (60, 44)),
            ("webp.webp", (56, 40)),
            ("gif.gif", (52, 36)),
            ("tiff.tif", (48, 32)),
            ("heic.heic", (68, 52)),
            ("jxl.jxl", (74, 58)),
        ];
        let mut nav = NavSet::new(
            (0i64..)
                .zip(files)
                .map(|(id, (name, _))| entry(id, fixtures.join(name), None))
                .collect(),
        );
        nav.set_cursor(3);
        let decode: Decode = Arc::new(crate::decode::decode);
        let mut prefetcher = Prefetcher::new(decode, 2, Limits::default(), || {}).unwrap();
        prefetcher.update(&nav);
        settle(&mut prefetcher);

        assert_eq!(prefetcher.cached(), ids(&[1, 2, 3, 4, 5]));
        for id in 1..=5 {
            let image = prefetcher.get(ImageId(id)).unwrap().as_ref().unwrap();
            assert_eq!(image.dimensions(), files[id as usize].1, "{id}");
        }
    }

    #[test]
    fn current_is_decoded_first() {
        let fake = Fake::open();
        let mut nav = nav(10);
        nav.set_cursor(5);
        let mut prefetcher = Prefetcher::new(fake.decode(), 1, Limits::default(), || {}).unwrap();
        prefetcher.update(&nav);
        settle(&mut prefetcher);
        assert_eq!(fake.calls(), [5, 6, 4, 7, 3]);
        assert_eq!(prefetcher.cached(), ids(&[3, 4, 5, 6, 7]));
    }

    #[test]
    fn jump_cancels_stale_decodes() {
        let fake = Fake::default();
        let mut nav = nav(20);
        nav.set_cursor(0);
        let notified = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&notified);
        let mut prefetcher = Prefetcher::new(fake.decode(), 1, Limits::default(), move || {
            counter.fetch_add(1, Ordering::SeqCst);
        })
        .unwrap();
        prefetcher.update(&nav);
        fake.wait_for_calls(1);

        nav.set_cursor(10);
        prefetcher.update(&nav);
        fake.release();
        settle(&mut prefetcher);

        // 0 was running and is dropped, 1 and 2 were queued and never start
        assert_eq!(fake.calls(), [0, 10, 11, 9, 12, 8]);
        assert_eq!(prefetcher.cached(), ids(&[8, 9, 10, 11, 12]));
        assert_eq!(notified.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn stepping_back_into_a_running_decode_keeps_it() {
        let fake = Fake::default();
        let mut nav = nav(20);
        nav.set_cursor(0);
        let mut prefetcher = Prefetcher::new(fake.decode(), 1, Limits::default(), || {}).unwrap();
        prefetcher.update(&nav);
        fake.wait_for_calls(1);
        nav.set_cursor(10);
        prefetcher.update(&nav);
        nav.set_cursor(0);
        prefetcher.update(&nav);
        fake.release();
        settle(&mut prefetcher);

        assert_eq!(fake.calls(), [0, 1, 2]);
        assert_eq!(prefetcher.cached(), ids(&[0, 1, 2]));
    }

    #[test]
    fn cache_is_bounded_by_count() {
        let fake = Fake::open();
        let mut nav = nav(30);
        let mut prefetcher = Prefetcher::new(fake.decode(), 2, Limits::default(), || {}).unwrap();
        for cursor in 0..30 {
            nav.set_cursor(cursor);
            prefetcher.update(&nav);
            settle(&mut prefetcher);
            let cached = prefetcher.cached();
            assert!(cached.len() <= 7, "{cursor}: {cached:?}");
            assert!(cached.contains(&ImageId(cursor as i64)));
        }
        // the most recently left images are kept: 23 to 29
        let calls = fake.calls().len();
        nav.set_cursor(24);
        prefetcher.update(&nav);
        settle(&mut prefetcher);
        assert_eq!(fake.calls()[calls..], [22]);
    }

    #[test]
    fn cache_is_bounded_by_bytes() {
        let fake = Fake::open();
        let mut nav = nav(30);
        // each image is 400 bytes: the current one and one neighbour fit
        let limits = Limits {
            images: 7,
            bytes: 1000,
        };
        let mut prefetcher = Prefetcher::new(fake.decode(), 2, limits, || {}).unwrap();
        for cursor in [5, 6, 7, 12, 11] {
            nav.set_cursor(cursor);
            prefetcher.update(&nav);
            settle(&mut prefetcher);
            assert!(prefetcher.cached_bytes() <= 1000, "{cursor}");
            let cached = prefetcher.cached();
            assert!(cached.contains(&ImageId(cursor as i64)), "{cursor}");
            assert!(cached.contains(&ImageId(cursor as i64 + 1)), "{cursor}");
        }
    }

    #[test]
    fn current_is_decoded_even_over_budget() {
        let fake = Fake::open();
        let mut nav = nav(5);
        nav.set_cursor(2);
        let limits = Limits {
            images: 7,
            bytes: 100,
        };
        let mut prefetcher = Prefetcher::new(fake.decode(), 1, limits, || {}).unwrap();
        prefetcher.update(&nav);
        settle(&mut prefetcher);
        assert_eq!(fake.calls(), [2]);
        assert_eq!(prefetcher.cached(), ids(&[2]));
    }

    #[test]
    fn errors_are_kept_and_not_retried() {
        let fake = Fake::open();
        let mut nav = NavSet::new(vec![entry(0, PathBuf::from("0.bad"), None)]);
        nav.set_cursor(0);
        let mut prefetcher = Prefetcher::new(fake.decode(), 1, Limits::default(), || {}).unwrap();
        prefetcher.update(&nav);
        settle(&mut prefetcher);
        prefetcher.update(&nav);
        settle(&mut prefetcher);
        assert_eq!(fake.calls(), [0]);
        let error = prefetcher.get(ImageId(0)).unwrap().as_ref().unwrap_err();
        assert!(error.contains("not an image"), "{error}");
    }

    #[test]
    fn videos_are_skipped_and_clear_empties_the_cache() {
        let fake = Fake::open();
        let mut entries: Vec<_> = nav(3).entries().to_vec();
        entries[1].kind = Kind::Video;
        let mut nav = NavSet::new(entries);
        nav.set_cursor(0);
        let mut prefetcher = Prefetcher::new(fake.decode(), 1, Limits::default(), || {}).unwrap();
        prefetcher.update(&nav);
        settle(&mut prefetcher);
        assert_eq!(fake.calls(), [0, 2]);

        prefetcher.clear();
        assert!(prefetcher.cached().is_empty());
        assert_eq!(prefetcher.pending(), 0);
    }
}
