use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, PoisonError};
use std::time::Duration;

/// A counting semaphore; a [`Slot`] gives its permit back on drop.
pub(crate) struct Slots {
    free: Mutex<usize>,
    freed: Condvar,
}

impl Slots {
    pub(crate) const fn new(count: usize) -> Self {
        Self {
            free: Mutex::new(count),
            freed: Condvar::new(),
        }
    }

    pub(crate) fn acquire(&self) -> Slot<'_> {
        let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
        while *free == 0 {
            free = self
                .freed
                .wait(free)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *free -= 1;
        Slot(self)
    }

    /// `None` once `cancel` is set, checked before and while waiting.
    pub(crate) fn acquire_unless(&self, cancel: &AtomicBool) -> Option<Slot<'_>> {
        let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            if *free > 0 {
                break;
            }
            free = self
                .freed
                .wait_timeout(free, Duration::from_millis(50))
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        *free -= 1;
        Some(Slot(self))
    }
}

pub(crate) struct Slot<'a>(&'a Slots);

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        *self.0.free.lock().unwrap_or_else(PoisonError::into_inner) += 1;
        self.0.freed.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::Instant;

    use super::*;

    #[test]
    fn slots_block_until_one_is_freed() {
        let slots = Slots::new(2);
        let held = [slots.acquire(), slots.acquire()];
        let taken = AtomicBool::new(false);
        thread::scope(|s| {
            s.spawn(|| {
                let _slot = slots.acquire();
                taken.store(true, Ordering::SeqCst);
            });
            thread::sleep(Duration::from_millis(50));
            assert!(!taken.load(Ordering::SeqCst));
            drop(held);
        });
        assert!(taken.load(Ordering::SeqCst));
    }

    #[test]
    fn slots_acquire_unless_returns_none_once_cancelled() {
        let slots = Slots::new(1);
        let cancel = AtomicBool::new(false);
        let held = slots.acquire_unless(&cancel);
        assert!(held.is_some());
        let start = Instant::now();
        thread::scope(|s| {
            let waiter = s.spawn(|| slots.acquire_unless(&cancel).is_none());
            thread::sleep(Duration::from_millis(50));
            cancel.store(true, Ordering::Relaxed);
            assert!(waiter.join().unwrap());
        });
        assert!(start.elapsed() < Duration::from_secs(1));
        drop(held);
        assert!(slots.acquire_unless(&cancel).is_none());
        assert!(slots.acquire_unless(&AtomicBool::new(false)).is_some());
    }
}
