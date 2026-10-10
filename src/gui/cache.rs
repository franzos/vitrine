use std::collections::HashMap;
use std::hash::Hash;

pub struct Lru<K, V> {
    capacity: usize,
    tick: u64,
    entries: HashMap<K, (V, u64)>,
}

impl<K: Eq + Hash + Clone, V> Lru<K, V> {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            tick: 0,
            entries: HashMap::with_capacity(capacity),
        }
    }

    /// Marks `key` as most recently used.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.tick += 1;
        let tick = self.tick;
        self.entries.get_mut(key).map(|(value, used)| {
            *used = tick;
            &*value
        })
    }

    pub fn insert(&mut self, key: K, value: V) {
        self.tick += 1;
        if !self.entries.contains_key(&key)
            && self.entries.len() >= self.capacity
            && let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(k, _)| k.clone())
        {
            self.entries.remove(&oldest);
        }
        self.entries.insert(key, (value, self.tick));
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_least_recently_used() {
        let mut lru = Lru::new(2);
        lru.insert('a', 1);
        lru.insert('b', 2);
        assert_eq!(lru.get(&'a'), Some(&1));
        lru.insert('c', 3);
        assert_eq!(lru.get(&'b'), None);
        assert_eq!(lru.get(&'a'), Some(&1));
        lru.insert('c', 4);
        assert_eq!(lru.len(), 2);
        lru.insert('d', 5);
        assert_eq!(lru.get(&'a'), None);
        assert_eq!(lru.get(&'c'), Some(&4));
    }
}
