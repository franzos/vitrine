//! The ordered set of entries the grid shows and the viewer steps through, with a cursor.

use crate::catalog::images::{Entry, ImageId};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NavSet {
    entries: Vec<Entry>,
    cursor: Option<usize>,
}

impl NavSet {
    #[must_use]
    pub fn new(entries: Vec<Entry>) -> Self {
        Self {
            entries,
            cursor: None,
        }
    }

    /// Replaces the entries, keeping the cursor on the same image when it's still present.
    pub fn replace(&mut self, entries: Vec<Entry>) {
        let current = self.current().map(|e| e.id);
        self.entries = entries;
        self.cursor = current.and_then(|id| self.position(id));
    }

    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Entry> {
        self.entries.get(index)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn ids(&self) -> impl ExactSizeIterator<Item = ImageId> + '_ {
        self.entries.iter().map(|e| e.id)
    }

    #[must_use]
    pub fn position(&self, id: ImageId) -> Option<usize> {
        self.entries.iter().position(|e| e.id == id)
    }

    #[must_use]
    pub fn cursor(&self) -> Option<usize> {
        self.cursor
    }

    #[must_use]
    pub fn current(&self) -> Option<&Entry> {
        self.cursor.and_then(|i| self.entries.get(i))
    }

    /// Moves the cursor to `index`; an index past the end is ignored.
    pub fn set_cursor(&mut self, index: usize) {
        if index < self.entries.len() {
            self.cursor = Some(index);
        }
    }

    /// Moves the cursor by `delta`, stopping at either end; without a cursor, starts at the first entry.
    pub fn step(&mut self, delta: isize) -> Option<usize> {
        let last = self.entries.len().checked_sub(1)?;
        let next = match self.cursor {
            None => 0,
            Some(i) => i.saturating_add_signed(delta).min(last),
        };
        self.cursor = Some(next);
        self.cursor
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::catalog::images::Kind;

    fn entries(ids: &[i64]) -> Vec<Entry> {
        ids.iter()
            .map(|&id| Entry {
                id: ImageId(id),
                path: PathBuf::from(format!("/p/{id}.jpg")),
                sha256: None,
                kind: Kind::Image,
                width: None,
                height: None,
            })
            .collect()
    }

    #[test]
    fn empty_set_has_no_cursor() {
        let mut nav = NavSet::default();
        assert!(nav.is_empty());
        assert_eq!(nav.step(1), None);
        assert_eq!(nav.step(-1), None);
        nav.set_cursor(0);
        assert_eq!(nav.cursor(), None);
        assert!(nav.current().is_none());
    }

    #[test]
    fn steps_stop_at_the_ends() {
        let mut nav = NavSet::new(entries(&[10, 20, 30]));
        assert_eq!(nav.cursor(), None);
        assert_eq!(nav.step(1), Some(0));
        assert_eq!(nav.step(1), Some(1));
        assert_eq!(nav.step(1), Some(2));
        assert_eq!(nav.step(1), Some(2));
        assert_eq!(nav.step(-5), Some(0));
        assert_eq!(nav.step(-1), Some(0));
        assert_eq!(nav.step(4), Some(2));
        assert_eq!(nav.current().map(|e| e.id), Some(ImageId(30)));
        assert_eq!(nav.ids().collect::<Vec<_>>(), [10, 20, 30].map(ImageId));
    }

    #[test]
    fn set_cursor_ignores_out_of_range() {
        let mut nav = NavSet::new(entries(&[1, 2]));
        nav.set_cursor(1);
        nav.set_cursor(2);
        assert_eq!(nav.cursor(), Some(1));
    }

    #[test]
    fn replace_keeps_cursor_on_same_image() {
        let mut nav = NavSet::new(entries(&[1, 2, 3]));
        nav.set_cursor(1);
        nav.replace(entries(&[0, 1, 2, 3]));
        assert_eq!(nav.cursor(), Some(2));
        assert_eq!(nav.position(ImageId(3)), Some(3));
        nav.replace(entries(&[1, 3]));
        assert_eq!(nav.cursor(), None);
    }
}
