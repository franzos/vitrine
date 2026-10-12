//! The ordered set of entries the grid shows and the viewer steps through, with a cursor and a selection.

use std::collections::{BTreeSet, HashSet};

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

    /// Like [`Self::replace`], but puts the cursor's image back at its index when `entries` lacks it; returns whether it did.
    pub fn replace_keeping_current(&mut self, mut entries: Vec<Entry>) -> bool {
        let missing = self
            .cursor
            .zip(self.current().cloned())
            .filter(|(_, current)| entries.iter().all(|e| e.id != current.id));
        let Some((index, current)) = missing else {
            self.replace(entries);
            return false;
        };
        let index = index.min(entries.len());
        entries.insert(index, current);
        self.entries = entries;
        self.cursor = Some(index);
        true
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

/// Images picked in the grid, independent of the cursor.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Selection {
    ids: BTreeSet<ImageId>,
    /// Where a range starts.
    anchor: Option<ImageId>,
}

impl Selection {
    #[must_use]
    pub fn ids(&self) -> &BTreeSet<ImageId> {
        &self.ids
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    #[must_use]
    pub fn contains(&self, id: ImageId) -> bool {
        self.ids.contains(&id)
    }

    #[must_use]
    pub fn anchor(&self) -> Option<ImageId> {
        self.anchor
    }

    /// Adds or removes `id` and makes it the anchor.
    pub fn toggle(&mut self, id: ImageId) {
        if !self.ids.remove(&id) {
            self.ids.insert(id);
        }
        self.anchor = Some(id);
    }

    /// Replaces the selection with every entry from the anchor to `to`, inclusive, in `nav` order.
    ///
    /// Without an anchor in `nav`, the range starts at the cursor, which becomes the anchor.
    pub fn select_range(&mut self, nav: &NavSet, to: ImageId) {
        let Some(end) = nav.position(to) else {
            return;
        };
        let start = self
            .anchor
            .and_then(|id| nav.position(id))
            .or(nav.cursor())
            .unwrap_or(end);
        let (low, high) = (start.min(end), start.max(end));
        self.ids = nav.entries()[low..=high].iter().map(|e| e.id).collect();
        self.anchor = nav.get(start).map(|e| e.id);
    }

    pub fn select_all(&mut self, nav: &NavSet) {
        self.ids = nav.ids().collect();
    }

    /// Drops ids that are no longer in `nav`, and the anchor if it left.
    pub fn prune(&mut self, nav: &NavSet) {
        let present: HashSet<ImageId> = nav.ids().collect();
        self.ids.retain(|id| present.contains(id));
        self.anchor = self.anchor.filter(|id| present.contains(id));
    }

    pub fn clear(&mut self) {
        self.ids.clear();
        self.anchor = None;
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
                duration_ms: None,
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

    #[test]
    fn replace_keeping_current_puts_the_cursor_image_back() {
        let id_list = |nav: &NavSet| nav.entries().iter().map(|e| e.id.0).collect::<Vec<_>>();
        let mut nav = NavSet::new(entries(&[1, 2, 3]));
        nav.set_cursor(1);
        assert!(nav.replace_keeping_current(entries(&[1, 3])));
        assert_eq!(id_list(&nav), [1, 2, 3]);
        assert_eq!(nav.cursor(), Some(1));
        assert!(!nav.replace_keeping_current(entries(&[0, 2])));
        assert_eq!(nav.cursor(), Some(1));
        nav.set_cursor(1);
        assert!(nav.replace_keeping_current(entries(&[])));
        assert_eq!(id_list(&nav), [2]);
        assert_eq!(nav.cursor(), Some(0));
    }

    fn ids(ids: &[i64]) -> BTreeSet<ImageId> {
        ids.iter().copied().map(ImageId).collect()
    }

    #[test]
    fn toggle_adds_removes_and_moves_the_anchor() {
        let mut sel = Selection::default();
        sel.toggle(ImageId(2));
        sel.toggle(ImageId(4));
        assert_eq!(sel.ids(), &ids(&[2, 4]));
        assert_eq!(sel.anchor(), Some(ImageId(4)));
        sel.toggle(ImageId(2));
        assert_eq!(sel.ids(), &ids(&[4]));
        assert_eq!(sel.anchor(), Some(ImageId(2)));
        assert!(sel.contains(ImageId(4)));
        assert!(!sel.contains(ImageId(2)));
    }

    #[test]
    fn range_runs_in_set_order_in_either_direction() {
        let nav = NavSet::new(entries(&[50, 10, 40, 20, 30]));
        let mut sel = Selection::default();
        sel.toggle(ImageId(40));
        sel.select_range(&nav, ImageId(30));
        assert_eq!(sel.ids(), &ids(&[40, 20, 30]));
        sel.select_range(&nav, ImageId(50));
        assert_eq!(sel.ids(), &ids(&[50, 10, 40]));
        assert_eq!(sel.anchor(), Some(ImageId(40)));
        sel.select_range(&nav, ImageId(40));
        assert_eq!(sel.ids(), &ids(&[40]));
    }

    #[test]
    fn range_replaces_the_selection() {
        let nav = NavSet::new(entries(&[1, 2, 3, 4, 5]));
        let mut sel = Selection::default();
        sel.toggle(ImageId(1));
        sel.toggle(ImageId(4));
        sel.select_range(&nav, ImageId(5));
        assert_eq!(sel.ids(), &ids(&[4, 5]));
    }

    #[test]
    fn range_without_anchor_starts_at_the_cursor() {
        let mut nav = NavSet::new(entries(&[1, 2, 3, 4]));
        let mut sel = Selection::default();
        sel.select_range(&nav, ImageId(3));
        assert_eq!(sel.ids(), &ids(&[3]));

        sel.clear();
        nav.set_cursor(3);
        sel.select_range(&nav, ImageId(2));
        assert_eq!(sel.ids(), &ids(&[2, 3, 4]));
        assert_eq!(sel.anchor(), Some(ImageId(4)));

        sel.select_range(&nav, ImageId(99));
        assert_eq!(sel.ids(), &ids(&[2, 3, 4]));
    }

    #[test]
    fn select_all_takes_the_whole_set() {
        let nav = NavSet::new(entries(&[3, 1, 2]));
        let mut sel = Selection::default();
        sel.toggle(ImageId(1));
        sel.select_all(&nav);
        assert_eq!(sel.ids(), &ids(&[1, 2, 3]));
        assert_eq!(sel.anchor(), Some(ImageId(1)));
        sel.clear();
        assert!(sel.is_empty());
        assert_eq!(sel.anchor(), None);
    }

    #[test]
    fn prune_drops_only_what_left_the_set() {
        let mut nav = NavSet::new(entries(&[1, 2, 3, 4]));
        let mut sel = Selection::default();
        sel.select_all(&nav);
        sel.toggle(ImageId(2));
        sel.toggle(ImageId(2));
        nav.replace(entries(&[0, 2, 3]));
        sel.prune(&nav);
        assert_eq!(sel.ids(), &ids(&[2, 3]));
        assert_eq!(sel.anchor(), Some(ImageId(2)));

        nav.replace(entries(&[3]));
        sel.prune(&nav);
        assert_eq!(sel.ids(), &ids(&[3]));
        assert_eq!(sel.len(), 1);
        assert_eq!(sel.anchor(), None);
    }
}
