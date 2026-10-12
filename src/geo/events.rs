//! Events: runs of images close in capture time, computed on demand and not stored.

use std::collections::HashSet;

use crate::catalog::images::ImageId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum When {
    /// Seconds since the Unix epoch of a naive local `taken_at`.
    Timed(i64),
    /// Days since the Unix epoch of a `YYYY-MM-DD` file name prefix.
    Day(i64),
}

impl When {
    /// `taken_at` wins; an invalid one falls back to the file stem.
    #[must_use]
    pub fn of(taken_at: Option<&str>, stem: &str) -> Option<Self> {
        taken_at
            .and_then(timed)
            .map(Self::Timed)
            .or_else(|| day(stem).map(Self::Day))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Item {
    pub id: ImageId,
    pub when: When,
}

/// Members of every event containing an anchor, each once, in capture order with day events last.
#[must_use]
pub fn events_of(items: &[Item], gap_secs: i64, anchors: &[ImageId]) -> Vec<ImageId> {
    let anchors: HashSet<ImageId> = anchors.iter().copied().collect();
    let mut timed = Vec::new();
    let mut days = Vec::new();
    for item in items {
        match item.when {
            When::Timed(secs) => timed.push((secs, item.id)),
            When::Day(day) => days.push((day, item.id)),
        }
    }
    timed.sort_unstable();
    days.sort_unstable();

    let mut out = Vec::new();
    let mut start = 0;
    while start < timed.len() {
        let mut end = start + 1;
        while end < timed.len() && timed[end].0 - timed[end - 1].0 <= gap_secs {
            end += 1;
        }
        push_if_anchored(&mut out, &timed[start..end], &anchors);
        start = end;
    }
    for run in days.chunk_by(|a, b| a.0 == b.0) {
        push_if_anchored(&mut out, run, &anchors);
    }
    out
}

fn push_if_anchored(out: &mut Vec<ImageId>, run: &[(i64, ImageId)], anchors: &HashSet<ImageId>) {
    if run.iter().any(|(_, id)| anchors.contains(id)) {
        out.extend(run.iter().map(|&(_, id)| id));
    }
}

/// `YYYY-MM-DDTHH:MM:SS`, as the catalog stores `taken_at`.
fn timed(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() != 19 || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let days = date(&b[..10])?;
    let (h, m, sec) = (num(&b[11..13])?, num(&b[14..16])?, num(&b[17..19])?);
    if h >= 24 || m >= 60 || sec >= 61 {
        return None;
    }
    Some(days * 86_400 + h * 3600 + m * 60 + sec)
}

/// A `YYYY-MM-DD` prefix not followed by another digit.
fn day(stem: &str) -> Option<i64> {
    let b = stem.as_bytes();
    if b.len() < 10 || b.get(10).is_some_and(u8::is_ascii_digit) {
        return None;
    }
    date(&b[..10])
}

fn date(b: &[u8]) -> Option<i64> {
    if b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let (y, m, d) = (num(&b[..4])?, num(&b[5..7])?, num(&b[8..10])?);
    if y == 0 || !(1..=12).contains(&m) || !(1..=month_len(y, m)).contains(&d) {
        return None;
    }
    Some(days_from_civil(y, m, d))
}

fn num(b: &[u8]) -> Option<i64> {
    b.iter().try_fold(0, |n, &c| {
        c.is_ascii_digit().then(|| n * 10 + i64::from(c - b'0'))
    })
}

fn month_len(y: i64, m: i64) -> i64 {
    match m {
        2 if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Howard Hinnant's `days_from_civil`, proleptic Gregorian.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * ((m + 9) % 12) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    const GAP: i64 = 6 * 3600;

    fn timed_item(id: i64, secs: i64) -> Item {
        Item {
            id: ImageId(id),
            when: When::Timed(secs),
        }
    }

    fn day_item(id: i64, day: i64) -> Item {
        Item {
            id: ImageId(id),
            when: When::Day(day),
        }
    }

    fn ids(raw: &[i64]) -> Vec<ImageId> {
        raw.iter().copied().map(ImageId).collect()
    }

    #[test]
    fn days_from_civil_matches_known_dates() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 3, 1), 11_017);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        assert_eq!(
            timed("2024-05-01T10:11:12"),
            Some(1_714_558_272),
            "2024-05-01T10:11:12Z"
        );
    }

    #[test]
    fn invalid_dates_are_dropped() {
        for s in [
            "2023-02-29T00:00:00",
            "2024-04-31T00:00:00",
            "2024-13-01T00:00:00",
            "2024-00-01T00:00:00",
            "2024-01-00T00:00:00",
            "0000-01-01T00:00:00",
            "2024-01-01T24:00:00",
            "2024-01-01T00:60:00",
            "2024:01:01 00:00:00",
            "2024-01-01T00:00",
            "2024-1a-01T00:00:00",
        ] {
            assert_eq!(timed(s), None, "{s}");
        }
        assert!(timed("2024-02-29T23:59:60").is_some());
        assert!(timed("2000-02-29T00:00:00").is_some());
        assert_eq!(timed("1900-02-29T00:00:00"), None);

        assert_eq!(day("2020-05-07_00002"), Some(days_from_civil(2020, 5, 7)));
        assert_eq!(day("2020-05-07"), Some(days_from_civil(2020, 5, 7)));
        for stem in [
            "2021-02-29_1",
            "2020-05-32_1",
            "2020-5-07_1",
            "20200507",
            "2020-05-071",
            "IMG_1234",
            "",
        ] {
            assert_eq!(day(stem), None, "{stem}");
        }
        assert_eq!(When::of(None, "IMG_1234"), None);
        assert_eq!(When::of(Some("garbage"), "IMG_1234"), None);
    }

    #[test]
    fn exif_time_wins_over_a_filename_date() {
        let at = "2024-05-01T10:11:12";
        assert_eq!(
            When::of(Some(at), "2020-05-07_00002"),
            Some(When::Timed(1_714_558_272))
        );
        assert_eq!(
            When::of(Some("2024-02-30T10:11:12"), "2020-05-07_00002"),
            Some(When::Day(days_from_civil(2020, 5, 7)))
        );
    }

    #[test]
    fn a_gap_equal_to_the_limit_stays_together() {
        let items = [timed_item(1, 0), timed_item(2, GAP), timed_item(3, 2 * GAP)];
        assert_eq!(events_of(&items, GAP, &ids(&[1])), ids(&[1, 2, 3]));
    }

    #[test]
    fn a_larger_gap_splits() {
        let items = [
            timed_item(3, GAP + 1 + 10),
            timed_item(1, 0),
            timed_item(2, 10),
        ];
        assert_eq!(events_of(&items, GAP, &ids(&[1])), ids(&[1, 2]));
        assert_eq!(events_of(&items, GAP, &ids(&[3])), ids(&[3]));
        assert!(events_of(&items, GAP, &ids(&[9])).is_empty());
        assert!(events_of(&items, GAP, &[]).is_empty());
    }

    #[test]
    fn day_items_group_by_day_and_never_merge_with_timed_items() {
        let d = days_from_civil(2020, 5, 7);
        let items = [
            day_item(1, d),
            day_item(2, d),
            day_item(3, d + 1),
            timed_item(4, d * 86_400 + 3600),
            timed_item(5, d * 86_400 + 7200),
        ];
        assert_eq!(events_of(&items, GAP, &ids(&[2])), ids(&[1, 2]));
        assert_eq!(events_of(&items, GAP, &ids(&[3])), ids(&[3]));
        assert_eq!(events_of(&items, GAP, &ids(&[4])), ids(&[4, 5]));
    }

    #[test]
    fn union_over_several_anchors() {
        let items = [
            timed_item(1, 0),
            timed_item(2, 100),
            timed_item(3, 10 * GAP),
            timed_item(4, 20 * GAP),
            day_item(5, 1),
            day_item(6, 1),
        ];
        assert_eq!(
            events_of(&items, GAP, &ids(&[6, 2, 1, 4])),
            ids(&[1, 2, 4, 5, 6])
        );
    }
}
