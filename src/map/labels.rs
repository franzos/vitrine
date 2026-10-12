//! City label thresholds and collision-free placement.

use super::Rect;

/// Up to this level the label candidates come from the in-memory list; above it from a box query.
pub const LIST_MAX_LEVEL: u32 = 10;
/// The population threshold of the in-memory list, `min_population(LIST_MAX_LEVEL)`.
pub const LIST_MIN_POPULATION: u32 = 10_000;

/// A labelled point on the map.
#[derive(Clone, Debug, PartialEq)]
pub struct City {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub population: u32,
}

/// The cities inside the box with at least `min_pop` inhabitants, in the order of `cities`,
/// which must be most populous first.
pub fn in_box(
    cities: &[City],
    min_lat: f64,
    max_lat: f64,
    min_lon: f64,
    max_lon: f64,
    min_pop: u32,
) -> impl Iterator<Item = &City> {
    cities
        .iter()
        .take_while(move |c| c.population >= min_pop)
        .filter(move |c| {
            (min_lat..=max_lat).contains(&c.lat) && (min_lon..=max_lon).contains(&c.lon)
        })
}

/// The smallest population a city needs to get a label at `level`; `None` below level 4.
#[must_use]
pub fn min_population(level: u32) -> Option<u32> {
    match level {
        0..=3 => None,
        4 => Some(1_000_000),
        5 => Some(500_000),
        6 => Some(200_000),
        7 => Some(100_000),
        8 => Some(50_000),
        9 => Some(20_000),
        10 => Some(10_000),
        _ => Some(1000),
    }
}

/// Keeps the candidates, in order, whose rect doesn't intersect one in `occupied`, and adds
/// each kept rect to `occupied`.
pub fn place<T>(
    candidates: impl IntoIterator<Item = (T, Rect)>,
    occupied: &mut Vec<Rect>,
) -> Vec<T> {
    let mut placed = Vec::new();
    for (item, rect) in candidates {
        if occupied.iter().any(|o| intersects(o, &rect)) {
            continue;
        }
        occupied.push(rect);
        placed.push(item);
    }
    placed
}

fn intersects(a: &Rect, b: &Rect) -> bool {
    a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: f64, y: f64) -> Rect {
        Rect {
            x,
            y,
            w: 40.0,
            h: 10.0,
        }
    }

    fn city(name: &str, lat: f64, lon: f64, population: u32) -> City {
        City {
            name: name.into(),
            lat,
            lon,
            population,
        }
    }

    #[test]
    fn in_box_keeps_the_box_threshold_and_order() {
        let cities = [
            city("Paris", 48.85, 2.35, 2_138_551),
            city("München", 48.14, 11.58, 1_260_391),
            city("Lisbon", 38.72, -9.14, 517_802),
            city("Porto", 41.15, -8.61, 249_633),
            city("Braga", 41.55, -8.42, 136_885),
            city("Faro", 37.02, -7.93, 41_355),
        ];
        let names = |found: Vec<&City>| found.iter().map(|c| c.name.clone()).collect::<Vec<_>>();
        assert_eq!(
            names(in_box(&cities, 37.0, 42.0, -10.0, -7.0, 0).collect()),
            ["Lisbon", "Porto", "Braga", "Faro"]
        );
        assert_eq!(
            names(in_box(&cities, 37.0, 42.0, -10.0, -7.0, 136_885).collect()),
            ["Lisbon", "Porto", "Braga"]
        );
        assert_eq!(
            names(in_box(&cities, 38.72, 41.15, -9.14, -8.61, 0).collect()),
            ["Lisbon", "Porto"]
        );
        assert!(
            in_box(&cities, 38.73, 41.14, -9.14, -8.61, 0)
                .next()
                .is_none()
        );
    }

    #[test]
    fn list_threshold_matches_its_level() {
        assert_eq!(min_population(LIST_MAX_LEVEL), Some(LIST_MIN_POPULATION));
    }

    #[test]
    fn thresholds_by_level() {
        assert_eq!(min_population(0), None);
        assert_eq!(min_population(3), None);
        assert_eq!(min_population(4), Some(1_000_000));
        assert_eq!(min_population(5), Some(500_000));
        assert_eq!(min_population(6), Some(200_000));
        assert_eq!(min_population(7), Some(100_000));
        assert_eq!(min_population(8), Some(50_000));
        assert_eq!(min_population(9), Some(20_000));
        assert_eq!(min_population(10), Some(10_000));
        assert_eq!(min_population(11), Some(1000));
        assert_eq!(min_population(17), Some(1000));
    }

    #[test]
    fn overlapping_labels_are_skipped_in_order() {
        let mut occupied = Vec::new();
        let placed = place(
            [
                ("a", rect(0.0, 0.0)),
                ("b", rect(30.0, 5.0)),
                ("c", rect(40.0, 0.0)),
                ("d", rect(0.0, 10.0)),
                ("e", rect(70.0, 5.0)),
            ],
            &mut occupied,
        );
        assert_eq!(placed, ["a", "c", "d"]);
        assert_eq!(occupied, [rect(0.0, 0.0), rect(40.0, 0.0), rect(0.0, 10.0)]);
    }

    #[test]
    fn occupied_rects_block_labels() {
        let mut occupied = vec![rect(0.0, 0.0)];
        let placed = place(
            [("a", rect(10.0, 2.0)), ("b", rect(100.0, 0.0))],
            &mut occupied,
        );
        assert_eq!(placed, ["b"]);
        assert_eq!(occupied.len(), 2);
    }
}
