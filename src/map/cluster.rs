//! Grid clustering of world points, one cell of `CELL_PX` screen points per cluster at a level.

use std::collections::HashMap;

use super::projection::{Rect, Viewport};

/// The highest clustering level; deeper zooms reuse it.
pub const MAX_LEVEL: u32 = 17;
const CELL_PX: f64 = 60.0;
const TILE: f64 = 256.0;

/// An image at world coordinates from `project`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub id: i64,
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cluster {
    /// Cell coordinates at the clustering level.
    pub key: (u32, u32),
    pub count: usize,
    /// Mean of the members in world coordinates.
    pub x: f64,
    pub y: f64,
    /// Id of the first member in input order.
    pub first: i64,
    /// World bbox of the members.
    pub min: (f64, f64),
    pub max: (f64, f64),
}

/// Clustering level for a zoom: rounded, within `0..=MAX_LEVEL`.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn level(zoom: f64) -> u32 {
    zoom.round().clamp(0.0, f64::from(MAX_LEVEL)) as u32
}

/// Largest bubble radius in screen points.
pub const MAX_RADIUS: f64 = 30.0;

/// Bubble radius in screen points for a cluster of `count` members.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn radius(count: usize) -> f64 {
    (14.0 + 4.0 * (count as f64).ln()).clamp(14.0, MAX_RADIUS)
}

fn cell_size(level: u32) -> f64 {
    CELL_PX / (TILE * f64::from(level.min(MAX_LEVEL)).exp2())
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn key(point: &Point, size: f64) -> (u32, u32) {
    ((point.x / size) as u32, (point.y / size) as u32)
}

/// Clusters `points` at `level`, sorted by ascending `(count, key)`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn cluster(points: &[Point], level: u32) -> Vec<Cluster> {
    let size = cell_size(level);
    let mut cells: HashMap<(u32, u32), Cluster> = HashMap::with_capacity(points.len());
    for p in points {
        let key = key(p, size);
        cells
            .entry(key)
            .and_modify(|c| {
                c.count += 1;
                c.x += p.x;
                c.y += p.y;
                c.min = (c.min.0.min(p.x), c.min.1.min(p.y));
                c.max = (c.max.0.max(p.x), c.max.1.max(p.y));
            })
            .or_insert(Cluster {
                key,
                count: 1,
                x: p.x,
                y: p.y,
                first: p.id,
                min: (p.x, p.y),
                max: (p.x, p.y),
            });
    }
    let mut clusters: Vec<Cluster> = cells
        .into_values()
        .map(|mut c| {
            c.x /= c.count as f64;
            c.y /= c.count as f64;
            c
        })
        .collect();
    clusters.sort_unstable_by_key(|c| (c.count, c.key));
    clusters
}

/// The points of the cluster `key` at `level`, in input order.
pub fn members(points: &[Point], level: u32, key: (u32, u32)) -> impl Iterator<Item = &Point> {
    let size = cell_size(level);
    points.iter().filter(move |p| self::key(p, size) == key)
}

/// The topmost cluster whose bubble contains `pointer`, given `clusters` drawn in slice order.
#[must_use]
pub fn hit<'a>(
    clusters: &'a [Cluster],
    view: &Viewport,
    rect: Rect,
    pointer: (f64, f64),
) -> Option<&'a Cluster> {
    clusters.iter().rev().find(|c| {
        let (sx, sy) = view.to_screen((c.x, c.y), rect);
        let r = radius(c.count);
        (sx - pointer.0).powi(2) + (sy - pointer.1).powi(2) <= r * r
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(id: i64, x: f64, y: f64) -> Point {
        Point { id, x, y }
    }

    #[test]
    fn level_rounds_and_caps() {
        assert_eq!(level(3.4), 3);
        assert_eq!(level(3.5), 4);
        assert_eq!(level(-1.2), 0);
        assert_eq!(level(16.0), 16);
        assert_eq!(level(21.0), MAX_LEVEL);
    }

    #[test]
    fn radius_is_clamped() {
        assert!((radius(1) - 14.0).abs() < f64::EPSILON);
        assert!((radius(10) - (14.0 + 4.0 * 10f64.ln())).abs() < 1e-12);
        assert!((radius(1_000_000) - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn membership_is_stable_for_a_level() {
        // Level 4: cells are 60 / 4096 world units wide.
        let points = vec![
            pt(1, 0.500, 0.500),
            pt(2, 0.505, 0.502),
            pt(3, 0.200, 0.300),
            pt(4, 0.501, 0.509),
        ];
        let a = cluster(&points, 4);
        assert_eq!(a, cluster(&points, 4));
        assert_eq!(a.len(), 2);
        assert_eq!(a.iter().map(|c| c.count).sum::<usize>(), points.len());

        let big = a[1];
        let ids: Vec<i64> = members(&points, 4, big.key).map(|p| p.id).collect();
        assert_eq!(ids, [1, 2, 4]);

        let mut more = points.clone();
        more.push(pt(5, 0.9, 0.1));
        let b = cluster(&more, 4);
        assert!(b.contains(&a[0]) && b.contains(&big));
    }

    #[test]
    fn first_is_the_first_member_in_input_order() {
        let points = vec![pt(30, 0.41, 0.41), pt(10, 0.40, 0.40), pt(20, 0.42, 0.42)];
        let clusters = cluster(&points, 0);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].first, 30);
    }

    #[test]
    fn bbox_and_mean() {
        let points = vec![pt(1, 0.40, 0.44), pt(2, 0.42, 0.40), pt(3, 0.41, 0.42)];
        let c = cluster(&points, 0)[0];
        assert_eq!(c.count, 3);
        assert!((c.x - 0.41).abs() < 1e-12 && (c.y - 0.42).abs() < 1e-12);
        assert_eq!(c.min, (0.40, 0.40));
        assert_eq!(c.max, (0.42, 0.44));
    }

    #[test]
    fn points_across_a_cell_edge_split() {
        let size = cell_size(10);
        let edge = 1000.0 * size;
        let points = vec![
            pt(1, edge - size * 1e-3, 0.5),
            pt(2, edge + size * 1e-3, 0.5),
        ];
        let clusters = cluster(&points, 10);
        assert_eq!(clusters.len(), 2);
        assert!(clusters.iter().all(|c| c.count == 1));
        assert_eq!(clusters[0].key.0 + 1, clusters[1].key.0);
        assert_eq!(cluster(&points, 5).len(), 1);
    }

    #[test]
    fn hit_returns_the_topmost_bubble() {
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 800.0,
            h: 600.0,
        };
        let view = Viewport {
            center: (0.5, 0.5),
            zoom: 10.0,
        };
        let px = 1.0 / view.world_px();
        // x = 0.5 sits 32 px into its level 10 cell, so 35 px to the right is the next cell.
        let mut points = vec![pt(1, 0.5, 0.5)];
        points.extend((0..20).map(|i| pt(10 + i, 0.5 + 35.0 * px, 0.5)));
        let clusters = cluster(&points, 10);
        assert_eq!(clusters.len(), 2);
        assert_eq!((clusters[0].count, clusters[1].count), (1, 20));
        let (cx, cy) = rect.center();

        // Radii 14 and 26 overlap between 9 and 14 px.
        let top = hit(&clusters, &view, rect, (cx + 11.0, cy)).unwrap();
        assert_eq!(top.first, 10);
        let lone = hit(&clusters, &view, rect, (cx - 10.0, cy)).unwrap();
        assert_eq!(lone.first, 1);
        assert_eq!(
            hit(&clusters, &view, rect, (cx + 30.0, cy)).unwrap().first,
            10
        );
        assert!(hit(&clusters, &view, rect, (cx - 15.0, cy)).is_none());
        assert!(hit(&clusters, &view, rect, (cx, cy + 40.0)).is_none());
    }

    #[test]
    #[ignore = "benchmark"]
    fn bench_cluster_100k() {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss)]
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            unit
        };
        let points: Vec<Point> = (0..100_000).map(|id| pt(id, next(), next())).collect();
        for level in [2, 8, 14] {
            let mut best = std::time::Duration::MAX;
            let mut count = 0;
            for _ in 0..5 {
                let start = std::time::Instant::now();
                count = cluster(&points, level).len();
                best = best.min(start.elapsed());
            }
            println!("level {level}: {count} clusters, best of 5 in {best:?}");
        }
    }
}
