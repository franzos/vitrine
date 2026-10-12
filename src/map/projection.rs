//! Web Mercator in world coordinates: `(0, 0)` is the north-west corner, `(1, 1)` the south-east.

use std::f64::consts::PI;

/// The latitude at which Web Mercator becomes square.
pub const MAX_LAT: f64 = 85.051_128_779_806_59;
pub const MAX_ZOOM: f64 = 16.0;
const TILE: f64 = 256.0;
/// Rect sides below this count as this, so an empty rect gives finite zooms.
const MIN_SIDE: f64 = 1.0;

/// World `(x, y)` of a latitude and longitude, both clamped to the projection's range.
#[must_use]
pub fn project(lat: f64, lon: f64) -> (f64, f64) {
    let lat = lat.clamp(-MAX_LAT, MAX_LAT).to_radians();
    let lon = lon.clamp(-180.0, 180.0);
    let x = (lon + 180.0) / 360.0;
    let y = 0.5 - (lat.tan() + 1.0 / lat.cos()).ln() / (2.0 * PI);
    (x, y.clamp(0.0, 1.0))
}

/// `(lat, lon)` of a world point, clamped to the world.
#[must_use]
pub fn unproject(x: f64, y: f64) -> (f64, f64) {
    let x = x.clamp(0.0, 1.0);
    let y = y.clamp(0.0, 1.0);
    let lat = (PI * (1.0 - 2.0 * y)).sinh().atan().to_degrees();
    (lat, x * 360.0 - 180.0)
}

/// A screen rectangle in points: top-left corner plus size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Rect {
    #[must_use]
    pub fn center(&self) -> (f64, f64) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }
}

/// An axis-aligned box in world coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub min: (f64, f64),
    pub max: (f64, f64),
}

impl Bounds {
    #[must_use]
    pub fn contains(&self, (x, y): (f64, f64)) -> bool {
        (self.min.0..=self.max.0).contains(&x) && (self.min.1..=self.max.1).contains(&y)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    /// World point at the centre of the screen rect.
    pub center: (f64, f64),
    pub zoom: f64,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            center: (0.5, 0.5),
            zoom: 0.0,
        }
    }
}

impl Viewport {
    /// Width and height of the whole world in points.
    #[must_use]
    pub fn world_px(&self) -> f64 {
        TILE * self.zoom.exp2()
    }

    #[must_use]
    pub fn to_screen(&self, (x, y): (f64, f64), rect: Rect) -> (f64, f64) {
        let scale = self.world_px();
        let (cx, cy) = rect.center();
        (
            cx + (x - self.center.0) * scale,
            cy + (y - self.center.1) * scale,
        )
    }

    #[must_use]
    pub fn to_world(&self, (sx, sy): (f64, f64), rect: Rect) -> (f64, f64) {
        let scale = self.world_px();
        let (cx, cy) = rect.center();
        (
            self.center.0 + (sx - cx) / scale,
            self.center.1 + (sy - cy) / scale,
        )
    }

    /// Scales by `factor`, keeping the world point under `pointer` fixed.
    pub fn zoom_at(&mut self, pointer: (f64, f64), factor: f64, rect: Rect) {
        let anchor = self.to_world(pointer, rect);
        self.zoom = (self.zoom + factor.log2()).clamp(min_zoom(rect), MAX_ZOOM);
        let moved = self.to_world(pointer, rect);
        self.center.0 += anchor.0 - moved.0;
        self.center.1 += anchor.1 - moved.1;
    }

    /// Moves the map by a screen delta in points, as a drag does.
    pub fn pan(&mut self, (dx, dy): (f64, f64)) {
        let scale = self.world_px();
        self.center.0 -= dx / scale;
        self.center.1 -= dy / scale;
    }

    /// The viewport that shows all of `bounds` in `rect`, at most at `max_zoom`.
    #[must_use]
    pub fn fit(bounds: Bounds, rect: Rect, max_zoom: f64) -> Self {
        let span_x = bounds.max.0 - bounds.min.0;
        let span_y = bounds.max.1 - bounds.min.1;
        // A zero span gives an infinite zoom, so a single point lands on max_zoom.
        let zoom = (rect.w.max(MIN_SIDE) / (span_x * TILE))
            .log2()
            .min((rect.h.max(MIN_SIDE) / (span_y * TILE)).log2())
            .min(max_zoom);
        Self {
            center: (
                (bounds.min.0 + bounds.max.0) / 2.0,
                (bounds.min.1 + bounds.max.1) / 2.0,
            ),
            zoom,
        }
    }

    /// Keeps the zoom between the rect's minimum and `MAX_ZOOM`, and the world covering the rect.
    pub fn clamp(&mut self, rect: Rect) {
        self.zoom = self.zoom.clamp(min_zoom(rect), MAX_ZOOM);
        let scale = self.world_px();
        self.center.0 = clamp_axis(self.center.0, rect.w / 2.0 / scale);
        self.center.1 = clamp_axis(self.center.1, rect.h / 2.0 / scale);
    }

    /// The world box visible in `rect`.
    #[must_use]
    pub fn bounds(&self, rect: Rect) -> Bounds {
        Bounds {
            min: self.to_world((rect.x, rect.y), rect),
            max: self.to_world((rect.x + rect.w, rect.y + rect.h), rect),
        }
    }
}

/// The zoom at which the world is exactly as tall as `rect`, capped at `MAX_ZOOM`.
fn min_zoom(rect: Rect) -> f64 {
    (rect.h.max(MIN_SIDE) / TILE).log2().min(MAX_ZOOM)
}

/// Centre coordinate on one axis, given half the visible span in world units.
fn clamp_axis(center: f64, half: f64) -> f64 {
    if half >= 0.5 {
        0.5
    } else {
        center.clamp(half, 1.0 - half)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECT: Rect = Rect {
        x: 10.0,
        y: 20.0,
        w: 800.0,
        h: 600.0,
    };

    fn close(a: (f64, f64), b: (f64, f64), eps: f64) -> bool {
        (a.0 - b.0).abs() < eps && (a.1 - b.1).abs() < eps
    }

    #[test]
    fn project_round_trips() {
        for (lat, lon) in [
            (0.0, 0.0),
            (38.72, -9.14),
            (-33.86, 151.21),
            (85.0, 179.9),
            (-85.0, -179.9),
        ] {
            let (x, y) = project(lat, lon);
            assert!(close(unproject(x, y), (lat, lon), 1e-9), "{lat}, {lon}");
        }
        assert!(close(project(0.0, 0.0), (0.5, 0.5), 1e-12));
    }

    #[test]
    fn project_clamps_latitude_and_longitude() {
        assert!(close(project(90.0, -200.0), (0.0, 0.0), 1e-9));
        assert!(close(project(-90.0, 200.0), (1.0, 1.0), 1e-9));
        assert!(close(project(89.0, 0.0), project(MAX_LAT, 0.0), 1e-12));
        let (lat, _) = unproject(0.5, -1.0);
        assert!((lat - MAX_LAT).abs() < 1e-9);
    }

    #[test]
    fn screen_and_world_round_trip() {
        let view = Viewport {
            center: (0.3, 0.6),
            zoom: 12.5,
        };
        let world = (0.300_01, 0.599_98);
        let screen = view.to_screen(world, RECT);
        assert!(close(view.to_world(screen, RECT), world, 1e-12));
        assert!(close(
            view.to_screen(view.center, RECT),
            RECT.center(),
            1e-9
        ));
    }

    #[test]
    fn zoom_at_keeps_the_point_under_the_pointer() {
        let pointer = (150.0, 500.0);
        for factor in [0.5, 1.1, 4.0] {
            let mut view = Viewport {
                center: (0.52, 0.47),
                zoom: 8.0,
            };
            let before = view.to_world(pointer, RECT);
            view.zoom_at(pointer, factor, RECT);
            assert!((view.zoom - (8.0 + factor.log2())).abs() < 1e-12);
            assert!(close(view.to_world(pointer, RECT), before, 1e-12));
        }
    }

    #[test]
    fn zoom_at_stops_at_max_zoom_with_the_pointer_fixed() {
        let pointer = (700.0, 100.0);
        let mut view = Viewport {
            center: (0.5, 0.5),
            zoom: 15.5,
        };
        let before = view.to_world(pointer, RECT);
        view.zoom_at(pointer, 8.0, RECT);
        assert!((view.zoom - MAX_ZOOM).abs() < f64::EPSILON);
        assert!(close(view.to_world(pointer, RECT), before, 1e-12));
    }

    #[test]
    fn pan_moves_the_map_with_the_drag() {
        let mut view = Viewport {
            center: (0.5, 0.5),
            zoom: 3.0,
        };
        let world = (0.55, 0.45);
        let before = view.to_screen(world, RECT);
        view.pan((40.0, -25.0));
        let after = view.to_screen(world, RECT);
        assert!(close(after, (before.0 + 40.0, before.1 - 25.0), 1e-9));
    }

    #[test]
    fn fit_contains_the_bounds() {
        let bounds = Bounds {
            min: project(43.0, -9.5),
            max: project(36.9, -6.2),
        };
        let view = Viewport::fit(bounds, RECT, 12.0);
        assert!(view.zoom < 12.0);
        let visible = view.bounds(RECT);
        let eps = 1e-12;
        assert!(visible.min.0 <= bounds.min.0 + eps && visible.min.1 <= bounds.min.1 + eps);
        assert!(visible.max.0 >= bounds.max.0 - eps && visible.max.1 >= bounds.max.1 - eps);
        let tight = Viewport {
            zoom: view.zoom + 0.01,
            ..view
        }
        .bounds(RECT);
        assert!(!(tight.contains(bounds.min) && tight.contains(bounds.max)));
    }

    #[test]
    fn fit_on_a_single_point_uses_max_zoom() {
        let point = project(38.72, -9.14);
        let view = Viewport::fit(
            Bounds {
                min: point,
                max: point,
            },
            RECT,
            12.0,
        );
        assert!((view.zoom - 12.0).abs() < f64::EPSILON);
        assert!(close(view.center, point, 1e-12));
    }

    #[test]
    fn clamp_honours_the_min_zoom() {
        let mut view = Viewport {
            center: (0.5, 0.5),
            zoom: 0.0,
        };
        view.clamp(RECT);
        assert!((view.zoom - (600.0_f64 / 256.0).log2()).abs() < 1e-12);
        assert!((view.world_px() - 600.0).abs() < 1e-9);
        // The world is narrower than the rect, so it stays centred horizontally.
        assert!(close(view.center, (0.5, 0.5), 1e-12));
    }

    #[test]
    fn empty_rects_give_finite_viewports() {
        let bounds = Bounds {
            min: project(43.0, -9.5),
            max: project(36.9, -6.2),
        };
        for (w, h) in [(0.0, 0.0), (0.0, 600.0), (800.0, 0.0), (-5.0, -5.0)] {
            let rect = Rect {
                x: 0.0,
                y: 0.0,
                w,
                h,
            };
            let mut view = Viewport::fit(bounds, rect, 12.0);
            assert!(view.zoom.is_finite(), "{w}x{h}: {view:?}");
            view.clamp(rect);
            assert!(view.zoom.is_finite(), "{w}x{h}: {view:?}");
            assert!(view.center.0.is_finite() && view.center.1.is_finite());
            view.zoom_at((0.0, 0.0), 2.0, rect);
            assert!(view.zoom.is_finite() && view.center.0.is_finite());
        }
    }

    #[test]
    fn clamp_caps_the_zoom_and_keeps_the_world_edge_in_the_rect() {
        let mut view = Viewport {
            center: (-0.2, 1.3),
            zoom: 20.0,
        };
        view.clamp(RECT);
        assert!((view.zoom - MAX_ZOOM).abs() < f64::EPSILON);
        let visible = view.bounds(RECT);
        assert!(visible.min.0.abs() < 1e-12);
        assert!((visible.max.1 - 1.0).abs() < 1e-12);

        let mut view = Viewport {
            center: (0.95, 0.02),
            zoom: 3.0,
        };
        view.clamp(RECT);
        let visible = view.bounds(RECT);
        assert!((visible.max.0 - 1.0).abs() < 1e-12);
        assert!(visible.min.1.abs() < 1e-12);
    }
}
