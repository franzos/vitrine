//! The map's background layer, behind a trait so another basemap can replace Natural Earth.

use std::sync::LazyLock;

use eframe::egui::{Color32, Mesh, Painter, Pos2, Rect, Shape, Stroke, Theme, pos2};

use crate::map::pack::{self, Pack, Point};
use crate::map::{self, Bounds, Viewport};

/// Natural Earth 1:50m, packed by `examples/pack_basemap.rs`.
pub(crate) const BASEMAP: &[u8] = include_bytes!("../../assets/basemap.vmap");

const COAST_WIDTH: f32 = 0.75;
const BORDER_WIDTH: f32 = 1.0;
const BORDER_MIN_ZOOM: f64 = 2.0;
/// Screen points beyond the rect within which strokes and triangles are kept.
const MARGIN: f64 = 4.0;

pub trait Basemap {
    fn paint(&self, painter: &Painter, rect: Rect, viewport: &Viewport);
    fn country_labels(&self) -> &[CountryLabel];
    fn attribution(&self) -> &str;
}

#[derive(Clone, Debug, PartialEq)]
pub struct CountryLabel {
    pub name: String,
    /// World coordinates.
    pub pos: (f64, f64),
    pub min_zoom: f64,
}

/// The embedded Natural Earth pack, decoded on first use.
#[derive(Clone, Copy, Debug, Default)]
pub struct NaturalEarth;

struct Data {
    pack: Pack,
    /// World bbox per boundary line, `None` for an empty line.
    line_bounds: Vec<Option<Bounds>>,
    labels: Vec<CountryLabel>,
}

static DATA: LazyLock<Data> = LazyLock::new(|| {
    // The asset is embedded and decoded by the tests, so failure is a build defect.
    let pack = pack::decode(BASEMAP).expect("embedded basemap pack decodes");
    let line_bounds = pack
        .lines
        .iter()
        .map(|l| bbox(l.iter().map(|&p| world(p))))
        .collect();
    let labels = pack
        .labels
        .iter()
        .map(|l| CountryLabel {
            name: l.name.clone(),
            pos: world(l.pos),
            min_zoom: f64::from(l.min_zoom),
        })
        .collect();
    Data {
        pack,
        line_bounds,
        labels,
    }
});

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Palette {
    ocean: Color32,
    land: Color32,
    coast: Color32,
    border: Color32,
}

impl Palette {
    fn new(dark: bool) -> Self {
        if dark {
            Self {
                ocean: Color32::from_rgb(27, 36, 46),
                land: Color32::from_rgb(52, 55, 58),
                coast: Color32::from_rgb(82, 100, 116),
                border: Color32::from_rgb(98, 98, 106),
            }
        } else {
            Self {
                ocean: Color32::from_rgb(178, 208, 222),
                land: Color32::from_rgb(238, 236, 228),
                coast: Color32::from_rgb(128, 156, 174),
                border: Color32::from_rgb(176, 164, 176),
            }
        }
    }
}

impl NaturalEarth {
    fn shapes(rect: Rect, viewport: &Viewport, palette: Palette) -> Vec<Shape> {
        let data = &*DATA;
        let view = map::Rect {
            x: f64::from(rect.min.x),
            y: f64::from(rect.min.y),
            w: f64::from(rect.width()),
            h: f64::from(rect.height()),
        };
        let visible = viewport.bounds(view);
        let clip = Bounds {
            min: (view.x - MARGIN, view.y - MARGIN),
            max: (view.x + view.w + MARGIN, view.y + view.h + MARGIN),
        };
        let world_px = viewport.world_px();
        let scale = world_px / f64::from(u16::MAX);
        let (cx, cy) = view.center();
        let origin = (
            cx - viewport.center.0 * world_px,
            cy - viewport.center.1 * world_px,
        );
        let to_screen = |(x, y): Point| {
            (
                origin.0 + f64::from(x) * scale,
                origin.1 + f64::from(y) * scale,
            )
        };
        let in_view = |min: Point, max: Point| {
            intersects(
                &Bounds {
                    min: world(min),
                    max: world(max),
                },
                &visible,
            )
        };
        let coast = Stroke::new(COAST_WIDTH, palette.coast);
        let border = Stroke::new(BORDER_WIDTH, palette.border);

        let mut shapes = vec![Shape::rect_filled(rect, 0.0, palette.ocean)];
        let mut coasts = Vec::new();
        let mut points = Vec::new();
        for chunk in data.pack.land.iter().filter(|c| in_view(c.min, c.max)) {
            points.clear();
            points.extend(chunk.vertices.iter().map(|&p| to_screen(p)));
            shapes.extend(mesh(&points, &chunk.triangles, &clip, palette.land));
            let mut start = 0;
            for &len in &chunk.rings {
                let ring = &points[start..start + usize::from(len)];
                start += usize::from(len);
                stroke(ring, true, &clip, coast, &mut coasts);
            }
        }
        for chunk in data.pack.lakes.iter().filter(|c| in_view(c.min, c.max)) {
            points.clear();
            points.extend(chunk.vertices.iter().map(|&p| to_screen(p)));
            shapes.extend(mesh(&points, &chunk.triangles, &clip, palette.ocean));
        }
        shapes.append(&mut coasts);
        if viewport.zoom >= BORDER_MIN_ZOOM {
            for (line, b) in data.pack.lines.iter().zip(&data.line_bounds) {
                if b.is_some_and(|b| intersects(&b, &visible)) {
                    points.clear();
                    points.extend(line.iter().map(|&p| to_screen(p)));
                    stroke(&points, false, &clip, border, &mut shapes);
                }
            }
        }
        shapes
    }
}

impl Basemap for NaturalEarth {
    fn paint(&self, painter: &Painter, rect: Rect, viewport: &Viewport) {
        let palette = Palette::new(painter.ctx().theme() == Theme::Dark);
        painter.extend(Self::shapes(rect, viewport, palette));
    }

    fn country_labels(&self) -> &[CountryLabel] {
        &DATA.labels
    }

    fn attribution(&self) -> &'static str {
        "Natural Earth"
    }
}

fn world((x, y): Point) -> (f64, f64) {
    let max = f64::from(u16::MAX);
    (f64::from(x) / max, f64::from(y) / max)
}

fn bbox(mut points: impl Iterator<Item = (f64, f64)>) -> Option<Bounds> {
    let first = points.next()?;
    Some(points.fold(
        Bounds {
            min: first,
            max: first,
        },
        |b, (x, y)| Bounds {
            min: (b.min.0.min(x), b.min.1.min(y)),
            max: (b.max.0.max(x), b.max.1.max(y)),
        },
    ))
}

fn intersects(a: &Bounds, b: &Bounds) -> bool {
    a.min.0 <= b.max.0 && a.max.0 >= b.min.0 && a.min.1 <= b.max.1 && a.max.1 >= b.min.1
}

fn overlaps(points: &[(f64, f64)], clip: &Bounds) -> bool {
    bbox(points.iter().copied()).is_some_and(|b| intersects(&b, clip))
}

#[allow(clippy::cast_possible_truncation)]
fn pos((x, y): (f64, f64)) -> Pos2 {
    pos2(x as f32, y as f32)
}

/// The triangles overlapping `clip`, or `None` if there are none.
fn mesh(
    points: &[(f64, f64)],
    triangles: &[[u16; 3]],
    clip: &Bounds,
    color: Color32,
) -> Option<Shape> {
    let mut mesh = Mesh::default();
    for t in triangles {
        let corners = t.map(|i| points[usize::from(i)]);
        if overlaps(&corners, clip) {
            let [a, b, c] = t.map(u32::from);
            mesh.add_triangle(a, b, c);
        }
    }
    if mesh.indices.is_empty() {
        return None;
    }
    mesh.reserve_vertices(points.len());
    for &p in points {
        mesh.colored_vertex(pos(p), color);
    }
    Some(Shape::mesh(mesh))
}

/// Strokes only the runs of segments overlapping `clip`, keeping far vertices out of f32.
fn stroke(
    points: &[(f64, f64)],
    closed: bool,
    clip: &Bounds,
    stroke: Stroke,
    out: &mut Vec<Shape>,
) {
    let n = points.len();
    if n < 2 {
        return;
    }
    let segments = if closed { n } else { n - 1 };
    let mut runs = Vec::new();
    let mut run: Vec<Pos2> = Vec::new();
    for i in 0..segments {
        let (a, b) = (points[i], points[(i + 1) % n]);
        if overlaps(&[a, b], clip) {
            if run.is_empty() {
                run.push(pos(a));
            }
            run.push(pos(b));
        } else if !run.is_empty() {
            runs.push(std::mem::take(&mut run));
        }
    }
    if closed && runs.is_empty() && run.len() == n + 1 {
        run.pop();
        out.push(Shape::closed_line(run, stroke));
        return;
    }
    // A ring's last run ends at vertex 0, where its first run starts.
    if closed && !run.is_empty() && !runs.is_empty() && overlaps(&points[..2], clip) {
        run.extend_from_slice(&runs[0][1..]);
        runs[0] = run;
    } else if !run.is_empty() {
        runs.push(run);
    }
    out.extend(runs.into_iter().map(|r| Shape::line(r, stroke)));
}

#[cfg(test)]
mod tests {
    use eframe::egui::Vec2;
    use egui_kittest::Harness;

    use super::*;
    use crate::map::pack::Chunk;
    use crate::map::project;

    fn pack() -> Pack {
        pack::decode(BASEMAP).unwrap()
    }

    fn quantise(lat: f64, lon: f64) -> Point {
        let (x, y) = project(lat, lon);
        ((x * 65535.0).round() as u16, (y * 65535.0).round() as u16)
    }

    /// Twice the signed area of the triangle `a`, `b`, `c`.
    fn cross(a: Point, b: Point, c: Point) -> i64 {
        let (ax, ay) = (i64::from(a.0), i64::from(a.1));
        let (bx, by) = (i64::from(b.0), i64::from(b.1));
        let (cx, cy) = (i64::from(c.0), i64::from(c.1));
        (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    }

    fn inside(chunks: &[Chunk], p: Point) -> bool {
        chunks.iter().any(|c| {
            c.triangles.iter().any(|t| {
                let [a, b, d] = t.map(|i| c.vertices[usize::from(i)]);
                let s = [cross(a, b, p), cross(b, d, p), cross(d, a, p)];
                s.iter().all(|&v| v >= 0) || s.iter().all(|&v| v <= 0)
            })
        })
    }

    #[test]
    fn decodes_and_re_encodes_identically() {
        assert!(BASEMAP.len() < 1_600_000, "{} bytes", BASEMAP.len());
        let pack = pack();
        assert!(!pack.land.is_empty() && !pack.lakes.is_empty() && !pack.lines.is_empty());
        assert_eq!(pack::encode(&pack).unwrap(), BASEMAP);
    }

    #[test]
    fn chunks_carry_their_bbox_and_ring_contract() {
        let pack = pack();
        for chunk in pack.land.iter().chain(&pack.lakes) {
            let xs = chunk.vertices.iter().map(|v| v.0);
            let ys = chunk.vertices.iter().map(|v| v.1);
            let min = (xs.clone().min().unwrap(), ys.clone().min().unwrap());
            let max = (xs.max().unwrap(), ys.max().unwrap());
            assert_eq!((chunk.min, chunk.max), (min, max));
        }
        assert!(pack.land.iter().all(|c| !c.rings.is_empty()));
        assert!(pack.lakes.iter().all(|c| c.rings.is_empty()));
    }

    #[test]
    fn land_triangles_cover_the_rings() {
        for (i, chunk) in pack().land.iter().enumerate() {
            let mut start = 0;
            let mut rings = 0;
            for &len in &chunk.rings {
                let ring = &chunk.vertices[start..start + usize::from(len)];
                rings += (0..ring.len())
                    .map(|j| cross((0, 0), ring[j], ring[(j + 1) % ring.len()]))
                    .sum::<i64>();
                start += usize::from(len);
            }
            let triangles: i64 = chunk
                .triangles
                .iter()
                .map(|t| {
                    let [a, b, c] = t.map(|i| chunk.vertices[usize::from(i)]);
                    cross(a, b, c).abs()
                })
                .sum();
            let rings = rings.abs();
            assert!(
                (triangles - rings).abs() * 10_000 <= rings,
                "chunk {i}: triangles {triangles}, rings {rings}"
            );
        }
    }

    #[test]
    fn land_and_lakes_are_where_they_belong() {
        let pack = pack();
        // 1:50m puts (38.72, -9.14) in the Tagus estuary.
        assert!(inside(&pack.land, quantise(38.77, -9.14)), "Lisbon");
        assert!(!inside(&pack.land, quantise(35.0, -30.0)), "Atlantic");
        assert!(inside(&pack.lakes, quantise(47.7, -87.5)), "Lake Superior");
    }

    #[test]
    fn labels_include_portugal() {
        assert!(pack().labels.iter().any(|l| l.name == "Portugal"));
    }

    const SIZE: Vec2 = Vec2::new(800.0, 600.0);

    fn at(lat: f64, lon: f64, zoom: f64) -> Viewport {
        Viewport {
            center: project(lat, lon),
            zoom,
        }
    }

    fn world_view() -> Viewport {
        Viewport {
            center: (0.5, 0.5),
            zoom: 0.0,
        }
    }

    fn shapes(viewport: &Viewport) -> Vec<Shape> {
        NaturalEarth::shapes(rect(), viewport, Palette::new(false))
    }

    fn is_mesh(shape: &Shape, color: Color32) -> bool {
        matches!(shape, Shape::Mesh(m) if m.vertices.first().is_some_and(|v| v.color == color))
    }

    fn meshes(shapes: &[Shape], color: Color32) -> usize {
        shapes.iter().filter(|s| is_mesh(s, color)).count()
    }

    fn strokes(shapes: &[Shape], width: f32) -> usize {
        shapes
            .iter()
            .filter(|s| matches!(s, Shape::Path(p) if p.stroke.width == width))
            .count()
    }

    #[test]
    fn paints_through_a_harness_at_world_europe_and_lisbon() {
        for viewport in [world_view(), at(38.72, -9.14, 4.0), at(38.72, -9.14, 10.0)] {
            let mut harness = Harness::builder().with_size(SIZE).build_ui(move |ui| {
                let rect = ui.max_rect();
                NaturalEarth.paint(&ui.painter_at(rect), rect, &viewport);
            });
            harness.run();
            let count = harness
                .output()
                .shapes
                .iter()
                .filter(|s| matches!(s.shape, Shape::Mesh(_)))
                .count();
            assert!(count > 0, "zoom {}", viewport.zoom);
        }
    }

    #[test]
    fn land_and_coasts_are_drawn_at_every_zoom() {
        let land = Palette::new(false).land;
        for viewport in [world_view(), at(38.72, -9.14, 4.0), at(38.72, -9.14, 10.0)] {
            let shapes = shapes(&viewport);
            assert!(meshes(&shapes, land) > 0, "zoom {}", viewport.zoom);
            assert!(strokes(&shapes, COAST_WIDTH) > 0, "zoom {}", viewport.zoom);
        }
    }

    #[test]
    fn borders_appear_from_zoom_two() {
        for zoom in [0.0, 1.0, 1.99] {
            let viewport = Viewport {
                zoom,
                ..world_view()
            };
            assert_eq!(strokes(&shapes(&viewport), BORDER_WIDTH), 0, "zoom {zoom}");
        }
        for viewport in [
            Viewport {
                zoom: 2.0,
                ..world_view()
            },
            at(38.72, -9.14, 6.0),
        ] {
            assert!(
                strokes(&shapes(&viewport), BORDER_WIDTH) > 0,
                "zoom {}",
                viewport.zoom
            );
        }
    }

    #[test]
    fn lakes_are_drawn_over_land_in_ocean_colour() {
        let palette = Palette::new(false);
        let shapes = shapes(&at(47.7, -87.5, 5.0));
        let last_land = shapes
            .iter()
            .rposition(|s| is_mesh(s, palette.land))
            .unwrap();
        let first_lake = shapes
            .iter()
            .position(|s| is_mesh(s, palette.ocean))
            .unwrap();
        assert!(last_land < first_lake);
    }

    #[test]
    fn culls_coasts_and_borders_outside_the_view() {
        let world = shapes(&Viewport {
            zoom: 2.0,
            ..world_view()
        });
        let lisbon = shapes(&at(38.72, -9.14, 10.0));
        for width in [COAST_WIDTH, BORDER_WIDTH] {
            let (world, lisbon) = (strokes(&world, width), strokes(&lisbon, width));
            assert!(
                lisbon < world,
                "width {width}: lisbon {lisbon}, world {world}"
            );
        }
    }

    #[test]
    fn country_labels_are_in_world_coordinates() {
        let labels = NaturalEarth.country_labels();
        let portugal = labels.iter().find(|l| l.name == "Portugal").unwrap();
        let (x, y) = portugal.pos;
        let (lat, lon) = crate::map::unproject(x, y);
        assert!(
            (36.0..43.0).contains(&lat) && (-10.0..-6.0).contains(&lon),
            "{lat}, {lon}"
        );
        assert_eq!(NaturalEarth.attribution(), "Natural Earth");
    }

    fn rect() -> Rect {
        Rect::from_min_size(Pos2::ZERO, SIZE)
    }

    /// The land vertex nearest to `lat`, `lon`, in world coordinates.
    fn nearest_coast(lat: f64, lon: f64) -> (f64, f64) {
        let target = project(lat, lon);
        let d = |p: &(f64, f64)| (p.0 - target.0).hypot(p.1 - target.1);
        pack()
            .land
            .iter()
            .flat_map(|c| c.vertices.iter().map(|&v| world(v)))
            .min_by(|a, b| d(a).total_cmp(&d(b)))
            .unwrap()
    }

    fn stroke_paths(shapes: &[Shape], width: f32) -> Vec<&[Pos2]> {
        shapes
            .iter()
            .filter_map(|s| match s {
                Shape::Path(p) if p.stroke.width == width => Some(p.points.as_slice()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn ocean_only_view_draws_no_land() {
        let shapes = shapes(&at(30.0, -45.0, 6.0));
        assert_eq!(meshes(&shapes, Palette::new(false).land), 0);
        assert_eq!(strokes(&shapes, COAST_WIDTH), 0);
    }

    #[test]
    fn deep_zoom_strokes_only_segments_crossing_the_view() {
        let viewport = Viewport {
            center: nearest_coast(38.69, -9.23),
            zoom: 16.0,
        };
        let shapes = shapes(&viewport);
        let paths = stroke_paths(&shapes, COAST_WIDTH);
        assert!(!paths.is_empty());
        let clip = rect().expand(MARGIN as f32 + 1.0);
        let mut emitted = 0;
        for path in &paths {
            emitted += path.len();
            assert!(path.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
            for pair in path.windows(2) {
                assert!(Rect::from_points(pair).intersects(clip), "{pair:?}");
            }
        }
        let total: usize = pack().land.iter().map(|c| c.vertices.len()).sum();
        assert!(emitted * 100 < total, "emitted {emitted} of {total}");
    }

    #[test]
    fn deep_zoom_vertices_land_where_f64_puts_them() {
        let vertex = nearest_coast(38.69, -9.23);
        let mut viewport = Viewport {
            center: vertex,
            zoom: 16.0,
        };
        viewport.pan((-123.4, 56.7));
        let view = map::Rect {
            x: 0.0,
            y: 0.0,
            w: f64::from(SIZE.x),
            h: f64::from(SIZE.y),
        };
        let (x, y) = viewport.to_screen(vertex, view);
        let land = Palette::new(false).land;
        let nearest = shapes(&viewport)
            .iter()
            .filter_map(|s| match s {
                Shape::Mesh(m) if m.vertices.first().is_some_and(|v| v.color == land) => {
                    Some(m.vertices.clone())
                }
                _ => None,
            })
            .flatten()
            .map(|v| (f64::from(v.pos.x) - x).hypot(f64::from(v.pos.y) - y))
            .fold(f64::INFINITY, f64::min);
        assert!(nearest < 0.01, "off by {nearest} px");
    }
}
