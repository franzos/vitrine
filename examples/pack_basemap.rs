//! Packs Natural Earth 1:50m GeoJSON into `assets/basemap.vmap`, the basemap the GUI embeds.
//!
//! Inputs are `ne_50m_land`, `ne_50m_lakes`, `ne_50m_admin_0_boundary_lines_land` and
//! `ne_50m_admin_0_countries` from `nvkelso/natural-earth-vector` at tag `v5.1.2`:
//!
//! ```text
//! https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/<name>.geojson
//! ```
//!
//! Run with `cargo run --release --example pack_basemap -- --in <dir> --out assets/basemap.vmap`.
//! The same inputs always give the same bytes.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use vitrine::map::pack::{self, Chunk, Label, Pack, Point};
use vitrine::map::project;

/// Douglas-Peucker tolerance in quanta.
const TOLERANCE: f64 = 1.0;
/// Rings smaller than this many square quanta are dropped.
const MIN_AREA: i64 = 2;
/// Polygons are grouped into chunks on a `GRID` x `GRID` world grid, so chunk bboxes cull.
const GRID: u32 = 8;
const MAX_VERTICES: usize = u16::MAX as usize;

/// Rings (outer first, then holes) and triangles indexing their concatenated vertices.
struct Polygon {
    rings: Vec<Vec<Point>>,
    triangles: Vec<[usize; 3]>,
}

impl Polygon {
    fn len(&self) -> usize {
        self.rings.iter().map(Vec::len).sum()
    }
}

#[derive(Default)]
struct Dropped {
    polygons: usize,
    holes: usize,
}

fn main() -> Result<()> {
    let (input, output) = args()?;
    let read = |name: &str| -> Result<Vec<Value>> {
        let path = input.join(format!("{name}.geojson"));
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let json: Value =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        match json.get("features") {
            Some(Value::Array(features)) => Ok(features.clone()),
            _ => bail!("{} has no features", path.display()),
        }
    };

    let mut dropped = Dropped::default();
    let land = polygons(&read("ne_50m_land")?, &mut dropped)?;
    let lakes = polygons(&read("ne_50m_lakes")?, &mut dropped)?;
    let pack = Pack {
        land: chunks(&land, true)?,
        lakes: chunks(&lakes, false)?,
        lines: lines(&read("ne_50m_admin_0_boundary_lines_land")?)?,
        labels: labels(&read("ne_50m_admin_0_countries")?)?,
    };
    let bytes = pack::encode(&pack)?;
    std::fs::write(&output, &bytes).with_context(|| format!("writing {}", output.display()))?;

    for (layer, chunks) in [("land", &pack.land), ("lakes", &pack.lakes)] {
        let vertices: usize = chunks.iter().map(|c| c.vertices.len()).sum();
        let triangles: usize = chunks.iter().map(|c| c.triangles.len()).sum();
        let largest = chunks.iter().map(|c| c.vertices.len()).max().unwrap_or(0);
        println!(
            "{layer}: {} chunks, {vertices} vertices (largest chunk {largest}), {triangles} triangles",
            chunks.len()
        );
    }
    let points: usize = pack.lines.iter().map(Vec::len).sum();
    println!("lines: {}, {points} points", pack.lines.len());
    println!("labels: {}", pack.labels.len());
    println!(
        "dropped: {} polygons, {} holes",
        dropped.polygons, dropped.holes
    );
    println!("{}: {} bytes", output.display(), bytes.len());
    Ok(())
}

fn args() -> Result<(PathBuf, PathBuf)> {
    let mut input = None;
    let mut output = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let slot = match arg.as_str() {
            "--in" => &mut input,
            "--out" => &mut output,
            _ => bail!("usage: pack_basemap --in <dir> --out <file>"),
        };
        *slot = Some(PathBuf::from(
            args.next()
                .with_context(|| format!("{arg} needs a value"))?,
        ));
    }
    match (input, output) {
        (Some(i), Some(o)) => Ok((i, o)),
        _ => bail!("usage: pack_basemap --in <dir> --out <file>"),
    }
}

fn quantise(v: f64) -> u16 {
    // project() keeps v in [0, 1], so the cast never saturates.
    (v * 65535.0).round() as u16
}

fn point(coord: &Value) -> Result<Point> {
    let lon = coord.get(0).and_then(Value::as_f64);
    let lat = coord.get(1).and_then(Value::as_f64);
    let (Some(lon), Some(lat)) = (lon, lat) else {
        bail!("bad coordinate {coord}");
    };
    let (x, y) = project(lat, lon);
    Ok((quantise(x), quantise(y)))
}

fn points(coords: &Value) -> Result<Vec<Point>> {
    let coords = coords.as_array().context("expected a coordinate array")?;
    let mut points = coords.iter().map(point).collect::<Result<Vec<_>>>()?;
    // Clamping to the Mercator limit collapses Antarctica's -90 edge into repeated points.
    points.dedup();
    Ok(points)
}

/// Twice the signed area.
fn area2(ring: &[Point]) -> i64 {
    let n = ring.len();
    (0..n)
        .map(|i| {
            let (x0, y0) = ring[i];
            let (x1, y1) = ring[(i + 1) % n];
            i64::from(x0) * i64::from(y1) - i64::from(x1) * i64::from(y0)
        })
        .sum()
}

/// A simplified ring without the closing point, or `None` when it degenerates.
fn ring(coords: &Value) -> Result<Option<Vec<Point>>> {
    let mut ring = points(coords)?;
    while ring.len() > 1 && ring.first() == ring.last() {
        ring.pop();
    }
    if ring.len() < 3 {
        return Ok(None);
    }
    let first = ring[0];
    let far = (1..ring.len())
        .max_by(|&a, &b| dist2(ring[a], first).cmp(&dist2(ring[b], first)))
        .unwrap_or(1);
    ring.push(first);
    let mut keep = vec![false; ring.len()];
    simplify(&ring, 0, far, &mut keep);
    simplify(&ring, far, ring.len() - 1, &mut keep);
    ring.pop();
    let ring: Vec<Point> = ring
        .into_iter()
        .zip(keep)
        .filter_map(|(p, k)| k.then_some(p))
        .collect();
    if ring.len() < 3 || area2(&ring).abs() < 2 * MIN_AREA {
        return Ok(None);
    }
    Ok(Some(ring))
}

fn dist2(a: Point, b: Point) -> i64 {
    let dx = i64::from(a.0) - i64::from(b.0);
    let dy = i64::from(a.1) - i64::from(b.1);
    dx * dx + dy * dy
}

/// Squared distance from `p` to the segment `a`-`b`.
fn segment_distance2(p: Point, a: Point, b: Point) -> f64 {
    let f = |q: Point| (f64::from(q.0), f64::from(q.1));
    let ((px, py), (ax, ay), (bx, by)) = (f(p), f(a), f(b));
    let (dx, dy) = (bx - ax, by - ay);
    let len2 = dx * dx + dy * dy;
    let t = if len2 == 0.0 {
        0.0
    } else {
        (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0)
    };
    let (ex, ey) = (px - ax - t * dx, py - ay - t * dy);
    ex * ex + ey * ey
}

/// Douglas-Peucker on `points[first..=last]`, marking kept points.
fn simplify(points: &[Point], first: usize, last: usize, keep: &mut [bool]) {
    keep[first] = true;
    keep[last] = true;
    let mut stack = vec![(first, last)];
    while let Some((a, b)) = stack.pop() {
        let mut worst = (0.0, a);
        for i in a + 1..b {
            let d = segment_distance2(points[i], points[a], points[b]);
            if d > worst.0 {
                worst = (d, i);
            }
        }
        if worst.0 > TOLERANCE * TOLERANCE {
            keep[worst.1] = true;
            stack.push((a, worst.1));
            stack.push((worst.1, b));
        }
    }
}

fn polygon(coords: &Value, dropped: &mut Dropped) -> Result<Option<Polygon>> {
    let rings = coords.as_array().context("expected polygon rings")?;
    let Some((outer, holes)) = rings.split_first() else {
        bail!("polygon without rings");
    };
    let Some(mut outer) = ring(outer)? else {
        dropped.polygons += 1;
        return Ok(None);
    };
    // Outer rings positive, holes negative, so a chunk's summed ring area is outer minus holes.
    if area2(&outer) < 0 {
        outer.reverse();
    }
    let mut rings = vec![outer];
    for hole in holes {
        match ring(hole)? {
            Some(mut hole) => {
                if area2(&hole) > 0 {
                    hole.reverse();
                }
                rings.push(hole);
            }
            None => dropped.holes += 1,
        }
    }

    let mut flat = Vec::new();
    let mut hole_starts = Vec::new();
    for (i, ring) in rings.iter().enumerate() {
        if i > 0 {
            hole_starts.push(flat.len() / 2);
        }
        flat.extend(ring.iter().flat_map(|&(x, y)| [f64::from(x), f64::from(y)]));
    }
    let indices =
        earcutr::earcut(&flat, &hole_starts, 2).map_err(|e| anyhow!("earcut failed: {e}"))?;
    let triangles = indices.as_chunks().0.to_vec();
    Ok(Some(Polygon { rings, triangles }))
}

fn polygons(features: &[Value], dropped: &mut Dropped) -> Result<Vec<Polygon>> {
    let mut out = Vec::new();
    for feature in features {
        let geometry = feature
            .get("geometry")
            .context("feature without geometry")?;
        let coords = geometry
            .get("coordinates")
            .context("geometry without coordinates")?;
        let parts = match geometry.get("type").and_then(Value::as_str) {
            Some("Polygon") => vec![coords],
            Some("MultiPolygon") => coords
                .as_array()
                .context("expected polygons")?
                .iter()
                .collect(),
            other => bail!("unexpected geometry {other:?}"),
        };
        for part in parts {
            out.extend(polygon(part, dropped)?);
        }
    }
    Ok(out)
}

/// Whole polygons grouped by grid cell, each chunk at most `MAX_VERTICES` vertices.
fn chunks(polygons: &[Polygon], with_rings: bool) -> Result<Vec<Chunk>> {
    let mut cells: BTreeMap<(u32, u32), Vec<&Polygon>> = BTreeMap::new();
    for p in polygons {
        if p.len() > MAX_VERTICES {
            bail!("a polygon has {} vertices, over {MAX_VERTICES}", p.len());
        }
        let (min, max) = bbox(p.rings.iter().flatten().copied());
        let cell = |a: u16, b: u16| (u32::from(a) + u32::from(b)) / 2 * GRID / 65536;
        cells
            .entry((cell(min.1, max.1), cell(min.0, max.0)))
            .or_default()
            .push(p);
    }

    let mut out = Vec::new();
    for group in cells.into_values() {
        let mut chunk = Chunk::default();
        for p in group {
            if chunk.vertices.len() + p.len() > MAX_VERTICES {
                out.push(finish(std::mem::take(&mut chunk)));
            }
            let base = chunk.vertices.len();
            for t in &p.triangles {
                // base + index < MAX_VERTICES, checked above.
                chunk
                    .triangles
                    .push(t.map(|i| u16::try_from(base + i).expect("index fits u16")));
            }
            for ring in &p.rings {
                chunk.vertices.extend_from_slice(ring);
                if with_rings {
                    chunk.rings.push(u16::try_from(ring.len())?);
                }
            }
        }
        if !chunk.vertices.is_empty() {
            out.push(finish(chunk));
        }
    }
    Ok(out)
}

fn finish(mut chunk: Chunk) -> Chunk {
    (chunk.min, chunk.max) = bbox(chunk.vertices.iter().copied());
    chunk
}

fn bbox(points: impl Iterator<Item = Point>) -> (Point, Point) {
    points.fold(((u16::MAX, u16::MAX), (0, 0)), |(min, max), (x, y)| {
        ((min.0.min(x), min.1.min(y)), (max.0.max(x), max.1.max(y)))
    })
}

fn lines(features: &[Value]) -> Result<Vec<Vec<Point>>> {
    let mut out = Vec::new();
    for feature in features {
        let geometry = feature
            .get("geometry")
            .context("feature without geometry")?;
        let coords = geometry
            .get("coordinates")
            .context("geometry without coordinates")?;
        let parts = match geometry.get("type").and_then(Value::as_str) {
            Some("LineString") => vec![coords],
            Some("MultiLineString") => coords
                .as_array()
                .context("expected lines")?
                .iter()
                .collect(),
            other => bail!("unexpected geometry {other:?}"),
        };
        for part in parts {
            let line = points(part)?;
            if line.len() < 2 {
                continue;
            }
            let mut keep = vec![false; line.len()];
            simplify(&line, 0, line.len() - 1, &mut keep);
            out.push(
                line.into_iter()
                    .zip(keep)
                    .filter_map(|(p, k)| k.then_some(p))
                    .collect(),
            );
        }
    }
    Ok(out)
}

fn labels(features: &[Value]) -> Result<Vec<Label>> {
    let mut ranked = Vec::new();
    for feature in features {
        let props = feature
            .get("properties")
            .context("feature without properties")?;
        let number = |key: &str| {
            props
                .get(key)
                .and_then(Value::as_f64)
                .with_context(|| format!("country without {key}"))
        };
        let name = props
            .get("NAME")
            .and_then(Value::as_str)
            .context("country without NAME")?;
        let (x, y) = project(number("LABEL_Y")?, number("LABEL_X")?);
        let min_zoom = number("MIN_LABEL")? as f32;
        ranked.push((
            number("LABELRANK")?,
            Label {
                name: name.to_owned(),
                pos: (quantise(x), quantise(y)),
                min_zoom,
            },
        ));
    }
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.name.cmp(&b.1.name)));
    Ok(ranked.into_iter().map(|(_, label)| label).collect())
}
