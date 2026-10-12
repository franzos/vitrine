//! The packed basemap: Natural Earth land, lakes, boundaries and country labels.
//!
//! Points are Web Mercator world `[0, 1]` quantised to `0..=65535` on both axes. Little-endian:
//!
//! ```text
//! pack     "VMAP"  version: u16 = 1
//!          land:   u32 count, chunk*
//!          lakes:  u32 count, chunk*
//!          lines:  u32 count, (u32 count, point*)*
//!          labels: u32 count, (u16 len, UTF-8 name, point, min_zoom: f32)*
//! chunk    bbox min: point, bbox max: point
//!          u16 count, point*            vertices
//!          u32 count, (u16 u16 u16)*    triangles, indices into the chunk's vertices
//!          u16 count, u16*              ring lengths, empty or summing to the vertex count
//! point    x: u16, y: u16
//! ```
//!
//! Rings (outer and holes) lie consecutively in a chunk's vertices, so coasts are stroked from the
//! same vertices the triangles fill. Nothing may follow the labels.

pub const MAGIC: &[u8; 4] = b"VMAP";
pub const VERSION: u16 = 1;

pub type Point = (u16, u16);

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PackError {
    #[error("not a basemap pack")]
    Magic,
    #[error("unsupported basemap pack version {0}")]
    Version(u16),
    #[error("basemap pack is truncated")]
    Truncated,
    #[error("invalid basemap pack: {0}")]
    Invalid(&'static str),
}

/// Whole polygons sharing one vertex array of at most 65,535 vertices.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Chunk {
    pub min: Point,
    pub max: Point,
    pub vertices: Vec<Point>,
    pub triangles: Vec<[u16; 3]>,
    pub rings: Vec<u16>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    pub name: String,
    pub pos: Point,
    pub min_zoom: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pack {
    pub land: Vec<Chunk>,
    pub lakes: Vec<Chunk>,
    pub lines: Vec<Vec<Point>>,
    pub labels: Vec<Label>,
}

/// Fails when a count doesn't fit its field or a chunk is inconsistent.
pub fn encode(pack: &Pack) -> Result<Vec<u8>, PackError> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    put_u16(&mut out, VERSION);
    for layer in [&pack.land, &pack.lakes] {
        put_u32(&mut out, len32(layer.len())?);
        for chunk in layer {
            check(chunk)?;
            put_point(&mut out, chunk.min);
            put_point(&mut out, chunk.max);
            put_u16(&mut out, len16(chunk.vertices.len())?);
            chunk.vertices.iter().for_each(|&p| put_point(&mut out, p));
            put_u32(&mut out, len32(chunk.triangles.len())?);
            for t in &chunk.triangles {
                t.iter().for_each(|&i| put_u16(&mut out, i));
            }
            put_u16(&mut out, len16(chunk.rings.len())?);
            chunk.rings.iter().for_each(|&r| put_u16(&mut out, r));
        }
    }
    put_u32(&mut out, len32(pack.lines.len())?);
    for line in &pack.lines {
        put_u32(&mut out, len32(line.len())?);
        line.iter().for_each(|&p| put_point(&mut out, p));
    }
    put_u32(&mut out, len32(pack.labels.len())?);
    for label in &pack.labels {
        put_u16(&mut out, len16(label.name.len())?);
        out.extend_from_slice(label.name.as_bytes());
        put_point(&mut out, label.pos);
        out.extend_from_slice(&label.min_zoom.to_le_bytes());
    }
    Ok(out)
}

pub fn decode(bytes: &[u8]) -> Result<Pack, PackError> {
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return Err(PackError::Magic);
    }
    let version = r.u16()?;
    if version != VERSION {
        return Err(PackError::Version(version));
    }
    let land = r.chunks()?;
    let lakes = r.chunks()?;
    let n = r.count(4)?;
    let mut lines = Vec::with_capacity(n);
    for _ in 0..n {
        let m = r.count(4)?;
        lines.push(r.points(m)?);
    }
    let n = r.count(10)?;
    let mut labels = Vec::with_capacity(n);
    for _ in 0..n {
        let len = usize::from(r.u16()?);
        let name = str::from_utf8(r.take(len)?)
            .map_err(|_| PackError::Invalid("label name is not UTF-8"))?
            .to_owned();
        let pos = r.point()?;
        let min_zoom = f32::from_le_bytes(r.array()?);
        labels.push(Label {
            name,
            pos,
            min_zoom,
        });
    }
    if !r.0.is_empty() {
        return Err(PackError::Invalid("trailing bytes"));
    }
    Ok(Pack {
        land,
        lakes,
        lines,
        labels,
    })
}

fn check(chunk: &Chunk) -> Result<(), PackError> {
    let n = chunk.vertices.len();
    if n > usize::from(u16::MAX) {
        return Err(PackError::Invalid("chunk has more than 65535 vertices"));
    }
    if chunk
        .triangles
        .iter()
        .flatten()
        .any(|&i| usize::from(i) >= n)
    {
        return Err(PackError::Invalid("triangle index out of range"));
    }
    let ring_sum: usize = chunk.rings.iter().map(|&r| usize::from(r)).sum();
    if !chunk.rings.is_empty() && ring_sum != n {
        return Err(PackError::Invalid(
            "ring lengths don't sum to the vertex count",
        ));
    }
    Ok(())
}

fn len16(n: usize) -> Result<u16, PackError> {
    u16::try_from(n).map_err(|_| PackError::Invalid("count exceeds u16"))
}

fn len32(n: usize) -> Result<u32, PackError> {
    u32::try_from(n).map_err(|_| PackError::Invalid("count exceeds u32"))
}

fn put_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_point(out: &mut Vec<u8>, (x, y): Point) {
    put_u16(out, x);
    put_u16(out, y);
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], PackError> {
        let (head, rest) = self.0.split_at_checked(n).ok_or(PackError::Truncated)?;
        self.0 = rest;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], PackError> {
        let mut a = [0; N];
        a.copy_from_slice(self.take(N)?);
        Ok(a)
    }

    fn u16(&mut self) -> Result<u16, PackError> {
        self.array().map(u16::from_le_bytes)
    }

    fn u32(&mut self) -> Result<u32, PackError> {
        self.array().map(u32::from_le_bytes)
    }

    fn point(&mut self) -> Result<Point, PackError> {
        Ok((self.u16()?, self.u16()?))
    }

    /// A u32 count, rejected unless `count * min_size` bytes remain, which bounds any allocation.
    fn count(&mut self, min_size: usize) -> Result<usize, PackError> {
        let n = usize::try_from(self.u32()?).map_err(|_| PackError::Truncated)?;
        match n.checked_mul(min_size) {
            Some(need) if need <= self.0.len() => Ok(n),
            _ => Err(PackError::Truncated),
        }
    }

    fn u16s(&mut self, n: usize) -> Result<Vec<u16>, PackError> {
        let raw = self.take(n.checked_mul(2).ok_or(PackError::Truncated)?)?;
        Ok(raw
            .as_chunks()
            .0
            .iter()
            .map(|&b| u16::from_le_bytes(b))
            .collect())
    }

    fn points(&mut self, n: usize) -> Result<Vec<Point>, PackError> {
        let raw = self.u16s(n.checked_mul(2).ok_or(PackError::Truncated)?)?;
        Ok(raw.as_chunks().0.iter().map(|&[x, y]| (x, y)).collect())
    }

    fn chunks(&mut self) -> Result<Vec<Chunk>, PackError> {
        // bbox, vertex count, triangle count and ring count
        let n = self.count(16)?;
        let mut chunks = Vec::with_capacity(n);
        for _ in 0..n {
            let min = self.point()?;
            let max = self.point()?;
            let nv = usize::from(self.u16()?);
            let vertices = self.points(nv)?;
            let nt = self.count(6)?;
            let triangles = self.u16s(nt * 3)?.as_chunks().0.to_vec();
            let nr = usize::from(self.u16()?);
            let rings = self.u16s(nr)?;
            let chunk = Chunk {
                min,
                max,
                vertices,
                triangles,
                rings,
            };
            check(&chunk)?;
            chunks.push(chunk);
        }
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Offsets into `sample()`'s encoding: header, land count, first chunk.
    const LAND_COUNT: usize = 6;
    const FIRST_VERTEX_COUNT: usize = 18;
    const FIRST_TRIANGLE_COUNT: usize = 20 + 5 * 4;
    const FIRST_INDEX: usize = FIRST_TRIANGLE_COUNT + 4;

    fn sample() -> Pack {
        Pack {
            land: vec![
                Chunk {
                    min: (10, 20),
                    max: (300, 400),
                    vertices: vec![(10, 20), (300, 20), (300, 400), (10, 400), (150, 200)],
                    triangles: vec![[0, 1, 4], [1, 2, 4], [2, 3, 4]],
                    rings: vec![4, 1],
                },
                Chunk {
                    min: (0, 0),
                    max: (65535, 65535),
                    vertices: vec![(0, 0), (65535, 0), (0, 65535)],
                    triangles: vec![[0, 1, 2]],
                    rings: vec![3],
                },
            ],
            lakes: vec![Chunk {
                min: (5, 5),
                max: (9, 9),
                vertices: vec![(5, 5), (9, 5), (9, 9)],
                triangles: vec![[2, 1, 0]],
                rings: vec![],
            }],
            lines: vec![vec![(1, 2), (3, 4), (5, 6)], vec![]],
            labels: vec![
                Label {
                    name: "Portugal".into(),
                    pos: (31000, 25000),
                    min_zoom: 1.7,
                },
                Label {
                    name: "Côte d'Ivoire".into(),
                    pos: (32000, 33000),
                    min_zoom: 3.0,
                },
            ],
        }
    }

    fn patch(at: usize, value: &[u8]) -> Vec<u8> {
        let mut bytes = encode(&sample()).unwrap();
        bytes[at..at + value.len()].copy_from_slice(value);
        bytes
    }

    #[test]
    fn round_trips() {
        let pack = sample();
        assert_eq!(decode(&encode(&pack).unwrap()), Ok(pack));
        let empty = Pack::default();
        assert_eq!(decode(&encode(&empty).unwrap()), Ok(empty));
    }

    #[test]
    fn rejects_bad_magic() {
        assert_eq!(decode(&patch(0, b"VMAQ")), Err(PackError::Magic));
    }

    #[test]
    fn rejects_wrong_version() {
        assert_eq!(
            decode(&patch(4, &2u16.to_le_bytes())),
            Err(PackError::Version(2))
        );
    }

    #[test]
    fn rejects_truncation_at_every_offset() {
        let bytes = encode(&sample()).unwrap();
        for len in 0..bytes.len() {
            assert!(
                decode(&bytes[..len]).is_err(),
                "prefix of {len} bytes decoded"
            );
        }
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut bytes = encode(&sample()).unwrap();
        bytes.push(0);
        assert!(decode(&bytes).is_err());
    }

    #[test]
    fn rejects_index_past_vertices() {
        assert_eq!(
            decode(&patch(FIRST_INDEX, &5u16.to_le_bytes())),
            Err(PackError::Invalid("triangle index out of range"))
        );
    }

    #[test]
    fn rejects_counts_overflowing_the_buffer() {
        let max32 = u32::MAX.to_le_bytes();
        assert_eq!(
            decode(&patch(LAND_COUNT, &max32)),
            Err(PackError::Truncated)
        );
        assert_eq!(
            decode(&patch(FIRST_TRIANGLE_COUNT, &max32)),
            Err(PackError::Truncated)
        );
        assert!(decode(&patch(FIRST_VERTEX_COUNT, &u16::MAX.to_le_bytes())).is_err());
    }

    #[test]
    fn rejects_invalid_utf8() {
        let mut pack = sample();
        pack.labels[0].name = "ab".into();
        let mut bytes = encode(&pack).unwrap();
        let at = bytes.windows(2).rposition(|w| w == b"ab").unwrap();
        bytes[at] = 0xff;
        assert!(matches!(decode(&bytes), Err(PackError::Invalid(_))));
    }

    #[test]
    fn rejects_mismatched_rings() {
        let mut pack = sample();
        pack.land[0].rings = vec![4];
        assert!(encode(&pack).is_err());
        let bytes = encode(&sample()).unwrap();
        let mut bad = bytes.clone();
        // first chunk's first ring length, after its 3 triangles and ring count
        let at = FIRST_INDEX + 3 * 6 + 2;
        bad[at..at + 2].copy_from_slice(&3u16.to_le_bytes());
        assert!(matches!(decode(&bad), Err(PackError::Invalid(_))));
    }

    #[test]
    fn rejects_oversized_chunks_on_encode() {
        let mut pack = Pack::default();
        pack.land.push(Chunk {
            vertices: vec![(0, 0); 65536],
            ..Chunk::default()
        });
        assert!(encode(&pack).is_err());
    }
}
