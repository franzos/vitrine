//! Geo stage: EXIF GPS coordinates of content not yet scanned.

use std::fs::File;
use std::io::{self, BufReader};
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Mutex, PoisonError};
use std::thread;

use exif::{Exif, In, Tag, Value};

use super::Failure;
use super::writer::Op;
use crate::db::Sha256;
use crate::geo::GeoError;

/// Header reads only; more threads mostly wait on the disk.
const WORKERS: usize = 4;

#[derive(Debug)]
pub(crate) struct Scan {
    pub sha256: Sha256,
    pub path: PathBuf,
    /// `Err` leaves the content pending for the next run.
    pub gps: Result<Option<(f64, f64)>, String>,
}

/// One [`Op::Gps`] per content; stops once `cancel` is set or the writer hangs up.
pub(crate) fn scan_all(pending: Vec<(Sha256, PathBuf)>, cancel: &AtomicBool, tx: &Sender<Op>) {
    let queue = Mutex::new(pending.into_iter());
    let queue = &queue;
    thread::scope(|s| {
        for _ in 0..WORKERS {
            let tx = tx.clone();
            s.spawn(move || {
                while !cancel.load(Ordering::Relaxed) {
                    let next = queue.lock().unwrap_or_else(PoisonError::into_inner).next();
                    let Some((sha256, path)) = next else {
                        break;
                    };
                    let gps = read_gps(&path).map_err(|e| format!("gps: {e}"));
                    if tx.send(Op::Gps(Scan { sha256, path, gps })).is_err() {
                        break;
                    }
                }
            });
        }
    });
}

pub(crate) fn failure(geo_db: &Path, error: GeoError) -> Failure {
    Failure {
        path: geo_db.to_owned(),
        error: format!("geo: {:#}", anyhow::Error::new(error)),
    }
}

/// Latitude and longitude in degrees, south and west negative. `Ok(None)` when the file has no usable GPS position; `Err` only when it can't be read.
pub(crate) fn read_gps(path: &Path) -> io::Result<Option<(f64, f64)>> {
    let file = File::open(path)?;
    // a malformed EXIF block can panic the reader; that's no usable GPS, same as one it rejects
    let exif = panic::catch_unwind(AssertUnwindSafe(|| {
        exif::Reader::new().read_from_container(&mut BufReader::new(file))
    }));
    match exif {
        Ok(Ok(exif)) => Ok(coordinates(&exif)),
        // a truncated file stays truncated; retrying it every run gains nothing
        Ok(Err(exif::Error::Io(e))) if e.kind() != io::ErrorKind::UnexpectedEof => Err(e),
        Ok(Err(_)) | Err(_) => Ok(None),
    }
}

fn coordinates(exif: &Exif) -> Option<(f64, f64)> {
    let lat = degrees(exif, Tag::GPSLatitude, Tag::GPSLatitudeRef, b'N', b'S')?;
    let lon = degrees(exif, Tag::GPSLongitude, Tag::GPSLongitudeRef, b'E', b'W')?;
    let valid = lat.abs() <= 90.0 && lon.abs() <= 180.0 && (lat, lon) != (0.0, 0.0);
    valid.then_some((lat, lon))
}

fn degrees(exif: &Exif, tag: Tag, reference: Tag, positive: u8, negative: u8) -> Option<f64> {
    let Value::Rational(dms) = &exif.get_field(tag, In::PRIMARY)?.value else {
        return None;
    };
    if dms.is_empty() || dms.len() > 3 {
        return None;
    }
    let value: f64 = dms
        .iter()
        .zip([1.0, 60.0, 3600.0])
        .map(|(r, scale)| r.to_f64() / scale)
        .sum();
    if !value.is_finite() {
        return None;
    }
    let Value::Ascii(refs) = &exif.get_field(reference, In::PRIMARY)?.value else {
        return None;
    };
    match refs.first()?.first()?.to_ascii_uppercase() {
        r if r == positive => Some(value),
        r if r == negative => Some(-value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::meta::exif_jpeg::{self, GPS_IFD, Value};
    use super::*;

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    fn gps_jpeg(lat: &[(u32, u32)], lat_ref: &[u8], lon: &[(u32, u32)], lon_ref: &[u8]) -> Vec<u8> {
        exif_jpeg::build(&[(
            GPS_IFD,
            Value::Ifd(&[
                (0x1, Value::Ascii(lat_ref)),
                (0x2, Value::Rational(lat)),
                (0x3, Value::Ascii(lon_ref)),
                (0x4, Value::Rational(lon)),
            ]),
        )])
    }

    fn read_built(bytes: &[u8]) -> Option<(f64, f64)> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("built.jpg");
        std::fs::write(&path, bytes).unwrap();
        read_gps(&path).unwrap()
    }

    fn close(got: Option<(f64, f64)>, want: (f64, f64)) -> bool {
        got.is_some_and(|(lat, lon)| (lat - want.0).abs() < 1e-4 && (lon - want.1).abs() < 1e-4)
    }

    #[test]
    fn reads_the_fixture_with_a_western_longitude() {
        let got = read_gps(&fixture("gps.jpg")).unwrap();
        assert!(close(got, (38.7223, -9.1393)), "{got:?}");
    }

    #[test]
    fn no_exif_is_no_gps() {
        assert_eq!(read_gps(&fixture("jpeg.jpg")).unwrap(), None);
        assert_eq!(read_gps(&fixture("orientation-6.jpg")).unwrap(), None);
    }

    #[test]
    fn dms_and_refs_convert_to_signed_degrees() {
        // 33 deg 51' 54" S, 151 deg 12' 36" E: Sydney
        let got = read_built(&gps_jpeg(
            &[(33, 1), (51, 1), (54, 1)],
            b"S\0",
            &[(151, 1), (12, 1), (36, 1)],
            b"E\0",
        ));
        assert!(close(got, (-33.865, 151.21)), "{got:?}");
        let got = read_built(&gps_jpeg(
            &[(387_223, 10_000)],
            b"N\0",
            &[(91_393, 10_000)],
            b"W\0",
        ));
        assert!(close(got, (38.7223, -9.1393)), "{got:?}");
    }

    #[test]
    fn invalid_coordinates_are_no_gps() {
        let zero = [(0, 1), (0, 1), (0, 1)];
        assert_eq!(read_built(&gps_jpeg(&zero, b"N\0", &zero, b"E\0")), None);
        let lisbon = [(38, 1), (43, 1), (20, 1)];
        let west = [(9, 1), (8, 1), (21, 1)];
        let infinite = [(38, 1), (43, 0), (20, 1)];
        assert_eq!(
            read_built(&gps_jpeg(&infinite, b"N\0", &west, b"W\0")),
            None
        );
        let undefined = [(0, 0), (8, 1), (21, 1)];
        assert_eq!(
            read_built(&gps_jpeg(&lisbon, b"N\0", &undefined, b"W\0")),
            None
        );
        let beyond = [(91, 1), (0, 1), (0, 1)];
        assert_eq!(read_built(&gps_jpeg(&beyond, b"N\0", &west, b"W\0")), None);
        assert_eq!(read_built(&gps_jpeg(&lisbon, b"X\0", &west, b"W\0")), None);
        assert!(read_built(&gps_jpeg(&lisbon, b"N\0", &west, b"W\0")).is_some());
    }

    #[test]
    fn exif_without_gps_is_no_gps() {
        let bytes = exif_jpeg::build(&[(
            exif_jpeg::EXIF_IFD,
            Value::Ifd(&[(0x9003, Value::Ascii(b"2024:05:01 10:11:12\0"))]),
        )]);
        assert_eq!(read_built(&bytes), None);
    }

    #[test]
    fn unreadable_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_gps(&dir.path().join("gone.jpg")).is_err());
        let garbage = dir.path().join("garbage.jpg");
        std::fs::write(&garbage, b"not an image").unwrap();
        assert_eq!(read_gps(&garbage).unwrap(), None);
    }
}
