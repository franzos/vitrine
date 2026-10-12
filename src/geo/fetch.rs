use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::{GeoError, Rows};

pub const BASE_URL: &str = "https://download.geonames.org/export/dump";

const CITIES_ZIP: &str = "cities1000.zip";
const CITIES: &str = "cities1000.txt";
const ALL_ZIP: &str = "allCountries.zip";
const ALL: &str = "allCountries.txt";
const ADMIN1: &str = "admin1CodesASCII.txt";
const COUNTRIES: &str = "countryInfo.txt";

/// Which GeoNames places go into the gazetteer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Coverage {
    /// Places with 1,000+ people (cities1000, about 165,000, 10 MB download)
    #[default]
    Cities,
    /// Cities plus inhabited islands with 1,000+ people (adds allCountries, 420 MB download)
    Islands,
    /// Every populated place and inhabited island (allCountries, about 5 million, 420 MB download)
    Full,
}

impl Coverage {
    /// Far fewer places than this means a broken download.
    pub fn min_places(self) -> u64 {
        match self {
            Self::Cities | Self::Islands => 100_000,
            Self::Full => 1_000_000,
        }
    }

    fn sources(self) -> &'static [Source] {
        match self {
            Self::Cities => &[CITIES_SOURCE],
            Self::Islands => &[CITIES_SOURCE, ISLANDS_SOURCE],
            Self::Full => &[ALL_SOURCE],
        }
    }
}

/// A zipped place table and which of its rows to keep.
struct Source {
    zip: &'static str,
    entry: &'static str,
    keep: fn(&[&str]) -> bool,
    /// Total download time, since ureq has no idle timeout.
    timeout: Duration,
}

const CITIES_SOURCE: Source = Source {
    zip: CITIES_ZIP,
    entry: CITIES,
    keep: |_| true,
    timeout: Duration::from_secs(600),
};

const ISLANDS_SOURCE: Source = Source {
    zip: ALL_ZIP,
    entry: ALL,
    keep: |f| inhabited_island(f, 1000),
    timeout: Duration::from_secs(3600),
};

const ALL_SOURCE: Source = Source {
    zip: ALL_ZIP,
    entry: ALL,
    keep: |f| populated(f) || inhabited_island(f, 1),
    timeout: Duration::from_secs(3600),
};

/// GeoNames files some inhabited islands, like most of the Maldives, as islands rather than places.
fn inhabited_island(f: &[&str], min_population: i64) -> bool {
    f.get(7) == Some(&"ISL")
        && f.get(14)
            .and_then(|p| p.parse::<i64>().ok())
            .is_some_and(|p| p >= min_population)
}

/// Populated places, without historical, abandoned and destroyed ones.
fn populated(f: &[&str]) -> bool {
    f.get(6) == Some(&"P") && !matches!(f.get(7), Some(&("PPLH" | "PPLQ" | "PPLW" | "PPLCH")))
}

/// Size caps in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub zip: u64,
    pub text: u64,
    /// Inflated size of a place table.
    pub entry: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            zip: 1 << 30,
            text: 4 << 20,
            entry: 4 << 30,
        }
    }
}

/// Downloads the GeoNames tables for `coverage` and builds the gazetteer at `dest`, returning the number of places.
pub fn fetch(dest: &Path, coverage: Coverage) -> Result<u64, GeoError> {
    fetch_with(
        BASE_URL,
        dest,
        coverage,
        Limits::default(),
        coverage.min_places(),
    )
}

/// Replaces `dest` only on success; on error `dest` is left as it was.
pub fn fetch_with(
    base_url: &str,
    dest: &Path,
    coverage: Coverage,
    limits: Limits,
    min_rows: u64,
) -> Result<u64, GeoError> {
    let sources = coverage.sources();
    let part = part_path(dest, None);
    let zips: Vec<PathBuf> = sources
        .iter()
        .map(|source| part_path(dest, Some(source.zip)))
        .collect();
    let remove_parts = || -> Result<(), GeoError> {
        for path in zips.iter().chain([&part]) {
            match fs::remove_file(path) {
                Err(e) if e.kind() != io::ErrorKind::NotFound => {
                    return Err(io_err(path)(e));
                }
                _ => {}
            }
        }
        Ok(())
    };
    remove_parts()?;

    let result: Result<u64, GeoError> = (|| {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(60)))
            // without it a stalled transfer blocks forever
            .timeout_recv_body(Some(Duration::from_secs(600)))
            .build()
            .into();
        if let Some(dir) = dest.parent() {
            fs::create_dir_all(dir).map_err(io_err(dir))?;
        }
        for (source, path) in sources.iter().zip(&zips) {
            download_to(&agent, base_url, source, limits.zip, path)?;
        }
        let admin1 = download(&agent, base_url, ADMIN1, limits.text)?;
        let countries = download(&agent, base_url, COUNTRIES, limits.text)?;

        let mut archives = sources
            .iter()
            .zip(&zips)
            .map(|(source, path)| {
                let file = File::open(path).map_err(io_err(path))?;
                zip::ZipArchive::new(BufReader::new(file)).map_err(|e| GeoError::Zip {
                    file: source.zip,
                    source: e,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tables = archives
            .iter_mut()
            .zip(sources)
            .map(|(archive, source)| {
                let entry = archive.by_name(source.entry).map_err(|e| GeoError::Zip {
                    file: source.zip,
                    source: e,
                })?;
                Ok(Rows {
                    file: source.entry,
                    reader: Box::new(BufReader::new(Capped {
                        inner: entry,
                        left: limits.entry,
                        file: source.entry,
                    })),
                    keep: source.keep,
                })
            })
            .collect::<Result<Vec<_>, GeoError>>()?;

        // build reads each table to EOF before it commits, so a CRC mismatch fails it
        let rows = super::build(
            &part,
            tables,
            admin1.as_slice(),
            countries.as_slice(),
            min_rows,
        )?;
        File::open(&part)
            .and_then(|file| file.sync_all())
            .map_err(io_err(&part))?;
        fs::rename(&part, dest).map_err(io_err(dest))?;
        if let Some(dir) = dest.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            File::open(dir)
                .and_then(|dir| dir.sync_all())
                .map_err(io_err(dir))?;
        }
        Ok(rows)
    })();
    let cleanup = remove_parts();
    let rows = result?;
    cleanup?;
    Ok(rows)
}

fn io_err(path: &Path) -> impl FnOnce(io::Error) -> GeoError {
    let path = path.to_owned();
    move |source| GeoError::Io { path, source }
}

/// `dest` with `.part`, or a download for it as `<dest>.<name>.part`.
fn part_path(dest: &Path, name: Option<&str>) -> PathBuf {
    let mut part = OsString::from(dest);
    if let Some(name) = name {
        part.push(".");
        part.push(name);
    }
    part.push(".part");
    part.into()
}

fn download(
    agent: &ureq::Agent,
    base_url: &str,
    name: &str,
    cap: u64,
) -> Result<Vec<u8>, GeoError> {
    let url = format!("{base_url}/{name}");
    agent
        .get(&url)
        .call()
        .and_then(|response| {
            response
                .into_body()
                .into_with_config()
                // +1: ureq errors on any read at the limit, even the one that sees EOF
                .limit(cap + 1)
                .read_to_vec()
        })
        .map_err(|source| GeoError::Download {
            url,
            source: Box::new(source),
        })
}

/// Streams `source.zip` to `path`, too large to hold in memory.
fn download_to(
    agent: &ureq::Agent,
    base_url: &str,
    source: &Source,
    cap: u64,
    path: &Path,
) -> Result<(), GeoError> {
    let url = format!("{base_url}/{}", source.zip);
    let download_err = |source: ureq::Error| GeoError::Download {
        url: url.clone(),
        source: Box::new(source),
    };
    let response = agent
        .get(&url)
        .config()
        .timeout_recv_body(Some(source.timeout))
        .build()
        .call()
        .map_err(download_err)?;
    let mut file = File::create(path).map_err(io_err(path))?;
    let mut body = response
        .into_body()
        .into_with_config()
        .limit(cap + 1)
        .reader();
    io::copy(&mut body, &mut file)
        .map_err(|e| download_err(e.downcast::<ureq::Error>().unwrap_or_else(ureq::Error::Io)))?;
    Ok(())
}

/// Fails reads once more than `left` bytes have come through, against zip bombs.
struct Capped<R> {
    inner: R,
    left: u64,
    file: &'static str,
}

impl<R: Read> Read for Capped<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.left = self.left.checked_sub(n as u64).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::FileTooLarge,
                format!("{} inflates past the size cap", self.file),
            )
        })?;
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, Cursor, Write};
    use std::net::TcpListener;
    use std::thread;

    use zip::CompressionMethod;
    use zip::write::SimpleFileOptions;

    use super::*;
    use crate::geo::GeoDb;
    use crate::geo::fixtures::{ADMIN1 as ADMIN1_TXT, COUNTRIES as COUNTRIES_TXT, city};

    const OLD_DB: &[u8] = b"previous geo.db";

    /// Serves `files` by path; anything else is a 404.
    fn serve(files: Vec<(&'static str, Vec<u8>)>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request = String::new();
                reader.read_line(&mut request).unwrap();
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap() > 2 {
                    line.clear();
                }
                let path = request.split(' ').nth(1).unwrap_or("");
                let found = files.iter().find(|(name, _)| path == format!("/{name}"));
                let (status, body) = match found {
                    Some((_, body)) => ("200 OK", body.as_slice()),
                    None => ("404 Not Found", &b""[..]),
                };
                write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(body).unwrap();
            }
        });
        base
    }

    fn cities(n: i64) -> String {
        (1..=n)
            .map(|id| {
                city(
                    id,
                    &format!("Town {id}"),
                    &format!("Town {id}"),
                    "",
                    41.0,
                    -8.0,
                    "PT",
                    "17",
                    1000,
                )
            })
            .collect()
    }

    fn zip_of(text: &str, method: CompressionMethod) -> Vec<u8> {
        zip_named(CITIES, text, method)
    }

    fn zip_named(entry: &str, text: &str, method: CompressionMethod) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        writer
            .start_file(
                entry,
                SimpleFileOptions::default().compression_method(method),
            )
            .unwrap();
        writer.write_all(text.as_bytes()).unwrap();
        writer.finish().unwrap().into_inner()
    }

    fn files(zip: Vec<u8>) -> Vec<(&'static str, Vec<u8>)> {
        vec![
            (CITIES_ZIP, zip),
            (ADMIN1, ADMIN1_TXT.as_bytes().to_vec()),
            (COUNTRIES, COUNTRIES_TXT.as_bytes().to_vec()),
        ]
    }

    fn geonames(n: i64) -> String {
        serve(files(zip_of(&cities(n), CompressionMethod::Deflated)))
    }

    /// A data dir holding a previous `geo.db` and an unrelated file.
    fn data_dir() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("geo.db"), OLD_DB).unwrap();
        fs::write(dir.path().join("catalog.db"), b"catalog").unwrap();
        let dest = dir.path().join("geo.db");
        (dir, dest)
    }

    fn assert_untouched(dest: &Path) {
        assert_eq!(fs::read(dest).unwrap(), OLD_DB);
        assert!(!part_path(dest, None).exists());
        let dir = dest.parent().unwrap();
        assert_eq!(fs::read(dir.join("catalog.db")).unwrap(), b"catalog");
        assert_eq!(fs::read_dir(dir).unwrap().count(), 2);
    }

    fn count(dest: &Path) -> u64 {
        GeoDb::open(dest).unwrap().unwrap().count()
    }

    #[test]
    fn downloads_and_builds() {
        let base = geonames(3);
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("data").join("geo.db");

        assert_eq!(
            fetch_with(&base, &dest, Coverage::Cities, Limits::default(), 3).unwrap(),
            3
        );
        assert_eq!(count(&dest), 3);
        assert_eq!(fs::read_dir(dest.parent().unwrap()).unwrap().count(), 1);
    }

    /// One allCountries row with feature class `class` and code `code`.
    fn feature(id: i64, class: &str, code: &str, population: u64) -> String {
        city(
            id,
            &format!("Place {id}"),
            "",
            "",
            0.2,
            73.2,
            "PT",
            "17",
            population,
        )
        .replace("\tP\tPPL\t", &format!("\t{class}\t{code}\t"))
    }

    fn all_countries() -> String {
        [
            feature(10, "P", "PPL", 0),
            feature(11, "P", "PPLH", 0),
            feature(12, "T", "ISL", 1104),
            feature(13, "T", "ISL", 500),
            feature(14, "T", "ISL", 0),
            feature(15, "S", "AIRP", 0),
        ]
        .concat()
    }

    fn ids(dest: &Path) -> Vec<i64> {
        rusqlite::Connection::open(dest)
            .unwrap()
            .prepare("SELECT geoname_id FROM places ORDER BY geoname_id")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    }

    #[test]
    fn coverage_picks_the_rows() {
        let mut files = files(zip_of(&cities(2), CompressionMethod::Deflated));
        files.push((
            ALL_ZIP,
            zip_named(ALL, &all_countries(), CompressionMethod::Deflated),
        ));
        let base = serve(files);
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("geo.db");

        fetch_with(&base, &dest, Coverage::Islands, Limits::default(), 1).unwrap();
        assert_eq!(ids(&dest), [1, 2, 12]);
        fetch_with(&base, &dest, Coverage::Full, Limits::default(), 1).unwrap();
        assert_eq!(ids(&dest), [10, 12, 13]);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn rerun_replaces_the_gazetteer() {
        let (_dir, dest) = data_dir();
        fs::write(part_path(&dest, None), b"stale").unwrap();

        assert_eq!(
            fetch_with(&geonames(2), &dest, Coverage::Cities, Limits::default(), 1).unwrap(),
            2
        );
        assert_eq!(
            fetch_with(&geonames(5), &dest, Coverage::Cities, Limits::default(), 1).unwrap(),
            5
        );
        assert_eq!(count(&dest), 5);
        assert!(!part_path(&dest, None).exists());
        assert_eq!(fs::read_dir(dest.parent().unwrap()).unwrap().count(), 2);
    }

    #[test]
    fn missing_file_keeps_the_old_gazetteer() {
        let mut files = files(zip_of(&cities(2), CompressionMethod::Deflated));
        files.pop();
        let base = serve(files);
        let (_dir, dest) = data_dir();
        fs::write(part_path(&dest, None), b"stale").unwrap();

        let err = fetch_with(&base, &dest, Coverage::Cities, Limits::default(), 1).unwrap_err();
        assert!(
            matches!(&err, GeoError::Download { url, .. } if url.ends_with(COUNTRIES)),
            "{err:?}"
        );
        assert_untouched(&dest);
    }

    #[test]
    fn too_few_rows_keep_the_old_gazetteer() {
        let (_dir, dest) = data_dir();

        let err =
            fetch_with(&geonames(2), &dest, Coverage::Cities, Limits::default(), 5).unwrap_err();
        assert!(
            matches!(err, GeoError::TooFewRows { rows: 2, min: 5 }),
            "{err:?}"
        );
        assert_untouched(&dest);
    }

    #[test]
    fn oversized_body_is_rejected() {
        let (_dir, dest) = data_dir();
        let limits = Limits {
            text: 16,
            ..Limits::default()
        };

        let err = fetch_with(&geonames(2), &dest, Coverage::Cities, limits, 1).unwrap_err();
        assert!(
            matches!(&err, GeoError::Download { url, source }
                if url.ends_with(ADMIN1) && matches!(**source, ureq::Error::BodyExceedsLimit(_))),
            "{err:?}"
        );
        assert_untouched(&dest);

        let limits = Limits {
            zip: 16,
            ..Limits::default()
        };
        let err = fetch_with(&geonames(2), &dest, Coverage::Cities, limits, 1).unwrap_err();
        assert!(
            matches!(&err, GeoError::Download { url, source }
                if url.ends_with(CITIES_ZIP) && matches!(**source, ureq::Error::BodyExceedsLimit(_))),
            "{err:?}"
        );
        assert_untouched(&dest);
    }

    #[test]
    fn entry_inflating_past_the_cap_is_rejected() {
        let text = cities(200);
        let zip = zip_of(&text, CompressionMethod::Deflated);
        assert!(zip.len() < text.len() / 4);
        let base = serve(files(zip));
        let (_dir, dest) = data_dir();
        let limits = Limits {
            entry: text.len() as u64 / 2,
            ..Limits::default()
        };

        let err = fetch_with(&base, &dest, Coverage::Cities, limits, 1).unwrap_err();
        assert!(
            matches!(&err, GeoError::Read { file: CITIES, source }
                if source.kind() == io::ErrorKind::FileTooLarge),
            "{err:?}"
        );
        assert_untouched(&dest);
    }

    #[test]
    fn crc_mismatch_is_rejected() {
        let mut zip = zip_of(&cities(3), CompressionMethod::Stored);
        let at = zip.windows(6).position(|w| w == b"Town 2").unwrap();
        zip[at] = b'G';
        let base = serve(files(zip));
        let (_dir, dest) = data_dir();

        let err = fetch_with(&base, &dest, Coverage::Cities, Limits::default(), 1).unwrap_err();
        assert!(
            matches!(&err, GeoError::Read { file: CITIES, source }
                if source.kind() == io::ErrorKind::InvalidData
                    && source.to_string().contains("checksum")),
            "{err:?}"
        );
        assert_untouched(&dest);
    }
}
