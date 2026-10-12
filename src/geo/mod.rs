//! The GeoNames gazetteer in `geo.db`: places by name and by nearest point.

pub mod build;
mod db;
pub mod events;
pub mod fetch;

use std::io;
use std::path::PathBuf;

pub use build::{Rows, build};
pub use db::GeoDb;

pub const ATTRIBUTION: &str = "Place data from GeoNames (https://www.geonames.org/), licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).";

#[derive(Debug, thiserror::Error)]
pub enum GeoError {
    #[error("failed to read {file}")]
    Read {
        file: &'static str,
        source: io::Error,
    },
    #[error("failed to access {}", path.display())]
    Io { path: PathBuf, source: io::Error },
    #[error("failed to open {}", path.display())]
    Open {
        path: PathBuf,
        source: rusqlite::Error,
    },
    #[error("gazetteer has {rows} places, expected at least {min}")]
    TooFewRows { rows: u64, min: u64 },
    #[error("failed to download {url}")]
    Download {
        url: String,
        #[source]
        source: Box<ureq::Error>,
    },
    #[error("failed to open {file}")]
    Zip {
        file: &'static str,
        source: zip::result::ZipError,
    },
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Place {
    pub geoname_id: i64,
    pub name: String,
    /// GeoNames admin1 code, unique within the country.
    pub region_code: String,
    /// Empty when GeoNames has no name for the region.
    pub region: String,
    /// ISO 3166-1 alpha-2.
    pub country_code: String,
    pub country: String,
    /// `(lat, lon)`; `None` when unknown.
    pub coords: Option<(f64, f64)>,
}

impl Place {
    /// `name, region, country`, leaving out empty parts.
    pub fn label(&self) -> String {
        [&self.name, &self.region, &self.country]
            .into_iter()
            .filter(|part| !part.is_empty())
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    use std::path::Path;

    use super::GeoDb;

    pub const ADMIN1: &str = "\
FR.11\tÎle-de-France\tIle-de-France\t3012874
PT.14\tLisbon\tLisbon\t2267056
PT.17\tPorto\tPorto\t2735941
DE.02\tBavaria\tBavaria\t2951839
US.TX\tTexas\tTexas\t4736286
";

    pub const COUNTRIES: &str = "\
# ISO\tISO3\tISO-Numeric\tfips\tCountry\tCapital
#
FR\tFRA\t250\tFR\tFrance\tParis
PT\tPRT\t620\tPO\tPortugal\tLisbon
DE\tDEU\t276\tGM\tGermany\tBerlin
US\tUSA\t840\tUS\tUnited States\tWashington
XX\tXXX\t000\tXX\tNowhere\t
";

    /// One cities1000 row with its 19 columns.
    #[allow(clippy::too_many_arguments)]
    pub fn city(
        id: i64,
        name: &str,
        ascii: &str,
        alternates: &str,
        lat: f64,
        lon: f64,
        country: &str,
        admin1: &str,
        population: u64,
    ) -> String {
        format!(
            "{id}\t{name}\t{ascii}\t{alternates}\t{lat}\t{lon}\tP\tPPL\t{country}\t\t{admin1}\t\t\t\t{population}\t\t0\tEurope/Lisbon\t2024-01-01\n"
        )
    }

    pub fn build(dir: &Path, cities: &str) -> GeoDb {
        let path = dir.join("geo.db");
        super::build(
            &path,
            vec![super::Rows::all("cities1000.txt", cities.as_bytes())],
            ADMIN1.as_bytes(),
            COUNTRIES.as_bytes(),
            1,
        )
        .unwrap();
        GeoDb::open(&path).unwrap().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn place(region: &str) -> Place {
        Place {
            geoname_id: 1,
            name: "Porto".into(),
            region_code: "17".into(),
            region: region.into(),
            country_code: "PT".into(),
            country: "Portugal".into(),
            coords: None,
        }
    }

    #[test]
    fn label_joins_name_region_and_country() {
        assert_eq!(
            place("Porto District").label(),
            "Porto, Porto District, Portugal"
        );
    }

    #[test]
    fn label_skips_an_empty_region() {
        assert_eq!(place("").label(), "Porto, Portugal");
    }
}
