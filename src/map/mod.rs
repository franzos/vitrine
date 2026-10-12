//! Map logic without egui: Web Mercator projection, the viewport, clustering, city labels and the packed basemap.

pub mod cluster;
pub mod labels;
pub mod pack;
pub mod projection;

pub use projection::{Bounds, MAX_LAT, MAX_ZOOM, Rect, Viewport, project, unproject};
