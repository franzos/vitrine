//! The map: the basemap, clustered image bubbles and labels, with pan and zoom.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;

use eframe::egui::{
    self, Color32, CursorIcon, FontId, Galley, Label, Painter, Pos2, Rect, RichText, Sense, Stroke,
    TextureHandle, Vec2, epaint::RectShape, pos2, vec2,
};

use super::basemap::Basemap;
use super::grid::{Grid, bucket_for};
use super::viewer::zoom_factor;
use crate::catalog::filter::Area;
use crate::catalog::images::{Entry, ImageId, Kind};
use crate::db::Sha256;
use crate::geo::GeoDb;
use crate::map::cluster::{self, Cluster, Point};
use crate::map::labels::{self, City, LIST_MAX_LEVEL, LIST_MIN_POPULATION};
use crate::map::{self, Bounds, Viewport, project, unproject};
use crate::thumbs::generate::BUCKETS;

const FIT_MAX_ZOOM: f64 = 12.0;
const FOCUS_ZOOM: f64 = 11.0;
/// Clusters nearest the centre that get a sample thumbnail.
const SAMPLES: usize = 64;
/// Smallest padding per axis of a clicked cluster's area, in degrees.
const MIN_PAD: f64 = 0.005;
const CITY_LIMIT: u32 = 500;
/// Label candidates measured per frame, per kind.
const LABEL_CANDIDATES: usize = 200;
const EMPTY: &str = "No located images in this view";
const GEONAMES: &str = " | GeoNames";

pub(super) const BUBBLE: Color32 = Color32::from_rgb(204, 88, 40);
const HIGHLIGHT: Color32 = Color32::from_rgb(255, 210, 0);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MapAction {
    /// A bubble was clicked; the area holds its members.
    Cluster(Area),
}

#[derive(Debug, Default)]
pub struct MapOutput {
    pub action: Option<MapAction>,
    /// Sample ids to resolve and pass to the next `show`.
    pub wanted: Vec<ImageId>,
    pub error: Option<String>,
}

pub struct MapView {
    basemap: Box<dyn Basemap>,
    viewport: Viewport,
    fitted: bool,
    fit_pending: bool,
    points: Vec<Point>,
    extent: Option<Bounds>,
    generation: u64,
    cluster_key: Option<(u32, u64)>,
    clusters: Vec<Cluster>,
    /// Sample entry per cluster `first`; `None` when the catalog has no active row for it.
    samples: HashMap<ImageId, Option<Entry>>,
    /// Ids returned as wanted by the last `show`.
    requested: Vec<ImageId>,
    highlight: Option<ImageId>,
    /// World position of the highlighted image, once it's among the points.
    highlight_pos: Option<(f64, f64)>,
    geo_db: Option<PathBuf>,
    gazetteer: Option<GeoDb>,
    city_list: CityList,
    city_box: Option<CityBox>,
    rect: map::Rect,
    area: Option<Area>,
    attribution: String,
}

/// Cities for levels up to `LIST_MAX_LEVEL`, loaded off the UI thread.
enum CityList {
    Idle,
    Loading(Receiver<Result<Vec<City>, String>>),
    Ready(Vec<City>),
    /// The loader thread couldn't start; not retried.
    Failed,
}

/// The result of a box query for one level.
struct CityBox {
    level: u32,
    area: Area,
    cities: Vec<City>,
}

struct Placed {
    galley: Arc<Galley>,
    pos: Pos2,
    dot: Option<Pos2>,
}

impl MapView {
    #[must_use]
    pub fn new(basemap: Box<dyn Basemap>) -> Self {
        let attribution = basemap.attribution().to_owned();
        Self {
            basemap,
            viewport: Viewport::default(),
            fitted: false,
            fit_pending: false,
            points: Vec::new(),
            extent: None,
            generation: 0,
            cluster_key: None,
            clusters: Vec::new(),
            samples: HashMap::new(),
            requested: Vec::new(),
            highlight: None,
            highlight_pos: None,
            geo_db: None,
            gazetteer: None,
            city_list: CityList::Idle,
            city_box: None,
            rect: map::Rect {
                x: 0.0,
                y: 0.0,
                w: 0.0,
                h: 0.0,
            },
            area: None,
            attribution,
        }
    }

    /// The `geo.db` for city labels; `None` leaves them out.
    pub fn set_geo_db(&mut self, path: Option<PathBuf>) {
        if self.geo_db != path {
            self.geo_db = path;
            self.gazetteer = None;
            self.city_list = CityList::Idle;
            self.city_box = None;
        }
    }

    /// Replaces the points, `(id, lat, lon)` in path order. Equal points keep the cluster and
    /// sample caches.
    pub fn set_points(&mut self, points: &[(ImageId, f64, f64)]) {
        let points: Vec<Point> = points
            .iter()
            .map(|&(id, lat, lon)| {
                let (x, y) = project(lat, lon);
                Point { id: id.0, x, y }
            })
            .collect();
        if points == self.points {
            return;
        }
        self.points = points;
        self.extent = self.points.split_first().map(|(first, rest)| {
            rest.iter().fold(
                Bounds {
                    min: (first.x, first.y),
                    max: (first.x, first.y),
                },
                |b, p| Bounds {
                    min: (b.min.0.min(p.x), b.min.1.min(p.y)),
                    max: (b.max.0.max(p.x), b.max.1.max(p.y)),
                },
            )
        });
        self.generation += 1;
        self.samples.clear();
        self.requested.clear();
        self.resolve_highlight();
    }

    /// Fits the points on the next frame that has any.
    pub fn request_fit(&mut self) {
        self.fit_pending = true;
    }

    /// Centres on `(lat, lon)` at city zoom and highlights `id`'s bubble; cancels any fit.
    pub fn focus(&mut self, id: ImageId, lat: f64, lon: f64) {
        self.fitted = true;
        self.fit_pending = false;
        self.viewport = Viewport {
            center: project(lat, lon),
            zoom: FOCUS_ZOOM,
        };
        self.highlight = Some(id);
        self.resolve_highlight();
    }

    /// Draws the map into the available rect. `area` is outlined. `samples` holds the entries
    /// resolved for the ids wanted by the previous call; `None` when they weren't resolved.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        grid: &mut Grid,
        area: Option<Area>,
        samples: Option<&[Entry]>,
    ) -> MapOutput {
        let mut output = MapOutput::default();
        let screen = ui.max_rect();
        let response = ui.allocate_rect(screen, Sense::click_and_drag());
        let rect = map_rect(screen);
        self.rect = rect;
        self.area = area;
        if let Some(samples) = samples {
            for id in self.requested.drain(..) {
                let entry = samples.iter().find(|e| e.id == id).cloned();
                self.samples.insert(id, entry);
            }
        }
        if !screen.is_positive() {
            // Keeps a pending fit for a real rect and still cancels the grid's stale queue.
            grid.thumbnails(ui.ctx(), &[], BUCKETS[0]);
            output.wanted.clone_from(&self.requested);
            return output;
        }

        if (!self.fitted || self.fit_pending)
            && let Some(extent) = self.extent
        {
            self.viewport = Viewport::fit(extent, fit_rect(rect), FIT_MAX_ZOOM);
            self.fitted = true;
            self.fit_pending = false;
        }
        self.viewport.clamp(rect);
        self.refresh_clusters();
        output.action = self.interact(ui, &response, rect);
        self.refresh_clusters();

        let painter = ui.painter_at(screen);
        self.basemap.paint(&painter, screen, &self.viewport);
        if let Some(area) = area {
            self.paint_area(&painter, rect, area);
        }
        let mut occupied = self.paint_bubbles(ui, &painter, grid, rect, &mut output.wanted);

        let visuals = ui.visuals();
        let font = FontId::proportional(11.0);
        let widest = painter.layout_no_wrap(
            format!("{}{GEONAMES}", self.basemap.attribution()),
            font.clone(),
            visuals.text_color(),
        );
        let corner = screen.right_bottom() - vec2(6.0, 4.0);
        occupied.push(map_rect(Rect::from_min_max(corner - widest.size(), corner)));

        let countries = self.country_labels(&painter, visuals, rect);
        let level = cluster::level(self.viewport.zoom);
        let view = lat_lon_box(self.viewport.bounds(rect));
        if let Err(e) = self.update_cities(ui.ctx(), level, view) {
            output.error = Some(format!("cannot load city labels: {e}"));
        }
        let cities = self.city_labels(&painter, visuals, rect, level, &view);
        let mut placed = labels::place(countries, &mut occupied);
        let first_city = placed.len();
        placed.extend(labels::place(cities, &mut occupied));
        let with_cities = placed.len() > first_city;
        for (i, label) in placed.into_iter().enumerate() {
            let color = if i < first_city {
                visuals.weak_text_color()
            } else {
                visuals.text_color()
            };
            if let Some(dot) = label.dot {
                painter.circle_filled(dot, 2.5, color);
            }
            painter.galley(label.pos, label.galley, color);
        }

        self.attribution = format!(
            "{}{}",
            self.basemap.attribution(),
            if with_cities { GEONAMES } else { "" }
        );
        let galley = painter.layout_no_wrap(self.attribution.clone(), font, visuals.text_color());
        let text = Rect::from_min_max(corner - galley.size(), corner);
        painter.rect_filled(
            text.expand2(vec2(4.0, 2.0)),
            2.0,
            visuals.extreme_bg_color.gamma_multiply(0.8),
        );
        painter.galley(text.min, galley, visuals.text_color());

        if self.points.is_empty() {
            ui.put(
                Rect::from_center_size(screen.center(), vec2(screen.width(), 24.0)),
                Label::new(RichText::new(EMPTY).strong()),
            );
        }
        output
    }

    #[cfg(test)]
    pub fn viewport(&self) -> Viewport {
        self.viewport
    }

    #[cfg(test)]
    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.fitted = true;
        self.viewport = viewport;
    }

    #[cfg(test)]
    pub fn attribution(&self) -> &str {
        &self.attribution
    }

    #[cfg(test)]
    pub fn outlined_area(&self) -> Option<Area> {
        self.area
    }

    /// Sample ids resolved so far, with or without an entry.
    #[cfg(test)]
    pub fn sampled(&self) -> usize {
        self.samples.len()
    }

    /// The highlighted image, once it's among the points.
    #[cfg(test)]
    pub fn highlighted(&self) -> Option<ImageId> {
        self.highlight.filter(|_| self.highlight_pos.is_some())
    }

    /// Screen centre, as of the last frame, of the bubble holding `id`.
    #[cfg(test)]
    pub fn cluster_pos(&self, id: ImageId) -> Option<Pos2> {
        let point = self.points.iter().find(|p| p.id == id.0)?;
        let cluster = self
            .clusters
            .iter()
            .find(|c| holds(c, (point.x, point.y)))?;
        Some(screen_pos(
            self.viewport.to_screen((cluster.x, cluster.y), self.rect),
        ))
    }

    fn resolve_highlight(&mut self) {
        self.highlight_pos = self.highlight.and_then(|id| {
            self.points
                .iter()
                .find(|p| p.id == id.0)
                .map(|p| (p.x, p.y))
        });
    }

    /// Reclusters only when the level or the points change.
    fn refresh_clusters(&mut self) {
        let key = (cluster::level(self.viewport.zoom), self.generation);
        if self.cluster_key != Some(key) {
            self.clusters = cluster::cluster(&self.points, key.0);
            self.cluster_key = Some(key);
        }
    }

    fn interact(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        rect: map::Rect,
    ) -> Option<MapAction> {
        let mut action = None;
        if response.clicked()
            && let Some(pos) = response.interact_pointer_pos()
        {
            let pointer = (f64::from(pos.x), f64::from(pos.y));
            if let Some(hit) = cluster::hit(&self.clusters, &self.viewport, rect, pointer) {
                action = Some(MapAction::Cluster(padded_area(hit)));
            } else if response.double_clicked() {
                self.viewport.zoom_at(pointer, 2.0, rect);
            }
        }
        let drag = response.drag_delta();
        if drag != Vec2::ZERO {
            self.viewport.pan((f64::from(drag.x), f64::from(drag.y)));
        }
        if let Some(pos) = response.hover_pos() {
            let pointer = (f64::from(pos.x), f64::from(pos.y));
            if response.dragged() {
                ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
            } else if cluster::hit(&self.clusters, &self.viewport, rect, pointer).is_some() {
                ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
            }
            let factor = ui.input(zoom_factor);
            if factor != 1.0 {
                self.viewport.zoom_at(pointer, f64::from(factor), rect);
            }
        }
        self.viewport.clamp(rect);
        action
    }

    /// Screen rect of `area`, its corners kept within 8 points of `rect`.
    fn area_rect(&self, rect: map::Rect, area: Area) -> Rect {
        // Far corners would otherwise leave the f32 range at high zoom.
        let near = |(x, y): (f64, f64)| {
            screen_pos((
                x.clamp(rect.x - 8.0, rect.x + rect.w + 8.0),
                y.clamp(rect.y - 8.0, rect.y + rect.h + 8.0),
            ))
        };
        Rect::from_min_max(
            near(
                self.viewport
                    .to_screen(project(area.north, area.west), rect),
            ),
            near(
                self.viewport
                    .to_screen(project(area.south, area.east), rect),
            ),
        )
    }

    fn paint_area(&self, painter: &Painter, rect: map::Rect, area: Area) {
        painter.rect_stroke(
            self.area_rect(rect, area),
            0.0,
            Stroke::new(2.0, BUBBLE),
            egui::StrokeKind::Middle,
        );
    }

    /// Draws the visible bubbles and returns their screen rects.
    fn paint_bubbles(
        &mut self,
        ui: &egui::Ui,
        painter: &Painter,
        grid: &mut Grid,
        rect: map::Rect,
        wanted: &mut Vec<ImageId>,
    ) -> Vec<map::Rect> {
        let visible: Vec<(usize, (f64, f64), f64)> = self
            .clusters
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let (x, y) = self.viewport.to_screen((c.x, c.y), rect);
                let r = cluster::radius(c.count);
                let inside = x + r >= rect.x
                    && x - r <= rect.x + rect.w
                    && y + r >= rect.y
                    && y - r <= rect.y + rect.h;
                inside.then_some((i, (x, y), r))
            })
            .collect();
        let (cx, cy) = rect.center();
        let distance = |&(_, (x, y), _): &(usize, (f64, f64), f64)| (x - cx).hypot(y - cy);
        let mut nearest: Vec<usize> = (0..visible.len()).collect();
        nearest.sort_by(|&a, &b| distance(&visible[a]).total_cmp(&distance(&visible[b])));
        nearest.truncate(SAMPLES);

        let mut items: Vec<(Sha256, &Path)> = Vec::new();
        let mut owners = Vec::new();
        for &v in &nearest {
            let id = ImageId(self.clusters[visible[v].0].first);
            match self.samples.get(&id) {
                None => wanted.push(id),
                Some(Some(entry)) if entry.kind == Kind::Image => {
                    if let Some(sha256) = entry.sha256 {
                        items.push((sha256, entry.path.as_path()));
                        owners.push(v);
                    }
                }
                Some(_) => {}
            }
        }
        let max_diameter = 2.0 * cluster::radius(usize::MAX) as f32;
        let bucket = bucket_for(max_diameter * ui.ctx().pixels_per_point());
        let mut textures: Vec<Option<TextureHandle>> = vec![None; visible.len()];
        for (v, texture) in owners
            .into_iter()
            .zip(grid.thumbnails(ui.ctx(), &items, bucket))
        {
            textures[v] = texture;
        }
        self.requested.clone_from(wanted);

        let font = FontId::proportional(11.0);
        let mut occupied = Vec::with_capacity(visible.len());
        for (&(i, centre, r), texture) in visible.iter().zip(&textures) {
            let c = &self.clusters[i];
            let centre = screen_pos(centre);
            #[allow(clippy::cast_possible_truncation)]
            let radius = r as f32;
            paint_bubble(painter, centre, radius, c.count, texture.as_ref(), &font);
            if self.highlight_pos.is_some_and(|p| holds(c, p)) {
                painter.circle_stroke(centre, radius + 3.0, Stroke::new(3.0, HIGHLIGHT));
            }
            occupied.push(map::Rect {
                x: f64::from(centre.x) - r,
                y: f64::from(centre.y) - r,
                w: 2.0 * r,
                h: 2.0 * r,
            });
        }
        occupied
    }

    fn country_labels(
        &self,
        painter: &Painter,
        visuals: &egui::Visuals,
        rect: map::Rect,
    ) -> Vec<(Placed, map::Rect)> {
        let font = FontId::proportional(13.0);
        self.basemap
            .country_labels()
            .iter()
            .filter(|l| self.viewport.zoom >= l.min_zoom)
            .filter_map(|l| {
                let (x, y) = self.viewport.to_screen(l.pos, rect);
                inside(rect, (x, y)).then(|| (l, screen_pos((x, y))))
            })
            .take(LABEL_CANDIDATES)
            .map(|(l, at)| {
                let galley =
                    painter.layout_no_wrap(l.name.clone(), font.clone(), visuals.weak_text_color());
                let pos = at - galley.size() / 2.0;
                let bounds = map_rect(Rect::from_min_size(pos, galley.size()));
                let placed = Placed {
                    galley,
                    pos,
                    dot: None,
                };
                (placed, bounds)
            })
            .collect()
    }

    /// Loads the in-memory list or runs the box query, whichever `level` needs.
    fn update_cities(&mut self, ctx: &egui::Context, level: u32, view: Area) -> Result<(), String> {
        let Some(path) = &self.geo_db else {
            return Ok(());
        };
        let Some(min_pop) = labels::min_population(level) else {
            return Ok(());
        };
        if level <= LIST_MAX_LEVEL {
            match &self.city_list {
                CityList::Idle => match load_city_list(ctx, path.clone()) {
                    Ok(rx) => self.city_list = CityList::Loading(rx),
                    Err(e) => {
                        self.city_list = CityList::Failed;
                        return Err(e);
                    }
                },
                CityList::Loading(rx) => match rx.try_recv() {
                    Ok(Ok(cities)) => self.city_list = CityList::Ready(cities),
                    Ok(Err(e)) => {
                        self.city_list = CityList::Ready(Vec::new());
                        return Err(e);
                    }
                    Err(TryRecvError::Empty) => {}
                    Err(TryRecvError::Disconnected) => {
                        self.city_list = CityList::Ready(Vec::new());
                        return Err("the loader stopped".into());
                    }
                },
                CityList::Ready(_) | CityList::Failed => {}
            }
            return Ok(());
        }
        if self
            .city_box
            .as_ref()
            .is_some_and(|b| b.level == level && contains(&b.area, &view))
        {
            return Ok(());
        }
        let grown = lat_lon_box(grow(self.viewport.bounds(self.rect), 0.25));
        // A failed query is cached as empty, so it's retried only once the view leaves the box.
        let (cities, result) = match self.query_cities(&grown, min_pop) {
            Ok(cities) => (cities, Ok(())),
            Err(e) => (Vec::new(), Err(e)),
        };
        self.city_box = Some(CityBox {
            level,
            area: grown,
            cities,
        });
        result
    }

    fn query_cities(&mut self, area: &Area, min_pop: u32) -> Result<Vec<City>, String> {
        if self.gazetteer.is_none()
            && let Some(path) = &self.geo_db
        {
            self.gazetteer = GeoDb::open(path).map_err(|e| e.to_string())?;
        }
        let Some(gazetteer) = &self.gazetteer else {
            return Ok(Vec::new());
        };
        gazetteer
            .cities_in(
                area.south, area.north, area.west, area.east, min_pop, CITY_LIMIT,
            )
            .map_err(|e| e.to_string())
    }

    fn city_labels(
        &self,
        painter: &Painter,
        visuals: &egui::Visuals,
        rect: map::Rect,
        level: u32,
        view: &Area,
    ) -> Vec<(Placed, map::Rect)> {
        let Some(min_pop) = labels::min_population(level).filter(|_| self.geo_db.is_some()) else {
            return Vec::new();
        };
        let cities: &[City] = if level <= LIST_MAX_LEVEL {
            match &self.city_list {
                CityList::Ready(cities) => cities,
                CityList::Idle | CityList::Loading(_) | CityList::Failed => &[],
            }
        } else {
            self.city_box.as_ref().map_or(&[], |b| &b.cities)
        };
        let font = FontId::proportional(12.0);
        labels::in_box(
            cities, view.south, view.north, view.west, view.east, min_pop,
        )
        .take(LABEL_CANDIDATES)
        .map(|c| {
            let dot = screen_pos(self.viewport.to_screen(project(c.lat, c.lon), rect));
            let galley = painter.layout_no_wrap(c.name.clone(), font.clone(), visuals.text_color());
            let pos = dot + vec2(5.0, -galley.size().y / 2.0);
            let bounds = map_rect(
                Rect::from_min_size(pos, galley.size())
                    .union(Rect::from_center_size(dot, Vec2::splat(5.0))),
            );
            let placed = Placed {
                galley,
                pos,
                dot: Some(dot),
            };
            (placed, bounds)
        })
        .collect()
    }
}

fn load_city_list(
    ctx: &egui::Context,
    path: PathBuf,
) -> Result<Receiver<Result<Vec<City>, String>>, String> {
    let (tx, rx) = mpsc::channel();
    let ctx = ctx.clone();
    thread::Builder::new()
        .name("city-labels".into())
        .spawn(move || {
            let cities = GeoDb::open(&path)
                .and_then(|db| db.map_or(Ok(Vec::new()), |db| db.major_cities(LIST_MIN_POPULATION)))
                .map_err(|e| e.to_string());
            // The map may be gone by now.
            _ = tx.send(cities);
            ctx.request_repaint();
        })
        .map_err(|e| e.to_string())?;
    Ok(rx)
}

fn paint_bubble(
    painter: &Painter,
    centre: Pos2,
    radius: f32,
    count: usize,
    texture: Option<&TextureHandle>,
    font: &FontId,
) {
    let ring = Stroke::new(2.0, Color32::WHITE);
    let Some(texture) = texture else {
        painter.circle(centre, radius, BUBBLE, ring);
        let text = painter.layout_no_wrap(count_text(count), font.clone(), Color32::WHITE);
        painter.galley(centre - text.size() / 2.0, text, Color32::WHITE);
        return;
    };
    let square = Rect::from_center_size(centre, Vec2::splat(2.0 * radius));
    painter.add(
        RectShape::filled(square, radius, Color32::WHITE)
            .with_texture(texture.id(), centre_square(texture.size_vec2())),
    );
    painter.circle_stroke(centre, radius, ring);
    if count > 1 {
        let text = painter.layout_no_wrap(count_text(count), font.clone(), Color32::WHITE);
        let badge = centre + vec2(radius, -radius) * 0.75;
        let badge_radius = (text.size().x / 2.0 + 4.0).max(9.0);
        painter.circle(
            badge,
            badge_radius,
            BUBBLE,
            Stroke::new(1.0, Color32::WHITE),
        );
        painter.galley(badge - text.size() / 2.0, text, Color32::WHITE);
    }
}

/// UV rect of the centred square crop of an image of `size`.
fn centre_square(size: Vec2) -> Rect {
    if size.x <= 0.0 || size.y <= 0.0 {
        return Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
    }
    let side = size.x.min(size.y);
    let (u, v) = (side / size.x, side / size.y);
    Rect::from_center_size(pos2(0.5, 0.5), vec2(u, v))
}

fn count_text(count: usize) -> String {
    if count < 10_000 {
        count.to_string()
    } else {
        format!("{}k", count / 1000)
    }
}

/// The cluster's member box as latitude and longitude, padded by `max(10% of span, MIN_PAD)`.
fn padded_area(cluster: &Cluster) -> Area {
    // World y grows southwards, so `min` is the north-west corner.
    let (north, west) = unproject(cluster.min.0, cluster.min.1);
    let (south, east) = unproject(cluster.max.0, cluster.max.1);
    let lat = ((north - south) * 0.1).max(MIN_PAD);
    let lon = ((east - west) * 0.1).max(MIN_PAD);
    Area {
        south: (south - lat).max(-90.0),
        north: (north + lat).min(90.0),
        west: (west - lon).max(-180.0),
        east: (east + lon).min(180.0),
    }
}

fn lat_lon_box(bounds: Bounds) -> Area {
    let (north, west) = unproject(bounds.min.0, bounds.min.1);
    let (south, east) = unproject(bounds.max.0, bounds.max.1);
    Area {
        south,
        north,
        west,
        east,
    }
}

/// `bounds` grown by `by` times its size on each side.
fn grow(bounds: Bounds, by: f64) -> Bounds {
    let dx = (bounds.max.0 - bounds.min.0) * by;
    let dy = (bounds.max.1 - bounds.min.1) * by;
    Bounds {
        min: (bounds.min.0 - dx, bounds.min.1 - dy),
        max: (bounds.max.0 + dx, bounds.max.1 + dy),
    }
}

fn contains(outer: &Area, inner: &Area) -> bool {
    outer.south <= inner.south
        && outer.north >= inner.north
        && outer.west <= inner.west
        && outer.east >= inner.east
}

/// Whether the world point lies in the cluster's member box; boxes of one level never overlap.
fn holds(cluster: &Cluster, point: (f64, f64)) -> bool {
    Bounds {
        min: cluster.min,
        max: cluster.max,
    }
    .contains(point)
}

fn inside(rect: map::Rect, (x, y): (f64, f64)) -> bool {
    (rect.x..=rect.x + rect.w).contains(&x) && (rect.y..=rect.y + rect.h).contains(&y)
}

pub(super) fn map_rect(rect: Rect) -> map::Rect {
    map::Rect {
        x: f64::from(rect.min.x),
        y: f64::from(rect.min.y),
        w: f64::from(rect.width()),
        h: f64::from(rect.height()),
    }
}

/// `rect` shrunk by the largest bubble radius on each side, so fitted edge bubbles stay whole;
/// an axis too short for that keeps its size.
fn fit_rect(rect: map::Rect) -> map::Rect {
    let inset = |pos: f64, len: f64| {
        if len > 2.0 * cluster::MAX_RADIUS {
            (pos + cluster::MAX_RADIUS, len - 2.0 * cluster::MAX_RADIUS)
        } else {
            (pos, len)
        }
    };
    let (x, w) = inset(rect.x, rect.w);
    let (y, h) = inset(rect.y, rect.h);
    map::Rect { x, y, w, h }
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn screen_pos((x, y): (f64, f64)) -> Pos2 {
    pos2(x as f32, y as f32)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use egui_kittest::Harness;
    use egui_kittest::kittest::Queryable;

    use super::*;
    use crate::geo::fixtures;
    use crate::gui::basemap::NaturalEarth;

    /// Three images in Lisbon, one in Paris.
    const POINTS: [(ImageId, f64, f64); 4] = [
        (ImageId(1), 38.70, -9.15),
        (ImageId(2), 38.71, -9.14),
        (ImageId(3), 38.72, -9.13),
        (ImageId(4), 48.85, 2.35),
    ];

    struct State {
        map: MapView,
        grid: Grid,
        area: Option<Area>,
        /// Entries the app would resolve wanted sample ids against.
        catalog: Vec<Entry>,
        /// Whether the app resolves wanted ids this frame.
        resolve: bool,
        wanted: Vec<ImageId>,
        requests: Vec<Vec<ImageId>>,
        actions: Vec<MapAction>,
    }

    fn harness(map: MapView) -> Harness<'static, State> {
        sized(map, egui::vec2(800.0, 600.0))
    }

    fn sized(map: MapView, size: Vec2) -> Harness<'static, State> {
        let state = State {
            map,
            grid: Grid::new(None),
            area: None,
            catalog: Vec::new(),
            resolve: true,
            wanted: Vec::new(),
            requests: Vec::new(),
            actions: Vec::new(),
        };
        // egui's double-click window is 0.3 s, shorter than the default step.
        let mut harness = Harness::builder()
            .with_size(size)
            .with_step_dt(1.0 / 60.0)
            .build_ui_state(
                |ui, s: &mut State| {
                    let samples: Vec<Entry> = s
                        .catalog
                        .iter()
                        .filter(|e| s.wanted.contains(&e.id))
                        .cloned()
                        .collect();
                    let samples = s.resolve.then_some(samples.as_slice());
                    let output = s.map.show(ui, &mut s.grid, s.area, samples);
                    assert_eq!(output.error, None);
                    if !output.wanted.is_empty() {
                        s.requests.push(output.wanted.clone());
                    }
                    s.wanted = output.wanted;
                    s.actions.extend(output.action);
                },
                state,
            );
        harness.run_steps(2);
        harness
    }

    fn with_points() -> Harness<'static, State> {
        let mut map = MapView::new(Box::new(NaturalEarth));
        map.set_points(&POINTS);
        harness(map)
    }

    fn pointer(pos: Pos2) -> (f64, f64) {
        (f64::from(pos.x), f64::from(pos.y))
    }

    fn button(pos: Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        }
    }

    fn close(a: (f64, f64), b: (f64, f64), eps: f64) -> bool {
        (a.0 - b.0).abs() < eps && (a.1 - b.1).abs() < eps
    }

    #[test]
    fn first_open_fits_the_points() {
        let harness = with_points();
        let map = &harness.state().map;
        let mut expected = Viewport::fit(map.extent.unwrap(), fit_rect(map.rect), FIT_MAX_ZOOM);
        expected.clamp(map.rect);
        assert!(close(map.viewport().center, expected.center, 1e-12));
        assert!((map.viewport().zoom - expected.zoom).abs() < 1e-12);
        // Lisbon to Paris, well above the 800x600 rect's min zoom of 1.2.
        assert!(
            (5.0..FIT_MAX_ZOOM).contains(&map.viewport().zoom),
            "{expected:?}"
        );
        assert!(harness.query_by_label(EMPTY).is_none());
        let rect = map.rect;
        let margin = cluster::MAX_RADIUS - 1e-3;
        for (id, ..) in POINTS {
            let pos = map.cluster_pos(id).unwrap();
            let (x, y) = (f64::from(pos.x), f64::from(pos.y));
            assert!(
                x >= rect.x + margin
                    && x <= rect.x + rect.w - margin
                    && y >= rect.y + margin
                    && y <= rect.y + rect.h - margin,
                "{id:?} at {pos:?} in {rect:?}"
            );
        }
    }

    #[test]
    fn fit_rect_insets_only_axes_that_stay_positive() {
        let rect = map::Rect {
            x: 10.0,
            y: 20.0,
            w: 800.0,
            h: 50.0,
        };
        let inset = fit_rect(rect);
        assert_eq!(
            (inset.x, inset.y, inset.w, inset.h),
            (40.0, 20.0, 740.0, 50.0)
        );
    }

    #[test]
    fn an_empty_rect_keeps_the_fit_pending() {
        let mut map = MapView::new(Box::new(NaturalEarth));
        map.set_points(&POINTS);
        let mut harness = sized(map, Vec2::ZERO);
        assert!(!harness.state().map.fitted);
        assert_eq!(harness.state().map.viewport(), Viewport::default());

        harness.set_size(egui::vec2(800.0, 600.0));
        harness.run_steps(2);
        let map = &harness.state().map;
        assert!(map.fitted);
        assert!(map.viewport().zoom > 5.0, "{:?}", map.viewport());
    }

    #[test]
    fn scroll_zooms_around_the_pointer() {
        let mut harness = with_points();
        let at = pos2(300.0, 200.0);
        let (before, world) = {
            let map = &harness.state().map;
            (
                map.viewport(),
                map.viewport().to_world(pointer(at), map.rect),
            )
        };
        harness.event(egui::Event::PointerMoved(at));
        harness.event(egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Line,
            delta: egui::vec2(0.0, 1.0),
            phase: egui::TouchPhase::Move,
            modifiers: egui::Modifiers::NONE,
        });
        harness.run_steps(60);
        let map = &harness.state().map;
        assert!(
            map.viewport().zoom > before.zoom + 0.2,
            "{:?}",
            map.viewport()
        );
        let after = map.viewport().to_world(pointer(at), map.rect);
        assert!(close(after, world, 1e-9), "{after:?} != {world:?}");
    }

    #[test]
    fn a_drag_from_a_bubble_pans() {
        let mut harness = with_points();
        harness.state_mut().map.set_viewport(Viewport {
            center: project(38.71, -9.14),
            zoom: 6.0,
        });
        harness.run_steps(1);
        let before = harness.state().map.viewport();
        let start = harness.state().map.cluster_pos(ImageId(1)).unwrap();
        assert!(start.distance(pos2(400.0, 300.0)) < 20.0, "{start:?}");
        harness.event(egui::Event::PointerMoved(start));
        harness.event(button(start, true));
        for i in 1..=10_u8 {
            let step = f32::from(i);
            harness.event(egui::Event::PointerMoved(
                start + egui::vec2(10.0 * step, 5.0 * step),
            ));
        }
        harness.event(button(start + egui::vec2(100.0, 50.0), false));
        harness.run_steps(1);
        let state = harness.state();
        let after = state.map.viewport();
        assert!((after.zoom - before.zoom).abs() < f64::EPSILON);
        let scale = before.world_px();
        let moved = (
            (before.center.0 - after.center.0) * scale,
            (before.center.1 - after.center.1) * scale,
        );
        assert!((80.0..=100.5).contains(&moved.0), "{moved:?}");
        assert!((40.0..=50.5).contains(&moved.1), "{moved:?}");
        assert!(state.actions.is_empty());
    }

    #[test]
    fn clicking_a_bubble_returns_its_padded_area() {
        let mut harness = with_points();
        let map = &harness.state().map;
        let at = map.cluster_pos(ImageId(1)).unwrap();
        assert_eq!(map.cluster_pos(ImageId(3)), Some(at));
        assert_ne!(map.cluster_pos(ImageId(4)), Some(at));
        harness.event(egui::Event::PointerMoved(at));
        harness.event(button(at, true));
        harness.event(button(at, false));
        harness.run_steps(1);

        let actions = &harness.state().actions;
        assert_eq!(actions.len(), 1, "{actions:?}");
        let MapAction::Cluster(area) = actions[0];
        // Spans of 0.02 degrees pad by the minimum.
        let expected = [38.695, 38.725, -9.155, -9.125];
        let got = [area.south, area.north, area.west, area.east];
        for (g, e) in got.iter().zip(expected) {
            assert!((g - e).abs() < 1e-6, "{got:?} != {expected:?}");
        }
    }

    #[test]
    fn padding_is_a_tenth_of_the_span_and_stays_in_range() {
        let wide = Cluster {
            key: (0, 0),
            count: 2,
            x: 0.0,
            y: 0.0,
            first: 1,
            min: project(40.0, -10.0),
            max: project(38.0, -8.0),
        };
        let area = padded_area(&wide);
        let got = [area.south, area.north, area.west, area.east];
        let expected = [37.8, 40.2, -10.2, -7.8];
        for (g, e) in got.iter().zip(expected) {
            assert!((g - e).abs() < 1e-9, "{got:?} != {expected:?}");
        }

        let corner = Cluster {
            min: (0.0, 0.0),
            max: (0.0, 0.0),
            ..wide
        };
        let area = padded_area(&corner);
        assert!(area.north <= 90.0 && area.west >= -180.0);
        assert!((area.east - (-180.0 + MIN_PAD)).abs() < 1e-9);
    }

    #[test]
    fn double_click_on_empty_map_zooms_in_one_level() {
        let mut harness = with_points();
        let at = pos2(30.0, 30.0);
        let (before, world) = {
            let map = &harness.state().map;
            assert!(cluster::hit(&map.clusters, &map.viewport, map.rect, pointer(at)).is_none());
            (
                map.viewport(),
                map.viewport().to_world(pointer(at), map.rect),
            )
        };
        harness.event(egui::Event::PointerMoved(at));
        for _ in 0..2 {
            harness.event(button(at, true));
            harness.event(button(at, false));
        }
        harness.run_steps(1);
        let state = harness.state();
        let after = state.map.viewport();
        assert!((after.zoom - (before.zoom + 1.0)).abs() < 1e-9, "{after:?}");
        assert!(close(
            after.to_world(pointer(at), state.map.rect),
            world,
            1e-9
        ));
        assert!(state.actions.is_empty());
    }

    #[test]
    fn empty_state_is_a_label() {
        let harness = harness(MapView::new(Box::new(NaturalEarth)));
        harness.get_by_label(EMPTY);
        assert_eq!(harness.state().map.attribution(), "Natural Earth");
    }

    #[test]
    fn attribution_names_geonames_only_with_city_labels() {
        let lisbon = (38.72, -9.14);
        let mut without_geo = with_points();
        for zoom in [7.0, 12.0] {
            without_geo.state_mut().map.set_viewport(Viewport {
                center: project(lisbon.0, lisbon.1),
                zoom,
            });
            without_geo.run_steps(2);
            assert_eq!(without_geo.state().map.attribution(), "Natural Earth");
        }

        let dir = tempfile::tempdir().unwrap();
        let cities = [
            fixtures::city(
                1, "Lisboa", "Lisboa", "", lisbon.0, lisbon.1, "PT", "14", 517_802,
            ),
            fixtures::city(2, "Porto", "Porto", "", 41.15, -8.61, "PT", "17", 249_633),
        ]
        .concat();
        drop(fixtures::build(dir.path(), &cities));
        let mut map = MapView::new(Box::new(NaturalEarth));
        map.set_geo_db(Some(dir.path().join("geo.db")));
        // Away from the bubbles, so the label isn't hidden behind one.
        map.set_viewport(Viewport {
            center: project(lisbon.0 + 1.0, lisbon.1),
            zoom: 7.0,
        });
        map.set_points(&POINTS);
        let mut harness = harness(map);

        let deadline = Instant::now() + Duration::from_secs(30);
        while !matches!(harness.state().map.city_list, CityList::Ready(_)) {
            assert!(Instant::now() < deadline, "the city list did not load");
            harness.step();
            std::thread::sleep(Duration::from_millis(10));
        }
        harness.run_steps(1);
        assert_eq!(
            harness.state().map.attribution(),
            "Natural Earth | GeoNames"
        );

        // Above the list levels the box query takes over.
        harness.state_mut().map.set_viewport(Viewport {
            center: project(41.15, -8.61),
            zoom: 12.0,
        });
        harness.run_steps(1);
        let map = &harness.state().map;
        assert_eq!(map.attribution(), "Natural Earth | GeoNames");
        let cached = map.city_box.as_ref().unwrap();
        assert_eq!(cached.level, 12);
        assert_eq!(cached.cities.len(), 1);
        assert_eq!(cached.cities[0].name, "Porto");

        // Out over the Atlantic there's nothing to name.
        harness.state_mut().map.set_viewport(Viewport {
            center: project(38.0, -20.0),
            zoom: 12.0,
        });
        harness.run_steps(1);
        assert_eq!(harness.state().map.attribution(), "Natural Earth");
    }

    #[test]
    fn focus_survives_the_first_points() {
        let mut map = MapView::new(Box::new(NaturalEarth));
        map.focus(ImageId(2), 38.71, -9.14);
        let mut harness = harness(map);
        assert_eq!(harness.state().map.highlighted(), None);

        harness.state_mut().map.set_points(&POINTS);
        harness.run_steps(2);
        let map = &harness.state().map;
        assert!(close(map.viewport().center, project(38.71, -9.14), 1e-12));
        assert!((map.viewport().zoom - FOCUS_ZOOM).abs() < f64::EPSILON);
        assert_eq!(map.highlighted(), Some(ImageId(2)));

        harness.state_mut().map.request_fit();
        harness.run_steps(1);
        assert!(harness.state().map.viewport().zoom < FOCUS_ZOOM);
    }

    #[test]
    fn samples_are_wanted_once_per_points() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let mut map = MapView::new(Box::new(NaturalEarth));
        map.set_points(&POINTS);
        let mut harness = harness(map);
        // Only image 1 is still active in the catalog.
        harness.state_mut().catalog = vec![Entry {
            id: ImageId(1),
            path: dir.join("jpeg.jpg"),
            sha256: Some([7; 32]),
            kind: Kind::Image,
            width: None,
            height: None,
            duration_ms: None,
        }];
        harness.state_mut().map.set_points(&[]);
        harness.state_mut().map.set_points(&POINTS);
        harness.state_mut().requests.clear();
        harness.run_steps(3);

        let state = harness.state();
        assert_eq!(state.requests.len(), 1, "{:?}", state.requests);
        let mut wanted = state.requests[0].clone();
        wanted.sort();
        assert_eq!(wanted, [ImageId(1), ImageId(4)]);
        assert!(matches!(state.map.samples.get(&ImageId(1)), Some(Some(_))));
        assert!(matches!(state.map.samples.get(&ImageId(4)), Some(None)));
    }

    #[test]
    fn equal_points_keep_the_caches() {
        let mut harness = with_points();
        harness.run_steps(2);
        let (generation, sampled) = {
            let map = &harness.state().map;
            (map.generation, map.sampled())
        };
        assert_eq!(sampled, 2);
        harness.state_mut().map.set_points(&POINTS);
        let map = &harness.state().map;
        assert_eq!((map.generation, map.sampled()), (generation, sampled));

        harness.state_mut().map.set_points(&POINTS[..3]);
        let map = &harness.state().map;
        assert_eq!((map.generation, map.sampled()), (generation + 1, 0));
    }

    #[test]
    fn area_is_outlined() {
        let mut harness = with_points();
        let area = Area {
            south: 38.0,
            north: 39.0,
            west: -10.0,
            east: -9.0,
        };
        harness.state_mut().map.set_viewport(Viewport {
            center: project(38.5, -9.5),
            zoom: 6.0,
        });
        harness.state_mut().area = Some(area);
        harness.run_steps(1);
        let map = &harness.state().map;
        assert_eq!(map.outlined_area(), Some(area));
        let expected = Rect::from_min_max(
            screen_pos(map.viewport().to_screen(project(39.0, -10.0), map.rect)),
            screen_pos(map.viewport().to_screen(project(38.0, -9.0), map.rect)),
        );
        assert_eq!(map.area_rect(map.rect, area), expected);
        let outlined = harness.output().shapes.iter().any(|clipped| {
            matches!(&clipped.shape, egui::Shape::Rect(r)
                if r.rect == expected && r.stroke.color == BUBBLE && r.fill == Color32::TRANSPARENT)
        });
        assert!(outlined, "no outline at {expected:?}");
    }

    #[test]
    fn far_area_corners_stay_near_the_rect() {
        let mut map = MapView::new(Box::new(NaturalEarth));
        map.set_viewport(Viewport {
            center: project(38.71, -9.14),
            zoom: 16.0,
        });
        let rect = map::Rect {
            x: 0.0,
            y: 0.0,
            w: 800.0,
            h: 600.0,
        };
        let world = Area {
            south: -80.0,
            north: 80.0,
            west: -170.0,
            east: 170.0,
        };
        assert_eq!(
            map.area_rect(rect, world),
            Rect::from_min_max(pos2(-8.0, -8.0), pos2(808.0, 608.0))
        );
    }

    #[test]
    fn unresolved_samples_stay_wanted() {
        let mut map = MapView::new(Box::new(NaturalEarth));
        map.set_points(&POINTS);
        let mut harness = harness(map);
        harness.state_mut().resolve = false;
        harness.state_mut().map.set_points(&[]);
        harness.state_mut().map.set_points(&POINTS);
        harness.state_mut().requests.clear();
        harness.run_steps(3);
        let state = harness.state();
        assert_eq!(state.requests.len(), 3, "{:?}", state.requests);
        assert!(state.map.samples.is_empty());

        harness.state_mut().resolve = true;
        harness.state_mut().requests.clear();
        harness.run_steps(2);
        let state = harness.state();
        assert!(state.requests.is_empty(), "{:?}", state.requests);
        assert_eq!(state.map.samples.len(), 2);
    }
}
