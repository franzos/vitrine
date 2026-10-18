use std::collections::BTreeSet;
use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::anyhow;
use eframe::egui::{self, CentralPanel, Frame, Id, Panel, Popup, PopupCloseBehavior, ScrollArea};
use eframe::glow;
use rusqlite::Connection;

use super::basemap::NaturalEarth;
use super::filter_bar::{FilterBar, PlaceOptions};
use super::grid::{Action, Grid};
use super::indexing::{Indexing, Update};
use super::info_panel::{Info, InfoData};
use super::loader::Loader;
use super::map::{MapAction, MapView};
use super::people_panel::{self, Edit, PeoplePanel};
use super::place_picker::{self, Pick, PlacePicker};
use super::tree::{self, Folder};
use super::viewer::{self, Places, Viewer};
use crate::catalog::faces::ModelId;
use crate::catalog::geo::{Location, LocationSummary};
use crate::catalog::images::{Entry, ImageId, Kind};
use crate::catalog::runs::RunStats;
use crate::catalog::{self, filter};
use crate::config::{self, Paths};
use crate::db::{self, Sha256};
use crate::faces::fetch::MODEL_NAME;
use crate::geo::events::events_of;
use crate::geo::{GeoDb, Place};
use crate::index::IndexEvent;
use crate::nav::{NavSet, Selection};

const MAX_WORKERS: usize = 4;
/// Shortest gap between catalog re-reads while an index run streams changes.
const REFRESH_INTERVAL: Duration = Duration::from_millis(500);

/// Offer to apply the last set place to the rest of its event.
struct Suggestion {
    place: Place,
    members: Vec<ImageId>,
    unlocated: usize,
    /// The grid shows `members` instead of the folder.
    shown: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MainView {
    Grid,
    Map,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Progress {
    Idle,
    Walking,
    Hashing { done: usize, pending: usize },
    Faces { done: usize, pending: usize },
    Geo { done: usize, pending: usize },
    Elsewhere,
    Done(RunStats),
    Failed(String),
}

pub struct VitrineApp {
    paths: Paths,
    roots: Vec<PathBuf>,
    faces: config::Faces,
    geo_config: config::Geo,
    catalog: Option<Connection>,
    /// Opened on the first manual edit; the GUI otherwise only reads.
    writer: Option<Connection>,
    /// `None` until the catalog knows it.
    model: Option<ModelId>,
    /// `None` until `geo.db` exists.
    geo: Option<GeoDb>,
    /// Location of the image under the cursor, keyed by its content.
    location: Option<(Sha256, Option<Location>)>,
    /// Map point of the content in `location`.
    point: Option<(f64, f64)>,
    /// Info panel data of the image under the cursor; loaded only while the panel is open.
    info: Option<InfoData>,
    /// Location summary of `targets()`, keyed by cursor and selection; `None` inside when there are no targets.
    target_location: Option<(Option<ImageId>, BTreeSet<ImageId>, Option<LocationSummary>)>,
    /// The nav holds the viewer's image although the filters exclude it.
    kept: bool,
    recent: Vec<Place>,
    suggestion: Option<Suggestion>,
    picker: PlacePicker,
    filter_bar: FilterBar,
    places: PlaceOptions,
    people_panel: PeoplePanel,
    folders: Vec<Folder>,
    selected: Option<PathBuf>,
    nav: NavSet,
    selection: Selection,
    grid: Grid,
    viewer: Viewer,
    view: MainView,
    map: MapView,
    /// Points no longer match the catalog, folder or filter.
    points_stale: bool,
    /// Folder and area-less filter of the loaded points; a change refits the map.
    points_key: Option<(PathBuf, filter::Filter)>,
    /// Entries for the sample ids the map wanted last frame.
    samples: Option<Vec<Entry>>,
    message: Option<String>,
    refresh_error: Option<String>,
    indexing: Option<Indexing>,
    progress: Progress,
    started: bool,
    /// The catalog changed since the last re-read.
    stale: bool,
    refreshed: Instant,
    refresh_cost: Duration,
}

impl VitrineApp {
    /// Indexing starts on the first frame.
    #[must_use]
    pub fn new(paths: &Paths, roots: &[PathBuf], config: &config::Config) -> Self {
        let mut message = None;
        if roots.is_empty() {
            message = Some(format!(
                "no roots configured in {}; add `roots` there or run `vitrine PATH`",
                paths.config_file.display()
            ));
        }
        let workers = thread::available_parallelism()
            .map_or(1, NonZero::get)
            .min(MAX_WORKERS);
        let loader = Loader::spawn(&paths.thumbs_db(), workers)
            .map_err(|e| message = Some(format!("cannot start thumbnail workers: {e}")))
            .ok();

        let mut app = Self {
            paths: paths.clone(),
            roots: roots.to_vec(),
            faces: config.faces.clone(),
            geo_config: config.geo.clone(),
            catalog: None,
            writer: None,
            model: None,
            geo: None,
            location: None,
            point: None,
            info: None,
            target_location: None,
            kept: false,
            recent: Vec::new(),
            suggestion: None,
            picker: PlacePicker::default(),
            filter_bar: FilterBar::default(),
            places: PlaceOptions::default(),
            people_panel: PeoplePanel::new(paths.catalog_db()),
            folders: Vec::new(),
            selected: None,
            nav: NavSet::default(),
            selection: Selection::default(),
            grid: Grid::new(loader),
            viewer: Viewer::new(config.viewer.external.clone(), config.viewer.player.clone()),
            view: MainView::Grid,
            map: MapView::new(Box::new(NaturalEarth)),
            points_stale: true,
            points_key: None,
            samples: None,
            message,
            refresh_error: None,
            indexing: None,
            progress: Progress::Idle,
            started: false,
            stale: false,
            refreshed: Instant::now(),
            refresh_cost: Duration::ZERO,
        };
        app.refresh();
        app
    }

    pub fn select_folder(&mut self, folder: &Path) {
        let Some(conn) = &self.catalog else {
            return;
        };
        match self.entries(conn, folder) {
            Ok(entries) => {
                self.nav = NavSet::new(entries);
                self.selection.clear();
                self.selected = Some(folder.to_owned());
                self.grid.reset_scroll();
                self.points_stale = true;
            }
            Err(e) => self.message = Some(format!("{}: {e}", folder.display())),
        }
    }

    /// A shown event ignores folder and filters; face criteria apply only when faces are enabled and the model is known.
    fn entries(&self, conn: &Connection, folder: &Path) -> crate::db::Result<Vec<Entry>> {
        if let Some(suggestion) = self.suggestion.as_ref().filter(|s| s.shown) {
            return catalog::images::entries_by_id(conn, &suggestion.members);
        }
        if !self.filter_bar.is_active() {
            return catalog::images::entries_under(conn, folder);
        }
        filter::entries(conn, folder, &self.filter())
    }

    /// Face criteria apply only when faces are enabled and the model is known.
    fn filter(&self) -> filter::Filter {
        let model = self.model.filter(|_| self.faces.enabled);
        self.filter_bar.filter(model, &self.faces)
    }

    /// Loads the map's points for the folder and the filter without its area.
    fn load_points(&mut self) {
        let (Some(conn), Some(folder)) = (&self.catalog, &self.selected) else {
            return;
        };
        let filter = points_filter(self.filter());
        self.points_stale = false;
        match filter::points(conn, folder, &filter) {
            Ok(points) => {
                self.map.set_points(&points);
                let key = (folder.clone(), filter);
                // an unfitted map fits by itself, and a fit would override a focus
                if self.points_key.as_ref().is_some_and(|k| *k != key) {
                    self.map.request_fit();
                }
                self.points_key = Some(key);
            }
            Err(e) => self.message = Some(format!("cannot read map points: {e}")),
        }
    }

    /// Loads the place lists for the current selection.
    fn reload_places(&mut self) {
        let Some(conn) = &self.catalog else {
            return;
        };
        if let Err(e) = self.places.reload(conn, &mut self.filter_bar.place) {
            self.message = Some(format!("cannot read places: {e}"));
        }
    }

    /// The cursor stays on its image if it's still in the set.
    fn apply_filter(&mut self) {
        self.points_stale = true;
        let (Some(conn), Some(folder)) = (&self.catalog, &self.selected) else {
            return;
        };
        match self.entries(conn, folder) {
            Ok(entries) => {
                self.replace_entries(entries);
                self.selection.prune(&self.nav);
                if self.nav.cursor().is_some() {
                    self.grid.reveal();
                } else {
                    self.grid.reset_scroll();
                }
            }
            Err(e) => self.message = Some(format!("{}: {e}", folder.display())),
        }
    }

    /// The viewer's image stays in the set until the viewer closes, even when it no longer matches the filters.
    fn replace_entries(&mut self, entries: Vec<Entry>) {
        if self.viewer.is_open() {
            self.kept = self.nav.replace_keeping_current(entries);
        } else {
            self.nav.replace(entries);
            self.kept = false;
        }
    }

    fn edit(&mut self, edit: &Edit) {
        self.write(|conn| edit.apply(conn));
    }

    /// Writes `pick` to every hash in one transaction; `false` when nothing was written.
    fn locate(&mut self, pick: &Pick, shas: &[Sha256]) -> bool {
        if shas.is_empty() {
            self.message = Some("no images to locate; videos have no location".to_owned());
            return false;
        }
        let set_at = db::now_ms();
        self.write(|conn| {
            let tx = conn.transaction()?;
            for sha256 in shas {
                match pick {
                    Pick::Set(place) => catalog::geo::set_manual(&tx, sha256, place, set_at)?,
                    Pick::Clear => catalog::geo::clear(&tx, sha256, set_at)?,
                }
            }
            tx.commit()?;
            Ok(())
        })
        .is_some()
    }

    /// The selection, else the cursor image.
    fn targets(&self) -> Vec<&Entry> {
        if self.selection.is_empty() {
            self.nav.current().into_iter().collect()
        } else {
            self.nav
                .entries()
                .iter()
                .filter(|e| self.selection.contains(e.id))
                .collect()
        }
    }

    /// Locates `targets`, then offers the place for the rest of their event.
    fn locate_event(&mut self, pick: &Pick, targets: &[Entry]) {
        let shas = distinct_content(targets);
        let anchors: Vec<ImageId> = targets.iter().map(|e| e.id).collect();
        if !self.locate(pick, &shas) {
            return;
        }
        let suggestion = match pick {
            Pick::Set(place) => self.suggest(place, &anchors),
            Pick::Clear => None,
        };
        let shown = match suggestion {
            Some(_) => self.suggestion.as_ref().is_some_and(|s| s.shown),
            None => self.hide_event(),
        };
        self.suggestion = suggestion.map(|s| Suggestion { shown, ..s });
        if shown {
            self.apply_filter();
        }
    }

    /// Returns the grid to the folder; `true` if the event was shown.
    fn hide_event(&mut self) -> bool {
        let Some(suggestion) = self.suggestion.as_mut().filter(|s| s.shown) else {
            return false;
        };
        suggestion.shown = false;
        true
    }

    /// Like [`Self::hide_event`], and closes the viewer when the event was shown.
    fn leave_event(&mut self, gl: Option<&glow::Context>) -> bool {
        if !self.hide_event() {
            return false;
        }
        if self.viewer.is_open() {
            self.viewer.close(gl);
            self.grid.reveal();
        }
        true
    }

    fn suggest(&mut self, place: &Place, anchors: &[ImageId]) -> Option<Suggestion> {
        let conn = self.catalog.as_ref()?;
        let gap_secs = i64::from(self.geo_config.event_gap_hours) * 3600;
        let found = catalog::geo::event_items(conn).and_then(|items| {
            let members = events_of(&items, gap_secs, anchors);
            let unlocated = catalog::geo::unlocated(conn, &members)?.len();
            Ok((members, unlocated))
        });
        match found {
            Ok((members, unlocated)) => (unlocated > 0).then(|| Suggestion {
                place: place.clone(),
                members,
                unlocated,
                shown: false,
            }),
            Err(e) => {
                self.message = Some(format!("cannot read the event: {e}"));
                None
            }
        }
    }

    /// Writes the suggested place to the members still without a location.
    fn apply_suggestion(&mut self, gl: Option<&glow::Context>) {
        let (Some(conn), Some(suggestion)) = (&self.catalog, &self.suggestion) else {
            return;
        };
        let shas = match catalog::geo::unlocated(conn, &suggestion.members) {
            Ok(shas) => shas,
            Err(e) => {
                self.message = Some(format!("cannot read the event: {e}"));
                return;
            }
        };
        self.leave_event(gl);
        let Some(suggestion) = self.suggestion.take() else {
            return;
        };
        let set_at = db::now_ms();
        self.write(|conn| {
            let tx = conn.transaction()?;
            catalog::geo::apply_event(&tx, &shas, &suggestion.place, set_at)?;
            tx.commit()?;
            Ok(())
        });
    }

    /// Bottom panel offering the last set place to the rest of its event.
    fn show_suggestion(&mut self, ui: &mut egui::Ui, gl: Option<&glow::Context>) {
        let Some(suggestion) = &self.suggestion else {
            return;
        };
        let images = if suggestion.unlocated == 1 {
            "image"
        } else {
            "images"
        };
        let prompt = format!(
            "Apply {} to {} more {images} from this event?",
            suggestion.place.name, suggestion.unlocated
        );
        let shown = suggestion.shown;
        let clicked = Panel::bottom("event_suggestion")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(prompt);
                    if ui.button("Apply").clicked() {
                        Some(SuggestionAction::Apply)
                    } else if ui.button(if shown { "Hide" } else { "Show" }).clicked() {
                        Some(SuggestionAction::Toggle)
                    } else if ui.button("Dismiss").clicked() {
                        Some(SuggestionAction::Dismiss)
                    } else {
                        None
                    }
                })
                .inner
            })
            .inner;
        let Some(clicked) = clicked else {
            return;
        };
        match clicked {
            SuggestionAction::Apply => self.apply_suggestion(gl),
            SuggestionAction::Toggle if shown => {
                self.leave_event(gl);
                self.apply_filter();
            }
            SuggestionAction::Toggle => {
                if let Some(suggestion) = &mut self.suggestion {
                    suggestion.shown = true;
                }
                self.view = MainView::Grid;
                if self.viewer.is_open() {
                    self.viewer.close(gl);
                }
                self.apply_filter();
                self.grid.reveal();
            }
            SuggestionAction::Dismiss => {
                if self.leave_event(gl) {
                    self.apply_filter();
                }
                self.suggestion = None;
            }
        }
        ui.ctx().request_repaint();
    }

    /// Runs `f` on `writer`, opened on first use, and re-reads the catalog; `None` on failure.
    fn write<T>(&mut self, f: impl FnOnce(&mut Connection) -> db::Result<T>) -> Option<T> {
        let writer = match &mut self.writer {
            Some(writer) => writer,
            None => match catalog::open(&self.paths.catalog_db()) {
                Ok(conn) => self.writer.insert(conn),
                Err(e) => {
                    self.message = Some(format!("cannot open the catalog for writing: {e}"));
                    return None;
                }
            },
        };
        let written = f(writer)
            .map_err(|e| self.message = Some(format!("edit failed: {e}")))
            .ok();
        self.refresh();
        written
    }

    /// Loads the location and map point of the image under the cursor unless they're cached.
    fn load_location(&mut self) {
        let Some(sha256) = self
            .nav
            .current()
            .filter(|e| e.kind == Kind::Image)
            .and_then(|e| e.sha256)
        else {
            return;
        };
        if self
            .location
            .as_ref()
            .is_some_and(|(sha, _)| *sha == sha256)
        {
            return;
        }
        let Some(conn) = &self.catalog else {
            return;
        };
        match catalog::geo::location_of(conn, &sha256) {
            Ok(location) => self.location = Some((sha256, location)),
            Err(e) => {
                self.location = Some((sha256, None));
                self.message = Some(format!("cannot read the location: {e}"));
            }
        }
        self.point = filter::point_of(conn, &sha256)
            .map_err(|e| self.message = Some(format!("cannot read the map point: {e}")))
            .ok()
            .flatten();
    }

    /// Loads the Info panel data of the image under the cursor unless it's cached.
    fn load_info(&mut self) {
        if !self.viewer.info_open() {
            return;
        }
        let Some(id) = self.nav.current().map(|e| e.id) else {
            return;
        };
        if self.info.as_ref().is_some_and(|i| i.id == id) {
            return;
        }
        let Some(conn) = &self.catalog else {
            return;
        };
        let image = match catalog::images::get(conn, id) {
            Ok(Some(image)) => image,
            Ok(None) => return,
            Err(e) => {
                self.message = Some(format!("cannot read the image details: {e}"));
                return;
            }
        };
        let faces = self
            .model
            .filter(|_| self.faces.enabled && image.kind == Kind::Image)
            .zip(image.sha256)
            .and_then(|(model, sha256)| {
                catalog::faces::of_image(
                    conn,
                    &sha256,
                    model,
                    self.faces.min_score,
                    self.faces.min_face_size,
                )
                .map_err(|e| self.message = Some(format!("cannot read the faces: {e}")))
                .ok()
            });
        self.info = Some(InfoData { id, image, faces });
    }

    /// Opens the catalog once it exists.
    fn refresh(&mut self) {
        let started = Instant::now();
        self.stale = false;
        self.points_stale = true;
        self.location = None;
        self.point = None;
        self.info = None;
        self.target_location = None;
        self.refresh_error = self.reread().err();
        self.refreshed = Instant::now();
        self.refresh_cost = self.refreshed - started;
    }

    fn reread(&mut self) -> Result<(), String> {
        if self.geo.is_none() {
            // a broken geo.db shouldn't keep the catalog from loading
            match GeoDb::open(&self.paths.geo_db()) {
                Ok(geo) => self.geo = geo,
                Err(e) => {
                    self.message
                        .get_or_insert_with(|| format!("cannot open the gazetteer: {e}"));
                }
            }
        }
        self.map
            .set_geo_db(self.geo.is_some().then(|| self.paths.geo_db()));
        if self.catalog.is_none() {
            let catalog_db = self.paths.catalog_db();
            if !catalog_db.exists() {
                return Ok(());
            }
            // the read-only connection can't apply pending migrations
            drop(catalog::open(&catalog_db).map_err(|e| e.to_string())?);
            self.catalog = Some(catalog::open_read_only(&catalog_db).map_err(|e| e.to_string())?);
        }
        let Some(conn) = &self.catalog else {
            return Ok(());
        };
        let mut folders = Vec::new();
        for root in &self.roots {
            let counts = catalog::images::folder_counts_under(conn, root)
                .map_err(|e| format!("{}: {e}", root.display()))?;
            folders.push(Folder::build(root, &counts));
        }
        self.folders = folders;
        self.recent = catalog::geo::recent(conn).map_err(|e| e.to_string())?;
        self.places
            .reload(conn, &mut self.filter_bar.place)
            .map_err(|e| e.to_string())?;
        if self.faces.enabled {
            self.model =
                catalog::faces::model_by_name(conn, MODEL_NAME).map_err(|e| e.to_string())?;
            match self.model {
                Some(model) => {
                    self.people_panel
                        .reload(conn, model)
                        .map_err(|e| e.to_string())?;
                    let people = self.people_panel.people();
                    self.filter_bar
                        .people
                        .retain(|id| people.iter().any(|p| p.id == *id && !p.ignored));
                }
                None => self.people_panel.clear(),
            }
        }
        let Some(folder) = self
            .selected
            .clone()
            .or_else(|| self.roots.first().cloned())
        else {
            return Ok(());
        };
        let entries = self
            .entries(conn, &folder)
            .map_err(|e| format!("{}: {e}", folder.display()))?;
        if self.selected.is_some() {
            self.replace_entries(entries);
            self.selection.prune(&self.nav);
        } else {
            self.nav = NavSet::new(entries);
            self.selection.clear();
            self.selected = Some(folder);
            self.grid.reset_scroll();
        }
        if self.view == MainView::Map {
            self.load_points();
        }
        Ok(())
    }

    pub fn rescan(&mut self, ctx: &egui::Context) {
        if self.indexing.is_some() || self.roots.is_empty() {
            return;
        }
        match Indexing::start(
            self.paths.clone(),
            self.roots.clone(),
            self.faces.clone(),
            ctx.clone(),
        ) {
            Ok(indexing) => {
                self.indexing = Some(indexing);
                self.progress = Progress::Walking;
            }
            Err(e) => self.message = Some(format!("cannot start indexing: {e}")),
        }
    }

    #[must_use]
    pub fn is_indexing(&self) -> bool {
        self.indexing.is_some()
    }

    /// Re-reads the catalog at most every [`REFRESH_INTERVAL`] while the run streams changes.
    fn poll_indexing(&mut self, ctx: &egui::Context) {
        let Some(indexing) = &self.indexing else {
            return;
        };
        let mut finished = false;
        for update in indexing.drain() {
            match update {
                Update::Held => {
                    self.progress = Progress::Elsewhere;
                    self.stale = true;
                }
                Update::Event(event) => {
                    self.stale = true;
                    match event {
                        IndexEvent::Walked { pending, .. } => {
                            self.progress = Progress::Hashing { done: 0, pending };
                        }
                        IndexEvent::Added { .. }
                        | IndexEvent::Changed { .. }
                        | IndexEvent::Moved { .. } => {
                            if let Progress::Hashing { done, .. } = &mut self.progress {
                                *done += 1;
                            }
                        }
                        IndexEvent::FacesPending { images } => {
                            self.progress = Progress::Faces {
                                done: 0,
                                pending: images,
                            };
                        }
                        IndexEvent::FacesScanned { .. } => {
                            if let Progress::Faces { done, .. } = &mut self.progress {
                                *done += 1;
                            }
                        }
                        IndexEvent::GeoPending { images } => {
                            self.progress = Progress::Geo {
                                done: 0,
                                pending: images,
                            };
                        }
                        IndexEvent::GeoScanned { .. } => {
                            if let Progress::Geo { done, .. } = &mut self.progress {
                                *done += 1;
                            }
                        }
                        IndexEvent::Restored { .. }
                        | IndexEvent::Missing { .. }
                        | IndexEvent::Failed { .. } => {}
                    }
                }
                Update::Finished(result) => {
                    finished = true;
                    self.stale = true;
                    self.progress = match result {
                        Ok(summary) => Progress::Done(summary.stats),
                        Err(e) => Progress::Failed(e),
                    };
                }
            }
        }
        if finished {
            self.indexing = None;
        }
        if self.stale {
            // re-reads block the frame; keep them under a tenth of the interval
            let interval = REFRESH_INTERVAL.max(self.refresh_cost * 10);
            let wait = interval.saturating_sub(self.refreshed.elapsed());
            if finished || wait.is_zero() {
                self.refresh();
            } else {
                ctx.request_repaint_after(wait);
            }
        }
    }

    #[must_use]
    pub fn nav(&self) -> &NavSet {
        &self.nav
    }

    #[cfg(test)]
    pub fn selection(&self) -> &Selection {
        &self.selection
    }

    #[must_use]
    pub fn opened(&self) -> Option<ImageId> {
        self.viewer
            .is_open()
            .then(|| self.nav.current().map(|e| e.id))
            .flatten()
    }

    /// Without `gl`, full images aren't drawn.
    pub fn show(&mut self, ui: &mut egui::Ui, gl: Option<&glow::Context>) {
        if !self.started {
            self.started = true;
            self.rescan(ui.ctx());
        }
        self.poll_indexing(ui.ctx());
        let in_viewer = self.viewer.is_open();
        if in_viewer {
            self.show_suggestion(ui, gl);
        }
        if self.viewer.is_open() {
            self.load_location();
            self.load_info();
            let places = Places {
                gazetteer: self.geo.as_ref(),
                location: self.location.as_ref(),
                coords: self.point,
                recent: &self.recent,
            };
            let (crops, people) = self.people_panel.crops_and_people();
            let mut info = Info {
                data: self.info.as_ref(),
                crops,
                people,
            };
            let action = CentralPanel::default()
                .frame(Frame::NONE)
                .show(ui, |ui| {
                    self.viewer
                        .show(ui, &mut self.nav, &mut self.grid, gl, &places, &mut info)
                })
                .inner;
            match action {
                Some(viewer::Action::Close) => {
                    self.grid.reveal();
                    ui.ctx().request_repaint();
                }
                Some(viewer::Action::Locate(pick)) => {
                    let current = Vec::from_iter(self.nav.current().cloned());
                    self.locate_event(&pick, &current);
                }
                Some(viewer::Action::ShowOnMap) => {
                    self.show_on_map(gl);
                    ui.ctx().request_repaint();
                }
                Some(viewer::Action::Edit(edit)) => self.edit(&edit),
                Some(viewer::Action::OpenPerson(person)) => {
                    self.viewer.close(gl);
                    self.grid.reveal();
                    if let Some(conn) = &self.catalog {
                        self.people_panel.open(conn, person);
                    }
                    ui.ctx().request_repaint();
                }
                None => {}
            }
            if !self.viewer.is_open() && std::mem::take(&mut self.kept) {
                self.apply_filter();
            }
            return;
        }
        let mut view = self.view;
        let (rescan, filtered, pick) = Panel::top("filter_bar")
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    let enabled = self.indexing.is_none() && !self.roots.is_empty();
                    let rescan = ui
                        .add_enabled(enabled, egui::Button::new("Rescan"))
                        .clicked();
                    ui.selectable_value(&mut view, MainView::Grid, "Grid");
                    ui.selectable_value(&mut view, MainView::Map, "Map");
                    ui.separator();
                    let people = (self.faces.enabled && self.model.is_some())
                        .then(|| self.people_panel.people());
                    let filtered = self.filter_bar.show(ui, people, &self.places);
                    if self.view == MainView::Map {
                        return (rescan, filtered, None);
                    }
                    if !self.selection.is_empty() {
                        ui.separator();
                        ui.label(format!("{} selected", self.selection.len()));
                        if ui.button("Clear selection").clicked() {
                            self.selection.clear();
                        }
                    }
                    ui.separator();
                    self.show_location_label(ui);
                    let pick = self.show_set_location(ui);
                    (rescan, filtered, pick)
                })
                .inner
            })
            .inner;
        if rescan {
            self.rescan(ui.ctx());
        }
        if view != self.view {
            self.view = view;
            if view == MainView::Map && self.leave_event(gl) {
                self.apply_filter();
            }
        }
        if let Some(pick) = pick {
            let targets: Vec<Entry> = self.targets().into_iter().cloned().collect();
            self.locate_event(&pick, &targets);
        }
        if filtered {
            self.leave_event(gl);
            self.reload_places();
            self.apply_filter();
        }
        Panel::bottom("status").show(ui, |ui| {
            ui.label(self.status());
        });
        if !in_viewer {
            self.show_suggestion(ui, gl);
        }
        let clicked = Panel::left("folders")
            .resizable(true)
            .default_size(260.0)
            .show(ui, |ui| {
                ScrollArea::both()
                    .auto_shrink(false)
                    .show(ui, |ui| {
                        tree::show(ui, &self.folders, self.selected.as_deref())
                    })
                    .inner
            })
            .inner;
        if let Some(folder) = clicked {
            self.leave_event(gl);
            self.select_folder(&folder);
        }
        if self.faces.enabled {
            self.show_people(ui, gl);
        }
        match self.view {
            MainView::Grid => {
                CentralPanel::default().show(ui, |ui| {
                    if let Some(Action::Open(index)) =
                        self.grid.show(ui, &mut self.nav, &mut self.selection)
                    {
                        self.nav.set_cursor(index);
                        self.viewer.open();
                        ui.ctx().request_repaint();
                    }
                });
            }
            MainView::Map => self.show_map(ui, gl),
        }
    }

    /// Closes the viewer and focuses the map on the image under the cursor.
    fn show_on_map(&mut self, gl: Option<&glow::Context>) {
        let (Some(entry), Some((lat, lon))) = (self.nav.current(), self.point) else {
            return;
        };
        let id = entry.id;
        self.viewer.close(gl);
        self.grid.reveal();
        self.view = MainView::Map;
        if self.leave_event(gl) {
            self.apply_filter();
        }
        // loading now lets a refit for a changed folder or filter run before the focus cancels it
        if self.points_stale {
            self.load_points();
        }
        self.map.focus(id, lat, lon);
    }

    fn show_map(&mut self, ui: &mut egui::Ui, gl: Option<&glow::Context>) {
        if self.points_stale {
            self.load_points();
        }
        let output = CentralPanel::default()
            .show(ui, |ui| {
                self.map.show(
                    ui,
                    &mut self.grid,
                    self.filter_bar.area,
                    self.samples.as_deref(),
                )
            })
            .inner;
        self.samples = None;
        if !output.wanted.is_empty()
            && let Some(conn) = &self.catalog
        {
            match catalog::images::entries_by_id(conn, &output.wanted) {
                Ok(entries) => {
                    self.samples = Some(entries);
                    ui.ctx().request_repaint();
                }
                Err(e) => self.message = Some(format!("cannot read map samples: {e}")),
            }
        }
        if let Some(error) = output.error {
            self.message = Some(error);
        }
        if let Some(MapAction::Cluster(area)) = output.action {
            self.filter_bar.area = Some(area);
            self.view = MainView::Grid;
            self.leave_event(gl);
            self.reload_places();
            self.apply_filter();
            ui.ctx().request_repaint();
        }
    }

    /// Loads the location summary of `targets()` unless the cursor and selection are unchanged.
    fn load_target_location(&mut self) {
        let cursor = self.nav.current().map(|e| e.id);
        if self
            .target_location
            .as_ref()
            .is_some_and(|(c, ids, _)| *c == cursor && ids == self.selection.ids())
        {
            return;
        }
        let targets = self.targets();
        let summary = match &self.catalog {
            Some(conn) if !targets.is_empty() => {
                Some(catalog::geo::summarize(conn, &distinct_content(targets)))
            }
            _ => None,
        };
        let summary = summary.and_then(|summary| {
            summary
                .map_err(|e| self.message = Some(format!("cannot read the locations: {e}")))
                .ok()
        });
        self.target_location = Some((cursor, self.selection.ids().clone(), summary));
    }

    fn show_location_label(&mut self, ui: &mut egui::Ui) {
        self.load_target_location();
        let Some((_, _, Some(summary))) = &self.target_location else {
            return;
        };
        let text = match summary {
            LocationSummary::None => "Not located".to_owned(),
            LocationSummary::One {
                place,
                source: Some(source),
            } => format!("{} ({})", place.label(), source.as_str()),
            LocationSummary::One {
                place,
                source: None,
            } => place.label(),
            LocationSummary::Mixed(n) => format!("Mixed ({n} places)"),
        };
        ui.label(text);
    }

    /// Applies to the selection, else to the cursor image.
    fn show_set_location(&mut self, ui: &mut egui::Ui) -> Option<Pick> {
        let target = !self.selection.is_empty() || self.nav.current().is_some();
        let button = ui.add_enabled(
            self.geo.is_some() && target,
            egui::Button::new("Set location"),
        );
        let Some(geo) = &self.geo else {
            button.on_disabled_hover_text(place_picker::FETCH_HINT);
            return None;
        };
        let shown = Popup::from_toggle_button_response(&button)
            .close_behavior(PopupCloseBehavior::CloseOnClickOutside)
            .width(place_picker::WIDTH)
            .show(|ui| {
                let current = match &self.target_location {
                    Some((_, _, Some(LocationSummary::One { place, .. }))) => Some(place),
                    _ => None,
                };
                self.picker
                    .show(ui, Id::new("grid_place"), geo, &self.recent, current, true)
            });
        match shown {
            Some(shown) => shown.inner,
            None => {
                self.picker.reset();
                None
            }
        }
    }

    fn show_people(&mut self, ui: &mut egui::Ui, gl: Option<&glow::Context>) {
        let action = Panel::right("people")
            .resizable(true)
            .default_size(280.0)
            .show(ui, |ui| {
                // keeps the panel from shrinking when a frame draws less than it holds
                ui.set_min_width(ui.available_width());
                self.show_people_content(ui)
            })
            .inner;
        match action {
            Some(people_panel::Action::Edit(edit)) => self.edit(&edit),
            Some(people_panel::Action::ToggleFilter(person)) => {
                self.filter_bar.toggle(person);
                self.leave_event(gl);
                self.apply_filter();
            }
            None => {}
        }
    }

    fn show_people_content(&mut self, ui: &mut egui::Ui) -> Option<people_panel::Action> {
        match (&self.catalog, self.model) {
            (Some(conn), Some(_)) => self.people_panel.show(ui, conn, &self.filter_bar.people),
            _ => {
                ui.heading("People");
                ui.weak("No faces indexed yet");
                None
            }
        }
    }

    fn status(&self) -> String {
        let mut parts = Vec::new();
        if self.suggestion.as_ref().is_some_and(|s| s.shown) {
            parts.push(format!("{} items in this event", self.nav.len()));
        } else if let Some(folder) = &self.selected {
            parts.push(format!("{} items in {}", self.nav.len(), folder.display()));
        }
        if let Some(entry) = self.nav.current() {
            parts.push(entry.path.display().to_string());
        }
        let pending = self.grid.pending();
        if pending > 0 {
            parts.push(format!("loading {pending} thumbnails"));
        }
        match &self.progress {
            Progress::Idle => {}
            Progress::Walking => parts.push("indexing: walking".to_owned()),
            Progress::Hashing { done, pending } => {
                parts.push(format!("indexing: {done} / {pending}"));
            }
            Progress::Faces { done, pending } => parts.push(format!("faces: {done} / {pending}")),
            Progress::Geo { done, pending } => parts.push(format!("gps: {done} / {pending}")),
            Progress::Elsewhere => parts.push("indexing elsewhere".to_owned()),
            Progress::Done(stats) => parts.push(format!(
                "indexed: {} added, {} changed, {} moved, {} missing, {} failed",
                stats.added, stats.changed, stats.moved, stats.vanished, stats.failed
            )),
            Progress::Failed(e) => parts.push(format!("indexing failed: {e}")),
        }
        parts.extend(self.refresh_error.iter().chain(&self.message).cloned());
        parts.join("  |  ")
    }
}

enum SuggestionAction {
    Apply,
    Toggle,
    Dismiss,
}

/// `filter` without its area, and without face settings when no face criterion uses them,
/// so the map's refit key changes only with what selects the points.
fn points_filter(mut filter: filter::Filter) -> filter::Filter {
    filter.area = None;
    if filter.count == filter::FaceCount::Any && filter.people.is_empty() && !filter.unknown {
        filter.model = None;
        filter.min_score = 0.0;
        filter.min_face_size = 0;
    }
    filter
}

fn distinct_content<'a>(entries: impl IntoIterator<Item = &'a Entry>) -> Vec<Sha256> {
    entries
        .into_iter()
        .filter(|e| e.kind == Kind::Image)
        .filter_map(|e| e.sha256)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

impl eframe::App for VitrineApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        self.show(ui, frame.gl().map(AsRef::as_ref));
    }

    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        self.indexing = None;
        if let Some(gl) = gl {
            self.viewer.destroy(gl);
        }
    }
}

/// Opens the viewer window on `roots` and blocks until it's closed.
pub fn run(paths: &Paths, roots: &[PathBuf], config: &config::Config) -> anyhow::Result<()> {
    let app = VitrineApp::new(paths, roots, config);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("vitrine")
            .with_app_id("vitrine")
            .with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native("vitrine", options, Box::new(|_cc| Ok(Box::new(app))))
        .map_err(|e| anyhow!("cannot open the viewer window: {e}"))
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use std::sync::atomic::AtomicBool;

    use egui_kittest::Harness;
    use egui_kittest::kittest::{NodeT, Queryable};
    use sha2::{Digest, Sha256};

    use crate::lock::IndexLock;

    use super::*;

    const A: &[&str] = &["jpeg.jpg", "png.png", "webp.webp", "gif.gif", "tiff.tif"];
    const B: &[&str] = &[
        "heic.heic",
        "avif.avif",
        "jxl.jxl",
        "dng.dng",
        "dng-orientation-6.dng",
        "orientation-6.jpg",
    ];

    struct Fixture {
        _dir: tempfile::TempDir,
        paths: Paths,
        tree: PathBuf,
    }

    /// Fixtures in `photos/a` and `photos/b`, plus a video in `b`, indexed into a temp catalog.
    fn fixture() -> Fixture {
        let fixture = unindexed();
        index(&fixture);
        fixture
    }

    fn unindexed() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let tree = root.join("photos");
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        for (folder, files) in [("a", A), ("b", B)] {
            fs::create_dir_all(tree.join(folder)).unwrap();
            for file in files {
                fs::copy(fixtures.join(file), tree.join(folder).join(file)).unwrap();
            }
        }
        fs::write(tree.join("b/clip.mp4"), b"not really a video").unwrap();
        Fixture {
            _dir: dir,
            paths: paths_in(&root),
            tree,
        }
    }

    fn paths_in(root: &Path) -> Paths {
        Paths {
            home_dir: root.to_owned(),
            config_file: root.join("config.toml"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        }
    }

    fn index(fixture: &Fixture) -> crate::index::Summary {
        crate::index::run(
            &fixture.paths,
            std::slice::from_ref(&fixture.tree),
            &config::Faces::default(),
            &AtomicBool::new(false),
            |_| {},
        )
        .unwrap()
    }

    fn harness(fixture: &Fixture) -> Harness<'static, VitrineApp> {
        sized_harness(fixture, egui::vec2(1000.0, 700.0))
    }

    fn sized_harness(fixture: &Fixture, size: egui::Vec2) -> Harness<'static, VitrineApp> {
        let app = VitrineApp::new(
            &fixture.paths,
            std::slice::from_ref(&fixture.tree),
            &config::Config::default(),
        );
        let mut harness = Harness::builder()
            .with_size(size)
            .build_ui_state(|ui, app: &mut VitrineApp| app.show(ui, None), app);
        harness.run_steps(2);
        harness
    }

    /// Steps frames until `done` holds, failing after 30 s.
    fn step_until(
        harness: &mut Harness<'_, VitrineApp>,
        mut done: impl FnMut(&VitrineApp) -> bool,
    ) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while !done(harness.state()) {
            assert!(Instant::now() < deadline, "{}", harness.state().status());
            harness.step();
            thread::sleep(Duration::from_millis(20));
        }
    }

    fn checksums(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
        let mut sums = BTreeMap::new();
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                sums.extend(checksums(&path));
            } else {
                sums.insert(
                    path.clone(),
                    Sha256::digest(fs::read(&path).unwrap()).to_vec(),
                );
            }
        }
        sums
    }

    #[test]
    fn gui_indexes_at_startup_and_leaves_sources_unchanged() {
        let fixture = unindexed();
        let before = checksums(&fixture.tree);
        assert_eq!(before.len(), A.len() + B.len() + 1);
        assert!(!fixture.paths.catalog_db().exists());

        let mut harness = sized_harness(&fixture, egui::vec2(700.0, 450.0));
        step_until(&mut harness, |app| !app.is_indexing());
        let app = harness.state();
        let Progress::Done(stats) = &app.progress else {
            panic!("{:?}", app.progress);
        };
        assert_eq!(stats.added, before.len() as u64);
        assert_eq!(app.nav().len(), before.len());
        assert_eq!(app.folders.len(), 1);
        assert_eq!(app.folders[0].count, before.len() as u64);
        assert!(app.grid.layout().rows > 2, "{:?}", app.grid.layout());
        harness.get_by_label_contains(&format!("indexed: {} added", before.len()));
        harness.get_by_label("Rescan");

        let first = harness.state().nav().entries()[0].path.clone();
        let first = first.file_name().unwrap().to_str().unwrap().to_owned();
        for _ in 0..10 {
            harness.get_by_label(&first).scroll_down();
            harness.run_steps(1);
            if harness.query_by_label(&first).is_none() {
                break;
            }
        }
        assert!(
            harness.query_by_label(&first).is_none(),
            "grid didn't scroll"
        );

        let names: Vec<String> = harness
            .state()
            .nav()
            .entries()
            .iter()
            .map(|e| e.path.file_name().unwrap().to_str().unwrap().to_owned())
            .collect();
        let visible = names
            .iter()
            .find(|name| *name != "clip.mp4" && harness.query_by_label(name).is_some())
            .unwrap();
        harness.get_by_label(visible).click();
        harness.run_steps(1);
        let id = harness.state().nav().current().unwrap().id;
        assert_ne!(harness.state().nav().cursor(), Some(0));
        harness.key_press(egui::Key::Enter);
        harness.run_steps(2);
        assert_eq!(harness.state().opened(), Some(id));
        harness.key_press(egui::Key::ArrowLeft);
        harness.run_steps(2);
        harness.key_press(egui::Key::ArrowRight);
        harness.run_steps(2);
        assert_eq!(harness.state().opened(), Some(id));
        harness.key_press(egui::Key::Escape);
        harness.run_steps(2);
        assert_eq!(harness.state().opened(), None);
        drop(harness);

        assert_eq!(checksums(&fixture.tree), before);
    }

    #[test]
    fn gui_migrates_a_v8_catalog_before_the_first_read() {
        let fixture = unindexed();
        let catalog_db = fixture.paths.catalog_db();
        fs::create_dir_all(catalog_db.parent().unwrap()).unwrap();
        let conn = catalog::open_at_version(&catalog_db, 8);
        conn.execute_batch(&format!(
            "INSERT INTO images (path, size, mtime_ns, sha256, kind)
             VALUES ('{}', 1, 0, zeroblob(32), 'image');
             INSERT INTO places VALUES (7, 'Lisbon', '14', 'Lisbon', 'PT', 'Portugal');
             INSERT INTO locations VALUES (zeroblob(32), 7, 'manual', 10);",
            fixture.tree.join("a/jpeg.jpg").display()
        ))
        .unwrap();
        drop(conn);
        // keeps the indexer from migrating first
        let lock = IndexLock::acquire(&fixture.paths.index_lock()).unwrap();

        let app = VitrineApp::new(
            &fixture.paths,
            std::slice::from_ref(&fixture.tree),
            &config::Config::default(),
        );
        let mut harness = Harness::builder()
            .with_size(egui::vec2(1000.0, 700.0))
            .build_ui_state(|ui, app: &mut VitrineApp| app.show(ui, None), app);
        harness.run_steps(1);
        let app = harness.state();
        assert_eq!(app.refresh_error, None);
        assert_eq!(app.folders.len(), 1);
        assert_eq!(app.folders[0].count, 1);
        assert_eq!(app.recent, [catalog::geo::tests::place(7, "Lisbon")]);
        drop(lock);
        step_until(&mut harness, |app| !app.is_indexing());
    }

    #[test]
    fn gui_waits_while_another_indexer_holds_the_lock() {
        let fixture = unindexed();
        let lock = IndexLock::acquire(&fixture.paths.index_lock()).unwrap();
        let mut harness = harness(&fixture);
        step_until(&mut harness, |app| app.progress == Progress::Elsewhere);
        harness.run_steps(1);
        harness.get_by_label_contains("indexing elsewhere");
        assert!(harness.state().is_indexing());
        assert!(harness.state().nav().is_empty());

        drop(lock);
        step_until(&mut harness, |app| !app.is_indexing());
        let app = harness.state();
        assert!(
            matches!(app.progress, Progress::Done(_)),
            "{:?}",
            app.progress
        );
        assert_eq!(app.nav().len(), A.len() + B.len() + 1);
    }

    #[test]
    fn gui_rescan_picks_up_new_files_and_keeps_the_viewer_on_its_image() {
        let fixture = fixture();
        let mut harness = harness(&fixture);
        step_until(&mut harness, |app| !app.is_indexing());
        harness.get_by_label("png.png").click();
        harness.run_steps(1);
        let id = harness.state().nav().current().unwrap().id;
        harness.key_press(egui::Key::Enter);
        harness.run_steps(1);
        assert_eq!(harness.state().opened(), Some(id));

        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        fs::copy(fixtures.join("jpeg.jpg"), fixture.tree.join("a/0-new.jpg")).unwrap();
        let ctx = harness.ctx.clone();
        harness.state_mut().rescan(&ctx);
        step_until(&mut harness, |app| !app.is_indexing());
        harness.run_steps(1);
        let app = harness.state();
        assert_eq!(app.nav().len(), A.len() + B.len() + 2);
        assert_eq!(app.opened(), Some(id));
        let Progress::Done(stats) = &app.progress else {
            panic!("{:?}", app.progress);
        };
        assert_eq!((stats.added, stats.vanished), (1, 0));
    }

    #[test]
    fn gui_selecting_a_folder_shows_its_entries() {
        let fixture = fixture();
        let mut harness = harness(&fixture);
        assert_eq!(harness.state().nav().len(), A.len() + B.len() + 1);

        harness.get_by_label("b (7)").click();
        harness.run_steps(2);

        let app = harness.state();
        assert_eq!(
            app.selected.as_deref(),
            Some(fixture.tree.join("b").as_path())
        );
        assert_eq!(app.nav().len(), B.len() + 1);
        assert!(
            app.nav()
                .entries()
                .iter()
                .all(|e| e.path.starts_with(fixture.tree.join("b")))
        );
        let layout = app.grid.layout();
        assert!(layout.columns > 1, "{layout:?}");
        assert_eq!(layout.rows, app.nav().len().div_ceil(layout.columns));
        assert!(layout.rows > 1, "{layout:?}");

        for name in B.iter().chain(&["clip.mp4"]) {
            harness.get_by_label(name);
        }
        for name in A {
            assert!(harness.query_by_label(name).is_none(), "{name} still shown");
        }
    }

    #[test]
    fn gui_thumbnails_load_for_images_but_not_broken_videos() {
        let fixture = fixture();
        let mut harness = harness(&fixture);
        let images = A.len() + B.len();
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            harness.step();
            let grid = &harness.state().grid;
            if grid.textures() == images && grid.pending() == 0 {
                break;
            }
            assert!(Instant::now() < deadline, "{} textures", grid.textures());
            thread::sleep(Duration::from_millis(20));
        }
        harness.run_steps(2);
        assert_eq!(harness.state().grid.textures(), images);
        assert_eq!(harness.state().grid.pending(), 0);
    }

    #[test]
    fn gui_click_selects_and_enter_opens() {
        let fixture = fixture();
        let mut harness = harness(&fixture);
        harness.get_by_label("png.png").click();
        harness.run_steps(1);
        let nav = harness.state().nav();
        let selected = nav.current().unwrap();
        assert!(selected.path.ends_with("a/png.png"));
        let id = selected.id;
        assert_eq!(harness.state().opened(), None);

        harness.key_press(egui::Key::Enter);
        harness.run_steps(1);
        assert_eq!(harness.state().opened(), Some(id));

        harness.key_press(egui::Key::ArrowRight);
        harness.run_steps(1);
        assert_eq!(
            harness.state().nav().cursor(),
            harness.state().nav().position(id).map(|i| i + 1)
        );
    }

    #[test]
    fn gui_viewer_steps_and_escape_returns_to_the_grid() {
        let fixture = fixture();
        let mut harness = harness(&fixture);
        harness.get_by_label("png.png").click();
        harness.run_steps(1);
        harness.key_press(egui::Key::Enter);
        harness.run_steps(2);

        let nav = harness.state().nav();
        let index = nav.cursor().unwrap();
        let ids: Vec<_> = nav.ids().collect();
        let total = ids.len();
        assert_eq!(harness.state().opened(), Some(ids[index]));
        assert!(
            harness.query_by_label("png.png").is_none(),
            "grid still shown"
        );
        harness.get_by_label_contains(&format!("png.png   {} / {total}", index + 1));
        harness.get_by_label("Open in external viewer");
        harness.get_by_label("Show in file manager");

        harness.key_press(egui::Key::ArrowRight);
        harness.run_steps(1);
        assert_eq!(harness.state().opened(), Some(ids[index + 1]));
        harness.get_by_label_contains("tiff.tif");

        harness.key_press(egui::Key::ArrowLeft);
        harness.key_press(egui::Key::ArrowLeft);
        harness.run_steps(1);
        assert_eq!(harness.state().opened(), Some(ids[index - 1]));

        for _ in 0..total {
            harness.key_press(egui::Key::ArrowRight);
        }
        harness.run_steps(1);
        assert_eq!(harness.state().opened(), Some(ids[total - 1]));
        harness.key_press(egui::Key::ArrowLeft);
        harness.run_steps(1);
        let last_shown = ids[total - 2];
        assert_eq!(harness.state().opened(), Some(last_shown));

        harness.key_press(egui::Key::Escape);
        harness.run_steps(2);
        let app = harness.state();
        assert_eq!(app.opened(), None);
        assert_eq!(app.nav().current().map(|e| e.id), Some(last_shown));
        assert!(app.grid.layout().rows > 0);
        let name = app.nav().current().unwrap().path.file_name().unwrap();
        let name = name.to_str().unwrap().to_owned();
        harness.get_by_label(&name);
        assert!(harness.query_by_label("Open in external viewer").is_none());
        assert!(harness.query_by_label("Show in file manager").is_none());
    }

    fn player_harness(fixture: &Fixture, player: Vec<String>) -> Harness<'static, VitrineApp> {
        let mut config = config::Config::default();
        config.viewer.player = player;
        let app = VitrineApp::new(&fixture.paths, std::slice::from_ref(&fixture.tree), &config);
        let mut harness = Harness::builder()
            .with_size(egui::vec2(1000.0, 700.0))
            .build_ui_state(|ui, app: &mut VitrineApp| app.show(ui, None), app);
        harness.run_steps(2);
        harness
    }

    /// A player that appends its arguments to `out`, one line per run.
    fn recording_player(out: &Path) -> Vec<String> {
        vec![
            "sh".to_owned(),
            "-c".to_owned(),
            format!("printf '%s\\n' \"$@\" >> '{}'", out.display()),
            "sh".to_owned(),
        ]
    }

    /// The lines in `out` once there are `n`, failing after 10 s.
    fn wait_for_lines(out: &Path, n: usize) -> Vec<String> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let lines: Vec<String> = fs::read_to_string(out)
                .unwrap_or_default()
                .lines()
                .map(str::to_owned)
                .collect();
            if lines.len() >= n {
                return lines;
            }
            assert!(Instant::now() < deadline, "{lines:?}");
            thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn gui_play_runs_the_player_once_per_click_or_enter() {
        let fixture = fixture();
        let out = fixture.paths.home_dir.join("played");
        let mut harness = player_harness(&fixture, recording_player(&out));
        step_until(&mut harness, |app| !app.is_indexing());

        open_in_viewer(&mut harness, "png.png");
        assert!(harness.query_by_label("Play").is_none());
        let id = harness.state().opened();
        harness.key_press(egui::Key::Enter);
        harness.run_steps(2);
        assert_eq!(harness.state().opened(), id);
        harness.key_press(egui::Key::Escape);
        harness.run_steps(2);

        open_in_viewer(&mut harness, "clip.mp4");
        let clip = fixture.tree.join("b/clip.mp4").display().to_string();
        harness.get_by_label("Play").click();
        harness.run_steps(2);
        assert_eq!(wait_for_lines(&out, 1), [clip.as_str()]);
        harness.key_press(egui::Key::Enter);
        harness.run_steps(2);
        assert_eq!(wait_for_lines(&out, 2), [clip.as_str(); 2]);
        harness.key_press(egui::Key::Space);
        harness.run_steps(2);
        // a third run, from a focused button's fake click, would land by now
        thread::sleep(Duration::from_millis(300));
        assert_eq!(wait_for_lines(&out, 2).len(), 2);
        assert_ne!(
            harness.state().nav().current().map(|e| e.path.clone()),
            Some(PathBuf::from(&clip))
        );
    }

    #[test]
    fn gui_play_shows_a_player_that_cannot_run() {
        let fixture = fixture();
        let mut harness = player_harness(&fixture, vec!["/nonexistent/player".to_owned()]);
        step_until(&mut harness, |app| !app.is_indexing());
        open_in_viewer(&mut harness, "clip.mp4");
        harness.get_by_label("Play").click();
        harness.run_steps(2);
        harness.get_by_label_contains("cannot run /nonexistent/player");
    }

    #[test]
    fn gui_video_gets_a_thumbnail_and_details() {
        if !crate::video::tools_present() {
            return;
        }
        let fixture = unindexed();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        fs::copy(fixtures.join("clip.mp4"), fixture.tree.join("b/clip.mp4")).unwrap();
        index(&fixture);
        let mut harness = harness(&fixture);
        let textures = A.len() + B.len() + 1;
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            harness.step();
            let grid = &harness.state().grid;
            if grid.textures() == textures && grid.pending() == 0 {
                break;
            }
            assert!(Instant::now() < deadline, "{} textures", grid.textures());
            thread::sleep(Duration::from_millis(20));
        }

        open_in_viewer(&mut harness, "clip.mp4");
        click_info(&mut harness);
        harness.get_by_label("Duration");
        harness.get_by_label("Codec");
    }

    fn selected(app: &VitrineApp, indices: &[usize]) -> BTreeSet<ImageId> {
        indices
            .iter()
            .map(|&i| app.nav().get(i).unwrap().id)
            .collect()
    }

    #[test]
    fn gui_command_click_toggles_and_shift_click_ranges() {
        let fixture = fixture();
        let mut harness = harness(&fixture);
        harness.get_by_label(&format!("a ({})", A.len())).click();
        harness.run_steps(2);
        let names: Vec<String> = harness
            .state()
            .nav()
            .entries()
            .iter()
            .map(|e| e.path.file_name().unwrap().to_str().unwrap().to_owned())
            .collect();
        assert_eq!(names.len(), A.len());
        let click = |harness: &mut Harness<'_, VitrineApp>, index: usize, modifiers| {
            harness
                .get_by_label(&names[index])
                .click_modifiers(modifiers);
            harness.run_steps(1);
        };

        click(&mut harness, 1, egui::Modifiers::COMMAND);
        click(&mut harness, 3, egui::Modifiers::COMMAND);
        let app = harness.state();
        assert_eq!(app.selection().ids(), &selected(app, &[1, 3]));
        assert_eq!(app.nav().cursor(), Some(3));
        harness.get_by_label("2 selected");

        click(&mut harness, 1, egui::Modifiers::COMMAND);
        let app = harness.state();
        assert_eq!(app.selection().ids(), &selected(app, &[3]));
        assert_eq!(app.nav().cursor(), Some(1));

        click(&mut harness, 4, egui::Modifiers::SHIFT);
        let app = harness.state();
        assert_eq!(app.selection().ids(), &selected(app, &[1, 2, 3, 4]));
        click(&mut harness, 0, egui::Modifiers::SHIFT);
        let app = harness.state();
        assert_eq!(app.selection().ids(), &selected(app, &[0, 1]));
        assert_eq!(app.nav().cursor(), Some(0));

        click(&mut harness, 2, egui::Modifiers::NONE);
        let app = harness.state();
        assert!(app.selection().is_empty());
        assert_eq!(app.nav().cursor(), Some(2));
        assert!(harness.query_by_label("Clear selection").is_none());

        click(&mut harness, 4, egui::Modifiers::SHIFT);
        let app = harness.state();
        assert_eq!(app.selection().ids(), &selected(app, &[2, 3, 4]));
        assert_eq!(app.opened(), None);
    }

    #[test]
    fn gui_command_a_selects_the_set_and_changing_folder_clears() {
        let fixture = fixture();
        let mut harness = harness(&fixture);
        let total = A.len() + B.len() + 1;
        harness.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
        harness.run_steps(1);
        let all: BTreeSet<_> = harness.state().nav().ids().collect();
        assert_eq!(all.len(), total);
        assert_eq!(harness.state().selection().ids(), &all);

        harness.get_by_label(&format!("{total} selected"));
        harness.get_by_label("Clear selection").click();
        harness.run_steps(1);
        assert!(harness.state().selection().is_empty());

        harness.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
        harness.run_steps(1);
        assert_eq!(harness.state().selection().len(), total);
        harness
            .get_by_label(&format!("b ({})", B.len() + 1))
            .click();
        harness.run_steps(2);
        assert_eq!(harness.state().nav().len(), B.len() + 1);
        assert!(harness.state().selection().is_empty());
        assert!(harness.query_by_label("Clear selection").is_none());
    }

    struct Faces {
        first: crate::catalog::people::PersonId,
        second: crate::catalog::people::PersonId,
        first_faces: Vec<crate::catalog::faces::FaceId>,
        second_faces: Vec<crate::catalog::faces::FaceId>,
    }

    /// [`fixture`] plus faces of the configured model, auto-assigned to two unnamed people:
    /// - first: `a/jpeg.jpg`, `a/png.png`
    /// - second: `a/png.png`, `b/heic.heic`
    /// - unassigned: `a/gif.gif`; `a/tiff.tif` scanned without faces
    fn face_fixture() -> (Fixture, Faces) {
        use crate::catalog::faces::{self, AssignedBy, NewFace, Rect};
        use crate::catalog::people;
        use crate::faces::fetch::{DETECTOR_NAME, EMBEDDING_DIM};

        let fixture = fixture();
        let mut conn = catalog::open(&fixture.paths.catalog_db()).unwrap();
        let model = faces::register_model(&conn, MODEL_NAME, DETECTOR_NAME, EMBEDDING_DIM).unwrap();
        let crop = fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/jpeg.jpg"))
            .unwrap();
        let embedding = vec![0.1_f32; EMBEDDING_DIM];
        let face = NewFace {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                w: 80.0,
                h: 80.0,
            },
            landmarks: [[0.0; 2]; 5],
            score: 0.9,
            embedding: &embedding,
            crop: &crop,
        };
        let mut scan = |file: &str, n: usize| {
            let path = fixture.tree.join(file);
            let sha = catalog::images::get_by_path(&conn, &path)
                .unwrap()
                .unwrap()
                .sha256
                .unwrap();
            let tx = conn.transaction().unwrap();
            let ids = faces::insert_scan(&tx, &sha, model, &vec![face.clone(); n]).unwrap();
            tx.commit().unwrap();
            ids
        };
        let jpeg = scan("a/jpeg.jpg", 1);
        let png = scan("a/png.png", 2);
        let heic = scan("b/heic.heic", 1);
        scan("a/gif.gif", 1);
        scan("a/tiff.tif", 0);
        let first = people::create(&conn, model).unwrap();
        let second = people::create(&conn, model).unwrap();
        let first_faces = vec![jpeg[0], png[0]];
        let second_faces = vec![png[1], heic[0]];
        for (faces, person) in [(&first_faces, first), (&second_faces, second)] {
            for face in faces {
                assert!(faces::assign(&conn, *face, person, AssignedBy::Auto).unwrap());
            }
        }
        (
            fixture,
            Faces {
                first,
                second,
                first_faces,
                second_faces,
            },
        )
    }

    /// Faces set to `enabled`, without the startup index run.
    fn face_harness(fixture: &Fixture, enabled: bool) -> Harness<'static, VitrineApp> {
        let mut config = config::Config::default();
        config.faces.enabled = enabled;
        let mut app = VitrineApp::new(&fixture.paths, std::slice::from_ref(&fixture.tree), &config);
        app.started = true;
        let mut harness = Harness::builder()
            .with_size(egui::vec2(1200.0, 800.0))
            .build_ui_state(|ui, app: &mut VitrineApp| app.show(ui, None), app);
        harness.run_steps(2);
        harness
    }

    fn people_in(fixture: &Fixture) -> Vec<crate::catalog::people::Person> {
        let conn = catalog::open_read_only(&fixture.paths.catalog_db()).unwrap();
        let model = catalog::faces::model_by_name(&conn, MODEL_NAME)
            .unwrap()
            .unwrap();
        catalog::people::list(&conn, model).unwrap()
    }

    /// `(person, assigned_by, not_a_face)` of `face`.
    fn face_row(
        fixture: &Fixture,
        face: crate::catalog::faces::FaceId,
    ) -> (Option<i64>, Option<String>, bool) {
        let conn = catalog::open_read_only(&fixture.paths.catalog_db()).unwrap();
        conn.query_row(
            "SELECT person_id, assigned_by, not_a_face FROM faces WHERE id = ?1",
            [face.0],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap()
    }

    fn file_names(app: &VitrineApp) -> Vec<String> {
        app.nav()
            .entries()
            .iter()
            .map(|e| {
                e.path
                    .strip_prefix(e.path.ancestors().nth(2).unwrap())
                    .unwrap()
                    .display()
                    .to_string()
            })
            .collect()
    }

    #[test]
    fn gui_naming_a_person_persists_and_pins_their_faces() {
        let (fixture, faces) = face_fixture();
        let mut harness = face_harness(&fixture, true);
        harness.get_by_label("unnamed #1 (2)").click();
        harness.run_steps(2);
        harness.get_by_label("2 faces");
        let name = harness.get_by_label("Name");
        name.focus();
        harness.run_steps(1);
        harness.get_by_label("Name").type_text("Ada");
        harness.run_steps(1);
        harness.key_press(egui::Key::Enter);
        harness.run_steps(2);

        let people = people_in(&fixture);
        assert_eq!(people[0].id, faces.first);
        assert_eq!(people[0].name.as_deref(), Some("Ada"));
        for face in &faces.first_faces {
            assert_eq!(
                face_row(&fixture, *face),
                (Some(faces.first.0), Some("manual".to_owned()), false)
            );
        }
        assert_eq!(harness.state().opened(), None, "Enter opened the viewer");
        harness.get_by_label("Back").click();
        harness.run_steps(2);
        harness.get_by_label("Ada (2)");
    }

    #[test]
    fn gui_merging_moves_faces_into_the_chosen_person() {
        let (fixture, faces) = face_fixture();
        let mut harness = face_harness(&fixture, true);
        harness.get_by_label("unnamed #1 (2)").click();
        harness.run_steps(2);
        harness.get_by_label("Merge into").click();
        harness.run_steps(2);
        harness.get_by_label("unnamed #2 (2)").click();
        harness.run_steps(2);
        harness.get_by_label("Merge").click();
        harness.run_steps(2);

        let people = people_in(&fixture);
        assert_eq!(people.len(), 1);
        assert_eq!((people[0].id, people[0].faces), (faces.second, 4));
        for face in faces.first_faces.iter() {
            assert_eq!(
                face_row(&fixture, *face),
                (Some(faces.second.0), Some("manual".to_owned()), false)
            );
        }
        assert_eq!(harness.state().people_panel.opened(), Some(faces.second));
        harness.get_by_label("4 faces");
    }

    #[test]
    fn gui_removing_marking_and_ignoring_write_manual_rows() {
        let (fixture, faces) = face_fixture();
        let mut harness = face_harness(&fixture, true);
        harness.get_by_label("unnamed #2 (2)").click();
        harness.run_steps(2);
        harness
            .get_all_by_label("Remove from person")
            .next()
            .unwrap()
            .click();
        harness.run_steps(2);
        harness.get_by_label("1 face");
        let removed = faces
            .second_faces
            .iter()
            .copied()
            .find(|f| face_row(&fixture, *f).0.is_none())
            .unwrap();
        assert_eq!(
            face_row(&fixture, removed),
            (None, Some("manual".to_owned()), false)
        );
        harness.get_by_label("Not a face").click();
        harness.run_steps(2);
        let kept = faces
            .second_faces
            .iter()
            .copied()
            .find(|f| *f != removed)
            .unwrap();
        assert_eq!(
            face_row(&fixture, kept),
            (None, Some("manual".to_owned()), true)
        );

        harness.get_by_label("Back").click();
        harness.run_steps(2);
        harness.get_by_label("unnamed #1 (2)").click();
        harness.run_steps(2);
        harness.get_by_label("Ignore").click();
        harness.run_steps(2);
        assert!(
            people_in(&fixture)
                .iter()
                .any(|p| p.id == faces.first && p.ignored)
        );
        assert!(harness.query_by_label("unnamed #1 (2)").is_none());
        harness.get_by_label("Show ignored").click();
        harness.run_steps(2);
        harness.get_by_label("unnamed #1 (2), ignored");
    }

    #[test]
    fn gui_filters_replace_the_set_the_grid_and_viewer_use() {
        use eframe::egui::accesskit::Role;

        let (fixture, faces) = face_fixture();
        let mut harness = face_harness(&fixture, true);
        assert_eq!(harness.state().nav().len(), A.len() + B.len() + 1);

        harness
            .get_by_role_and_label(Role::Button, "People")
            .click();
        harness.run_steps(2);
        harness
            .get_by_role_and_label(Role::CheckBox, "unnamed #2 (2)")
            .click();
        harness.run_steps(2);
        harness.key_press(egui::Key::Escape);
        harness.run_steps(2);
        harness.get_by_label("People: unnamed #2");
        assert_eq!(harness.state().filter_bar.people.len(), 1);
        assert!(harness.state().filter_bar.people.contains(&faces.second));
        assert_eq!(file_names(harness.state()), ["a/png.png", "b/heic.heic"]);
        harness.get_by_label("png.png");
        assert!(harness.query_by_label("jpeg.jpg").is_none());

        harness.get_by_label("png.png").click();
        harness.run_steps(1);
        harness.key_press(egui::Key::Enter);
        harness.run_steps(2);
        harness.key_press(egui::Key::ArrowRight);
        harness.run_steps(2);
        let opened = harness.state().opened().unwrap();
        assert!(
            harness
                .state()
                .nav()
                .current()
                .unwrap()
                .path
                .ends_with("b/heic.heic")
        );
        harness.key_press(egui::Key::Escape);
        harness.run_steps(2);

        harness.get_by_label("Faces").click();
        harness.run_steps(2);
        harness.get_by_label("1").click();
        harness.run_steps(2);
        assert_eq!(file_names(harness.state()), ["b/heic.heic"]);
        assert_eq!(harness.state().nav().current().map(|e| e.id), Some(opened));

        harness.get_by_label("Clear filters").click();
        harness.run_steps(2);
        harness.get_by_label("Unknown faces only").click();
        harness.run_steps(2);
        assert_eq!(
            file_names(harness.state()),
            ["a/gif.gif", "a/jpeg.jpg", "a/png.png", "b/heic.heic"]
        );
        harness.get_by_label("Faces").click();
        harness.run_steps(2);
        harness.get_by_label("0").click();
        harness.run_steps(2);
        assert!(harness.state().nav().is_empty());
        harness.get_by_label("Unknown faces only").click();
        harness.run_steps(2);
        assert_eq!(file_names(harness.state()), ["a/tiff.tif"]);
    }

    #[test]
    fn gui_face_controls_are_absent_when_faces_are_disabled() {
        let (fixture, _) = face_fixture();
        let mut harness = face_harness(&fixture, false);
        harness.run_steps(2);
        for label in [
            "People",
            "Faces",
            "Unknown faces only",
            "Show ignored",
            "unnamed #1 (2)",
        ] {
            assert!(harness.query_by_label(label).is_none(), "{label} shown");
        }
        harness.get_by_label("Rescan");
        harness.get_by_label("Country");
        assert_eq!(harness.state().nav().len(), A.len() + B.len() + 1);

        let mut harness = face_harness(&fixture, true);
        harness.run_steps(2);
        harness.get_by_label("Unknown faces only");
        harness.get_by_label("unnamed #1 (2)");
    }

    #[test]
    fn gui_place_filter_cascades_without_faces() {
        let fixture = fixture();
        let porto = Place {
            region_code: "17".to_owned(),
            region: "Porto".to_owned(),
            ..catalog::geo::tests::place(PORTO, "Porto")
        };
        let paris = paris(None);
        let mut conn = catalog::open(&fixture.paths.catalog_db()).unwrap();
        let tx = conn.transaction().unwrap();
        let lisbon = catalog::geo::tests::place(LISBON, "Lisbon");
        let sha = |file| sha_of(&fixture, file);
        catalog::geo::set_manual(&tx, &sha("a/png.png"), &lisbon, 1).unwrap();
        catalog::geo::set_manual(&tx, &sha("b/heic.heic"), &porto, 1).unwrap();
        catalog::geo::set_manual(&tx, &sha("a/jpeg.jpg"), &paris, 1).unwrap();
        catalog::geo::clear(&tx, &sha("a/gif.gif"), 1).unwrap();
        tx.commit().unwrap();

        let mut harness = face_harness(&fixture, false);
        let all = A.len() + B.len() + 1;
        assert_eq!(harness.state().nav().len(), all);
        let choose = |harness: &mut Harness<'_, VitrineApp>, combo: &str, option: &str| {
            harness.get_by_label(combo).click();
            harness.run_steps(2);
            harness.get_by_label(option).click();
            harness.run_steps(2);
        };
        assert!(harness.query_by_label("Region").is_none());

        choose(&mut harness, "Country", "Portugal (2)");
        assert_eq!(file_names(harness.state()), ["a/png.png", "b/heic.heic"]);
        assert!(harness.query_by_label("City").is_none());
        choose(&mut harness, "Region", "Lisbon (1)");
        assert_eq!(file_names(harness.state()), ["a/png.png"]);
        choose(&mut harness, "City", "Lisbon (1)");
        assert_eq!(harness.state().filter_bar.place.city, Some(LISBON));
        assert_eq!(file_names(harness.state()), ["a/png.png"]);
        harness.get_by_label("Country: Portugal");
        harness.get_by_label("City: Lisbon");
        choose(&mut harness, "Region: Lisbon", "Porto (1)");
        assert_eq!(harness.state().filter_bar.place.city, None);
        assert_eq!(file_names(harness.state()), ["b/heic.heic"]);

        let tx = conn.transaction().unwrap();
        catalog::geo::clear(&tx, &sha("b/heic.heic"), 2).unwrap();
        tx.commit().unwrap();
        harness.state_mut().refresh();
        harness.run_steps(2);
        assert_eq!(harness.state().filter_bar.place.region, None);
        assert_eq!(file_names(harness.state()), ["a/png.png"]);

        harness.get_by_label("No location").click();
        harness.run_steps(2);
        assert_eq!(harness.state().filter_bar.place.country, None);
        let mut unlocated = file_names(harness.state());
        unlocated.sort();
        let mut expected: Vec<String> = A
            .iter()
            .map(|f| format!("a/{f}"))
            .chain(B.iter().map(|f| format!("b/{f}")))
            .filter(|f| f != "a/png.png" && f != "a/jpeg.jpg")
            .collect();
        expected.sort();
        assert_eq!(unlocated, expected);

        choose(&mut harness, "Country", "France (1)");
        assert!(!harness.state().filter_bar.place.none);
        assert_eq!(file_names(harness.state()), ["a/jpeg.jpg"]);

        harness.get_by_label("Clear filters").click();
        harness.run_steps(2);
        assert_eq!(harness.state().filter_bar, FilterBar::default());
        assert_eq!(harness.state().nav().len(), all);
    }

    #[test]
    fn gui_country_search_enter_applies_without_opening_the_viewer() {
        let fixture = fixture();
        let mut conn = catalog::open(&fixture.paths.catalog_db()).unwrap();
        let tx = conn.transaction().unwrap();
        let lisbon = catalog::geo::tests::place(LISBON, "Lisbon");
        catalog::geo::set_manual(&tx, &sha_of(&fixture, "a/png.png"), &lisbon, 1).unwrap();
        catalog::geo::set_manual(&tx, &sha_of(&fixture, "a/jpeg.jpg"), &paris(None), 1).unwrap();
        tx.commit().unwrap();

        let mut harness = face_harness(&fixture, false);
        harness.get_by_label("png.png").click();
        harness.run_steps(1);
        harness.get_by_label("Country").click();
        harness.run_steps(2);
        harness.get_by_label("Portugal (1)");
        harness.get_by_label("Search").type_text("fra");
        harness.run_steps(1);
        harness.get_by_label("France (1)");
        assert!(harness.query_by_label("Portugal (1)").is_none());
        harness.key_press(egui::Key::Enter);
        harness.run_steps(2);

        assert_eq!(
            harness.state().filter_bar.place.country.as_deref(),
            Some("FR")
        );
        assert_eq!(file_names(harness.state()), ["a/jpeg.jpg"]);
        assert!(harness.query_by_label("Search").is_none(), "popup open");
        assert_eq!(harness.state().opened(), None);
        harness.get_by_label("Country: France");

        harness.get_by_label("jpeg.jpg").click();
        harness.run_steps(1);
        harness.key_press(egui::Key::Enter);
        harness.run_steps(2);
        assert!(harness.state().opened().is_some());
    }

    fn paris(coords: Option<(f64, f64)>) -> Place {
        Place {
            geoname_id: 2_988_507,
            name: "Paris".to_owned(),
            region_code: "11".to_owned(),
            region: "Ile-de-France".to_owned(),
            country_code: "FR".to_owned(),
            country: "France".to_owned(),
            coords,
        }
    }

    const LISBON: i64 = 2_267_057;
    const PORTO: i64 = 2_735_943;
    const LISBON_LABEL: &str = "Lisbon, Lisbon, Portugal";

    fn gazetteer(paths: &Paths) {
        use crate::geo::fixtures::{build, city};
        let cities = [
            city(
                LISBON, "Lisbon", "Lisbon", "Lisboa", 38.71667, -9.13333, "PT", "14", 517_802,
            ),
            city(
                PORTO, "Porto", "Porto", "", 41.14961, -8.61099, "PT", "17", 249_633,
            ),
        ]
        .concat();
        fs::create_dir_all(&paths.data_dir).unwrap();
        build(&paths.data_dir, &cities);
    }

    fn sha_of(fixture: &Fixture, file: &str) -> db::Sha256 {
        let conn = catalog::open_read_only(&fixture.paths.catalog_db()).unwrap();
        catalog::images::get_by_path(&conn, &fixture.tree.join(file))
            .unwrap()
            .unwrap()
            .sha256
            .unwrap()
    }

    fn location(fixture: &Fixture, file: &str) -> Option<Location> {
        let conn = catalog::open_read_only(&fixture.paths.catalog_db()).unwrap();
        catalog::geo::location_of(&conn, &sha_of(fixture, file)).unwrap()
    }

    /// `(sha256, place_id, source)` of every location row.
    fn location_rows(fixture: &Fixture) -> Vec<(db::Sha256, Option<i64>, String)> {
        let conn = catalog::open_read_only(&fixture.paths.catalog_db()).unwrap();
        let mut stmt = conn
            .prepare("SELECT sha256, place_id, source FROM locations ORDER BY sha256")
            .unwrap();
        stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
    }

    fn open_in_viewer(harness: &mut Harness<'_, VitrineApp>, name: &str) -> ImageId {
        harness.get_by_label(name).click();
        harness.run_steps(1);
        harness.key_press(egui::Key::Enter);
        harness.run_steps(2);
        let id = harness.state().opened().unwrap();
        assert!(
            harness
                .state()
                .nav()
                .current()
                .unwrap()
                .path
                .ends_with(name)
        );
        id
    }

    fn click_info(harness: &mut Harness<'_, VitrineApp>) {
        harness.get_by_label("Info").click();
        harness.run_steps(2);
    }

    /// Opens the Info panel's place picker.
    fn change_location(harness: &mut Harness<'_, VitrineApp>) {
        harness.get_by_label("Change location...").click();
        harness.run_steps(2);
    }

    fn manual(place: Option<i64>) -> Option<(Option<i64>, catalog::geo::Source)> {
        Some((place, catalog::geo::Source::Manual))
    }

    fn located(fixture: &Fixture, file: &str) -> Option<(Option<i64>, catalog::geo::Source)> {
        location(fixture, file).map(|l| (l.place.map(|p| p.geoname_id), l.source))
    }

    #[test]
    fn gui_typing_a_place_leaves_the_viewer_on_its_image_and_picking_writes_manual() {
        let fixture = fixture();
        gazetteer(&fixture.paths);
        let mut harness = face_harness(&fixture, false);
        let id = open_in_viewer(&mut harness, "png.png");
        harness.get_by_label("No location");
        for label in ["Edit", "Clear", "Show on map", "Change location..."] {
            assert!(harness.query_by_label(label).is_none(), "{label} shown");
        }
        click_info(&mut harness);
        change_location(&mut harness);
        harness.get_by_label("Place").type_text("lis");
        harness.run_steps(1);
        for key in [
            egui::Key::Space,
            egui::Key::Backspace,
            egui::Key::O,
            egui::Key::I,
        ] {
            harness.key_press(key);
            harness.run_steps(1);
        }
        harness.run_steps(1);
        assert_eq!(harness.state().opened(), Some(id));
        assert!(harness.state().viewer.info_open(), "i closed the panel");

        harness.get_by_label(LISBON_LABEL).click();
        harness.run_steps(2);
        assert_eq!(located(&fixture, "a/png.png"), manual(Some(LISBON)));
        assert_eq!(harness.state().opened(), Some(id));
        harness.get_by_label(&format!("{LISBON_LABEL} (manual)"));
        harness.get_by_label(&format!("Place: {LISBON_LABEL}"));
        harness.get_by_label("Set manually");
        assert!(
            harness.query_by_label("Place").is_none(),
            "picker still open"
        );

        change_location(&mut harness);
        harness.get_by_label("Recent");
        harness.get_by_label(LISBON_LABEL);
    }

    #[test]
    fn gui_locating_an_image_the_filter_then_excludes_keeps_it_in_the_viewer() {
        let fixture = fixture();
        gazetteer(&fixture.paths);
        let mut harness = face_harness(&fixture, false);
        harness.get_by_label("No location").click();
        harness.run_steps(2);
        assert!(file_names(harness.state()).contains(&"a/png.png".to_owned()));
        let id = open_in_viewer(&mut harness, "png.png");
        click_info(&mut harness);
        change_location(&mut harness);
        harness.get_by_label("Place").type_text("lis");
        harness.run_steps(2);
        harness.get_by_label(LISBON_LABEL).click();
        harness.run_steps(2);
        assert_eq!(located(&fixture, "a/png.png"), manual(Some(LISBON)));
        assert_eq!(harness.state().opened(), Some(id));
        harness.get_by_label(&format!("Place: {LISBON_LABEL}"));

        harness.key_press(egui::Key::Escape);
        harness.run_steps(2);
        assert_eq!(harness.state().opened(), None);
        assert!(!file_names(harness.state()).contains(&"a/png.png".to_owned()));
    }

    #[test]
    fn gui_location_is_read_only_without_a_gazetteer() {
        let fixture = fixture();
        let mut conn = catalog::open(&fixture.paths.catalog_db()).unwrap();
        let tx = conn.transaction().unwrap();
        let lisbon = catalog::geo::tests::place(LISBON, "Lisbon");
        catalog::geo::set_manual(&tx, &sha_of(&fixture, "a/png.png"), &lisbon, 1).unwrap();
        tx.commit().unwrap();
        let mut harness = face_harness(&fixture, false);
        harness.get_by_label("Set location").click();
        harness.run_steps(2);
        assert!(harness.query_by_label("Place").is_none(), "picker opened");

        open_in_viewer(&mut harness, "png.png");
        harness.get_by_label(&format!("{LISBON_LABEL} (manual)"));
        click_info(&mut harness);
        harness.get_by_label(place_picker::FETCH_HINT);
        harness.get_by_label(&format!("Place: {LISBON_LABEL}"));
        assert!(
            harness.query_by_label("Change location...").is_none(),
            "picker offered"
        );
    }

    #[test]
    fn gui_cleared_exif_location_stays_cleared_after_reindexing() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let tree = root.join("photos");
        fs::create_dir_all(&tree).unwrap();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        fs::copy(fixtures.join("gps.jpg"), tree.join("gps.jpg")).unwrap();
        let fixture = Fixture {
            _dir: dir,
            paths: paths_in(&root),
            tree,
        };
        gazetteer(&fixture.paths);
        assert_eq!(index(&fixture).stats.places_resolved, 1);
        assert_eq!(
            located(&fixture, "gps.jpg"),
            Some((Some(LISBON), catalog::geo::Source::Exif))
        );

        let mut harness = face_harness(&fixture, false);
        open_in_viewer(&mut harness, "gps.jpg");
        harness.get_by_label(&format!("{LISBON_LABEL} (exif)"));
        click_info(&mut harness);
        harness.get_by_label("From EXIF");
        change_location(&mut harness);
        harness.get_by_label("Remove location").click();
        harness.run_steps(2);
        assert_eq!(located(&fixture, "gps.jpg"), manual(None));
        harness.get_by_label("No location (manual)");
        harness.get_by_label("Not located");
        drop(harness);

        assert_eq!(index(&fixture).stats.places_resolved, 0);
        assert_eq!(located(&fixture, "gps.jpg"), manual(None));
    }

    #[test]
    fn gui_picker_shows_the_current_place_and_remove_writes_the_marker() {
        let fixture = fixture();
        gazetteer(&fixture.paths);
        let mut conn = catalog::open(&fixture.paths.catalog_db()).unwrap();
        let tx = conn.transaction().unwrap();
        let lisbon = catalog::geo::tests::place(LISBON, "Lisbon");
        catalog::geo::set_manual(&tx, &sha_of(&fixture, "a/png.png"), &lisbon, 1).unwrap();
        tx.commit().unwrap();
        let mut harness = face_harness(&fixture, false);
        let id = open_in_viewer(&mut harness, "png.png");
        click_info(&mut harness);
        change_location(&mut harness);
        harness.get_by_label(&format!("Now: {}", lisbon.label()));
        harness.get_by_label("Hides the EXIF location too");
        harness.get_by_label("Remove location").click();
        harness.run_steps(2);
        assert_eq!(located(&fixture, "a/png.png"), manual(None));
        assert_eq!(harness.state().opened(), Some(id));
        harness.get_by_label("No location (manual)");
    }

    #[test]
    fn gui_remove_location_is_disabled_in_the_viewer_for_an_unlocated_image() {
        let fixture = fixture();
        gazetteer(&fixture.paths);
        let mut harness = face_harness(&fixture, false);
        open_in_viewer(&mut harness, "png.png");
        harness.get_by_label("No location");
        click_info(&mut harness);
        change_location(&mut harness);
        let button = harness.get_by_label("Remove location");
        assert!(button.accesskit_node().is_disabled());
        button.click();
        harness.run_steps(2);
        assert_eq!(located(&fixture, "a/png.png"), None);
        assert!(harness.query_by_label_contains("Now: ").is_none());
    }

    #[test]
    fn gui_set_location_writes_the_selection_else_the_cursor_image() {
        let fixture = unindexed();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        fs::copy(fixtures.join("jpeg.jpg"), fixture.tree.join("a/copy.jpg")).unwrap();
        index(&fixture);
        gazetteer(&fixture.paths);
        let mut harness = face_harness(&fixture, false);
        harness.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
        harness.run_steps(1);
        let entries = harness.state().nav().entries();
        let content = distinct_content(entries);
        let images = entries.iter().filter(|e| e.kind == Kind::Image).count();
        assert_eq!(harness.state().selection().len(), entries.len());
        assert!(content.len() < images, "{} of {images}", content.len());

        let set = |harness: &mut Harness<'_, VitrineApp>, query: &str| {
            harness.get_by_label("Set location").click();
            harness.run_steps(2);
            harness.get_by_label("Place").type_text(query);
            harness.run_steps(1);
            harness.key_press(egui::Key::Enter);
            harness.run_steps(2);
            assert_eq!(harness.state().opened(), None, "Enter opened the viewer");
        };
        set(&mut harness, "lis");
        let rows = location_rows(&fixture);
        assert_eq!(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
            content,
            "one row per distinct content"
        );
        assert!(rows.iter().all(|r| r.1 == Some(LISBON) && r.2 == "manual"));

        harness.get_by_label("png.png").click();
        // the toolbar drops the selection controls a frame later
        harness.run_steps(2);
        assert!(harness.state().selection().is_empty());
        set(&mut harness, "porto");
        assert_eq!(located(&fixture, "a/png.png"), manual(Some(PORTO)));
        assert_eq!(located(&fixture, "a/jpeg.jpg"), manual(Some(LISBON)));
        assert_eq!(location_rows(&fixture).len(), content.len());
    }

    #[test]
    fn gui_grid_location_label_summarizes_the_selection_else_the_cursor_image() {
        let fixture = fixture();
        gazetteer(&fixture.paths);
        let mut conn = catalog::open(&fixture.paths.catalog_db()).unwrap();
        let tx = conn.transaction().unwrap();
        let lisbon = catalog::geo::tests::place(LISBON, "Lisbon");
        let porto = catalog::geo::tests::place(PORTO, "Porto");
        assert!(
            catalog::geo::insert_exif(&tx, &sha_of(&fixture, "a/jpeg.jpg"), &lisbon, 1).unwrap()
        );
        for file in ["a/png.png", "a/webp.webp"] {
            catalog::geo::set_manual(&tx, &sha_of(&fixture, file), &lisbon, 1).unwrap();
        }
        catalog::geo::set_manual(&tx, &sha_of(&fixture, "a/tiff.tif"), &porto, 1).unwrap();
        catalog::geo::clear(&tx, &sha_of(&fixture, "a/gif.gif"), 1).unwrap();
        tx.commit().unwrap();
        drop(conn);

        let mut harness = face_harness(&fixture, false);
        assert!(harness.state().nav().current().is_none());
        assert!(harness.query_by_label("Not located").is_none());
        assert!(harness.query_by_label_contains("Lisbon").is_none());

        let click = |harness: &mut Harness<'_, VitrineApp>, name: &str, modifiers| {
            harness.get_by_label(name).click_modifiers(modifiers);
            harness.run_steps(2);
        };
        click(&mut harness, "jpeg.jpg", egui::Modifiers::NONE);
        harness.get_by_label(&format!("{LISBON_LABEL} (exif)"));

        click(&mut harness, "png.png", egui::Modifiers::COMMAND);
        click(&mut harness, "webp.webp", egui::Modifiers::COMMAND);
        harness.get_by_label("2 selected");
        harness.get_by_label(&format!("{LISBON_LABEL} (manual)"));

        click(&mut harness, "jpeg.jpg", egui::Modifiers::COMMAND);
        harness.get_by_label(LISBON_LABEL);

        click(&mut harness, "gif.gif", egui::Modifiers::COMMAND);
        harness.get_by_label("Mixed (2 places)");
        click(&mut harness, "tiff.tif", egui::Modifiers::COMMAND);
        harness.get_by_label("Mixed (3 places)");

        harness.get_by_label("Clear selection").click();
        harness.run_steps(2);
        click(&mut harness, "gif.gif", egui::Modifiers::NONE);
        harness.get_by_label("Not located");

        harness.get_by_label("Set location").click();
        harness.run_steps(2);
        assert!(harness.query_by_label_contains("Now: ").is_none());
        harness.get_by_label("Place").type_text("lis");
        harness.run_steps(1);
        harness.key_press(egui::Key::Enter);
        harness.run_steps(2);
        assert_eq!(located(&fixture, "a/gif.gif"), manual(Some(LISBON)));
        assert!(harness.query_by_label("Not located").is_none());
        harness.get_by_label(&format!("{LISBON_LABEL} (manual)"));

        harness.get_by_label("Set location").click();
        harness.run_steps(2);
        harness.get_by_label(&format!("Now: {LISBON_LABEL}"));
    }

    #[test]
    fn gui_event_suggestion_shows_the_event_and_applies_only_to_unlocated_members() {
        let fixture = fixture();
        gazetteer(&fixture.paths);
        let mut conn = catalog::open(&fixture.paths.catalog_db()).unwrap();
        conn.execute("UPDATE images SET taken_at = NULL", [])
            .unwrap();
        for (file, taken_at) in [
            ("a/jpeg.jpg", "2020-05-07T10:00:00"),
            ("a/png.png", "2020-05-07T11:00:00"),
            ("a/webp.webp", "2020-05-07T12:00:00"),
            ("b/heic.heic", "2020-05-07T13:00:00"),
            ("b/avif.avif", "2020-05-07T14:00:00"),
            ("a/gif.gif", "2021-01-01T10:00:00"),
        ] {
            let path = fixture.tree.join(file);
            let changed = conn
                .execute(
                    "UPDATE images SET taken_at = ?1 WHERE path = ?2",
                    [taken_at, path.to_str().unwrap()],
                )
                .unwrap();
            assert_eq!(changed, 1, "{file}");
        }
        let (heic, webp) = (
            sha_of(&fixture, "b/heic.heic"),
            sha_of(&fixture, "a/webp.webp"),
        );
        let tx = conn.transaction().unwrap();
        let porto = catalog::geo::tests::place(PORTO, "Porto");
        assert!(catalog::geo::insert_exif(&tx, &heic, &porto, 1).unwrap());
        catalog::geo::clear(&tx, &webp, 1).unwrap();
        tx.commit().unwrap();
        let event = [
            "a/jpeg.jpg",
            "a/png.png",
            "a/webp.webp",
            "b/heic.heic",
            "b/avif.avif",
        ];
        let prompt = "Apply Lisbon to 2 more images from this event?";

        let mut harness = face_harness(&fixture, false);
        let set = |harness: &mut Harness<'_, VitrineApp>, name: &str, query: &str| {
            harness.get_by_label(name).click();
            harness.run_steps(2);
            harness.get_by_label("Set location").click();
            harness.run_steps(2);
            harness.get_by_label("Place").type_text(query);
            harness.run_steps(1);
            harness.key_press(egui::Key::Enter);
            harness.run_steps(2);
        };
        let click = |harness: &mut Harness<'_, VitrineApp>, label: &str| {
            harness.get_by_label(label).click();
            harness.run_steps(2);
        };
        set(&mut harness, "png.png", "lis");
        harness.get_by_label(prompt);

        click(&mut harness, "Show");
        assert_eq!(file_names(harness.state()), event);
        harness.get_by_label_contains("5 items in this event");
        click(&mut harness, "Hide");
        assert_eq!(harness.state().nav().len(), A.len() + B.len() + 1);
        click(&mut harness, "Show");
        click(&mut harness, &format!("b ({})", B.len() + 1));
        assert_eq!(harness.state().nav().len(), B.len() + 1);
        harness.get_by_label(prompt);
        harness.get_by_label("Show");

        click(&mut harness, "Show");
        click(&mut harness, "No location");
        let names = file_names(harness.state());
        assert_eq!(names.len(), B.len() - 1, "{names:?}");
        assert!(
            names
                .iter()
                .all(|n| n.starts_with("b/") && n != "b/heic.heic")
        );
        harness.get_by_label(prompt);
        harness.get_by_label("Show");
        click(&mut harness, "Clear filters");

        click(&mut harness, "Show");
        assert_eq!(file_names(harness.state()), event);
        click(&mut harness, "Dismiss");
        assert!(
            harness.query_by_label(prompt).is_none(),
            "prompt still shown"
        );
        assert_eq!(harness.state().nav().len(), B.len() + 1);

        click(&mut harness, &format!("a ({})", A.len()));
        open_in_viewer(&mut harness, "png.png");
        click_info(&mut harness);
        change_location(&mut harness);
        harness.get_by_label("Place").type_text("lis");
        harness.run_steps(1);
        click(&mut harness, LISBON_LABEL);
        harness.get_by_label(prompt);
        click(&mut harness, "Show");
        assert_eq!(harness.state().opened(), None);
        assert_eq!(file_names(harness.state()), event);
        open_in_viewer(&mut harness, "png.png");
        click(&mut harness, "Dismiss");
        assert_eq!(harness.state().opened(), None);
        assert_eq!(harness.state().nav().len(), A.len());

        set(&mut harness, "png.png", "lis");
        click(&mut harness, "Show");
        open_in_viewer(&mut harness, "png.png");
        click(&mut harness, "Apply");
        assert_eq!(harness.state().opened(), None);
        assert!(
            harness.query_by_label(prompt).is_none(),
            "prompt still shown"
        );
        assert_eq!(harness.state().nav().len(), A.len());
        for file in ["a/jpeg.jpg", "a/png.png", "b/avif.avif"] {
            assert_eq!(located(&fixture, file), manual(Some(LISBON)), "{file}");
        }
        assert_eq!(
            located(&fixture, "b/heic.heic"),
            Some((Some(PORTO), catalog::geo::Source::Exif))
        );
        assert_eq!(located(&fixture, "a/webp.webp"), manual(None));
        assert_eq!(located(&fixture, "a/gif.gif"), None);
        assert_eq!(location_rows(&fixture).len(), 5);

        set(&mut harness, "jpeg.jpg", "porto");
        assert_eq!(located(&fixture, "a/jpeg.jpg"), manual(Some(PORTO)));
        assert!(harness.state().suggestion.is_none());
        set(&mut harness, "gif.gif", "porto");
        assert_eq!(located(&fixture, "a/gif.gif"), manual(Some(PORTO)));
        assert!(harness.state().suggestion.is_none());
        assert!(harness.query_by_label("Dismiss").is_none());
    }

    /// `lisbon/gps.jpg` with unresolved EXIF Lisbon and `paris/jpeg.jpg` set to Paris by hand.
    fn map_fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let tree = root.join("photos");
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        for (folder, file) in [("lisbon", "gps.jpg"), ("paris", "jpeg.jpg")] {
            fs::create_dir_all(tree.join(folder)).unwrap();
            fs::copy(fixtures.join(file), tree.join(folder).join(file)).unwrap();
        }
        let fixture = Fixture {
            _dir: dir,
            paths: paths_in(&root),
            tree,
        };
        index(&fixture);
        let sha = sha_of(&fixture, "paris/jpeg.jpg");
        let mut conn = catalog::open(&fixture.paths.catalog_db()).unwrap();
        let tx = conn.transaction().unwrap();
        catalog::geo::set_manual(&tx, &sha, &paris(Some((48.85, 2.35))), 1).unwrap();
        tx.commit().unwrap();
        fixture
    }

    fn id_of(fixture: &Fixture, file: &str) -> ImageId {
        let conn = catalog::open_read_only(&fixture.paths.catalog_db()).unwrap();
        catalog::images::get_by_path(&conn, &fixture.tree.join(file))
            .unwrap()
            .unwrap()
            .id
    }

    fn press(harness: &mut Harness<'_, VitrineApp>, pos: egui::Pos2, pressed: bool) {
        harness.event(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        });
    }

    fn click_label(harness: &mut Harness<'_, VitrineApp>, label: &str) {
        harness.get_by_label(label).click();
        harness.run_steps(2);
    }

    /// Drags from between the two bubbles, up and right; returns the new viewport.
    fn pan(
        harness: &mut Harness<'_, VitrineApp>,
        lisbon: ImageId,
        paris: ImageId,
    ) -> crate::map::Viewport {
        let map = &harness.state().map;
        let at_lisbon = map.cluster_pos(lisbon).unwrap();
        let at_paris = map.cluster_pos(paris).unwrap();
        // Lisbon sits bottom-left of Paris; panning up and right keeps it in view.
        let start = at_lisbon + (at_paris - at_lisbon) / 2.0;
        harness.event(egui::Event::PointerMoved(start));
        press(harness, start, true);
        for i in 1..=10_u8 {
            let step = f32::from(i);
            harness.event(egui::Event::PointerMoved(
                start + egui::vec2(6.0 * step, -4.0 * step),
            ));
        }
        press(harness, start + egui::vec2(60.0, -40.0), false);
        harness.run_steps(1);
        harness.state().map.viewport()
    }

    #[test]
    fn gui_map_toggle_keeps_the_viewport_and_a_cluster_click_filters_the_grid() {
        let fixture = map_fixture();
        let lisbon = id_of(&fixture, "lisbon/gps.jpg");
        let paris = id_of(&fixture, "paris/jpeg.jpg");
        let mut harness = face_harness(&fixture, false);
        harness.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
        harness.run_steps(2);
        harness.get_by_label("2 selected");

        click_label(&mut harness, "Map");
        let app = harness.state();
        assert_eq!(app.view, MainView::Map);
        for label in [
            "gps.jpg",
            "jpeg.jpg",
            "2 selected",
            "Clear selection",
            "Set location",
        ] {
            assert!(harness.query_by_label(label).is_none(), "{label} shown");
        }
        let fitted = app.map.viewport();
        let at_lisbon = app.map.cluster_pos(lisbon).unwrap();
        let at_paris = app.map.cluster_pos(paris).unwrap();
        assert_ne!(at_lisbon, at_paris);

        click_label(&mut harness, "Grid");
        assert_eq!(harness.state().view, MainView::Grid);
        harness.get_by_label("gps.jpg");
        click_label(&mut harness, "Map");
        assert_eq!(harness.state().map.viewport(), fitted);

        let panned = pan(&mut harness, lisbon, paris);
        assert_ne!(panned, fitted);
        assert_eq!(harness.state().view, MainView::Map);

        let at = harness.state().map.cluster_pos(lisbon).unwrap();
        harness.event(egui::Event::PointerMoved(at));
        press(&mut harness, at, true);
        press(&mut harness, at, false);
        harness.run_steps(2);
        let app = harness.state();
        assert_eq!(app.view, MainView::Grid);
        let area = app.filter_bar.area.expect("cluster click sets the area");
        assert_eq!(file_names(app), ["lisbon/gps.jpg"]);
        harness.get_by_label("gps.jpg");
        harness.get_by_label("Area");
        assert!(harness.query_by_label("jpeg.jpg").is_none());

        click_label(&mut harness, "Map");
        let map = &harness.state().map;
        assert_eq!(map.viewport(), panned);
        assert_eq!(map.outlined_area(), Some(area));
        assert!(map.cluster_pos(lisbon).is_some());
        assert!(map.cluster_pos(paris).is_some());

        click_label(&mut harness, "Clear filters");
        let app = harness.state();
        assert_eq!(app.filter_bar.area, None);
        assert_eq!(app.map.outlined_area(), None);
        assert_eq!(app.map.viewport(), panned);
        assert_eq!(app.nav().len(), 2);
        assert!(harness.query_by_label("Area").is_none());
    }

    #[test]
    fn gui_map_folder_change_narrows_the_points_and_refits() {
        let fixture = map_fixture();
        let lisbon = id_of(&fixture, "lisbon/gps.jpg");
        let paris = id_of(&fixture, "paris/jpeg.jpg");
        let mut harness = face_harness(&fixture, false);
        click_label(&mut harness, "Map");
        let fitted = harness.state().map.viewport();
        assert!(harness.state().map.cluster_pos(paris).is_some());

        click_label(&mut harness, "lisbon (1)");
        let app = harness.state();
        assert_eq!(app.view, MainView::Map);
        assert!(app.map.cluster_pos(lisbon).is_some());
        assert!(app.map.cluster_pos(paris).is_none());
        let viewport = app.map.viewport();
        assert_ne!(viewport, fitted);
        // a single point fits at the maximum fit zoom
        assert!((viewport.zoom - 12.0).abs() < 1e-9, "{viewport:?}");
        let (lat, lon) = crate::map::unproject(viewport.center.0, viewport.center.1);
        assert!(
            (lat - 38.7223).abs() < 1e-3 && (lon + 9.1393).abs() < 1e-3,
            "{lat}, {lon}"
        );
    }

    #[test]
    fn gui_map_refresh_reloads_the_points_without_a_refit() {
        let fixture = map_fixture();
        let lisbon = id_of(&fixture, "lisbon/gps.jpg");
        let paris = id_of(&fixture, "paris/jpeg.jpg");
        let mut harness = face_harness(&fixture, false);
        click_label(&mut harness, "Map");
        let panned = pan(&mut harness, lisbon, paris);

        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        fs::copy(
            fixtures.join("png.png"),
            fixture.tree.join("lisbon/png.png"),
        )
        .unwrap();
        index(&fixture);
        let sha = sha_of(&fixture, "lisbon/png.png");
        let porto = Place {
            coords: Some((41.15, -8.61)),
            ..catalog::geo::tests::place(PORTO, "Porto")
        };
        let mut conn = catalog::open(&fixture.paths.catalog_db()).unwrap();
        let tx = conn.transaction().unwrap();
        catalog::geo::set_manual(&tx, &sha, &porto, 1).unwrap();
        tx.commit().unwrap();
        let added = id_of(&fixture, "lisbon/png.png");
        assert!(harness.state().map.cluster_pos(added).is_none());

        harness.state_mut().refresh();
        harness.run_steps(2);
        let map = &harness.state().map;
        assert!(map.cluster_pos(added).is_some());
        assert_eq!(map.viewport(), panned);
    }

    #[test]
    fn gui_map_first_points_keep_a_focus() {
        let fixture = map_fixture();
        let paris = id_of(&fixture, "paris/jpeg.jpg");
        let mut harness = face_harness(&fixture, false);
        harness.state_mut().map.focus(paris, 48.85, 2.35);
        click_label(&mut harness, "Map");
        assert_focused(&harness.state().map, paris, (48.85, 2.35));
    }

    fn assert_focused(map: &MapView, id: ImageId, (lat, lon): (f64, f64)) {
        let viewport = map.viewport();
        assert!((viewport.zoom - 11.0).abs() < 1e-9, "{viewport:?}");
        let expected = crate::map::project(lat, lon);
        assert!(
            (viewport.center.0 - expected.0).abs() < 1e-12
                && (viewport.center.1 - expected.1).abs() < 1e-12,
            "{viewport:?}"
        );
        assert_eq!(map.highlighted(), Some(id));
    }

    #[test]
    fn gui_info_panel_narrows_the_image() {
        let fixture = fixture();
        let mut harness = face_harness(&fixture, false);
        open_in_viewer(&mut harness, "png.png");
        // the top bar's buttons sit at the image rect's right edge
        let edge = |harness: &Harness<'_, VitrineApp>| {
            harness
                .get_by_label("Open in external viewer")
                .rect()
                .right()
        };
        let full = edge(&harness);
        harness.get_by_label("Info").click();
        harness.run_steps(2);
        assert!(harness.state().viewer.info_open());
        harness.get_by_label("Details");
        assert!(edge(&harness) < full - 200.0, "{} {full}", edge(&harness));
    }

    fn mini_map_shown(harness: &Harness<'_, VitrineApp>) -> bool {
        use super::super::basemap::{Basemap, NaturalEarth};
        harness.output().shapes.iter().any(|s| {
            matches!(&s.shape, egui::Shape::Text(t) if t.galley.text() == NaturalEarth.attribution())
        })
    }

    #[test]
    fn gui_info_panel_stays_open_across_images_and_reopening() {
        let fixture = fixture();
        let mut harness = face_harness(&fixture, false);
        open_in_viewer(&mut harness, "png.png");
        assert!(harness.query_by_label("Details").is_none());
        harness.key_press(egui::Key::I);
        harness.run_steps(2);
        assert!(harness.state().viewer.info_open());
        harness.get_by_label("Details");

        harness.key_press(egui::Key::ArrowRight);
        harness.run_steps(2);
        let current = &harness.state().nav().current().unwrap().path;
        assert!(!current.ends_with("png.png"), "{}", current.display());
        harness.get_by_label("Details");
        harness.key_press(egui::Key::ArrowLeft);
        harness.run_steps(2);
        harness.get_by_label("Details");

        harness.key_press(egui::Key::Escape);
        harness.run_steps(2);
        assert_eq!(harness.state().opened(), None);
        open_in_viewer(&mut harness, "png.png");
        harness.get_by_label("Details");
        harness.key_press(egui::Key::I);
        harness.run_steps(2);
        assert!(!harness.state().viewer.info_open());
        assert!(harness.query_by_label("Details").is_none());
    }

    #[test]
    fn gui_info_panel_shows_the_place_and_a_mini_map() {
        let fixture = map_fixture();
        let mut harness = face_harness(&fixture, false);
        open_in_viewer(&mut harness, "jpeg.jpg");
        assert!(!mini_map_shown(&harness));
        click_info(&mut harness);
        harness.get_by_label(&format!("Place: {}", paris(None).label()));
        harness.get_by_label("Set manually");
        harness.get_by_label(place_picker::FETCH_HINT);
        assert!(mini_map_shown(&harness));
    }

    #[test]
    fn gui_info_panel_without_a_location_has_no_mini_map() {
        let fixture = fixture();
        let mut harness = face_harness(&fixture, false);
        open_in_viewer(&mut harness, "png.png");
        click_info(&mut harness);
        harness.get_by_label("Not located");
        harness.get_by_label(place_picker::FETCH_HINT);
        assert!(!mini_map_shown(&harness));
    }

    #[test]
    fn gui_info_panel_lists_the_people_of_the_image() {
        let (fixture, faces) = face_fixture();
        let mut harness = face_harness(&fixture, true);
        // the cached data belongs to one image; stepping away and back reloads it
        let reload = |harness: &mut Harness<'_, VitrineApp>| {
            for key in [egui::Key::ArrowRight, egui::Key::ArrowLeft] {
                harness.key_press(key);
                harness.run_steps(2);
            }
            assert!(
                harness
                    .state()
                    .nav()
                    .current()
                    .unwrap()
                    .path
                    .ends_with("png.png")
            );
        };
        open_in_viewer(&mut harness, "png.png");
        click_info(&mut harness);
        harness.get_by_label("People (2)");
        harness.get_by_label("unnamed #1");
        harness.get_by_label("unnamed #2");

        let mut conn = catalog::open(&fixture.paths.catalog_db()).unwrap();
        let tx = conn.transaction().unwrap();
        catalog::people::set_ignored(&tx, faces.second, true).unwrap();
        tx.commit().unwrap();
        reload(&mut harness);
        harness.get_by_label("People (2)");
        harness.get_by_label("unnamed #1");
        harness.get_by_label("unnamed #2, ignored");

        conn.execute(
            "UPDATE faces SET person_id = NULL, assigned_by = NULL WHERE id IN (?1, ?2)",
            [faces.first_faces[1].0, faces.second_faces[0].0],
        )
        .unwrap();
        reload(&mut harness);
        harness.get_by_label("People (2)");
        assert_eq!(harness.get_all_by_label("Unknown").count(), 2);
        assert!(harness.query_by_label("unnamed #1").is_none());

        harness.key_press(egui::Key::Escape);
        harness.run_steps(2);
        open_in_viewer(&mut harness, "tiff.tif");
        harness.get_by_label("People (0)");
        harness.get_by_label("No faces");
    }

    #[test]
    fn gui_info_panel_has_no_people_with_faces_disabled() {
        let (fixture, _) = face_fixture();
        let mut harness = face_harness(&fixture, false);
        open_in_viewer(&mut harness, "png.png");
        click_info(&mut harness);
        harness.get_by_label("Details");
        assert!(harness.query_by_label_contains("People (").is_none());
        assert!(harness.query_by_label("unnamed #1").is_none());
    }

    #[test]
    fn gui_show_in_people_from_the_info_panel_opens_them() {
        let (fixture, faces) = face_fixture();
        let mut harness = face_harness(&fixture, true);
        let id = open_in_viewer(&mut harness, "png.png");
        click_info(&mut harness);
        harness.get_all_by_label("...").next().unwrap().click();
        harness.run_steps(2);
        harness.get_by_label("Show in People").click();
        harness.run_steps(2);
        let app = harness.state();
        assert_eq!(app.opened(), None);
        assert_eq!(app.people_panel.opened(), Some(faces.first));
        assert_eq!(app.nav().current().map(|e| e.id), Some(id));
    }

    /// Opens the picker of the face row captioned `caption` and types `query`.
    fn pick_face(harness: &mut Harness<'_, VitrineApp>, caption: &str, query: &str) {
        harness.get_by_label(caption).click();
        harness.run_steps(2);
        harness.get_by_label("Search").type_text(query);
        harness.run_steps(1);
    }

    #[test]
    fn gui_naming_an_unnamed_person_in_the_info_panel_names_the_whole_person() {
        let (fixture, faces) = face_fixture();
        let mut harness = face_harness(&fixture, true);
        let id = open_in_viewer(&mut harness, "png.png");
        click_info(&mut harness);
        pick_face(&mut harness, "unnamed #1", "Ada");
        harness.get_by_label("New: Ada").click();
        harness.run_steps(2);
        harness.get_by_label("unnamed #1 has 2 faces. Name all of them \"Ada\", or only this one?");
        assert!(harness.query_by_label_contains("low-confidence").is_none());
        assert_eq!(people_in(&fixture)[0].name, None, "named before confirming");
        harness.get_by_label("All 2 faces").click();
        harness.run_steps(2);
        assert!(harness.query_by_label("All 2 faces").is_none());
        let ada = people_in(&fixture)
            .into_iter()
            .find(|p| p.id == faces.first)
            .unwrap();
        assert_eq!((ada.name.as_deref(), ada.faces), (Some("Ada"), 2));
        assert_eq!(harness.state().opened(), Some(id));
        harness.get_by_label("Ada");
        harness.get_by_label("unnamed #2");
    }

    #[test]
    fn gui_only_this_face_names_one_face_and_low_scores_warn() {
        let (fixture, faces) = face_fixture();
        let conn = catalog::open(&fixture.paths.catalog_db()).unwrap();
        conn.execute(
            "UPDATE faces SET score = 0.61 WHERE person_id = ?1",
            [faces.first.0],
        )
        .unwrap();
        let mut harness = face_harness(&fixture, true);
        open_in_viewer(&mut harness, "png.png");
        click_info(&mut harness);
        pick_face(&mut harness, "unnamed #1", "Ada");
        harness.key_press(egui::Key::Enter);
        harness.run_steps(2);
        harness.get_by_label_contains("2 of the 2 are low-confidence detections");

        harness.get_by_label("Cancel").click();
        harness.run_steps(2);
        assert!(harness.query_by_label_contains("low-confidence").is_none());
        assert_eq!(people_in(&fixture).len(), 2);

        pick_face(&mut harness, "unnamed #1", "Ada");
        harness.key_press(egui::Key::Enter);
        harness.run_steps(2);
        harness.get_by_label("Only this face").click();
        harness.run_steps(2);
        let people = people_in(&fixture);
        let ada = people
            .iter()
            .find(|p| p.name.as_deref() == Some("Ada"))
            .unwrap();
        assert_eq!(ada.faces, 1);
        assert_eq!(face_row(&fixture, faces.first_faces[1]).0, Some(ada.id.0));
        let first = people.iter().find(|p| p.id == faces.first).unwrap();
        assert_eq!((first.name.as_deref(), first.faces), (None, 1));
    }

    #[test]
    fn gui_picking_a_person_for_an_unnamed_one_merges_them() {
        let (fixture, faces) = face_fixture();
        let mut harness = face_harness(&fixture, true);
        open_in_viewer(&mut harness, "png.png");
        click_info(&mut harness);
        pick_face(&mut harness, "unnamed #2", "#1");
        harness.key_press(egui::Key::Enter);
        harness.run_steps(2);
        harness.get_by_label("All 2 faces").click();
        harness.run_steps(2);
        let people = people_in(&fixture);
        assert_eq!(people.len(), 1);
        assert_eq!((people[0].id, people[0].faces), (faces.first, 4));
        assert!(
            harness.state().opened().is_some(),
            "Enter closed the viewer"
        );
    }

    #[test]
    fn gui_naming_an_unknown_face_in_the_info_panel_creates_a_person() {
        let (fixture, faces) = face_fixture();
        let conn = catalog::open(&fixture.paths.catalog_db()).unwrap();
        catalog::faces::unassign(&conn, faces.second_faces[0]).unwrap();
        let mut harness = face_harness(&fixture, true);
        open_in_viewer(&mut harness, "png.png");
        click_info(&mut harness);
        pick_face(&mut harness, "Unknown", "Bea");
        harness.key_press(egui::Key::Enter);
        harness.run_steps(2);
        let bea = people_in(&fixture)
            .into_iter()
            .find(|p| p.name.as_deref() == Some("Bea"))
            .unwrap();
        assert_eq!(bea.faces, 1);
        assert_eq!(
            face_row(&fixture, faces.second_faces[0]),
            (Some(bea.id.0), Some("manual".to_owned()), false)
        );
        harness.get_by_label("Bea");

        // a named person's row moves only that face
        pick_face(&mut harness, "Bea", "unnamed #1");
        harness.key_press(egui::Key::Enter);
        harness.run_steps(2);
        assert_eq!(
            face_row(&fixture, faces.second_faces[0]),
            (Some(faces.first.0), Some("manual".to_owned()), false)
        );
        assert!(people_in(&fixture).iter().any(|p| p.id == bea.id));
    }

    #[test]
    fn gui_ignoring_unknown_faces_gives_each_its_own_ignored_person() {
        let (fixture, faces) = face_fixture();
        let conn = catalog::open(&fixture.paths.catalog_db()).unwrap();
        let unknown = [faces.first_faces[1], faces.second_faces[0]];
        for face in unknown {
            catalog::faces::unassign(&conn, face).unwrap();
        }
        let mut harness = face_harness(&fixture, true);
        open_in_viewer(&mut harness, "png.png");
        click_info(&mut harness);
        for row in 0..unknown.len() {
            harness.get_all_by_label("...").nth(row).unwrap().click();
            harness.run_steps(2);
            harness.get_by_label("Ignore").click();
            harness.run_steps(2);
        }
        let people = people_in(&fixture);
        let owners: Vec<_> = unknown
            .iter()
            .map(|face| {
                let (person, by, _) = face_row(&fixture, *face);
                assert_eq!(by.as_deref(), Some("manual"));
                people.iter().find(|p| Some(p.id.0) == person).unwrap()
            })
            .collect();
        assert_ne!(owners[0].id, owners[1].id);
        for owner in &owners {
            assert_eq!(
                (owner.name.as_deref(), owner.ignored, owner.faces),
                (None, true, 1)
            );
        }
        harness.get_by_label("People (2)");
        assert_eq!(harness.query_all_by_label_contains(", ignored").count(), 2);

        harness.get_all_by_label("...").next().unwrap().click();
        harness.run_steps(2);
        harness.get_by_label("Unignore person").click();
        harness.run_steps(2);
        assert_eq!(people_in(&fixture).iter().filter(|p| p.ignored).count(), 1);
    }

    #[test]
    fn gui_info_panel_marks_a_face_as_not_a_face() {
        let (fixture, faces) = face_fixture();
        let mut harness = face_harness(&fixture, true);
        open_in_viewer(&mut harness, "png.png");
        click_info(&mut harness);
        harness.get_all_by_label("...").next().unwrap().click();
        harness.run_steps(2);
        harness.get_by_label("Not a face").click();
        harness.run_steps(2);
        assert_eq!(
            face_row(&fixture, faces.first_faces[1]),
            (None, Some("manual".to_owned()), true)
        );
        harness.get_by_label("People (1)");
    }

    #[test]
    fn gui_show_on_map_is_disabled_without_coordinates() {
        let fixture = fixture();
        let mut harness = face_harness(&fixture, false);
        open_in_viewer(&mut harness, "png.png");
        harness.get_by_label("No location");
        click_info(&mut harness);
        let button = harness.get_by_label("Show on map");
        assert!(button.accesskit_node().is_disabled());
        button.click();
        harness.run_steps(2);
        assert!(harness.state().opened().is_some());
        assert_eq!(harness.state().view, MainView::Grid);
    }

    #[test]
    fn gui_show_on_map_closes_the_viewer_and_focuses_an_unopened_map() {
        let fixture = map_fixture();
        let paris = id_of(&fixture, "paris/jpeg.jpg");
        let mut harness = face_harness(&fixture, false);
        open_in_viewer(&mut harness, "jpeg.jpg");
        assert!(harness.state().points_key.is_none());
        click_info(&mut harness);
        // read-only without geo.db, which keeps the button
        harness.get_by_label(place_picker::FETCH_HINT);
        let button = harness.get_by_label("Show on map");
        assert!(!button.accesskit_node().is_disabled());
        button.click();
        harness.run_steps(2);
        let app = harness.state();
        assert_eq!(app.opened(), None);
        assert_eq!(app.view, MainView::Map);
        assert!(app.points_key.is_some());
        assert!(app.map.cluster_pos(paris).is_some());
        assert_focused(&app.map, paris, (48.85, 2.35));
    }

    #[test]
    fn gui_show_on_map_wins_over_a_refit_for_a_new_folder() {
        let fixture = map_fixture();
        let paris = id_of(&fixture, "paris/jpeg.jpg");
        let mut harness = face_harness(&fixture, false);
        click_label(&mut harness, "Map");
        click_label(&mut harness, "Grid");
        click_label(&mut harness, "paris (1)");
        open_in_viewer(&mut harness, "jpeg.jpg");
        click_info(&mut harness);
        click_label(&mut harness, "Show on map");
        let app = harness.state();
        assert_eq!(app.view, MainView::Map);
        assert!(
            app.points_key
                .as_ref()
                .is_some_and(|(folder, _)| folder.ends_with("paris"))
        );
        assert_focused(&app.map, paris, (48.85, 2.35));
    }

    #[test]
    fn gui_map_resolves_samples_without_input() {
        let fixture = map_fixture();
        let mut harness = face_harness(&fixture, false);
        step_until(&mut harness, |app| {
            app.grid.textures() == 2 && app.grid.pending() == 0
        });
        harness.run_steps(2);

        // a single frame resolves the samples; only a repaint hands them to the map
        harness.state_mut().view = MainView::Map;
        harness.run_ok();
        let app = harness.state();
        assert_eq!(app.map.sampled(), 2);
        assert!(app.samples.is_none());
    }

    #[test]
    fn points_filter_drops_the_area_and_unused_face_settings() {
        let area = filter::Area {
            south: 38.0,
            north: 39.0,
            west: -10.0,
            east: -9.0,
        };
        let filter = filter::Filter {
            model: Some(ModelId(1)),
            min_score: 0.5,
            min_face_size: 40,
            count: filter::FaceCount::Any,
            people: Vec::new(),
            unknown: false,
            place: filter::PlaceSel::default(),
            area: Some(area),
        };
        let plain = points_filter(filter.clone());
        assert_eq!(
            plain,
            filter::Filter {
                model: None,
                min_score: 0.0,
                min_face_size: 0,
                area: None,
                ..filter.clone()
            }
        );
        assert_eq!(
            points_filter(filter::Filter {
                model: None,
                ..filter.clone()
            }),
            plain
        );

        let faces = filter::Filter {
            count: filter::FaceCount::One,
            ..filter.clone()
        };
        assert_eq!(
            points_filter(faces.clone()),
            filter::Filter {
                area: None,
                ..faces
            }
        );
    }

    #[test]
    fn location_writes_are_one_transaction() {
        let fixture = fixture();
        let mut app = VitrineApp::new(
            &fixture.paths,
            std::slice::from_ref(&fixture.tree),
            &config::Config::default(),
        );
        let content = distinct_content(app.nav().entries());
        assert!(content.len() > 2);
        app.write(|conn| {
            Ok(conn.execute_batch(
                "CREATE TEMP TRIGGER second_fails BEFORE INSERT ON main.locations
                 WHEN (SELECT count(*) FROM main.locations) > 0
                 BEGIN SELECT RAISE(ABORT, 'second write fails'); END",
            )?)
        })
        .unwrap();
        let lisbon = catalog::geo::tests::place(LISBON, "Lisbon");
        assert!(!app.locate(&Pick::Set(lisbon), &content));
        assert!(
            app.message
                .as_deref()
                .is_some_and(|m| m.contains("second write fails")),
            "{:?}",
            app.message
        );
        assert_eq!(location_rows(&fixture), []);
        let conn = catalog::open_read_only(&fixture.paths.catalog_db()).unwrap();
        let places: u64 = conn
            .query_row("SELECT count(*) FROM places", [], |r| r.get(0))
            .unwrap();
        assert_eq!(places, 0);
    }
}
