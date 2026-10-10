use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::anyhow;
use eframe::egui::{self, CentralPanel, Frame, Panel, ScrollArea};
use eframe::glow;
use rusqlite::Connection;

use super::filter_bar::FilterBar;
use super::grid::{Action, Grid};
use super::indexing::{Indexing, Update};
use super::loader::Loader;
use super::people_panel::{self, Edit, PeoplePanel};
use super::tree::{self, Folder};
use super::viewer::{self, Viewer};
use crate::catalog::faces::ModelId;
use crate::catalog::images::{Entry, ImageId};
use crate::catalog::runs::RunStats;
use crate::catalog::{self, filter};
use crate::config::{self, Paths};
use crate::faces::fetch::MODEL_NAME;
use crate::index::IndexEvent;
use crate::nav::NavSet;

const MAX_WORKERS: usize = 4;
/// Shortest gap between catalog re-reads while an index run streams changes.
const REFRESH_INTERVAL: Duration = Duration::from_millis(500);

#[derive(Debug, Clone, PartialEq, Eq)]
enum Progress {
    Idle,
    Walking,
    Hashing { done: usize, pending: usize },
    Faces { done: usize, pending: usize },
    Elsewhere,
    Done(RunStats),
    Failed(String),
}

pub struct VitrineApp {
    paths: Paths,
    roots: Vec<PathBuf>,
    faces: config::Faces,
    catalog: Option<Connection>,
    /// Opened on the first manual edit; the GUI otherwise only reads.
    writer: Option<Connection>,
    /// `None` until the catalog knows it.
    model: Option<ModelId>,
    filter_bar: FilterBar,
    people_panel: PeoplePanel,
    folders: Vec<Folder>,
    selected: Option<PathBuf>,
    nav: NavSet,
    grid: Grid,
    viewer: Viewer,
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
            catalog: None,
            writer: None,
            model: None,
            filter_bar: FilterBar::default(),
            people_panel: PeoplePanel::new(paths.catalog_db()),
            folders: Vec::new(),
            selected: None,
            nav: NavSet::default(),
            grid: Grid::new(loader),
            viewer: Viewer::new(config.viewer.external.clone()),
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
                self.selected = Some(folder.to_owned());
                self.grid.reset_scroll();
            }
            Err(e) => self.message = Some(format!("{}: {e}", folder.display())),
        }
    }

    /// Face-filtered only when faces are enabled, the model is known and the filter is active.
    fn entries(&self, conn: &Connection, folder: &Path) -> crate::db::Result<Vec<Entry>> {
        match self.model {
            Some(model) if self.faces.enabled && self.filter_bar.is_active() => {
                let filter = self.filter_bar.filter(model, &self.faces);
                filter::entries(conn, folder, &filter)
            }
            _ => catalog::images::entries_under(conn, folder),
        }
    }

    /// The cursor stays on its image if it's still in the set.
    fn apply_filter(&mut self) {
        let (Some(conn), Some(folder)) = (&self.catalog, &self.selected) else {
            return;
        };
        match self.entries(conn, folder) {
            Ok(entries) => {
                self.nav.replace(entries);
                if self.nav.cursor().is_some() {
                    self.grid.reveal();
                } else {
                    self.grid.reset_scroll();
                }
            }
            Err(e) => self.message = Some(format!("{}: {e}", folder.display())),
        }
    }

    /// Opens `writer` on first use.
    fn edit(&mut self, edit: &Edit) {
        let writer = match &mut self.writer {
            Some(writer) => writer,
            None => match catalog::open(&self.paths.catalog_db()) {
                Ok(conn) => self.writer.insert(conn),
                Err(e) => {
                    self.message = Some(format!("cannot open the catalog for writing: {e}"));
                    return;
                }
            },
        };
        if let Err(e) = edit.apply(writer) {
            self.message = Some(format!("edit failed: {e}"));
        }
        self.refresh();
    }

    /// Opens the catalog once it exists.
    fn refresh(&mut self) {
        let started = Instant::now();
        self.stale = false;
        self.refresh_error = self.reread().err();
        self.refreshed = Instant::now();
        self.refresh_cost = self.refreshed - started;
    }

    fn reread(&mut self) -> Result<(), String> {
        if self.catalog.is_none() {
            let catalog_db = self.paths.catalog_db();
            if !catalog_db.exists() {
                return Ok(());
            }
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
            self.nav.replace(entries);
        } else {
            self.nav = NavSet::new(entries);
            self.selected = Some(folder);
            self.grid.reset_scroll();
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
        if self.viewer.is_open() {
            CentralPanel::default().frame(Frame::NONE).show(ui, |ui| {
                if self.viewer.show(ui, &mut self.nav, &mut self.grid, gl)
                    == Some(viewer::Action::Close)
                {
                    self.grid.reveal();
                    ui.ctx().request_repaint();
                }
            });
            return;
        }
        let (rescan, filtered) = Panel::top("filter_bar")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let enabled = self.indexing.is_none() && !self.roots.is_empty();
                    let rescan = ui
                        .add_enabled(enabled, egui::Button::new("Rescan"))
                        .clicked();
                    let filtered = self.faces.enabled
                        && self.model.is_some()
                        && self.filter_bar.show(ui, self.people_panel.people());
                    (rescan, filtered)
                })
                .inner
            })
            .inner;
        if rescan {
            self.rescan(ui.ctx());
        }
        if filtered {
            self.apply_filter();
        }
        Panel::bottom("status").show(ui, |ui| {
            ui.label(self.status());
        });
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
            self.select_folder(&folder);
        }
        if self.faces.enabled {
            self.show_people(ui);
        }
        CentralPanel::default().show(ui, |ui| {
            if let Some(Action::Open(index)) = self.grid.show(ui, &mut self.nav) {
                self.nav.set_cursor(index);
                self.viewer.open();
                ui.ctx().request_repaint();
            }
        });
    }

    fn show_people(&mut self, ui: &mut egui::Ui) {
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
        if let Some(folder) = &self.selected {
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
    use std::collections::BTreeMap;
    use std::fs;
    use std::sync::atomic::AtomicBool;

    use egui_kittest::Harness;
    use egui_kittest::kittest::Queryable;
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
        let tree = std::slice::from_ref(&fixture.tree);
        crate::index::run(
            &fixture.paths,
            tree,
            &config::Faces::default(),
            &AtomicBool::new(false),
            |_| {},
        )
        .unwrap();
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
        let paths = Paths {
            home_dir: root.clone(),
            config_file: root.join("config.toml"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        Fixture {
            _dir: dir,
            paths,
            tree,
        }
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
    fn gui_thumbnails_load_for_images_but_not_videos() {
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
        assert_eq!(harness.state().nav().len(), A.len() + B.len() + 1);

        let mut harness = face_harness(&fixture, true);
        harness.run_steps(2);
        harness.get_by_label("Unknown faces only");
        harness.get_by_label("unnamed #1 (2)");
    }
}
