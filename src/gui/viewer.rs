use std::fmt::Write as _;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui::{
    self, Align, Align2, Color32, FontId, Key, Layout, Pos2, Rect, Sense, TextureHandle, UiBuilder,
    Vec2, pos2, vec2,
};
use eframe::{egui_glow, glow};
use image::DynamicImage;

use super::grid::Grid;
use crate::catalog::images::{Entry, ImageId, Kind};
use crate::nav::NavSet;
use crate::prefetch::{Decode, Limits, Prefetcher};
use crate::render::{self, GlowRenderer, GlowTile, TexFormat, TileRegion};

const WORKERS: usize = 2;
/// Images passed to the external viewer on each side of the current one.
const EXTERNAL_WINDOW: usize = 500;
/// Time spent uploading tiles per frame; at least one tile is uploaded.
const UPLOAD_BUDGET: Duration = Duration::from_millis(8);
const MIN_SCALE: f32 = 0.01;
const MAX_SCALE: f32 = 64.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Close,
}

pub struct Viewer {
    open: bool,
    external: Vec<String>,
    prefetcher: Option<Prefetcher>,
    /// `None` until the first frame with a GL context.
    renderer: Option<Result<GlowRenderer, String>>,
    upload: Option<Upload>,
    shown: Option<Shown>,
    view: View,
    wanted: Option<(ImageId, usize)>,
    message: Option<String>,
    /// Failures of actions that run on their own thread, shown in the overlay.
    notices: (Sender<String>, Receiver<String>),
}

/// Tiles are uploaded a few per frame.
struct Upload {
    id: ImageId,
    image: Arc<DynamicImage>,
    format: TexFormat,
    regions: Vec<TileRegion>,
    tiles: Vec<GlowTile>,
}

struct Shown {
    id: ImageId,
    size: Vec2,
    format: TexFormat,
    tiles: Vec<GlowTile>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum View {
    Fit,
    /// `scale` in points per image pixel; `pan` is the image centre's offset from the view centre, in points.
    Free {
        scale: f32,
        pan: Vec2,
    },
}

impl Viewer {
    #[must_use]
    pub fn new(external: Vec<String>) -> Self {
        Self {
            open: false,
            external,
            prefetcher: None,
            renderer: None,
            upload: None,
            shown: None,
            view: View::Fit,
            wanted: None,
            message: None,
            notices: mpsc::channel(),
        }
    }

    #[must_use]
    pub fn is_open(&self) -> bool {
        self.open
    }

    pub fn open(&mut self) {
        self.open = true;
        self.view = View::Fit;
        self.wanted = None;
        self.message = None;
    }

    /// Frees the GL resources; call with the context the viewer drew with.
    pub fn destroy(&mut self, gl: &glow::Context) {
        self.release(gl);
        if let Some(Ok(renderer)) = self.renderer.take() {
            renderer.destroy(gl);
        }
    }

    /// Frees the image on [`Action::Close`]; without a GL context full images aren't drawn.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        nav: &mut NavSet,
        grid: &mut Grid,
        gl: Option<&glow::Context>,
    ) -> Option<Action> {
        if let Some(gl) = gl
            && self.renderer.is_none()
        {
            self.renderer = Some(GlowRenderer::new(gl).map_err(|e| format!("{e:#}")));
        }
        if self.prefetcher.is_none() {
            let ctx = ui.ctx().clone();
            let decode: Decode =
                Arc::new(|path| crate::decode::decode(path).map(render::into_texture_layout));
            match Prefetcher::new(decode, WORKERS, Limits::default(), move || {
                ctx.request_repaint()
            }) {
                Ok(prefetcher) => self.prefetcher = Some(prefetcher),
                Err(e) => self.message = Some(format!("cannot start decoding: {e}")),
            }
        }

        if let Some(notice) = self.notices.1.try_iter().last() {
            self.message = Some(notice);
        }
        if self.keyboard(ui, nav) {
            self.close(gl);
            return Some(Action::Close);
        }
        let Some((index, entry)) = nav.cursor().zip(nav.current().cloned()) else {
            self.close(gl);
            return Some(Action::Close);
        };

        if self.wanted != Some((entry.id, nav.len())) {
            if self.wanted.is_some_and(|(id, _)| id != entry.id) {
                self.view = View::Fit;
            }
            self.wanted = Some((entry.id, nav.len()));
            if let Some(prefetcher) = &mut self.prefetcher {
                prefetcher.update(nav);
            }
        }
        if let Some(prefetcher) = &mut self.prefetcher {
            prefetcher.poll();
        }
        if let Some(gl) = gl {
            self.upload(gl, entry.id);
            if self.upload.as_ref().is_some_and(|u| u.id == entry.id) {
                ui.ctx().request_repaint();
            }
        }

        let rect = ui.max_rect();
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        ui.painter().rect_filled(rect, 0.0, Color32::BLACK);

        let error = self
            .prefetcher
            .as_ref()
            .and_then(|p| p.get(entry.id))
            .and_then(|d| d.as_ref().err().cloned());
        let size = self
            .shown
            .as_ref()
            .filter(|s| s.id == entry.id && !s.tiles.is_empty())
            .map(|s| s.size);
        if let Some(size) = size {
            self.navigate(ui, &response, rect, size);
            let placement = self.placement(rect, size, ui.ctx().pixels_per_point());
            self.paint(ui, rect, placement);
        } else if entry.kind == Kind::Video {
            centred(ui, rect, "video");
        } else if let Some(error) = &error {
            centred(ui, rect, error);
        } else if let Some(texture) = grid.thumbnail(ui.ctx(), &entry) {
            paint_thumbnail(ui, rect, &texture, &entry);
        }

        self.overlay(ui, rect, nav, index, &entry, size);
        None
    }

    /// Returns whether to close.
    fn keyboard(&mut self, ui: &egui::Ui, nav: &mut NavSet) -> bool {
        let (delta, close, fit, actual, external) = ui.input(|i| {
            let delta = [
                (Key::ArrowLeft, -1),
                (Key::PageUp, -1),
                (Key::Backspace, -1),
                (Key::ArrowRight, 1),
                (Key::PageDown, 1),
                (Key::Space, 1),
            ]
            .into_iter()
            .filter(|(key, _)| i.key_pressed(*key))
            .map(|(_, d)| d)
            .sum::<isize>();
            (
                delta,
                i.key_pressed(Key::Escape),
                i.key_pressed(Key::Num0),
                i.key_pressed(Key::Num1),
                i.key_pressed(Key::O),
            )
        });
        if delta != 0 {
            nav.step(delta);
        }
        if fit {
            self.view = View::Fit;
        }
        if actual {
            self.view = View::Free {
                scale: 1.0 / ui.ctx().pixels_per_point(),
                pan: Vec2::ZERO,
            };
        }
        if external {
            self.open_external(nav);
        }
        close
    }

    fn navigate(&mut self, ui: &egui::Ui, response: &egui::Response, rect: Rect, size: Vec2) {
        let (scroll, zoom, pointer) = ui.input(|i| {
            (
                i.smooth_scroll_delta.y,
                i.zoom_delta(),
                i.pointer.hover_pos(),
            )
        });
        let factor = zoom * (scroll / 200.0).exp();
        let drag = response.drag_delta();
        if factor == 1.0 && drag == Vec2::ZERO {
            return;
        }
        let (scale, pan) = self.scale_and_pan(rect, size, ui.ctx().pixels_per_point());
        let pointer = pointer
            .filter(|p| rect.contains(*p))
            .map_or(Vec2::ZERO, |p| p - rect.center());
        let (scale, pan) = zoom_at(scale, pan, factor, pointer);
        self.view = View::Free {
            scale,
            pan: pan + drag,
        };
    }

    fn scale_and_pan(&self, rect: Rect, size: Vec2, ppp: f32) -> (f32, Vec2) {
        match self.view {
            View::Fit => (fit_scale(rect, size, ppp), Vec2::ZERO),
            View::Free { scale, pan } => (scale, pan),
        }
    }

    /// Points per image pixel, and where the image's top left corner goes.
    fn placement(&self, rect: Rect, size: Vec2, ppp: f32) -> (f32, Pos2) {
        let (scale, pan) = self.scale_and_pan(rect, size, ppp);
        (scale, rect.center() + pan - size * scale / 2.0)
    }

    fn paint(&self, ui: &egui::Ui, rect: Rect, (scale, origin): (f32, Pos2)) {
        let (Some(Ok(renderer)), Some(shown)) = (&self.renderer, &self.shown) else {
            return;
        };
        let quads: Vec<_> = shown
            .tiles
            .iter()
            .map(|t| t.quad([origin.x, origin.y], scale))
            .collect();
        let shader = renderer.image_shader();
        let format = shown.format;
        let callback = egui_glow::CallbackFn::new(move |info, painter| {
            render::paint_quads(painter.gl(), shader, &info, format, &quads);
        });
        ui.painter().add(egui::PaintCallback {
            rect,
            callback: Arc::new(callback),
        });
    }

    /// Uploads tiles until the frame budget is spent.
    fn upload(&mut self, gl: &glow::Context, id: ImageId) {
        let Some(Ok(renderer)) = &self.renderer else {
            return;
        };
        if self.shown.as_ref().is_some_and(|s| s.id == id) {
            return;
        }
        if self.upload.as_ref().is_none_or(|u| u.id != id) {
            let Some(Ok(image)) = self.prefetcher.as_ref().and_then(|p| p.get(id)) else {
                return;
            };
            let image = Arc::clone(image);
            let regions = renderer.tile_regions(image.width(), image.height());
            let format = render::texture_layout(&image).0;
            self.release(gl);
            self.upload = Some(Upload {
                id,
                image,
                format,
                tiles: Vec::with_capacity(regions.len()),
                regions,
            });
        }
        let Some(upload) = &mut self.upload else {
            return;
        };
        let started = Instant::now();
        let (_, layout) = render::texture_layout(&upload.image);
        let bytes = layout.as_bytes();
        let mut failed = None;
        while let Some(&region) = upload.regions.get(upload.tiles.len()) {
            match GlowRenderer::create_tile(gl, bytes, upload.image.width(), region, upload.format)
            {
                Ok(tile) => upload.tiles.push(tile),
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
            if started.elapsed() > UPLOAD_BUDGET {
                break;
            }
        }
        drop(layout);
        if let Some(e) = failed
            && let Some(mut upload) = self.upload.take()
        {
            self.message = Some(format!("cannot upload the image: {e:#}"));
            GlowRenderer::delete_tiles(gl, std::mem::take(&mut upload.tiles));
            upload.regions.clear();
            self.upload = Some(upload);
        }
        if let Some(upload) = &self.upload
            && upload.tiles.len() == upload.regions.len()
            && let Some(upload) = self.upload.take()
        {
            self.shown = Some(Shown {
                id: upload.id,
                size: vec2(upload.image.width() as f32, upload.image.height() as f32),
                format: upload.format,
                tiles: upload.tiles,
            });
        }
    }

    fn release(&mut self, gl: &glow::Context) {
        if let Some(shown) = self.shown.take() {
            GlowRenderer::delete_tiles(gl, shown.tiles);
        }
        if let Some(upload) = self.upload.take() {
            GlowRenderer::delete_tiles(gl, upload.tiles);
        }
    }

    fn close(&mut self, gl: Option<&glow::Context>) {
        self.open = false;
        self.wanted = None;
        if let Some(gl) = gl {
            self.release(gl);
        }
        if let Some(prefetcher) = &mut self.prefetcher {
            prefetcher.clear();
        }
    }

    fn overlay(
        &mut self,
        ui: &mut egui::Ui,
        rect: Rect,
        nav: &NavSet,
        index: usize,
        entry: &Entry,
        size: Option<Vec2>,
    ) {
        let bar = Rect::from_min_size(rect.min, vec2(rect.width(), 28.0));
        ui.painter()
            .rect_filled(bar, 0.0, Color32::from_black_alpha(160));
        let mut parts = vec![file_name(entry), format!("{} / {}", index + 1, nav.len())];
        let dimensions = size
            .map(|s| (s.x as u32, s.y as u32))
            .or(entry.width.zip(entry.height));
        if let Some((w, h)) = dimensions {
            parts.push(format!("{w} x {h}"));
        }
        if let Some(size) = size {
            let ppp = ui.ctx().pixels_per_point();
            let (scale, _) = self.placement(rect, size, ppp);
            parts.push(format!("{:.0}%", scale * ppp * 100.0));
        }
        if let Some(message) = &self.message {
            parts.push(message.clone());
        }
        let line = parts.join("   ");
        let (mut external, mut show) = (false, false);
        ui.scope_builder(
            UiBuilder::new()
                .max_rect(bar.shrink2(vec2(8.0, 2.0)))
                .layout(Layout::left_to_right(Align::Center)),
            |ui| {
                ui.visuals_mut().override_text_color = Some(Color32::from_gray(220));
                ui.label(line);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    external = ui.button("Open in external viewer").clicked();
                    show = ui.button("Show in file manager").clicked();
                });
            },
        );
        if external {
            self.open_external(nav);
        }
        if show {
            self.show_in_file_manager(ui.ctx(), entry);
        }
    }

    fn open_external(&mut self, nav: &NavSet) {
        self.message = spawn_external(&self.external, external_paths(nav))
            .err()
            .map(|e| format!("cannot run {}: {e}", self.external.join(" ")));
    }

    /// On a thread, since a file manager may take a moment to answer.
    fn show_in_file_manager(&mut self, ctx: &egui::Context, entry: &Entry) {
        let path = entry.path.clone();
        let notices = self.notices.0.clone();
        let ctx = ctx.clone();
        let spawned = thread::Builder::new()
            .name("reveal".to_owned())
            .spawn(move || {
                if let Err(e) = reveal(&path) {
                    _ = notices.send(format!("cannot show in file manager: {e}"));
                    ctx.request_repaint();
                }
            });
        self.message = spawned
            .err()
            .map(|e| format!("cannot show in file manager: {e}"));
    }
}

/// Rotated to the cursor (the external viewer opens the first), at most [`EXTERNAL_WINDOW`] per side for the kernel's argv limit.
fn external_paths(nav: &NavSet) -> Vec<PathBuf> {
    let entries = nav.entries();
    let start = nav.cursor().unwrap_or(0).min(entries.len());
    let mut around: Vec<&Entry> = entries[start..]
        .iter()
        .chain(&entries[..start])
        .filter(|e| e.kind == Kind::Image)
        .collect();
    if around.len() > 2 * EXTERNAL_WINDOW + 1 {
        around.drain(EXTERNAL_WINDOW + 1..around.len() - EXTERNAL_WINDOW);
    }
    around.into_iter().map(|e| e.path.clone()).collect()
}

/// Doesn't wait for the child.
fn spawn_external(argv: &[String], paths: Vec<PathBuf>) -> io::Result<()> {
    let Some((program, args)) = argv.split_first() else {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "no command"));
    };
    let child = Command::new(program)
        .args(args)
        .args(paths)
        .stdin(Stdio::null())
        .spawn()?;
    detach(child)
}

/// Reaps `child` on a thread of its own.
fn detach(mut child: Child) -> io::Result<()> {
    thread::Builder::new()
        .name("external-viewer".to_owned())
        .spawn(move || child.wait())?;
    Ok(())
}

/// Selects `path` in the file manager via the freedesktop `FileManager1` D-Bus interface; falls back to `xdg-open` on its folder.
fn reveal(path: &Path) -> io::Result<()> {
    let shown = Command::new("dbus-send")
        .args([
            "--session",
            "--print-reply",
            "--reply-timeout=5000",
            "--dest=org.freedesktop.FileManager1",
            "/org/freedesktop/FileManager1",
            "org.freedesktop.FileManager1.ShowItems",
        ])
        .arg(format!("array:string:{}", file_uri(path)))
        .arg("string:")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if shown {
        return Ok(());
    }
    let folder = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(path);
    let child = Command::new("xdg-open")
        .arg(folder)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    detach(child)
}

/// Percent-encodes all but unreserved characters and `/`; dbus-send splits array values on commas.
fn file_uri(path: &Path) -> String {
    let mut uri = String::from("file://");
    for &byte in path.as_os_str().as_encoded_bytes() {
        if byte.is_ascii_alphanumeric() || b"/-._~".contains(&byte) {
            uri.push(char::from(byte));
        } else {
            let _ = write!(uri, "%{byte:02X}");
        }
    }
    uri
}

/// Keeps the image point under `pointer` fixed; `pan` and `pointer` are relative to the view centre, in points.
fn zoom_at(scale: f32, pan: Vec2, factor: f32, pointer: Vec2) -> (f32, Vec2) {
    let zoomed = (scale * factor).clamp(MIN_SCALE, MAX_SCALE);
    let from_centre = pointer - pan;
    (zoomed, pan + from_centre - from_centre * (zoomed / scale))
}

/// Points per image pixel, never above 100%.
fn fit_scale(rect: Rect, size: Vec2, ppp: f32) -> f32 {
    if size.x <= 0.0 || size.y <= 0.0 {
        return 1.0;
    }
    (rect.width() / size.x)
        .min(rect.height() / size.y)
        .min(1.0 / ppp)
}

/// The thumbnail fitted like the full image would be, so the switch to it doesn't jump.
fn paint_thumbnail(ui: &egui::Ui, rect: Rect, texture: &TextureHandle, entry: &Entry) {
    let thumb = texture.size_vec2();
    let ppp = ui.ctx().pixels_per_point();
    // catalog dimensions may predate orientation; take the aspect from the thumbnail
    let longest = entry
        .width
        .max(entry.height)
        .map_or(f32::INFINITY, |l| l as f32);
    let full = thumb * (longest / thumb.x.max(thumb.y));
    let size = full * fit_scale(rect, full, ppp);
    ui.painter().image(
        texture.id(),
        Rect::from_center_size(rect.center(), size),
        Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
        Color32::WHITE,
    );
}

fn centred(ui: &egui::Ui, rect: Rect, text: &str) {
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        text,
        FontId::proportional(16.0),
        Color32::from_gray(180),
    );
}

fn file_name(entry: &Entry) -> String {
    entry.path.file_name().map_or_else(
        || entry.path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nav(kinds: &[Kind]) -> NavSet {
        NavSet::new(
            (0i64..)
                .zip(kinds)
                .map(|(id, &kind)| Entry {
                    id: ImageId(id),
                    path: PathBuf::from(format!("/p/{id}")),
                    sha256: None,
                    kind,
                    width: None,
                    height: None,
                })
                .collect(),
        )
    }

    #[test]
    fn external_paths_start_at_the_cursor_and_skip_videos() {
        let mut nav = nav(&[Kind::Image, Kind::Video, Kind::Image, Kind::Image]);
        assert_eq!(
            external_paths(&nav),
            ["/p/0", "/p/2", "/p/3"].map(PathBuf::from)
        );
        nav.set_cursor(2);
        assert_eq!(
            external_paths(&nav),
            ["/p/2", "/p/3", "/p/0"].map(PathBuf::from)
        );
    }

    #[test]
    fn external_paths_are_cut_around_the_cursor() {
        let n = 2 * EXTERNAL_WINDOW + 5;
        let mut nav = nav(&vec![Kind::Image; n]);
        nav.set_cursor(2);
        let paths = external_paths(&nav);
        assert_eq!(paths.len(), 2 * EXTERNAL_WINDOW + 1);
        assert_eq!(paths[0], PathBuf::from("/p/2"));
        assert_eq!(
            paths[EXTERNAL_WINDOW],
            PathBuf::from(format!("/p/{}", 2 + EXTERNAL_WINDOW))
        );
        assert_eq!(
            paths[EXTERNAL_WINDOW + 1],
            PathBuf::from(format!("/p/{}", 2 + EXTERNAL_WINDOW + 5))
        );
        assert_eq!(paths[paths.len() - 1], PathBuf::from("/p/1"));
    }

    #[test]
    fn file_uri_percent_encodes_everything_but_the_path_characters() {
        assert_eq!(
            file_uri(Path::new("/a b/c,d/é~x.jpg")),
            "file:///a%20b/c%2Cd/%C3%A9~x.jpg"
        );
    }

    #[test]
    fn fit_shrinks_but_never_enlarges() {
        let rect = Rect::from_min_size(Pos2::ZERO, vec2(1000.0, 500.0));
        assert!((fit_scale(rect, vec2(4000.0, 1000.0), 1.0) - 0.25).abs() < 1e-6);
        assert!((fit_scale(rect, vec2(1000.0, 1000.0), 1.0) - 0.5).abs() < 1e-6);
        assert!((fit_scale(rect, vec2(100.0, 50.0), 1.0) - 1.0).abs() < 1e-6);
        assert!((fit_scale(rect, vec2(100.0, 50.0), 2.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn zoom_keeps_the_point_under_the_pointer() {
        let image_point = |scale: f32, pan: Vec2, pointer: Vec2| (pointer - pan) / scale;
        let (scale, pan, pointer) = (0.25, vec2(30.0, -10.0), vec2(-200.0, 150.0));
        for factor in [0.5, 1.5, 4.0] {
            let (zoomed, moved) = zoom_at(scale, pan, factor, pointer);
            assert!((zoomed - scale * factor).abs() < 1e-6);
            let before = image_point(scale, pan, pointer);
            let after = image_point(zoomed, moved, pointer);
            assert!((before - after).length() < 1e-3, "{before:?} {after:?}");
        }
        assert_eq!(zoom_at(MAX_SCALE, Vec2::ZERO, 2.0, pointer).0, MAX_SCALE);
    }

    #[test]
    fn external_command_runs_with_paths_appended() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("args");
        let argv = [
            "sh".to_owned(),
            "-c".to_owned(),
            format!("printf '%s\\n' \"$@\" > {}", out.display()),
            "sh".to_owned(),
        ];
        spawn_external(&argv, vec![PathBuf::from("/a b"), PathBuf::from("/c")]).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Ok(text) = std::fs::read_to_string(&out)
                && text.lines().count() == 2
            {
                assert_eq!(text, "/a b\n/c\n");
                break;
            }
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(10));
        }
        assert!(spawn_external(&["/nonexistent/viewer".to_owned()], Vec::new()).is_err());
    }
}
