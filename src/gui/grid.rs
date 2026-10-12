use std::collections::HashSet;
use std::path::Path;

use eframe::egui::{
    self, Align2, Color32, Event, FontId, Key as KeyCode, Modifiers, PointerButton, Rect,
    ScrollArea, Sense, Stroke, StrokeKind, TextureHandle, TextureOptions, Vec2, WidgetInfo,
    WidgetType, pos2, vec2,
};

use super::cache::Lru;
use super::loader::{Key, Loader, Request};
use crate::catalog::images::{Entry, Kind};
use crate::db::Sha256;
use crate::nav::{NavSet, Selection};
use crate::thumbs::generate::BUCKETS;
use crate::video;

/// Tile edge in points.
pub const TILE: f32 = 192.0;
const TEXTURES: usize = 600;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Layout {
    pub columns: usize,
    pub rows: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Open(usize),
}

pub struct Grid {
    loader: Option<Loader>,
    textures: Lru<Key, TextureHandle>,
    failed: HashSet<Key>,
    layout: Layout,
    scroll_to: Option<f32>,
    reveal: bool,
}

impl Grid {
    #[must_use]
    pub fn new(loader: Option<Loader>) -> Self {
        Self {
            loader,
            textures: Lru::new(TEXTURES),
            failed: HashSet::new(),
            layout: Layout::default(),
            scroll_to: None,
            reveal: false,
        }
    }

    #[cfg(test)]
    pub fn layout(&self) -> Layout {
        self.layout
    }

    #[cfg(test)]
    pub fn textures(&self) -> usize {
        self.textures.len()
    }

    #[must_use]
    pub fn pending(&self) -> usize {
        self.loader.as_ref().map_or(0, Loader::pending)
    }

    /// Scrolls to the top on the next frame.
    pub fn reset_scroll(&mut self) {
        self.scroll_to = Some(0.0);
    }

    /// Scrolls the cursor's row into view on the next frame.
    pub fn reveal(&mut self) {
        self.reveal = true;
    }

    /// Largest cached thumbnail of `entry`; requests the largest bucket when it's missing.
    pub fn thumbnail(&mut self, ctx: &egui::Context, entry: &Entry) -> Option<TextureHandle> {
        self.receive(ctx);
        let sha256 = entry.sha256?;
        let mut wanted = Vec::new();
        let largest = BUCKETS[BUCKETS.len() - 1];
        let texture = self.lookup(sha256, &entry.path, entry.kind, largest, &mut wanted);
        if !wanted.is_empty()
            && let Some(loader) = &self.loader
        {
            loader.request(ctx, wanted);
        }
        texture
    }

    /// Cached texture per image at `bucket`, else the largest cached bucket. Replaces the
    /// loader queue with the missing keys, so an empty `items` cancels pending requests.
    pub fn thumbnails(
        &mut self,
        ctx: &egui::Context,
        items: &[(Sha256, &Path)],
        bucket: u32,
    ) -> Vec<Option<TextureHandle>> {
        self.receive(ctx);
        let mut wanted = Vec::new();
        let textures = items
            .iter()
            .map(|&(sha256, path)| self.lookup(sha256, path, Kind::Image, bucket, &mut wanted))
            .collect();
        if let Some(loader) = &self.loader {
            loader.request(ctx, wanted);
        }
        textures
    }

    /// Texture at `bucket`, else the largest cached one; pushes the key to `wanted` unless it's cached or failed.
    fn lookup(
        &mut self,
        sha256: Sha256,
        path: &Path,
        kind: Kind,
        bucket: u32,
        wanted: &mut Vec<Request>,
    ) -> Option<TextureHandle> {
        let key = (sha256, bucket);
        if let Some(texture) = self.textures.get(&key).cloned() {
            return Some(texture);
        }
        if !self.failed.contains(&key) {
            wanted.push(Request {
                key,
                path: path.to_owned(),
                kind,
            });
        }
        BUCKETS
            .iter()
            .rev()
            .find_map(|&b| self.textures.get(&(sha256, b)).cloned())
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        nav: &mut NavSet,
        selection: &mut Selection,
    ) -> Option<Action> {
        self.receive(ui.ctx());
        let spacing = ui.spacing().item_spacing;
        let width = ui.available_width() - ui.spacing().scroll.allocated_width();
        let columns = (((width + spacing.x) / (TILE + spacing.x)).floor() as usize).max(1);
        let rows = nav.len().div_ceil(columns);
        self.layout = Layout { columns, rows };
        let bucket = bucket_for(TILE * ui.ctx().pixels_per_point());

        let mut action = self.keyboard(ui, nav, selection, columns);

        let mut area = ScrollArea::vertical().auto_shrink(false).id_salt("grid");
        if let Some(offset) = self.scroll_to.take() {
            area = area.vertical_scroll_offset(offset);
        }
        let mut wanted = Vec::new();
        let output = area.show_rows(ui, TILE, rows, |ui, range| {
            for row in range {
                ui.horizontal(|ui| {
                    for index in row * columns..((row + 1) * columns).min(nav.len()) {
                        let entry = &nav.entries()[index];
                        let id = entry.id;
                        let cursor = nav.cursor() == Some(index);
                        let picked = selection.contains(id);
                        let response = self.tile(ui, entry, bucket, cursor, picked, &mut wanted);
                        let modifiers = if response.clicked() {
                            click_modifiers(ui)
                        } else {
                            Modifiers::NONE
                        };
                        if response.double_clicked() && !modifiers.shift && !modifiers.command {
                            nav.set_cursor(index);
                            action = Some(Action::Open(index));
                        } else if response.clicked() {
                            if modifiers.shift {
                                selection.select_range(nav, id);
                            } else if modifiers.command {
                                selection.toggle(id);
                            } else {
                                selection.clear();
                            }
                            nav.set_cursor(index);
                        }
                    }
                });
            }
        });

        if std::mem::take(&mut self.reveal)
            && let Some(cursor) = nav.cursor()
        {
            let pitch = TILE + spacing.y;
            let top = (cursor / columns) as f32 * pitch;
            let offset = output.state.offset.y;
            let height = output.inner_rect.height();
            if top < offset {
                self.scroll_to = Some(top);
            } else if top + TILE > offset + height {
                self.scroll_to = Some(top + TILE - height);
            }
            if self.scroll_to.is_some() {
                ui.ctx().request_repaint();
            }
        }

        if let Some(loader) = &self.loader {
            loader.request(ui.ctx(), wanted);
        }
        action
    }

    fn receive(&mut self, ctx: &egui::Context) {
        let Some(loader) = &self.loader else {
            return;
        };
        for loaded in loader.drain() {
            match loaded.image {
                Ok(image) => {
                    let name = format!("thumb-{}", loaded.key.1);
                    let texture = ctx.load_texture(name, image, TextureOptions::LINEAR);
                    self.textures.insert(loaded.key, texture);
                }
                Err(_) => {
                    self.failed.insert(loaded.key);
                }
            }
        }
    }

    fn keyboard(
        &mut self,
        ui: &egui::Ui,
        nav: &mut NavSet,
        selection: &mut Selection,
        columns: usize,
    ) -> Option<Action> {
        if ui.memory(|m| m.focused().is_some())
            || egui::Popup::is_any_open(ui.ctx())
            || nav.is_empty()
        {
            return None;
        }
        if ui.input_mut(|i| i.consume_key(Modifiers::COMMAND, KeyCode::A)) {
            selection.select_all(nav);
        }
        let columns = isize::try_from(columns).unwrap_or(1);
        let (delta, open) = ui.input(|i| {
            let delta = [
                (KeyCode::ArrowLeft, -1),
                (KeyCode::ArrowRight, 1),
                (KeyCode::ArrowUp, -columns),
                (KeyCode::ArrowDown, columns),
            ]
            .into_iter()
            .filter(|(key, _)| i.key_pressed(*key))
            .map(|(_, d)| d)
            .sum::<isize>();
            (delta, i.key_pressed(KeyCode::Enter))
        });
        if delta != 0 {
            nav.step(delta);
            self.reveal = true;
        }
        if open {
            return nav.cursor().map(Action::Open);
        }
        None
    }

    fn tile(
        &mut self,
        ui: &mut egui::Ui,
        entry: &Entry,
        bucket: u32,
        cursor: bool,
        picked: bool,
        wanted: &mut Vec<Request>,
    ) -> egui::Response {
        let (rect, response) = ui.allocate_exact_size(Vec2::splat(TILE), Sense::click());
        let name = file_name(entry);
        response
            .widget_info(|| WidgetInfo::selected(WidgetType::SelectableLabel, true, cursor, &name));
        let visuals = ui.visuals();
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, visuals.extreme_bg_color);

        let key = entry.sha256.map(|sha256| (sha256, bucket));
        let texture = key.and_then(|key| self.textures.get(&key));
        if let Some(texture) = texture {
            let size = texture.size_vec2();
            let scale = (TILE / size.x).min(TILE / size.y);
            let fitted = Rect::from_center_size(rect.center(), size * scale);
            painter.image(
                texture.id(),
                fitted,
                Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                Color32::WHITE,
            );
        } else {
            if let Some(key) = key
                && !self.failed.contains(&key)
            {
                wanted.push(Request {
                    key,
                    path: entry.path.clone(),
                    kind: entry.kind,
                });
            }
            let text = visuals.weak_text_color();
            if entry.kind == Kind::Video {
                painter.text(
                    rect.center_top() + vec2(0.0, 12.0),
                    Align2::CENTER_TOP,
                    "video",
                    FontId::proportional(12.0),
                    text,
                );
            }
            let galley = painter.layout(name, FontId::proportional(12.0), text, TILE - 16.0);
            let pos = rect.center() - galley.size() / 2.0;
            painter.galley(pos, galley, text);
        }
        if let Some(ms) = entry.duration_ms {
            duration_badge(&painter, rect, ms);
        }

        if picked {
            painter.rect_filled(rect, 4.0, visuals.selection.bg_fill.gamma_multiply(0.45));
        }
        if cursor {
            painter.rect_stroke(
                rect.shrink(1.0),
                4.0,
                Stroke::new(2.0, visuals.selection.stroke.color),
                StrokeKind::Inside,
            );
        } else if response.hovered() {
            painter.rect_stroke(
                rect.shrink(1.0),
                4.0,
                visuals.widgets.hovered.bg_stroke,
                StrokeKind::Inside,
            );
        }
        response
    }
}

fn duration_badge(painter: &egui::Painter, tile: Rect, ms: u64) {
    let galley =
        painter.layout_no_wrap(video::clock(ms), FontId::proportional(11.0), Color32::WHITE);
    let pad = vec2(4.0, 2.0);
    let size = galley.size() + pad * 2.0;
    let badge = Rect::from_min_size(tile.right_bottom() - size - vec2(6.0, 6.0), size);
    painter.rect_filled(badge, 3.0, Color32::from_black_alpha(170));
    painter.galley(badge.min + pad, galley, Color32::WHITE);
}

/// Modifiers of the latest primary release; `i.modifiers` may be reset later in the same frame.
fn click_modifiers(ui: &egui::Ui) -> Modifiers {
    ui.input(|i| {
        i.events
            .iter()
            .rev()
            .find_map(|e| match e {
                Event::PointerButton {
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers,
                    ..
                } => Some(*modifiers),
                _ => None,
            })
            .unwrap_or(i.modifiers)
    })
}

pub(super) fn bucket_for(pixels: f32) -> u32 {
    BUCKETS
        .iter()
        .copied()
        .find(|&b| b as f32 >= pixels)
        .unwrap_or(BUCKETS[BUCKETS.len() - 1])
}

fn file_name(entry: &Entry) -> String {
    entry.path.file_name().map_or_else(
        || entry.path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::thread;
    use std::time::{Duration, Instant};

    use sha2::Digest;

    use super::*;

    const IMAGES: &[&str] = &["jpeg.jpg", "png.png", "webp.webp", "gif.gif"];

    fn fixtures() -> Vec<(Sha256, PathBuf)> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        IMAGES
            .iter()
            .map(|name| {
                let path = dir.join(name);
                let sha256 = sha2::Sha256::digest(std::fs::read(&path).unwrap()).into();
                (sha256, path)
            })
            .collect()
    }

    fn grid(dir: &Path, workers: usize) -> Grid {
        Grid::new(Some(
            Loader::spawn(&dir.join("thumbs.db"), workers).unwrap(),
        ))
    }

    #[test]
    fn thumbnails_loads_every_item_in_one_call() {
        let dir = tempfile::tempdir().unwrap();
        let mut grid = grid(dir.path(), 2);
        let ctx = egui::Context::default();
        let fixtures = fixtures();
        let items: Vec<(Sha256, &Path)> = fixtures.iter().map(|(s, p)| (*s, p.as_path())).collect();
        let distinct: HashSet<Sha256> = items.iter().map(|(s, _)| *s).collect();
        assert_eq!(distinct.len(), items.len());

        let deadline = Instant::now() + Duration::from_secs(30);
        let textures = loop {
            let textures = grid.thumbnails(&ctx, &items, BUCKETS[0]);
            if textures.iter().all(Option::is_some) {
                break textures;
            }
            assert!(Instant::now() < deadline, "thumbnails did not load");
            thread::sleep(Duration::from_millis(10));
        };
        assert_eq!(textures.len(), items.len());
        assert_eq!(grid.textures(), items.len());
        assert_eq!(grid.pending(), 0);
    }

    #[test]
    fn empty_thumbnails_call_clears_the_queue() {
        let dir = tempfile::tempdir().unwrap();
        let mut grid = grid(dir.path(), 1);
        let ctx = egui::Context::default();
        let fixtures = fixtures();
        let items: Vec<(Sha256, &Path)> = fixtures.iter().map(|(s, p)| (*s, p.as_path())).collect();

        grid.thumbnails(&ctx, &items, BUCKETS[0]);
        let loader = grid.loader.as_ref().unwrap();
        assert!(loader.queued() > 0);
        grid.thumbnails(&ctx, &[], BUCKETS[0]);
        assert_eq!(grid.loader.as_ref().unwrap().queued(), 0);
    }

    #[test]
    fn thumbnail_loads_for_a_hashed_video() {
        if !crate::video::tools_present() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let mut grid = grid(dir.path(), 1);
        let ctx = egui::Context::default();
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/clip.mp4");
        let entry = Entry {
            id: crate::catalog::images::ImageId(1),
            sha256: Some(sha2::Sha256::digest(std::fs::read(&path).unwrap()).into()),
            path,
            kind: Kind::Video,
            width: Some(72),
            height: Some(56),
            duration_ms: Some(1000),
        };

        let deadline = Instant::now() + Duration::from_secs(30);
        let texture = loop {
            if let Some(texture) = grid.thumbnail(&ctx, &entry) {
                break texture;
            }
            assert!(Instant::now() < deadline, "video thumbnail did not load");
            thread::sleep(Duration::from_millis(10));
        };
        assert_eq!(texture.size(), [72, 56]);
    }

    #[test]
    fn video_tile_paints_its_duration() {
        let dir = tempfile::tempdir().unwrap();
        let mut grid = Grid::new(None);
        let entry = Entry {
            id: crate::catalog::images::ImageId(1),
            path: dir.path().join("clip.mp4"),
            sha256: None,
            kind: Kind::Video,
            width: None,
            height: None,
            duration_ms: Some(83_000),
        };
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            grid.tile(ui, &entry, BUCKETS[0], false, false, &mut Vec::new());
        });
        output.textures_delta.clear();
        let texts: Vec<String> = output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Text(text) => Some(text.galley.text().to_owned()),
                _ => None,
            })
            .collect();
        assert!(texts.iter().any(|t| t == "1:23"), "{texts:?}");
    }

    #[test]
    fn bucket_covers_tile_pixels() {
        assert_eq!(bucket_for(192.0), 256);
        assert_eq!(bucket_for(256.0), 256);
        assert_eq!(bucket_for(288.0), 512);
        assert_eq!(bucket_for(768.0), 512);
    }
}
