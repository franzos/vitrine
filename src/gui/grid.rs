use std::collections::HashSet;

use eframe::egui::{
    self, Align2, Color32, FontId, Key as KeyCode, Rect, ScrollArea, Sense, Stroke, StrokeKind,
    TextureHandle, TextureOptions, Vec2, WidgetInfo, WidgetType, pos2, vec2,
};

use super::cache::Lru;
use super::loader::{Key, Loader, Request};
use crate::catalog::images::{Entry, Kind};
use crate::nav::NavSet;
use crate::thumbs::generate::BUCKETS;

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
        let sha256 = entry.sha256.filter(|_| entry.kind == Kind::Image)?;
        let key = (sha256, BUCKETS[BUCKETS.len() - 1]);
        if self.textures.get(&key).is_none()
            && !self.failed.contains(&key)
            && let Some(loader) = &self.loader
        {
            loader.request(
                ctx,
                vec![Request {
                    key,
                    path: entry.path.clone(),
                }],
            );
        }
        BUCKETS
            .iter()
            .rev()
            .find_map(|&bucket| self.textures.get(&(sha256, bucket)).cloned())
    }

    pub fn show(&mut self, ui: &mut egui::Ui, nav: &mut NavSet) -> Option<Action> {
        self.receive(ui.ctx());
        let spacing = ui.spacing().item_spacing;
        let width = ui.available_width() - ui.spacing().scroll.allocated_width();
        let columns = (((width + spacing.x) / (TILE + spacing.x)).floor() as usize).max(1);
        let rows = nav.len().div_ceil(columns);
        self.layout = Layout { columns, rows };
        let bucket = bucket_for(TILE * ui.ctx().pixels_per_point());

        let mut action = self.keyboard(ui, nav, columns);

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
                        let selected = nav.cursor() == Some(index);
                        let response = self.tile(ui, entry, bucket, selected, &mut wanted);
                        if response.clicked() {
                            nav.set_cursor(index);
                        }
                        if response.double_clicked() {
                            nav.set_cursor(index);
                            action = Some(Action::Open(index));
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

    fn keyboard(&mut self, ui: &egui::Ui, nav: &mut NavSet, columns: usize) -> Option<Action> {
        if ui.memory(|m| m.focused().is_some()) || nav.is_empty() {
            return None;
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
        selected: bool,
        wanted: &mut Vec<Request>,
    ) -> egui::Response {
        let (rect, response) = ui.allocate_exact_size(Vec2::splat(TILE), Sense::click());
        let name = file_name(entry);
        response.widget_info(|| {
            WidgetInfo::selected(WidgetType::SelectableLabel, true, selected, &name)
        });
        let visuals = ui.visuals();
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, visuals.extreme_bg_color);

        let key = match (entry.kind, entry.sha256) {
            (Kind::Image, Some(sha256)) => Some((sha256, bucket)),
            _ => None,
        };
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

        if selected {
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

fn bucket_for(pixels: f32) -> u32 {
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
    use super::*;

    #[test]
    fn bucket_covers_tile_pixels() {
        assert_eq!(bucket_for(192.0), 256);
        assert_eq!(bucket_for(256.0), 256);
        assert_eq!(bucket_for(288.0), 512);
        assert_eq!(bucket_for(768.0), 512);
    }
}
