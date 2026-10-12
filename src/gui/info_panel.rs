//! The viewer's side panel: location, people and file details of the image under the cursor.

use eframe::egui::{
    self, CollapsingHeader, Id, Popup, PopupCloseBehavior, Rect, ScrollArea, Vec2, pos2,
};

use super::crops::Crops;
use super::mini_map;
use super::people_panel::{self, crop};
use super::place_picker::{self, PlacePicker};
use super::viewer::{Action, Places};
use crate::catalog::faces::{self, ImageFace};
use crate::catalog::geo::Source;
use crate::catalog::images::{Entry, Image, ImageId, Kind};
use crate::video;

pub const WIDTH: f32 = 300.0;
const THUMB: f32 = 40.0;
/// Keeps Details reachable below a long People list.
const PEOPLE_HEIGHT: f32 = 320.0;

/// Catalog data of one image, loaded by the app while the panel is open.
pub struct InfoData {
    pub id: ImageId,
    pub image: Image,
    /// `None` when faces are disabled or no model is loaded.
    pub faces: Option<Vec<ImageFace>>,
}

pub struct Info<'a> {
    /// Belongs to the previous image for a frame after the cursor moves.
    pub data: Option<&'a InfoData>,
    pub crops: &'a mut Crops,
}

#[derive(Default)]
pub(super) struct Output {
    pub action: Option<Action>,
    /// Face of the row under the pointer, in image pixels.
    pub hovered: Option<faces::Rect>,
    /// The place picker's popup is open.
    pub picking: bool,
}

/// `size` is the decoded image size while the full image is shown.
pub(super) fn show(
    ui: &mut egui::Ui,
    entry: &Entry,
    size: Option<Vec2>,
    info: &mut Info<'_>,
    places: &Places<'_>,
    picker: &mut PlacePicker,
) -> Output {
    let mut out = Output::default();
    let data = info.data.filter(|d| d.id == entry.id);
    if data.is_none() {
        ui.ctx().request_repaint();
    }
    ScrollArea::vertical()
        .id_salt("viewer_info")
        .auto_shrink(false)
        .show(ui, |ui| {
            if entry.kind == Kind::Image {
                location(ui, entry, places, picker, &mut out);
                if let Some(faces) = data.and_then(|d| d.faces.as_deref()) {
                    people(ui, faces, info.crops, &mut out);
                }
            }
            if let Some(data) = data {
                details(ui, &data.image, size);
            }
        });
    out
}

fn location(
    ui: &mut egui::Ui,
    entry: &Entry,
    places: &Places<'_>,
    picker: &mut PlacePicker,
    out: &mut Output,
) {
    CollapsingHeader::new("Location")
        .id_salt("info_location")
        .default_open(true)
        .show(ui, |ui| {
            let Some(sha256) = entry.sha256 else {
                ui.label("Not located");
                return;
            };
            let Some((_, location)) = places.location.filter(|(sha, _)| *sha == sha256) else {
                ui.ctx().request_repaint();
                return;
            };
            if let Some((lat, lon)) = places.coords {
                mini_map::show(ui, lat, lon);
            }
            let current = location.as_ref().and_then(|l| l.place.as_ref());
            match location.as_ref().zip(current) {
                Some((location, place)) => {
                    ui.label(format!("Place: {}", place.label()));
                    ui.weak(match location.source {
                        Source::Exif => "From EXIF",
                        Source::Manual => "Set manually",
                    });
                }
                None => {
                    ui.label("Not located");
                }
            }
            match places.gazetteer {
                Some(geo) => {
                    let button = ui.button("Change location...");
                    let shown = Popup::from_toggle_button_response(&button)
                        .close_behavior(PopupCloseBehavior::CloseOnClickOutside)
                        .width(place_picker::WIDTH)
                        .show(|ui| {
                            picker.show(
                                ui,
                                Id::new("viewer_info_place"),
                                geo,
                                places.recent,
                                current,
                                current.is_some(),
                            )
                        });
                    if let Some(shown) = shown {
                        out.picking = true;
                        if let Some(pick) = shown.inner {
                            out.action = Some(Action::Locate(pick));
                        }
                    }
                }
                None => {
                    ui.weak(place_picker::FETCH_HINT);
                }
            }
            if ui
                .add_enabled(places.coords.is_some(), egui::Button::new("Show on map"))
                .clicked()
            {
                out.action = Some(Action::ShowOnMap);
            }
        });
}

fn people(ui: &mut egui::Ui, faces: &[ImageFace], crops: &mut Crops, out: &mut Output) {
    CollapsingHeader::new(format!("People ({})", faces.len()))
        .id_salt("info_people")
        .default_open(true)
        .show(ui, |ui| {
            if faces.is_empty() {
                ui.weak("No faces");
                return;
            }
            ScrollArea::vertical()
                .id_salt("info_people_list")
                .max_height(PEOPLE_HEIGHT)
                .auto_shrink([false, true])
                .show_rows(ui, THUMB, faces.len(), |ui, range| {
                    for face in &faces[range] {
                        let row = ui.horizontal(|ui| {
                            crop(ui, crops, Some(face.id), THUMB);
                            match &face.person {
                                Some(person) => ui
                                    .selectable_label(false, people_panel::name(person))
                                    .clicked()
                                    .then_some(Action::OpenPerson(person.id)),
                                None => {
                                    // lines up with the selectable names' text
                                    ui.add_space(ui.spacing().button_padding.x);
                                    ui.label("Unknown");
                                    None
                                }
                            }
                        });
                        if let Some(action) = row.inner {
                            out.action = Some(action);
                        }
                        let rect = row.response.rect;
                        let full =
                            Rect::from_min_max(rect.min, pos2(ui.max_rect().right(), rect.max.y));
                        if ui.rect_contains_pointer(full) {
                            out.hovered = Some(face.rect);
                        }
                    }
                });
        });
}

fn details(ui: &mut egui::Ui, image: &Image, size: Option<Vec2>) {
    CollapsingHeader::new("Details")
        .id_salt("info_details")
        .default_open(true)
        .show(ui, |ui| {
            field(ui, "Path", &image.path.display().to_string());
            let taken = image
                .taken_at
                .as_deref()
                .map_or_else(|| "Not recorded".to_owned(), |t| t.replacen('T', " ", 1));
            field(ui, "Date taken", &taken);
            field(ui, "File size", &file_size(image.size));
            if let Some(ms) = image.duration_ms {
                field(ui, "Duration", &video::clock(ms));
            }
            if let Some(codec) = &image.codec {
                field(ui, "Codec", codec);
            }
            // a video shows its thumbnail, not a full-size frame
            let shown = size.filter(|_| image.kind == Kind::Image);
            let resolution = shown
                .map(|s| (s.x as u32, s.y as u32))
                .or(image.width.zip(image.height));
            if let Some((w, h)) = resolution {
                field(ui, "Resolution", &format!("{w} x {h}"));
            }
        });
}

fn field(ui: &mut egui::Ui, caption: &str, value: &str) {
    ui.weak(caption);
    ui.label(value);
    ui.add_space(4.0);
}

/// Decimal units, as file managers show them.
fn file_size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["KB", "MB", "GB", "TB"];
    if bytes < 1000 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64 / 1000.0;
    let mut unit = 0;
    // 999.95 rounds to "1000.0" at one decimal
    while value >= 999.95 && unit + 1 < UNITS.len() {
        value /= 1000.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_size_picks_a_decimal_unit() {
        assert_eq!(file_size(0), "0 B");
        assert_eq!(file_size(999), "999 B");
        assert_eq!(file_size(1000), "1.0 KB");
        assert_eq!(file_size(4_321_000), "4.3 MB");
        assert_eq!(file_size(999_960), "1.0 MB");
        assert_eq!(file_size(2_500_000_000), "2.5 GB");
        assert_eq!(file_size(5_000_000_000_000_000), "5000.0 TB");
    }
}
