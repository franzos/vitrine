//! The viewer's side panel: location, people and file details of the image under the cursor.

use eframe::egui::{
    self, CollapsingHeader, Id, Popup, PopupCloseBehavior, Rect, ScrollArea, Vec2, pos2,
};

use super::crops::Crops;
use super::mini_map;
use super::people_panel::{self, Edit, crop};
use super::place_picker::{self, PlacePicker};
use super::search_pick::{self, Pick};
use super::viewer::{Action, Places};
use crate::catalog::faces::{self, FaceId, ImageFace};
use crate::catalog::geo::Source;
use crate::catalog::images::{Entry, Image, ImageId, Kind};
use crate::catalog::people::{Person, PersonId};
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
    /// Choices of the face pickers.
    pub people: &'a [Person],
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
                    people(ui, faces, info.crops, info.people, &mut out);
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

fn people(
    ui: &mut egui::Ui,
    faces: &[ImageFace],
    crops: &mut Crops,
    people: &[Person],
    out: &mut Output,
) {
    CollapsingHeader::new(format!("People ({})", faces.len()))
        .id_salt("info_people")
        .default_open(true)
        .show(ui, |ui| {
            if faces.is_empty() {
                ui.weak("No faces");
                return;
            }
            let confirm_id = Id::new("info_name_confirm");
            let mut pending = ui
                .data(|d| d.get_temp::<(FaceId, Pick<PersonId>)>(confirm_id))
                .and_then(|(id, pick)| Some((faces.iter().find(|f| f.id == id)?, pick)));
            if let Some((face, pick)) = &pending {
                out.hovered = Some(face.rect);
                if let Some(whole) = confirm(ui, face, pick, people) {
                    out.action = whole.map(|whole| Action::Edit(edit(face, pick, whole)));
                    pending = None;
                }
            }
            let rows: Vec<_> = people
                .iter()
                .filter(|p| !p.ignored)
                .map(|p| search_pick::Row {
                    key: p.id,
                    label: people_panel::name(p).into(),
                    count: Some(p.faces),
                })
                .collect();
            ScrollArea::vertical()
                .id_salt("info_people_list")
                .max_height(PEOPLE_HEIGHT)
                .auto_shrink([false, true])
                .show_rows(ui, THUMB, faces.len(), |ui, range| {
                    for face in &faces[range] {
                        let row = ui.horizontal(|ui| {
                            crop(ui, crops, Some(face.id), THUMB);
                            // before the name, which takes the rest of the row
                            let menu = face_menu(ui, face);
                            if let Some(pick) = name_picker(ui, face, &rows) {
                                if cluster_size(face) > 1 {
                                    pending = Some((face, pick));
                                } else {
                                    return Some(Action::Edit(edit(face, &pick, true)));
                                }
                            }
                            menu
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
            let pending = pending.map(|(face, pick)| (face.id, pick));
            ui.data_mut(|d| match pending {
                Some(pending) => {
                    d.insert_temp(confirm_id, pending);
                }
                None => d.remove::<(FaceId, Pick<PersonId>)>(confirm_id),
            });
        });
}

/// Faces of the unnamed person holding `face`, which a pick would name as a whole; 0 otherwise.
fn cluster_size(face: &ImageFace) -> u64 {
    face.person
        .as_ref()
        .filter(|p| p.name.is_none())
        .map_or(0, |p| p.faces)
}

/// `whole` names or merges the face's unnamed person; otherwise only the face moves.
fn edit(face: &ImageFace, pick: &Pick<PersonId>, whole: bool) -> Edit {
    let unnamed = face
        .person
        .as_ref()
        .filter(|p| whole && p.name.is_none())
        .map(|p| p.id);
    match (pick.clone(), unnamed) {
        (Pick::Row(into), Some(from)) => Edit::Merge { from, into },
        (Pick::New(name), Some(person)) => Edit::Rename(person, Some(name)),
        (Pick::Row(person), None) => Edit::Assign(face.id, person),
        (Pick::New(name), None) => Edit::NameFace(face.id, name),
    }
}

/// Asks whether a pick applies to the face's whole unnamed person; `Some(None)` cancels.
fn confirm(
    ui: &mut egui::Ui,
    face: &ImageFace,
    pick: &Pick<PersonId>,
    people: &[Person],
) -> Option<Option<bool>> {
    let person = face.person.as_ref()?;
    let target = match pick {
        Pick::New(name) => name.clone(),
        Pick::Row(id) => people
            .iter()
            .find(|p| p.id == *id)
            .map_or_else(|| format!("#{}", id.0), people_panel::name),
    };
    let n = person.faces;
    ui.group(|ui| {
        ui.label(format!(
            "{} has {n} faces. Name all of them \"{target}\", or only this one?",
            people_panel::name(person)
        ));
        if face.low_faces * 2 >= n {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                format!(
                    "{} of the {n} are low-confidence detections; this group is probably several people.",
                    face.low_faces
                ),
            );
        }
        let mut answer = None;
        ui.horizontal_wrapped(|ui| {
            if ui.button("Only this face").clicked() {
                answer = Some(Some(false));
            }
            if ui.button(format!("All {n} faces")).clicked() {
                answer = Some(Some(true));
            }
            if ui.button("Cancel").clicked() {
                answer = Some(None);
            }
        });
        answer
    })
    .inner
}

fn name_picker(
    ui: &mut egui::Ui,
    face: &ImageFace,
    rows: &[search_pick::Row<'_, PersonId>],
) -> Option<Pick<PersonId>> {
    let caption = match &face.person {
        Some(person) if person.ignored => format!("{}, ignored", people_panel::name(person)),
        Some(person) => people_panel::name(person),
        None => "Unknown".to_owned(),
    };
    let caption = people_panel::two_lines(ui, caption);
    search_pick::pick_or_new(ui, ("info_face", face.id.0), caption, rows).1
}

fn face_menu(ui: &mut egui::Ui, face: &ImageFace) -> Option<Action> {
    ui.menu_button("...", |ui| {
        let person = face.person.as_ref();
        let mut action = None;
        if let Some(person) = person
            && ui.button("Show in People").clicked()
        {
            action = Some(Action::OpenPerson(person.id));
        }
        if ui.button("Not a face").clicked() {
            action = Some(Action::Edit(Edit::NotAFace(face.id)));
        }
        match person {
            Some(person) => {
                if ui.button("Unassign").clicked() {
                    action = Some(Action::Edit(Edit::Unassign(face.id)));
                }
                let ignore = if person.ignored {
                    "Unignore person"
                } else {
                    "Ignore person"
                };
                if ui.button(ignore).clicked() {
                    action = Some(Action::Edit(Edit::Ignore(person.id, !person.ignored)));
                }
            }
            None => {
                if ui.button("Ignore").clicked() {
                    action = Some(Action::Edit(Edit::IgnoreFace(face.id)));
                }
            }
        }
        action
    })
    .inner
    .flatten()
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
