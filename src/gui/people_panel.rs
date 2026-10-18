use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui::text::TextWrapping;
use eframe::egui::{
    self, Align, Color32, ComboBox, FontSelection, Galley, Id, Key, Modifiers, Rect, ScrollArea,
    Sense, TextEdit, TextStyle, Vec2, WidgetText, pos2,
};
use rusqlite::{Connection, Transaction};

use super::crops::Crops;
use crate::catalog::faces::{self, AssignedBy, FaceId};
use crate::catalog::people::{self, Person, PersonId};
use crate::db;

const ROW: f32 = 52.0;
const THUMB: f32 = 48.0;
const COVER: f32 = 96.0;
const FACE: f32 = 72.0;

/// A manual catalog change; affected faces get `assigned_by = 'manual'`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Edit {
    Rename(PersonId, Option<String>),
    Ignore(PersonId, bool),
    /// `into` keeps its name, or takes `from`'s when it has none.
    Merge {
        from: PersonId,
        into: PersonId,
    },
    Unassign(FaceId),
    NotAFace(FaceId),
    Assign(FaceId, PersonId),
    /// A new person with this name, holding just the face.
    NameFace(FaceId, String),
    /// A new ignored person holding just the face, so it isn't merged with other strangers.
    IgnoreFace(FaceId),
}

impl Edit {
    /// Writes the edit in one transaction.
    pub fn apply(&self, conn: &mut Connection) -> db::Result<()> {
        match self {
            Self::Rename(person, name) => {
                in_transaction(conn, |tx| people::rename(tx, *person, name.as_deref()))
            }
            Self::Ignore(person, ignored) => {
                in_transaction(conn, |tx| people::set_ignored(tx, *person, *ignored))
            }
            Self::Merge { from, into } => {
                in_transaction(conn, |tx| people::merge(tx, *from, *into))
            }
            Self::Unassign(face) => faces::unassign(conn, *face),
            Self::NotAFace(face) => faces::set_not_a_face(conn, *face),
            Self::Assign(face, person) => {
                faces::assign(conn, *face, *person, AssignedBy::Manual).map(drop)
            }
            Self::NameFace(face, name) => in_transaction(conn, |tx| {
                people::create_from_face(tx, *face, Some(name)).map(drop)
            }),
            Self::IgnoreFace(face) => in_transaction(conn, |tx| {
                let person = people::create_from_face(tx, *face, None)?;
                people::set_ignored(tx, person, true)
            }),
        }
    }
}

fn in_transaction(
    conn: &mut Connection,
    f: impl FnOnce(&Transaction<'_>) -> db::Result<()>,
) -> db::Result<()> {
    let tx = conn.transaction()?;
    f(&tx)?;
    tx.commit()?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Edit(Edit),
    ToggleFilter(PersonId),
}

#[must_use]
pub fn label(person: &Person) -> String {
    format!("{} ({})", name(person), person.faces)
}

#[must_use]
pub fn name(person: &Person) -> String {
    match &person.name {
        Some(name) => name.clone(),
        None => format!("unnamed #{}", person.id.0),
    }
}

pub struct PeoplePanel {
    people: Vec<Person>,
    show_ignored: bool,
    open: Option<Open>,
    crops: Crops,
    error: Option<String>,
}

struct Open {
    id: PersonId,
    faces: Vec<FaceId>,
    name: String,
    /// Stored name `name` was last reset to; `None` forces a reset.
    synced: Option<Option<String>>,
    merge_into: Option<PersonId>,
}

impl Open {
    fn new(id: PersonId) -> Self {
        Self {
            id,
            faces: Vec::new(),
            name: String::new(),
            synced: None,
            merge_into: None,
        }
    }
}

impl PeoplePanel {
    #[must_use]
    pub fn new(catalog_db: PathBuf) -> Self {
        Self {
            people: Vec::new(),
            show_ignored: false,
            open: None,
            crops: Crops::new(catalog_db),
            error: None,
        }
    }

    /// Every person of the model, ignored ones included.
    #[must_use]
    pub fn people(&self) -> &[Person] {
        &self.people
    }

    pub fn crops_and_people(&mut self) -> (&mut Crops, &[Person]) {
        (&mut self.crops, &self.people)
    }

    #[cfg(test)]
    pub fn opened(&self) -> Option<PersonId> {
        self.open.as_ref().map(|o| o.id)
    }

    /// Closes the open person when it's gone or hidden.
    pub fn reload(&mut self, conn: &Connection, model: faces::ModelId) -> db::Result<()> {
        self.people = people::list(conn, model)?;
        if let Some(open) = &mut self.open {
            let visible = self
                .people
                .iter()
                .any(|p| p.id == open.id && (self.show_ignored || !p.ignored));
            if visible {
                open.faces = faces::of_person(conn, open.id)?;
                open.merge_into = open
                    .merge_into
                    .filter(|into| self.people.iter().any(|p| p.id == *into));
            } else {
                self.open = None;
            }
        }
        Ok(())
    }

    /// For a catalog without the configured model.
    pub fn clear(&mut self) {
        self.people.clear();
        self.open = None;
    }

    /// Draws the list or the open person; `filtered` is the people filter.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        conn: &Connection,
        filtered: &BTreeSet<PersonId>,
    ) -> Option<Action> {
        if let Some(error) = &self.error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
        if self.open.is_some() {
            self.show_person(ui, filtered)
        } else {
            self.show_list(ui, conn);
            None
        }
    }

    fn show_list(&mut self, ui: &mut egui::Ui, conn: &Connection) {
        ui.heading("People");
        ui.checkbox(&mut self.show_ignored, "Show ignored");
        let shown: Vec<usize> = (0..self.people.len())
            .filter(|&i| self.show_ignored || !self.people[i].ignored)
            .collect();
        if shown.is_empty() {
            ui.weak("No people yet");
            return;
        }
        let mut clicked = None;
        ScrollArea::vertical()
            .id_salt("people")
            .auto_shrink(false)
            .show_rows(ui, ROW, shown.len(), |ui, range| {
                for &i in &shown[range] {
                    let person = &self.people[i];
                    ui.horizontal(|ui| {
                        crop(ui, &mut self.crops, person.cover, THUMB);
                        let mut text = label(person);
                        if person.ignored {
                            text.push_str(", ignored");
                        }
                        if ui.selectable_label(false, two_lines(ui, text)).clicked() {
                            clicked = Some(person.id);
                        }
                    });
                }
            });
        if let Some(id) = clicked {
            self.open(conn, id);
        }
    }

    /// Opens `id`; a failed face lookup opens it without faces and sets the error.
    pub fn open(&mut self, conn: &Connection, id: PersonId) {
        self.error = None;
        let mut open = Open::new(id);
        match faces::of_person(conn, id) {
            Ok(faces) => open.faces = faces,
            Err(e) => self.error = Some(e.to_string()),
        }
        self.open = Some(open);
    }

    fn show_person(&mut self, ui: &mut egui::Ui, filtered: &BTreeSet<PersonId>) -> Option<Action> {
        let open = self.open.as_mut()?;
        let Some(person) = self.people.iter().find(|p| p.id == open.id) else {
            self.open = None;
            return None;
        };
        let mut action = None;
        if ui.button("Back").clicked() {
            self.open = None;
            return None;
        }
        crop(ui, &mut self.crops, person.cover, COVER);

        let name_id = Id::new(("person_name", open.id));
        if !ui.memory(|m| m.has_focus(name_id)) && open.synced.as_ref() != Some(&person.name) {
            open.name = person.name.clone().unwrap_or_default();
            open.synced = Some(person.name.clone());
        }
        ui.horizontal(|ui| {
            let caption = ui.label("Name");
            let response = ui
                .add(
                    TextEdit::singleline(&mut open.name)
                        .id(name_id)
                        .desired_width(ui.available_width())
                        .hint_text("unnamed"),
                )
                .labelled_by(caption.id);
            if response.lost_focus() {
                let (enter, escape) = ui.input_mut(|i| {
                    (
                        i.consume_key(Modifiers::NONE, Key::Enter),
                        i.key_pressed(Key::Escape),
                    )
                });
                let trimmed = open.name.trim();
                let name = (!trimmed.is_empty()).then(|| trimmed.to_owned());
                if escape && !enter {
                    open.synced = None;
                } else if name != person.name {
                    action = Some(Action::Edit(Edit::Rename(person.id, name)));
                }
            }
        });
        ui.label(match person.faces {
            1 => "1 face".to_owned(),
            n => format!("{n} faces"),
        });

        let mut in_filter = filtered.contains(&person.id);
        if ui
            .checkbox(&mut in_filter, "Filter by this person")
            .changed()
        {
            action = Some(Action::ToggleFilter(person.id));
        }

        ui.horizontal(|ui| {
            let target = open
                .merge_into
                .and_then(|id| self.people.iter().find(|p| p.id == id));
            let caption = ui.label("Merge into");
            ComboBox::from_id_salt("merge_into")
                .selected_text(target.map_or_else(|| "choose".to_owned(), label))
                .show_ui(ui, |ui| {
                    for other in self
                        .people
                        .iter()
                        .filter(|p| p.id != person.id && (self.show_ignored || !p.ignored))
                    {
                        ui.selectable_value(&mut open.merge_into, Some(other.id), label(other));
                    }
                })
                .response
                .labelled_by(caption.id);
        });
        if ui
            .add_enabled(open.merge_into.is_some(), egui::Button::new("Merge"))
            .clicked()
            && let Some(into) = open.merge_into
        {
            action = Some(Action::Edit(Edit::Merge {
                from: person.id,
                into,
            }));
        }

        let ignore = if person.ignored {
            "Stop ignoring"
        } else {
            "Ignore"
        };
        if ui.button(ignore).clicked() {
            action = Some(Action::Edit(Edit::Ignore(person.id, !person.ignored)));
        }

        ui.separator();
        let faces = &open.faces;
        ScrollArea::vertical()
            .id_salt("person_faces")
            .auto_shrink(false)
            .show_rows(ui, FACE, faces.len(), |ui, range| {
                for &face in &faces[range] {
                    ui.horizontal(|ui| {
                        crop(ui, &mut self.crops, Some(face), FACE);
                        ui.vertical(|ui| {
                            if ui.small_button("Remove from person").clicked() {
                                action = Some(Action::Edit(Edit::Unassign(face)));
                            }
                            if ui.small_button("Not a face").clicked() {
                                action = Some(Action::Edit(Edit::NotAFace(face)));
                            }
                        });
                    });
                }
            });

        if let Some(Action::Edit(Edit::Merge { into, .. })) = &action {
            self.open = Some(Open::new(*into));
        }
        action
    }
}

/// Draws the crop of `face` in a `size` square; empty while loading, failed or `None`.
/// Button text wrapped to the remaining width, cut with an ellipsis after two lines to fit a fixed row.
pub(super) fn two_lines(ui: &egui::Ui, text: impl Into<WidgetText>) -> Arc<Galley> {
    let wrapping = TextWrapping {
        max_width: ui.available_width() - 2.0 * ui.spacing().button_padding.x,
        max_rows: 2,
        break_anywhere: false,
        overflow_character: Some('…'),
    };
    text.into().into_galley_impl(
        ui.ctx(),
        ui.style(),
        wrapping,
        FontSelection::Style(TextStyle::Button),
        Align::Min,
    )
}

pub(super) fn crop(ui: &mut egui::Ui, crops: &mut Crops, face: Option<FaceId>, size: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(size), Sense::hover());
    let texture = face.and_then(|face| crops.get(ui.ctx(), face));
    match texture {
        Some(texture) => {
            ui.painter().image(
                texture.id(),
                rect,
                Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                Color32::WHITE,
            );
        }
        None => {
            ui.painter()
                .rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn person(id: i64, name: Option<&str>) -> Person {
        Person {
            id: PersonId(id),
            name: name.map(str::to_owned),
            ignored: false,
            faces: 3,
            cover: None,
        }
    }

    #[test]
    fn name_omits_face_count() {
        assert_eq!(name(&person(7, Some("Ada"))), "Ada");
        assert_eq!(name(&person(7, None)), "unnamed #7");
        assert_eq!(label(&person(7, Some("Ada"))), "Ada (3)");
        assert_eq!(label(&person(7, None)), "unnamed #7 (3)");
    }
}
