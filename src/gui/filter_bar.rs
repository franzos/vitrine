use std::collections::BTreeSet;

use eframe::egui::{self, ComboBox};

use super::people_panel::label;
use crate::catalog::faces::ModelId;
use crate::catalog::filter::{FaceCount, Filter};
use crate::catalog::people::{Person, PersonId};
use crate::config;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FilterBar {
    pub count: FaceCount,
    pub people: BTreeSet<PersonId>,
    pub unknown: bool,
}

impl FilterBar {
    #[must_use]
    pub fn is_active(&self) -> bool {
        *self != Self::default()
    }

    #[must_use]
    pub fn filter(&self, model: ModelId, faces: &config::Faces) -> Filter {
        Filter {
            model,
            min_score: faces.min_score,
            min_face_size: faces.min_face_size,
            count: self.count,
            people: self.people.iter().copied().collect(),
            unknown: self.unknown,
        }
    }

    pub fn toggle(&mut self, person: PersonId) {
        if !self.people.remove(&person) {
            self.people.insert(person);
        }
    }

    /// Draws the controls; returns whether the filter changed. Ignored people aren't offered.
    pub fn show(&mut self, ui: &mut egui::Ui, people: &[Person]) -> bool {
        let before = self.clone();
        let caption = ui.label("Faces");
        ComboBox::from_id_salt("face_count")
            .selected_text(self.count.label())
            .width(60.0)
            .show_ui(ui, |ui| {
                for count in FaceCount::ALL {
                    ui.selectable_value(&mut self.count, count, count.label());
                }
            })
            .response
            .labelled_by(caption.id);

        let selected: Vec<&str> = people
            .iter()
            .filter(|p| self.people.contains(&p.id))
            .map(|p| p.name.as_deref().unwrap_or("unnamed"))
            .collect();
        let title = if selected.is_empty() {
            "People".to_owned()
        } else {
            format!("People: {}", selected.join(", "))
        };
        ui.menu_button(title, |ui| {
            egui::ScrollArea::vertical()
                .max_height(400.0)
                .show(ui, |ui| {
                    for person in people.iter().filter(|p| !p.ignored) {
                        let mut on = self.people.contains(&person.id);
                        if ui.checkbox(&mut on, label(person)).changed() {
                            self.toggle(person.id);
                        }
                    }
                });
        });

        ui.checkbox(&mut self.unknown, "Unknown faces only");
        if ui
            .add_enabled(self.is_active(), egui::Button::new("Clear filters"))
            .clicked()
        {
            *self = Self::default();
        }
        *self != before
    }
}
