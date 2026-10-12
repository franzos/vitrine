use eframe::egui::{self, Id, Key, Modifiers, TextEdit};

use crate::geo::{GeoDb, Place};

pub const WIDTH: f32 = 320.0;
pub const FETCH_HINT: &str = "run `vitrine geo fetch` to edit locations";

#[derive(Debug, Clone, PartialEq)]
pub enum Pick {
    Set(Place),
    Clear,
}

/// Search state of one picker; [`Self::reset`] it while its popup is closed.
#[derive(Default)]
pub struct PlacePicker {
    query: String,
    /// Query `results` belong to.
    searched: String,
    results: Vec<Place>,
    error: Option<String>,
    focused: bool,
}

impl PlacePicker {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Results while there's a query, `recent` otherwise; Enter picks the first result.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        id: Id,
        geo: &GeoDb,
        recent: &[Place],
        current: Option<&Place>,
        can_remove: bool,
    ) -> Option<Pick> {
        if let Some(place) = current {
            ui.label(format!("Now: {}", place.label()));
        }
        let caption = ui.label("Place");
        let response = ui
            .add(
                TextEdit::singleline(&mut self.query)
                    .id(id)
                    .desired_width(f32::INFINITY)
                    .hint_text("search by name"),
            )
            .labelled_by(caption.id);
        if !self.focused {
            response.request_focus();
            self.focused = true;
        }
        if self.searched != self.query {
            self.searched.clone_from(&self.query);
            match geo.search(&self.query) {
                Ok(results) => {
                    self.results = results;
                    self.error = None;
                }
                Err(e) => {
                    self.results.clear();
                    self.error = Some(e.to_string());
                }
            }
        }
        let mut pick = None;
        // consumed so the grid doesn't open the viewer on the same Enter
        if response.lost_focus() && ui.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Enter)) {
            pick = self.results.first().cloned().map(Pick::Set);
        }
        if let Some(error) = &self.error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
        let (places, heading, empty) = if self.query.trim().is_empty() {
            (recent, "Recent", "No recent places")
        } else {
            (self.results.as_slice(), "Matches", "No matches")
        };
        ui.weak(if places.is_empty() { empty } else { heading });
        for place in places {
            if ui.button(place.label()).clicked() {
                pick = Some(Pick::Set(place.clone()));
            }
        }
        ui.separator();
        if ui
            .add_enabled(can_remove, egui::Button::new("Remove location"))
            .clicked()
        {
            pick = Some(Pick::Clear);
        }
        ui.weak("Hides the EXIF location too");
        if pick.is_some() {
            ui.close();
        }
        pick
    }
}
