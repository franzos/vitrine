use std::collections::BTreeSet;

use eframe::egui::{self, ComboBox};
use rusqlite::Connection;

use super::people_panel;
use super::search_pick::{self, Row};
use crate::catalog::faces::ModelId;
use crate::catalog::filter::{Area, FaceCount, Filter, PlaceSel};
use crate::catalog::geo;
use crate::catalog::people::{Person, PersonId};
use crate::config;
use crate::db;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FilterBar {
    pub count: FaceCount,
    pub people: BTreeSet<PersonId>,
    pub unknown: bool,
    pub place: PlaceSel,
    pub area: Option<Area>,
}

/// The catalog's places in use, as `(key, name, images)`, for the selected country and region.
#[derive(Debug, Default)]
pub struct PlaceOptions {
    countries: Vec<(String, String, u64)>,
    regions: Vec<(String, String, u64)>,
    cities: Vec<(i64, String, u64)>,
}

impl PlaceOptions {
    /// Drops selections that are no longer in use.
    pub fn reload(&mut self, conn: &Connection, sel: &mut PlaceSel) -> db::Result<()> {
        self.countries = geo::countries(conn)?;
        self.regions.clear();
        self.cities.clear();
        keep_known(&mut sel.country, &self.countries);
        let Some(country) = &sel.country else {
            sel.region = None;
            sel.city = None;
            return Ok(());
        };
        self.regions = geo::regions(conn, country)?;
        for (code, name, _) in &mut self.regions {
            if name.is_empty() {
                *name = format!("region {code}");
            }
        }
        keep_known(&mut sel.region, &self.regions);
        let Some(region) = &sel.region else {
            sel.city = None;
            return Ok(());
        };
        self.cities = geo::cities(conn, country, region)?;
        keep_known(&mut sel.city, &self.cities);
        Ok(())
    }
}

fn keep_known<K: PartialEq>(value: &mut Option<K>, options: &[(K, String, u64)]) {
    if value
        .as_ref()
        .is_some_and(|v| !options.iter().any(|(k, ..)| k == v))
    {
        *value = None;
    }
}

impl FilterBar {
    #[must_use]
    pub fn is_active(&self) -> bool {
        *self != Self::default()
    }

    #[must_use]
    pub fn filter(&self, model: Option<ModelId>, faces: &config::Faces) -> Filter {
        Filter {
            model,
            min_score: faces.min_score,
            min_face_size: faces.min_face_size,
            count: self.count,
            people: self.people.iter().copied().collect(),
            unknown: self.unknown,
            place: self.place.clone(),
            area: self.area,
        }
    }

    pub fn toggle(&mut self, person: PersonId) {
        if !self.people.remove(&person) {
            self.people.insert(person);
        }
    }

    /// Draws the controls, the face ones only with `people`; returns whether the filter changed. Ignored people aren't offered.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        people: Option<&[Person]>,
        places: &PlaceOptions,
    ) -> bool {
        let before = self.clone();
        if let Some(people) = people {
            self.show_faces(ui, people);
            ui.separator();
        }
        self.show_places(ui, places, &before.place);
        if self.area.is_some() {
            ui.separator();
            ui.label("Area");
            if ui.button("Remove").clicked() {
                self.area = None;
            }
        }
        if ui
            .add_enabled(self.is_active(), egui::Button::new("Clear filters"))
            .clicked()
        {
            *self = Self::default();
        }
        *self != before
    }

    fn show_faces(&mut self, ui: &mut egui::Ui, people: &[Person]) {
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

        let rows: Vec<Row<'_, PersonId>> = people
            .iter()
            .filter(|p| !p.ignored)
            .map(|p| Row {
                key: p.id,
                label: people_panel::name(p).into(),
                count: Some(p.faces),
            })
            .collect();
        search_pick::multi(ui, "people", "People", &mut self.people, &rows);

        ui.checkbox(&mut self.unknown, "Unknown faces only");
    }

    /// "No location" and a place exclude each other; a new country or region resets what's below it.
    fn show_places(&mut self, ui: &mut egui::Ui, places: &PlaceOptions, before: &PlaceSel) {
        let place = &mut self.place;
        // the lists match `before`, so it decides which levels are drawn
        let countries = rows(&places.countries);
        search_pick::single(ui, "country", "Country", &mut place.country, &countries);
        if before.country.is_some() {
            let regions = rows(&places.regions);
            search_pick::single(ui, "region", "Region", &mut place.region, &regions);
            if before.region.is_some() {
                let cities = rows(&places.cities);
                search_pick::single(ui, "city", "City", &mut place.city, &cities);
            }
        }
        ui.checkbox(&mut place.none, "No location");

        if place.none && !before.none {
            place.country = None;
            place.region = None;
            place.city = None;
        } else if place.country != before.country {
            place.none = false;
            place.region = None;
            place.city = None;
        } else if place.region != before.region {
            place.city = None;
        }
    }
}

fn rows<K: Clone>(options: &[(K, String, u64)]) -> Vec<Row<'_, K>> {
    options
        .iter()
        .map(|(key, name, images)| Row {
            key: key.clone(),
            label: name.as_str().into(),
            count: Some(*images),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use egui_kittest::Harness;
    use egui_kittest::kittest::Queryable;

    use super::*;

    const AREA: Area = Area {
        south: 38.6,
        north: 38.8,
        west: -9.3,
        east: -9.0,
    };

    fn harness(bar: FilterBar) -> Harness<'static, FilterBar> {
        let places = PlaceOptions::default();
        let mut harness = Harness::new_ui_state(
            move |ui, bar: &mut FilterBar| {
                ui.horizontal(|ui| bar.show(ui, None, &places));
            },
            bar,
        );
        harness.run();
        harness
    }

    #[test]
    fn filter_carries_area() {
        let bar = FilterBar {
            area: Some(AREA),
            ..FilterBar::default()
        };
        let filter = bar.filter(None, &config::Faces::default());
        assert_eq!(filter.area, Some(AREA));
    }

    #[test]
    fn area_alone_is_active() {
        let bar = FilterBar {
            area: Some(AREA),
            ..FilterBar::default()
        };
        assert!(bar.is_active());
    }

    #[test]
    fn gui_clear_filters_resets_area() {
        let mut harness = harness(FilterBar {
            area: Some(AREA),
            ..FilterBar::default()
        });
        harness.get_by_label("Area");
        harness.get_by_label("Clear filters").click();
        harness.run();
        assert_eq!(*harness.state(), FilterBar::default());
        assert!(harness.query_by_label("Area").is_none());
    }

    #[test]
    fn gui_remove_area_keeps_other_filters() {
        let others = FilterBar {
            count: FaceCount::Two,
            unknown: true,
            place: PlaceSel {
                none: true,
                ..PlaceSel::default()
            },
            ..FilterBar::default()
        };
        let mut harness = harness(FilterBar {
            area: Some(AREA),
            ..others.clone()
        });
        harness.get_by_label("Remove").click();
        harness.run();
        assert_eq!(*harness.state(), others);
        assert!(harness.query_by_label("Area").is_none());
    }
}
