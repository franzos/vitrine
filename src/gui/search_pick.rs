use std::borrow::Cow;
use std::collections::BTreeSet;

use eframe::egui::{self, Id, Key, Modifiers, Popup, PopupCloseBehavior, ScrollArea, TextEdit};

const WIDTH: f32 = 240.0;
const MAX_HEIGHT: f32 = 300.0;

pub struct Row<'a, K> {
    pub key: K,
    pub label: Cow<'a, str>,
    pub count: Option<u64>,
}

impl<K> Row<'_, K> {
    fn text(&self) -> Cow<'_, str> {
        match self.count {
            Some(count) => format!("{} ({count})", self.label).into(),
            None => Cow::Borrowed(&self.label),
        }
    }
}

/// A button opening a searchable list with an "any" row; returns whether `value` changed.
pub fn single<K: Clone + PartialEq>(
    ui: &mut egui::Ui,
    salt: impl egui::AsIdSalt,
    caption: &str,
    value: &mut Option<K>,
    rows: &[Row<'_, K>],
) -> bool {
    let selected = value
        .as_ref()
        .and_then(|v| rows.iter().find(|r| r.key == *v));
    let title = match selected {
        Some(row) => format!("{caption}: {}", row.label),
        None => caption.to_owned(),
    };
    let id = ui.make_persistent_id(salt);
    let button = ui.button(title);
    let picked = popup(ui, &button, id, rows, |ui, matches, enter, _| {
        let mut picked = None;
        if enter && let Some(row) = matches.first() {
            picked = Some(Some(row.key.clone()));
        }
        if ui.selectable_label(value.is_none(), "any").clicked() {
            picked = Some(None);
        }
        list(ui, matches, |ui, row| {
            let on = value.as_ref() == Some(&row.key);
            if ui.selectable_label(on, row.text()).clicked() {
                picked = Some(Some(row.key.clone()));
            }
        });
        if picked.is_some() {
            ui.close();
        }
        picked
    });
    match picked.flatten() {
        Some(new) if new != *value => {
            *value = new;
            true
        }
        _ => false,
    }
}

/// A button opening a searchable checkbox list; returns the toggled key. Enter toggles the first match and closes.
pub fn multi<K: Clone + Ord>(
    ui: &mut egui::Ui,
    salt: impl egui::AsIdSalt,
    caption: &str,
    value: &mut BTreeSet<K>,
    rows: &[Row<'_, K>],
) -> Option<K> {
    let title = match value.len() {
        0 => caption.to_owned(),
        1 => match rows.iter().find(|r| value.contains(&r.key)) {
            Some(row) => format!("{caption}: {}", row.label),
            None => format!("{caption}: 1 selected"),
        },
        n => format!("{caption}: {n} selected"),
    };
    let id = ui.make_persistent_id(salt);
    let button = ui.button(title);
    let toggled = popup(ui, &button, id, rows, |ui, matches, enter, _| {
        let mut toggled = None;
        if enter && let Some(row) = matches.first() {
            toggled = Some(row.key.clone());
            ui.close();
        }
        list(ui, matches, |ui, row| {
            let mut on = value.contains(&row.key);
            if ui.checkbox(&mut on, row.text()).changed() {
                toggled = Some(row.key.clone());
            }
        });
        toggled
    })
    .flatten()?;
    if !value.remove(&toggled) {
        value.insert(toggled.clone());
    }
    Some(toggled)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pick<K> {
    Row(K),
    New(String),
}

/// A `caption` button opening a searchable list with a "New: <query>" row unless a row matches exactly; Enter picks the first row shown.
pub fn pick_or_new<K: Clone>(
    ui: &mut egui::Ui,
    salt: impl egui::AsIdSalt,
    caption: impl Into<egui::WidgetText>,
    rows: &[Row<'_, K>],
) -> (egui::Response, Option<Pick<K>>) {
    let id = ui.make_persistent_id(salt);
    let button = ui.button(caption);
    let picked = popup(ui, &button, id, rows, |ui, matches, enter, query| {
        let mut picked = None;
        list(ui, matches, |ui, row| {
            let text = row.text();
            let button = ui.add(egui::Button::selectable(false, text.as_ref()).truncate());
            if button.on_hover_text(text).clicked() {
                picked = Some(Pick::Row(row.key.clone()));
            }
        });
        let exact = matches.iter().any(|r| r.label.eq_ignore_ascii_case(query));
        if !query.is_empty()
            && !exact
            && ui
                .selectable_label(false, format!("New: {query}"))
                .clicked()
        {
            picked = Some(Pick::New(query.to_owned()));
        }
        if enter && picked.is_none() {
            picked = match matches.first() {
                Some(row) => Some(Pick::Row(row.key.clone())),
                None if !query.is_empty() => Some(Pick::New(query.to_owned())),
                None => None,
            };
        }
        if picked.is_some() {
            ui.close();
        }
        picked
    });
    (button, picked.flatten())
}

/// The search field and `rows`; `None` while closed. The query lives in temp memory until the popup closes.
fn popup<'r, 'a, K, R>(
    ui: &egui::Ui,
    button: &egui::Response,
    id: Id,
    rows: &'r [Row<'a, K>],
    add_rows: impl FnOnce(&mut egui::Ui, &[&'r Row<'a, K>], bool, &str) -> R,
) -> Option<R> {
    let query_id = id.with("query");
    let shown = Popup::from_toggle_button_response(button)
        .id(id)
        .close_behavior(PopupCloseBehavior::CloseOnClickOutside)
        .width(WIDTH)
        .show(|ui| {
            let stored = ui.data(|d| d.get_temp::<String>(query_id));
            let opened = stored.is_none();
            let mut query = stored.unwrap_or_default();
            let caption = ui.label("Search");
            let field = ui
                .add(
                    TextEdit::singleline(&mut query)
                        .id(id.with("field"))
                        .desired_width(f32::INFINITY),
                )
                .labelled_by(caption.id);
            if opened {
                field.request_focus();
            }
            // consumed so the grid doesn't open the viewer on the same Enter
            let enter =
                field.lost_focus() && ui.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Enter));
            if ui.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
                ui.close();
            }
            let needle = query.to_lowercase();
            let matches: Vec<&Row<'a, K>> = rows
                .iter()
                .filter(|r| r.label.to_lowercase().contains(&needle))
                .collect();
            let picked = add_rows(ui, &matches, enter, query.trim());
            ui.data_mut(|d| d.insert_temp(query_id, query));
            picked
        });
    if shown.is_none() {
        ui.data_mut(|d| d.remove::<String>(query_id));
    }
    shown.map(|shown| shown.inner)
}

fn list<K>(
    ui: &mut egui::Ui,
    rows: &[&Row<'_, K>],
    mut add_row: impl FnMut(&mut egui::Ui, &Row<'_, K>),
) {
    let height = ui.spacing().interact_size.y;
    // full width, else it resizes to the rows in view while scrolling
    ScrollArea::vertical()
        .max_height(MAX_HEIGHT)
        .auto_shrink([false, true])
        .show_rows(ui, height, rows.len(), |ui, range| {
            for row in &rows[range] {
                add_row(ui, row);
            }
        });
}

#[cfg(test)]
mod tests {
    use egui_kittest::Harness;
    use egui_kittest::kittest::Queryable;

    use super::*;

    const PLACES: [(u32, &str, u64); 3] = [(1, "Lisbon", 4), (2, "Porto", 2), (3, "Lagos", 1)];

    fn rows() -> Vec<Row<'static, u32>> {
        PLACES
            .iter()
            .map(|&(key, label, count)| Row {
                key,
                label: label.into(),
                count: Some(count),
            })
            .collect()
    }

    #[derive(Default)]
    struct Single {
        value: Option<u32>,
        changes: usize,
        leaked: Vec<Key>,
    }

    #[derive(Default)]
    struct Multi {
        value: BTreeSet<u32>,
        toggled: Vec<u32>,
    }

    fn leaked(ui: &egui::Ui) -> Vec<Key> {
        [Key::Enter, Key::Escape]
            .into_iter()
            .filter(|key| ui.input(|i| i.key_pressed(*key)))
            .collect()
    }

    fn single_harness(value: Option<u32>) -> Harness<'static, Single> {
        let rows = rows();
        let mut harness = Harness::new_ui_state(
            move |ui, state: &mut Single| {
                if single(ui, "place", "Country", &mut state.value, &rows) {
                    state.changes += 1;
                }
                state.leaked.extend(leaked(ui));
            },
            Single {
                value,
                ..Single::default()
            },
        );
        harness.run();
        harness
    }

    fn multi_harness() -> Harness<'static, Multi> {
        let rows = rows();
        let mut harness = Harness::new_ui_state(
            move |ui, state: &mut Multi| {
                state
                    .toggled
                    .extend(multi(ui, "people", "People", &mut state.value, &rows));
            },
            Multi::default(),
        );
        harness.run();
        harness
    }

    fn open<S>(harness: &mut Harness<'_, S>, button: &str) {
        harness.get_by_label(button).click();
        harness.run_steps(2);
    }

    #[test]
    fn gui_search_filters_case_insensitively() {
        let mut harness = single_harness(None);
        open(&mut harness, "Country");
        harness.get_by_label("Porto (2)");
        harness.get_by_label("Search").type_text("LIS");
        harness.run_steps(1);
        harness.get_by_label("Lisbon (4)");
        harness.get_by_label("any");
        for label in ["Porto (2)", "Lagos (1)"] {
            assert!(harness.query_by_label(label).is_none(), "{label} shown");
        }
    }

    #[test]
    fn gui_search_field_is_focused_on_open() {
        let mut harness = single_harness(None);
        open(&mut harness, "Country");
        assert!(harness.get_by_label("Search").is_focused());
    }

    #[test]
    fn gui_picking_a_row_and_any() {
        let mut harness = single_harness(None);
        open(&mut harness, "Country");
        harness.get_by_label("Porto (2)").click();
        harness.run_steps(2);
        assert_eq!(harness.state().value, Some(2));
        assert_eq!(harness.state().changes, 1);
        assert!(harness.query_by_label("Search").is_none(), "popup open");

        open(&mut harness, "Country: Porto");
        harness.get_by_label("any").click();
        harness.run_steps(2);
        assert_eq!(harness.state().value, None);
        assert_eq!(harness.state().changes, 2);
        harness.get_by_label("Country");
    }

    #[test]
    fn gui_enter_picks_the_first_match_and_is_consumed() {
        let mut harness = single_harness(Some(2));
        open(&mut harness, "Country: Porto");
        harness.get_by_label("Search").type_text("l");
        harness.run_steps(1);
        harness.key_press(Key::Enter);
        harness.run_steps(2);
        assert_eq!(harness.state().value, Some(1));
        assert!(harness.query_by_label("Search").is_none(), "popup open");
        assert!(harness.state().leaked.is_empty());
    }

    #[test]
    fn gui_escape_closes_the_popup_and_the_query_resets() {
        let mut harness = single_harness(None);
        open(&mut harness, "Country");
        harness.get_by_label("Search").type_text("lis");
        harness.run_steps(1);
        harness.key_press(Key::Escape);
        harness.run_steps(2);
        assert!(harness.query_by_label("Search").is_none(), "popup open");
        assert!(harness.state().leaked.is_empty());
        assert_eq!(harness.state().value, None);

        open(&mut harness, "Country");
        assert_eq!(harness.get_by_label("Search").value().as_deref(), Some(""));
        harness.get_by_label("Porto (2)");
    }

    #[test]
    fn gui_multi_toggles_and_stays_open() {
        let mut harness = multi_harness();
        open(&mut harness, "People");
        harness.get_by_label("Lisbon (4)").click();
        harness.run_steps(2);
        assert_eq!(harness.state().toggled, [1]);
        harness.get_by_label("People: Lisbon");

        harness.get_by_label("Lagos (1)").click();
        harness.run_steps(2);
        harness.get_by_label("People: 2 selected");

        harness.get_by_label("Lisbon (4)").click();
        harness.run_steps(2);
        assert_eq!(harness.state().toggled, [1, 3, 1]);
        assert_eq!(harness.state().value, BTreeSet::from([3]));
        harness.get_by_label("People: Lagos");
    }

    #[test]
    fn gui_multi_enter_toggles_the_first_match() {
        let mut harness = multi_harness();
        open(&mut harness, "People");
        harness.get_by_label("Search").type_text("la");
        harness.run_steps(1);
        harness.key_press(Key::Enter);
        harness.run_steps(2);
        assert_eq!(harness.state().value, BTreeSet::from([3]));
        assert!(harness.query_by_label("Search").is_none(), "popup open");
    }
}
