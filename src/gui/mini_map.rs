//! A small static map centred on one location.

use eframe::egui::{self, Align2, Color32, FontId, Sense, Stroke, Vec2, vec2};

use super::basemap::{Basemap, NaturalEarth};
use super::map::{BUBBLE, map_rect, screen_pos};
use crate::map::{Viewport, project};

const SIZE: Vec2 = Vec2::new(260.0, 195.0);
const ZOOM: f64 = 4.0;
const DOT_RADIUS: f32 = 5.0;

/// Draws the basemap around `(lat, lon)` with a dot on it.
pub fn show(ui: &mut egui::Ui, lat: f64, lon: f64) {
    let (screen, _) = ui.allocate_exact_size(SIZE, Sense::hover());
    let rect = map_rect(screen);
    let point = project(lat, lon);
    let mut viewport = Viewport {
        center: point,
        zoom: ZOOM,
    };
    viewport.clamp(rect);

    let painter = ui.painter_at(screen);
    NaturalEarth.paint(&painter, screen, &viewport);
    painter.circle(
        screen_pos(viewport.to_screen(point, rect)),
        DOT_RADIUS,
        BUBBLE,
        Stroke::new(2.0, Color32::WHITE),
    );
    painter.text(
        screen.right_bottom() - vec2(4.0, 2.0),
        Align2::RIGHT_BOTTOM,
        NaturalEarth.attribution(),
        FontId::proportional(10.0),
        ui.visuals().weak_text_color(),
    );
}

#[cfg(test)]
mod tests {
    use eframe::egui::epaint::ClippedShape;
    use eframe::egui::{Rect, Shape};
    use egui_kittest::Harness;

    use super::*;

    fn render(lat: f64, lon: f64) -> Vec<ClippedShape> {
        let mut harness = Harness::builder()
            .with_size(vec2(400.0, 300.0))
            .build_ui(move |ui| show(ui, lat, lon));
        harness.run();
        harness.output().shapes.clone()
    }

    /// The map rect, from the clip of the marker, and the marker's centre.
    fn marker(shapes: &[ClippedShape]) -> (Rect, egui::Pos2) {
        shapes
            .iter()
            .find_map(|s| match &s.shape {
                Shape::Circle(c) if c.fill == BUBBLE => Some((s.clip_rect, c.center)),
                _ => None,
            })
            .expect("marker drawn")
    }

    fn expected(rect: Rect, lat: f64, lon: f64) -> egui::Pos2 {
        let rect = map_rect(rect);
        let point = project(lat, lon);
        let mut viewport = Viewport {
            center: point,
            zoom: ZOOM,
        };
        viewport.clamp(rect);
        screen_pos(viewport.to_screen(point, rect))
    }

    #[test]
    fn marker_sits_on_the_point_over_the_basemap() {
        let shapes = render(38.72, -9.14);
        let (rect, centre) = marker(&shapes);
        assert_eq!(rect.size(), SIZE);
        assert_eq!(centre, expected(rect, 38.72, -9.14));
        assert!((centre - rect.center()).length() < 0.01, "{centre:?}");
        let meshes = shapes
            .iter()
            .filter(|s| matches!(s.shape, Shape::Mesh(_)))
            .count();
        assert!(meshes > 0);
        assert!(shapes.iter().any(
            |s| matches!(&s.shape, Shape::Text(t) if t.galley.text() == NaturalEarth.attribution())
        ));
    }

    #[test]
    fn near_the_pole_the_view_clamps_and_the_marker_follows() {
        let (rect, centre) = marker(&render(85.0, 20.0));
        assert_eq!(centre, expected(rect, 85.0, 20.0));
        assert!(
            centre.y < rect.center().y && rect.contains(centre),
            "{centre:?}"
        );
    }
}
