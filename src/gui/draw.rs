//! Custom chart painting toolkit. Replaces egui_plot: every panel is a
//! fixed-size card whose painter is clipped to its rect, with explicit
//! pixel mapping — geometry cannot shift, auto-fit, or overflow.

use eframe::egui::{
    self, Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, Vec2, pos2, vec2,
};

use super::DIM;

pub const CARD_BG: Color32 = Color32::from_rgb(23, 20, 18);
pub const GRID: Color32 = Color32::from_rgb(48, 42, 38);

pub struct Card {
    /// Content area (inside padding and title).
    pub rect: Rect,
    /// Painter clipped to the card — drawing cannot escape it.
    pub painter: egui::Painter,
}

/// Allocate an exact-size card with background and title. The returned
/// painter is clipped to the card's outer rect.
pub fn card(ui: &mut egui::Ui, size: Vec2, title: &str) -> Card {
    let (outer, _) = ui.allocate_exact_size(size, Sense::hover());
    let painter = ui.painter_at(outer);
    painter.rect_filled(outer, CornerRadius::same(8), CARD_BG);
    painter.text(
        outer.left_top() + vec2(12.0, 8.0),
        Align2::LEFT_TOP,
        title,
        FontId::proportional(11.0),
        DIM,
    );
    let rect = Rect::from_min_max(outer.min + vec2(12.0, 30.0), outer.max - vec2(12.0, 12.0));
    Card { rect, painter }
}

/// Linear mapping from a data range onto a pixel rect.
#[derive(Clone, Copy)]
pub struct Scale {
    pub rect: Rect,
    pub x0: f32,
    pub x1: f32,
    pub y0: f32,
    pub y1: f32,
}

impl Scale {
    pub fn new(rect: Rect, x: (f32, f32), y: (f32, f32)) -> Self {
        Self {
            rect,
            x0: x.0,
            x1: x.1,
            y0: y.0,
            y1: y.1,
        }
    }

    pub fn x(&self, v: f32) -> f32 {
        self.rect.left() + (v - self.x0) / (self.x1 - self.x0) * self.rect.width()
    }

    pub fn y(&self, v: f32) -> f32 {
        // y grows downward on screen.
        self.rect.bottom() - (v - self.y0) / (self.y1 - self.y0) * self.rect.height()
    }

    pub fn pos(&self, x: f32, y: f32) -> Pos2 {
        pos2(self.x(x), self.y(y))
    }
}

pub fn hgrid(painter: &egui::Painter, s: &Scale, values: &[f32], label: impl Fn(f32) -> String) {
    for &v in values {
        let y = s.y(v);
        painter.line_segment(
            [pos2(s.rect.left(), y), pos2(s.rect.right(), y)],
            Stroke::new(1.0, GRID),
        );
        painter.text(
            pos2(s.rect.left() + 2.0, y - 2.0),
            Align2::LEFT_BOTTOM,
            label(v),
            FontId::proportional(9.0),
            DIM,
        );
    }
}

pub fn vgrid(painter: &egui::Painter, s: &Scale, values: &[f32], label: impl Fn(f32) -> String) {
    for &v in values {
        let x = s.x(v);
        painter.line_segment(
            [pos2(x, s.rect.top()), pos2(x, s.rect.bottom())],
            Stroke::new(1.0, GRID),
        );
        painter.text(
            pos2(x, s.rect.bottom() + 1.0),
            Align2::CENTER_TOP,
            label(v),
            FontId::proportional(9.0),
            DIM,
        );
    }
}

/// Polyline through data points (skips if fewer than 2 points).
pub fn polyline(painter: &egui::Painter, points: Vec<Pos2>, stroke: Stroke) {
    if points.len() >= 2 {
        painter.add(egui::Shape::line(points, stroke));
    }
}

/// Filled area between a curve and the bottom of the scale rect.
pub fn area_fill(painter: &egui::Painter, s: &Scale, curve: &[Pos2], fill: Color32) {
    let bottom = s.rect.bottom();
    for w in curve.windows(2) {
        painter.add(egui::Shape::convex_polygon(
            vec![w[0], w[1], pos2(w[1].x, bottom), pos2(w[0].x, bottom)],
            fill,
            Stroke::NONE,
        ));
    }
}

pub fn dashed_vline(painter: &egui::Painter, s: &Scale, x: f32, color: Color32) {
    let x = s.x(x);
    painter.extend(egui::Shape::dashed_line(
        &[pos2(x, s.rect.top()), pos2(x, s.rect.bottom())],
        Stroke::new(1.0, color),
        5.0,
        4.0,
    ));
}

/// Vertical bars (x, height, color) rising from the bottom of the scale.
pub fn bars(painter: &egui::Painter, s: &Scale, values: &[(f32, f32, Color32)], width_frac: f32) {
    let slot = s.rect.width() / (s.x1 - s.x0);
    let half_w = slot * width_frac / 2.0;
    for &(x, h, color) in values {
        let cx = s.x(x);
        let top = s.y(h.clamp(s.y0, s.y1));
        if top < s.rect.bottom() - 0.5 {
            painter.rect_filled(
                Rect::from_min_max(pos2(cx - half_w, top), pos2(cx + half_w, s.rect.bottom())),
                CornerRadius::same(2),
                color,
            );
        }
    }
}
