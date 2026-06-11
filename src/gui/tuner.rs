//! Tuner strip: big note readout, cents bar, strings row / vibrato card.

use eframe::egui::{
    self, Align2, Color32, CornerRadius, FontId, Rect, RichText, Sense, Stroke, Vec2, pos2, vec2,
};

use super::{DIM, FLAMENCO_RED, GREEN, Mode, PANEL, TRACK, TunerApp, zone_color};
use crate::dsp::vibrato;
use crate::{IN_TUNE_CENTS, tuning};

const SPAN_CENTS: f32 = 50.0;

pub fn tuner_strip(ui: &mut egui::Ui, app: &TunerApp, narrow: bool) {
    let (note, color, sub) = match &app.reading {
        Some(r) => (
            tuning::note_name(r.target_midi),
            zone_color(r.cents),
            format!("{:.1} Hz → {:.1} Hz", r.freq, r.target_freq),
        ),
        None => ("—".to_string(), DIM, "listening…".to_string()),
    };

    if narrow {
        // Phone layout: note readout on its own line, bar and strings below.
        ui.horizontal(|ui| {
            ui.label(RichText::new(note).size(44.0).strong().color(color));
            ui.add_space(8.0);
            ui.label(RichText::new(sub).color(DIM).size(12.0));
        });
        cents_bar(ui, app);
        ui.add_space(6.0);
        match app.mode {
            Mode::Guitar => strings_row(ui, app),
            Mode::Voice => vibrato_row(ui, app),
        }
        return;
    }

    ui.horizontal(|ui| {
        ui.allocate_ui(vec2(180.0, 110.0), |ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new(note).size(64.0).strong().color(color));
                ui.label(RichText::new(sub).color(DIM).size(12.0));
            });
        });

        ui.vertical(|ui| {
            cents_bar(ui, app);
            ui.add_space(8.0);
            match app.mode {
                Mode::Guitar => strings_row(ui, app),
                Mode::Voice => vibrato_row(ui, app),
            }
        });
    });
}

fn cents_bar(ui: &mut egui::Ui, app: &TunerApp) {
    let width = ui.available_width();
    let (response, painter) = ui.allocate_painter(vec2(width, 56.0), Sense::hover());
    let rect = response.rect;
    let track = Rect::from_min_size(pos2(rect.left(), rect.top() + 14.0), vec2(width, 22.0));

    painter.rect_filled(track, CornerRadius::same(11), TRACK);

    let x_of =
        |cents: f32| track.left() + (cents + SPAN_CENTS) / (2.0 * SPAN_CENTS) * track.width();

    let zone = Rect::from_min_max(
        pos2(x_of(-IN_TUNE_CENTS), track.top()),
        pos2(x_of(IN_TUNE_CENTS), track.bottom()),
    );
    painter.rect_filled(zone, CornerRadius::ZERO, Color32::from_rgb(28, 62, 42));

    for t in (-50..=50).step_by(10) {
        let x = x_of(t as f32);
        let h = if t == 0 { track.height() } else { 6.0 };
        painter.line_segment(
            [pos2(x, track.bottom() - h), pos2(x, track.bottom())],
            Stroke::new(1.0, Color32::from_rgb(70, 62, 56)),
        );
        if t % 50 == 0 || t == 0 {
            // Keep the ±50 labels inside the track so narrow layouts
            // (full-bleed bar) don't clip them at the screen edge.
            let tx = x.clamp(track.left() + 12.0, track.right() - 12.0);
            painter.text(
                pos2(tx, track.bottom() + 10.0),
                Align2::CENTER_CENTER,
                format!("{t:+}").replace("+0", "0"),
                FontId::proportional(10.0),
                DIM,
            );
        }
    }

    if app.reading.is_some() {
        let c = app.bar_cents.clamp(-SPAN_CENTS, SPAN_CENTS);
        let x = x_of(c);
        painter.rect_filled(
            Rect::from_center_size(pos2(x, track.center().y), vec2(5.0, track.height() + 10.0)),
            CornerRadius::same(2),
            zone_color(c),
        );
    }

    let verdict_pos = pos2(track.center().x, rect.top() + 2.0);
    match &app.reading {
        Some(r) if r.cents.abs() <= IN_TUNE_CENTS => {
            painter.text(
                verdict_pos,
                Align2::CENTER_CENTER,
                "✓ in tune — ¡olé!",
                FontId::proportional(13.0),
                GREEN,
            );
        }
        Some(r) => {
            let dir = if r.cents > 0.0 {
                "tune down"
            } else {
                "tune up"
            };
            painter.text(
                verdict_pos,
                Align2::CENTER_CENTER,
                format!("{:+.0}¢ · {dir}", r.cents),
                FontId::proportional(13.0),
                zone_color(r.cents),
            );
        }
        None => {}
    }
}

fn strings_row(ui: &mut egui::Ui, app: &TunerApp) {
    let active = app.reading.as_ref().map(|r| r.string_no);
    let zone = app.reading.as_ref().map_or(GREEN, |r| zone_color(r.cents));

    ui.horizontal(|ui| {
        // Each badge is followed by add_space(6) plus egui's item spacing;
        // budget all six gaps or the row overflows narrow screens (which
        // also ratchets the scroll area's content width wider every frame).
        let gap = 6.0 + ui.spacing().item_spacing.x;
        let badge = vec2(
            ((ui.available_width() - 6.0 * gap) / 6.0).clamp(40.0, 78.0),
            30.0,
        );
        // Tight badges (phone widths) drop the separators and shrink the font.
        let compact = badge.x < 54.0;
        for (i, &open) in app.tuning.open_strings().iter().enumerate() {
            let no = 6 - i;
            let is_active = Some(no) == active;
            let (rect, _) = ui.allocate_exact_size(badge, Sense::hover());
            let fill = if is_active { zone } else { PANEL };
            let text_color = if is_active { Color32::BLACK } else { DIM };
            ui.painter().rect_filled(rect, CornerRadius::same(6), fill);
            let name = tuning::note_name(open + app.capo);
            let (label, font) = if compact {
                (format!("{no}·{name}"), FontId::proportional(11.0))
            } else {
                (format!("{no} · {name}"), FontId::proportional(13.0))
            };
            ui.painter().text(
                rect.center(),
                Align2::CENTER_CENTER,
                label,
                font,
                text_color,
            );
            ui.add_space(6.0);
        }
    });
}

fn vibrato_row(ui: &mut egui::Ui, app: &TunerApp) {
    let history: Vec<[f64; 2]> = app.history.iter().copied().collect();
    ui.horizontal(|ui| match vibrato::analyze(&history, 3.0) {
        Some(v) => {
            stat_badge(ui, "vibrato", &format!("{:.1} Hz", v.rate_hz));
            stat_badge(ui, "depth", &format!("±{:.0}¢", v.depth_cents));
            stat_badge(ui, "center", &format!("{:+.0}¢", v.mean_cents));
        }
        None => {
            ui.label(
                RichText::new("voice mode — hold a note to analyze vibrato")
                    .color(DIM)
                    .size(12.0),
            );
        }
    });
}

fn stat_badge(ui: &mut egui::Ui, label: &str, value: &str) {
    let (rect, _) = ui.allocate_exact_size(vec2(108.0, 30.0), Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::same(6), PANEL);
    ui.painter().text(
        pos2(rect.left() + 8.0, rect.center().y),
        Align2::LEFT_CENTER,
        label,
        FontId::proportional(10.0),
        DIM,
    );
    ui.painter().text(
        pos2(rect.right() - 8.0, rect.center().y),
        Align2::RIGHT_CENTER,
        value,
        FontId::proportional(13.0),
        super::FLAMENCO_GOLD,
    );
    ui.add_space(6.0);
}

pub fn level_pill(ui: &mut egui::Ui, level: f32, label: &str) {
    let (response, painter) = ui.allocate_painter(vec2(110.0, 12.0), Sense::hover());
    let rect = response.rect;
    painter.rect_filled(rect, CornerRadius::same(6), TRACK);
    let ratio = (level * 4.0).clamp(0.0, 1.0);
    if ratio > 0.0 {
        let fill = Rect::from_min_size(rect.min, Vec2::new(rect.width() * ratio, rect.height()));
        painter.rect_filled(fill, CornerRadius::same(6), FLAMENCO_RED);
    }
    ui.label(RichText::new(label).color(DIM).size(11.0));
}
