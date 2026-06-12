//! Tuner strip: big note readout, cents bar, strings row / vibrato card.

use eframe::egui::{
    self, Align2, Color32, CornerRadius, FontId, Rect, RichText, Sense, Stroke, StrokeKind, Vec2,
    pos2, vec2,
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
            Mode::Guitar => {
                strings_row(ui, app);
                duet_row(ui, app);
            }
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
                Mode::Guitar => {
                    strings_row(ui, app);
                    duet_row(ui, app);
                }
                Mode::Voice => vibrato_row(ui, app),
            }
        });
    });
}

/// Duet sources that are genuinely sounding (a seeded source whose string
/// has faded reads near-zero level), as string matches with their level
/// relative to the loudest, loudest first.
fn duet_voices(app: &TunerApp) -> Vec<(tuning::Match, f32)> {
    let loudest = app.duet.iter().map(|r| r.level).fold(0.0f32, f32::max);
    if loudest <= 0.0 {
        return Vec::new();
    }
    let mut voices: Vec<(tuning::Match, f32)> = app
        .duet
        .iter()
        .filter(|r| r.level > 0.1 * loudest)
        .map(|r| {
            (
                tuning::nearest_string(r.freq, app.tuning, app.capo, app.a4),
                r.level / loudest,
            )
        })
        .collect();
    voices.sort_by(|a, b| b.1.total_cmp(&a.1));
    voices
}

/// Pinned-height strip under the string badges: one pill per ringing
/// string (note, cents, level bar), or a dim hint while the Kalman bank
/// has fewer than two sources locked.
fn duet_row(ui: &mut egui::Ui, app: &TunerApp) {
    ui.add_space(4.0);
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 26.0), Sense::hover());
    let p = ui.painter_at(rect);
    p.text(
        pos2(rect.left(), rect.center().y),
        Align2::LEFT_CENTER,
        "DUET",
        FontId::proportional(10.0),
        DIM,
    );
    let mut x = rect.left() + 42.0;

    let voices = duet_voices(app);
    if voices.len() < 2 {
        p.text(
            pos2(x, rect.center().y),
            Align2::LEFT_CENTER,
            "— two strings ringing together read separately here",
            FontId::proportional(11.0),
            DIM,
        );
        return;
    }

    for (m, level) in &voices {
        let pill = Rect::from_min_size(pos2(x, rect.top() + 1.0), vec2(132.0, 24.0));
        if pill.right() > rect.right() {
            break;
        }
        let color = zone_color(m.cents);
        p.rect_filled(pill, CornerRadius::same(6), PANEL);
        p.rect_stroke(
            pill,
            CornerRadius::same(6),
            Stroke::new(1.0, color.linear_multiply(0.7)),
            StrokeKind::Inside,
        );
        p.text(
            pos2(pill.left() + 8.0, pill.top() + 10.0),
            Align2::LEFT_CENTER,
            format!("{} · {}", m.string_no, tuning::note_name(m.target_midi)),
            FontId::proportional(12.0),
            color,
        );
        p.text(
            pos2(pill.right() - 8.0, pill.top() + 10.0),
            Align2::RIGHT_CENTER,
            format!("{:+.1}¢", m.cents),
            FontId::proportional(12.0),
            color,
        );
        // Source level along the pill's bottom edge.
        let bar = Rect::from_min_max(
            pos2(pill.left() + 6.0, pill.bottom() - 6.0),
            pos2(pill.right() - 6.0, pill.bottom() - 4.0),
        );
        p.rect_filled(bar, CornerRadius::same(1), TRACK);
        p.rect_filled(
            Rect::from_min_size(
                bar.min,
                vec2(bar.width() * level.clamp(0.05, 1.0), bar.height()),
            ),
            CornerRadius::same(1),
            color.linear_multiply(0.8),
        );
        x = pill.right() + 8.0;
    }

    if app.level > 1e-4 {
        let fit = 20.0 * (app.sep_residual / app.level.max(1e-6)).log10();
        p.text(
            pos2(x + 4.0, rect.center().y),
            Align2::LEFT_CENTER,
            format!("residual {fit:.0} dB"),
            FontId::proportional(10.0),
            DIM,
        );
    }
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

    // Duet: the second ringing string gets its own slimmer needle with a
    // string-number tag, so both tuning errors show on one bar.
    if let Some(r) = &app.reading {
        for (m, _) in duet_voices(app) {
            if m.string_no == r.string_no {
                continue;
            }
            let c = m.cents.clamp(-SPAN_CENTS, SPAN_CENTS);
            let x = x_of(c);
            let color = zone_color(c);
            painter.rect_filled(
                Rect::from_center_size(pos2(x, track.center().y), vec2(3.0, track.height() + 6.0)),
                CornerRadius::same(2),
                color.linear_multiply(0.85),
            );
            painter.text(
                pos2(x, rect.top() + 6.0),
                Align2::CENTER_CENTER,
                m.string_no.to_string(),
                FontId::proportional(9.0),
                color,
            );
        }
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
    // Strings the Kalman bank hears ringing get an outline in their own
    // tuning color, so both of a duet light up at once.
    let duet: Vec<(usize, Color32)> = duet_voices(app)
        .iter()
        .map(|(m, _)| (m.string_no, zone_color(m.cents)))
        .collect();

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
            let duet_hit = (!is_active)
                .then(|| duet.iter().find(|(n, _)| *n == no).map(|(_, c)| *c))
                .flatten();
            let (rect, _) = ui.allocate_exact_size(badge, Sense::hover());
            let fill = if is_active { zone } else { PANEL };
            let text_color = match (is_active, duet_hit) {
                (true, _) => Color32::BLACK,
                (_, Some(c)) => c,
                _ => DIM,
            };
            ui.painter().rect_filled(rect, CornerRadius::same(6), fill);
            if let Some(c) = duet_hit {
                ui.painter().rect_stroke(
                    rect,
                    CornerRadius::same(6),
                    Stroke::new(1.5, c),
                    StrokeKind::Inside,
                );
            }
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
