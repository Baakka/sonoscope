//! Chromagram + chord card, pitch history, waveform, visibility metrics —
//! all custom-painted into fixed, clipped cards.

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Stroke, Vec2, pos2};

use super::draw::{self, Scale};
use super::{DIM, FLAMENCO_GOLD, FLAMENCO_RED, GREEN, HISTORY_SECS, TunerApp};
use crate::IN_TUNE_CENTS;
use crate::dsp::chroma::PITCH_CLASSES;
use crate::dsp::visibility::HIST_BINS;

pub fn chroma_section(ui: &mut egui::Ui, app: &TunerApp, size: Vec2) {
    let card = draw::card(ui, size, "CHROMAGRAM · pitch classes");
    let p = &card.painter;
    // Reserve a strip at the bottom for class labels + chord line.
    let plot = Rect::from_min_max(
        card.rect.min,
        pos2(card.rect.right(), card.rect.bottom() - 44.0),
    );
    let s = Scale::new(plot, (-0.7, 11.7), (0.0, 1.05));

    let dominant = app
        .chroma
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .filter(|(_, v)| **v > 0.2)
        .map(|(i, _)| i);

    let values: Vec<(f32, f32, Color32)> = app
        .chroma
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let color = if Some(i) == dominant {
                FLAMENCO_RED
            } else {
                FLAMENCO_GOLD.linear_multiply(0.55)
            };
            (i as f32, v, color)
        })
        .collect();
    draw::bars(p, &s, &values, 0.7);

    // Class labels.
    for (i, name) in PITCH_CLASSES.iter().enumerate() {
        p.text(
            pos2(s.x(i as f32), plot.bottom() + 2.0),
            Align2::CENTER_TOP,
            *name,
            FontId::proportional(10.0),
            DIM,
        );
    }

    // Fixed chord/key line.
    let line_y = card.rect.bottom() - 10.0;
    let chord_text = match &app.chord {
        Some(c) => format!("♬ {}   ({:.0}% match)", c.name, c.confidence * 100.0),
        None => "♬ —".to_string(),
    };
    let chord_color = if app.chord.is_some() { GREEN } else { DIM };
    p.text(
        pos2(card.rect.left(), line_y),
        Align2::LEFT_CENTER,
        chord_text,
        FontId::proportional(16.0),
        chord_color,
    );
    let key_text = match &app.key {
        Some(k) => format!("key: {k}"),
        None => "key: listening…".to_string(),
    };
    p.text(
        pos2(card.rect.right(), line_y),
        Align2::RIGHT_CENTER,
        key_text,
        FontId::proportional(12.0),
        FLAMENCO_GOLD,
    );
}

pub fn history_section(ui: &mut egui::Ui, app: &TunerApp, size: Vec2) {
    let card = draw::card(ui, size, "PITCH HISTORY (¢)");
    let p = &card.painter;
    let now = app.started.elapsed().as_secs_f64();
    let s = Scale::new(
        card.rect,
        ((now - HISTORY_SECS) as f32, now as f32),
        (-50.0, 50.0),
    );

    draw::hgrid(p, &s, &[-25.0, 0.0, 25.0], |v| format!("{v:+.0}"));
    // In-tune band.
    let band = Rect::from_min_max(
        pos2(s.rect.left(), s.y(IN_TUNE_CENTS)),
        pos2(s.rect.right(), s.y(-IN_TUNE_CENTS)),
    );
    p.rect_filled(band, 0.0, Color32::from_rgba_unmultiplied(40, 110, 70, 36));

    // Polyline split at NaN gaps.
    let mut segment: Vec<Pos2> = Vec::new();
    for &[t, cents] in &app.history {
        if cents.is_nan() {
            draw::polyline(
                p,
                std::mem::take(&mut segment),
                Stroke::new(1.8, FLAMENCO_GOLD),
            );
        } else {
            segment.push(s.pos(t as f32, cents.clamp(-50.0, 50.0) as f32));
        }
    }
    draw::polyline(p, segment, Stroke::new(1.8, FLAMENCO_GOLD));
}

/// Samples shown after the trigger point (~85 ms at 48 kHz).
const WAVE_VIEW: usize = 4096;

pub fn waveform_section(ui: &mut egui::Ui, app: &TunerApp, size: Vec2) {
    let card = draw::card(ui, size, "WAVEFORM · triggered");
    let p = &card.painter;
    let s = Scale::new(card.rect, (0.0, WAVE_VIEW as f32), (-0.55, 0.55));

    // Center line.
    p.line_segment(
        [s.pos(0.0, 0.0), s.pos(WAVE_VIEW as f32, 0.0)],
        Stroke::new(1.0, draw::GRID),
    );

    let window = app.window_samples();
    let tail = &window[window.len() - WAVE_VIEW * 2..];
    // Oscilloscope trigger: anchor at a rising zero crossing so a periodic
    // signal draws in place instead of scrolling.
    let trigger = tail[..WAVE_VIEW]
        .windows(2)
        .position(|w| w[0] <= 0.0 && w[1] > 0.0)
        .unwrap_or(0);
    let view = &tail[trigger..trigger + WAVE_VIEW];

    let step = 8;
    let points: Vec<Pos2> = view
        .iter()
        .step_by(step)
        .enumerate()
        .map(|(i, &v)| s.pos((i * step) as f32, v.clamp(-0.55, 0.55)))
        .collect();
    draw::polyline(p, points, Stroke::new(1.0, FLAMENCO_RED));
}

pub fn visibility_section(ui: &mut egui::Ui, app: &TunerApp, size: Vec2) {
    let card = draw::card(ui, size, "TEXTURE · visibility graph");
    let p = &card.painter;
    let m = &app.vis;

    // Fixed metric line.
    p.text(
        pos2(card.rect.left(), card.rect.top() + 2.0),
        Align2::LEFT_TOP,
        format!(
            "mean deg {:>5.2}    density {:>6.4}    max {:>3}    {}n·{}e",
            m.mean_degree, m.density, m.max_degree, m.nodes, m.edges
        ),
        FontId::monospace(11.0),
        FLAMENCO_GOLD,
    );

    // Degree histogram, fixed scale (counts can't exceed the envelope length).
    let plot = Rect::from_min_max(
        pos2(card.rect.left(), card.rect.top() + 24.0),
        pos2(card.rect.right(), card.rect.bottom() - 14.0),
    );
    let s = Scale::new(
        plot,
        (-0.7, HIST_BINS as f32 - 0.3),
        (0.0, super::ENVELOPE_LEN as f32 * 1.05),
    );
    let values: Vec<(f32, f32, Color32)> = m
        .histogram
        .iter()
        .enumerate()
        .map(|(d, &count)| (d as f32, count as f32, FLAMENCO_GOLD.linear_multiply(0.6)))
        .collect();
    draw::bars(p, &s, &values, 0.7);
    for d in 0..HIST_BINS {
        let label = if d == HIST_BINS - 1 {
            format!("{d}+")
        } else {
            d.to_string()
        };
        p.text(
            pos2(s.x(d as f32), plot.bottom() + 2.0),
            Align2::CENTER_TOP,
            label,
            FontId::proportional(9.0),
            DIM,
        );
    }
}
