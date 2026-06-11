//! FFT spectrum with tracked peak overlays and a fixed-size peak table.
//! Custom-painted into a clipped card: geometry is pixel-pinned.

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Stroke, Vec2, pos2};

use super::draw::{self, Scale};
use super::{DIM, FLAMENCO_GOLD, FLAMENCO_RED, GREEN, TEXT, TunerApp};
use crate::tuning;

pub const SPECTRUM_MAX_HZ: f64 = 1500.0;
const DB_MIN: f32 = -90.0;
const DB_MAX: f32 = 5.0;
/// Peaks must survive this many frames before being labeled.
const MIN_TRACK_AGE: u32 = 8;
const TABLE_ROWS: usize = 5;
const TABLE_H: f32 = 74.0;

pub fn section(ui: &mut egui::Ui, app: &TunerApp, size: Vec2) {
    let extra = app.fused_mics - 1;
    let title = if app.beam_mics >= 2 {
        let fused = if extra > 0 {
            format!(" +{extra} fused")
        } else {
            String::new()
        };
        format!("SPECTRUM · beam ×{} mics{fused} · tracked peaks", app.beam_mics)
    } else if extra > 0 {
        format!("SPECTRUM · fused ×{} mics · tracked peaks", app.fused_mics)
    } else {
        "SPECTRUM · tracked peaks".to_string()
    };
    let card = draw::card(ui, size, &title);
    let p = &card.painter;
    let plot = Rect::from_min_max(
        card.rect.min,
        pos2(card.rect.right(), card.rect.bottom() - TABLE_H),
    );
    let s = Scale::new(plot, (0.0, SPECTRUM_MAX_HZ as f32), (DB_MIN, DB_MAX));

    draw::hgrid(p, &s, &[-80.0, -60.0, -40.0, -20.0, 0.0], |v| {
        format!("{v:.0} dB")
    });
    draw::vgrid(p, &s, &[250.0, 500.0, 750.0, 1000.0, 1250.0], |v| {
        format!("{v:.0}")
    });

    // Spectrum curve with translucent fill.
    let curve: Vec<Pos2> = app
        .spectrum
        .iter()
        .map(|&[hz, db]| s.pos(hz as f32, db as f32))
        .collect();
    draw::area_fill(
        p,
        &s,
        &curve,
        Color32::from_rgba_unmultiplied(255, 196, 0, 22),
    );
    draw::polyline(p, curve, Stroke::new(1.5, FLAMENCO_GOLD));

    // Target pitch marker.
    if let Some(r) = &app.reading {
        draw::dashed_vline(p, &s, r.target_freq, GREEN);
    }

    // Tracked peaks: markers + frequency labels.
    let stable: Vec<_> = app
        .tracker
        .tracks()
        .iter()
        .filter(|t| t.age >= MIN_TRACK_AGE && (t.freq as f64) < SPECTRUM_MAX_HZ)
        .take(TABLE_ROWS)
        .copied()
        .collect();
    for t in &stable {
        let pos = s.pos(t.freq, t.db.clamp(DB_MIN, DB_MAX));
        p.circle_filled(pos, 3.0, FLAMENCO_RED);
        p.text(
            pos2(pos.x, (pos.y - 8.0).max(plot.top() + 8.0)),
            Align2::CENTER_BOTTOM,
            format!("{:.0}", t.freq),
            FontId::proportional(10.0),
            TEXT,
        );
    }

    // Fixed five-row peak table under the plot.
    for i in 0..TABLE_ROWS {
        let y = plot.bottom() + 12.0 + i as f32 * 13.0;
        let text = match stable.get(i) {
            Some(t) => {
                let m = tuning::nearest_note(t.freq, app.a4);
                format!(
                    "{:>7.1} Hz   {:<4} {:+6.1}¢   {:>6.1} dB",
                    t.freq,
                    tuning::note_name(m.target_midi),
                    m.cents,
                    t.db,
                )
            }
            None => "      —".to_string(),
        };
        p.text(
            pos2(card.rect.left(), y),
            Align2::LEFT_TOP,
            text,
            FontId::monospace(10.0),
            DIM,
        );
    }
}
