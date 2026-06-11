//! Scrolling CQT (variable-Q) spectrogram: time × semitone heatmap.
//!
//! egui_plot has no heatmap, so we maintain a CPU pixel buffer (one column
//! per frame, one row per semitone band) and push it to a texture.

use eframe::egui::{
    self, Align2, Color32, ColorImage, FontId, TextureHandle, TextureOptions, Vec2, vec2,
};

use super::draw;
use super::{DIM, TunerApp};
use crate::dsp::cqt::{MIDI_LO, N_BANDS};

const COLS: usize = 480;
const DB_FLOOR: f32 = -70.0;

pub struct Spectrogram {
    /// Row-major pixels, row 0 = highest band (top of image).
    pixels: Vec<Color32>,
    texture: Option<TextureHandle>,
}

impl Spectrogram {
    pub fn new() -> Self {
        Self {
            pixels: vec![heat(0.0); COLS * N_BANDS],
            texture: None,
        }
    }

    /// Append one CQT column (scrolls left). Silent frames draw the floor
    /// color so pauses read as gaps.
    pub fn push(&mut self, bands: &[f32; N_BANDS], audible: bool) {
        for row in 0..N_BANDS {
            let start = row * COLS;
            self.pixels.copy_within(start + 1..start + COLS, start);
            let band = N_BANDS - 1 - row; // top row = highest pitch
            let t = if audible {
                let db = 20.0 * (bands[band] + 1e-9).log10();
                ((db - DB_FLOOR) / -DB_FLOOR).clamp(0.0, 1.0)
            } else {
                0.0
            };
            self.pixels[start + COLS - 1] = heat(t);
        }
    }

    fn image(&self) -> ColorImage {
        ColorImage {
            size: [COLS, N_BANDS],
            source_size: vec2(COLS as f32, N_BANDS as f32),
            pixels: self.pixels.clone(),
        }
    }
}

pub fn section(ui: &mut egui::Ui, app: &mut TunerApp, size: Vec2) {
    let image = app.spectrogram.image();
    let ctx = ui.ctx().clone();
    let card = draw::card(ui, size, "CQT SPECTROGRAM · C2–C7, semitone bins");
    let texture = app.spectrogram.texture.get_or_insert_with(|| {
        ctx.load_texture("cqt-spectrogram", image.clone(), TextureOptions::NEAREST)
    });
    texture.set(image, TextureOptions::NEAREST);

    let rect = card.rect;
    let painter = &card.painter;
    painter.image(
        texture.id(),
        rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        Color32::WHITE,
    );

    // Octave gridlines + labels at every C.
    for midi in (MIDI_LO..=MIDI_LO + N_BANDS as i32 - 1).filter(|m| m % 12 == 0) {
        let band = (midi - MIDI_LO) as f32;
        let y = rect.bottom() - (band + 0.5) / N_BANDS as f32 * rect.height();
        painter.line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.left() + 6.0, y)],
            egui::Stroke::new(1.0, DIM),
        );
        painter.text(
            egui::pos2(rect.left() + 8.0, y),
            Align2::LEFT_CENTER,
            crate::tuning::note_name(midi),
            FontId::proportional(10.0),
            DIM,
        );
    }
}

/// Flamenco-tinted heat color map (dark → wine → red → orange → pale gold).
fn heat(t: f32) -> Color32 {
    const STOPS: [(f32, [f32; 3]); 5] = [
        (0.00, [16.0, 14.0, 13.0]),
        (0.35, [80.0, 18.0, 48.0]),
        (0.62, [206.0, 36.0, 62.0]),
        (0.85, [255.0, 140.0, 30.0]),
        (1.00, [255.0, 232.0, 170.0]),
    ];
    let t = t.clamp(0.0, 1.0);
    for pair in STOPS.windows(2) {
        let (t0, c0) = pair[0];
        let (t1, c1) = pair[1];
        if t <= t1 {
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            return Color32::from_rgb(
                (c0[0] + (c1[0] - c0[0]) * f) as u8,
                (c0[1] + (c1[1] - c0[1]) * f) as u8,
                (c0[2] + (c1[2] - c0[2]) * f) as u8,
            );
        }
    }
    Color32::from_rgb(255, 232, 170)
}
