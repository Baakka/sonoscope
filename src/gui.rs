use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui::{
    self, Align2, Color32, CornerRadius, FontId, Rect, RichText, Sense, Stroke, Vec2, pos2, vec2,
};
use egui_plot::{HLine, Line, Plot, PlotPoints, VLine};

use crate::pitch::{self, FFT_LEN};
use crate::tuning::{self, Tuning};
use crate::{HOLD_FRAMES, IN_TUNE_CENTS, RMS_GATE};

const SPAN_CENTS: f32 = 50.0; // tuner bar range is ±SPAN_CENTS
const SPECTRUM_MAX_HZ: f64 = 1500.0;
const HISTORY_SECS: f64 = 12.0;

const BG: Color32 = Color32::from_rgb(16, 14, 13);
const PANEL: Color32 = Color32::from_rgb(30, 26, 24);
const TRACK: Color32 = Color32::from_rgb(44, 38, 34);
const TEXT: Color32 = Color32::from_rgb(228, 218, 208);
const FLAMENCO_RED: Color32 = Color32::from_rgb(206, 36, 62);
const FLAMENCO_GOLD: Color32 = Color32::from_rgb(255, 196, 0);
const GREEN: Color32 = Color32::from_rgb(80, 220, 120);
const DIM: Color32 = Color32::from_rgb(130, 118, 108);

pub struct Reading {
    pub freq: f32,
    pub string_no: usize,
    pub target_midi: i32,
    pub target_freq: f32,
    pub cents: f32,
}

pub struct TunerApp {
    buffer: Arc<Mutex<VecDeque<f32>>>,
    sample_rate: f32,
    // Keeps the cpal stream alive for the lifetime of the window.
    _stream: cpal::Stream,
    window: Vec<f32>,
    detector: pitch::Detector,

    tuning: Tuning,
    capo: i32,
    a4: f32,

    reading: Option<Reading>,
    level: f32,
    /// Frames the display holds after the note decays below the gate.
    hold: u32,
    /// Eased bar position for a physical feel.
    bar_cents: f32,

    /// Spectrum in (Hz, dB), exponentially averaged for a calm display.
    spectrum: Vec<[f64; 2]>,
    /// Pitch deviation history as (seconds, cents); NaN rows break the line.
    history: VecDeque<[f64; 2]>,
    started: Instant,
    styled: bool,
}

impl TunerApp {
    pub fn new(buffer: Arc<Mutex<VecDeque<f32>>>, sample_rate: f32, stream: cpal::Stream) -> Self {
        Self {
            buffer,
            sample_rate,
            _stream: stream,
            window: vec![0.0; FFT_LEN],
            detector: pitch::Detector::new(sample_rate),
            tuning: Tuning::Standard,
            capo: 0,
            a4: 440.0,
            reading: None,
            level: 0.0,
            hold: 0,
            bar_cents: 0.0,
            spectrum: Vec::new(),
            history: VecDeque::new(),
            started: Instant::now(),
            styled: false,
        }
    }

    fn process_audio(&mut self) {
        let filled = {
            let buf = self.buffer.lock().unwrap();
            if buf.len() < FFT_LEN {
                false
            } else {
                for (dst, &src) in self.window.iter_mut().zip(buf.iter()) {
                    *dst = src;
                }
                true
            }
        };

        self.level = if filled {
            pitch::rms(&self.window)
        } else {
            0.0
        };
        if filled {
            self.detector.analyze_spectrum(&self.window);
            self.update_spectrum_display();
        }

        let now = self.started.elapsed().as_secs_f64();
        let mut detected = false;

        if filled && self.level > RMS_GATE {
            if let Some(freq) = self.detector.track(&self.window) {
                let m = tuning::nearest_string(freq, self.tuning, self.capo, self.a4);
                self.history.push_back([now, m.cents as f64]);
                self.reading = Some(Reading {
                    freq,
                    string_no: m.string_no,
                    target_midi: m.target_midi,
                    target_freq: m.target_freq,
                    cents: m.cents,
                });
                self.hold = HOLD_FRAMES;
                detected = true;
            }
        } else {
            self.detector.relax();
        }
        if !detected {
            if self.hold > 0 {
                self.hold -= 1;
            } else {
                if self.reading.is_some() {
                    // Break the history line while silent.
                    self.history.push_back([now, f64::NAN]);
                }
                self.reading = None;
            }
        }

        while self
            .history
            .front()
            .is_some_and(|p| p[0] < now - HISTORY_SECS)
        {
            self.history.pop_front();
        }
    }

    fn update_spectrum_display(&mut self) {
        let bin_hz = self.detector.bin_hz() as f64;
        let mags = self.detector.mags();
        let n_bins = ((SPECTRUM_MAX_HZ / bin_hz) as usize).min(mags.len());
        if self.spectrum.is_empty() {
            self.spectrum = (0..n_bins).map(|i| [i as f64 * bin_hz, -90.0]).collect();
        }
        for (point, &mag) in self.spectrum.iter_mut().zip(mags) {
            let db = (20.0 * (mag + 1e-9).log10()).clamp(-90.0, 0.0) as f64;
            // Fast attack, slow decay keeps harmonics readable.
            point[1] = if db > point[1] {
                db
            } else {
                point[1] * 0.82 + db * 0.18
            };
        }
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        ctx.input(|i| {
            if i.key_pressed(egui::Key::T) {
                self.tuning = self.tuning.toggle();
                self.detector.reset();
            }
            if i.key_pressed(egui::Key::ArrowUp) {
                self.capo = (self.capo + 1).min(9);
            }
            if i.key_pressed(egui::Key::ArrowDown) {
                self.capo = (self.capo - 1).max(0);
            }
        });
    }
}

impl eframe::App for TunerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_audio();
        self.handle_keys(ctx);

        let target = self.reading.as_ref().map_or(0.0, |r| r.cents);
        self.bar_cents += (target - self.bar_cents) * 0.35;

        if !self.styled {
            let mut style = (*ctx.style()).clone();
            style.visuals.panel_fill = BG;
            style.visuals.override_text_color = Some(TEXT);
            ctx.set_style(style);
            self.styled = true;
        }

        egui::TopBottomPanel::top("controls")
            .frame(egui::Frame::default().fill(PANEL).inner_margin(10.0))
            .show(ctx, |ui| self.controls(ui));

        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(BG).inner_margin(14.0))
            .show(ctx, |ui| {
                self.tuner_section(ui);
                ui.add_space(12.0);
                self.spectrum_section(ui);
                ui.add_space(10.0);
                self.bottom_graphs(ui);
            });

        ctx.request_repaint_after(Duration::from_millis(50));
    }
}

impl TunerApp {
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("♪ Flamenco Tuner")
                    .color(FLAMENCO_RED)
                    .strong()
                    .size(16.0),
            );
            ui.add_space(12.0);

            ui.label(RichText::new("Tuning").color(DIM));
            if ui
                .selectable_label(self.tuning == Tuning::Standard, "Standard")
                .clicked()
            {
                self.tuning = Tuning::Standard;
                self.detector.reset();
            }
            if ui
                .selectable_label(self.tuning == Tuning::Rondena, "Rondeña")
                .clicked()
            {
                self.tuning = Tuning::Rondena;
                self.detector.reset();
            }

            ui.add_space(12.0);
            ui.label(RichText::new("Cejilla").color(DIM));
            ui.add(egui::Slider::new(&mut self.capo, 0..=9).integer());

            ui.add_space(12.0);
            ui.label(RichText::new("A4").color(DIM));
            ui.add(
                egui::DragValue::new(&mut self.a4)
                    .speed(0.5)
                    .range(415.0..=466.0)
                    .suffix(" Hz"),
            );

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                self.level_pill(ui);
            });
        });
    }

    // ── Tuner ────────────────────────────────────────────────────────────

    fn tuner_section(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // Big note readout on the left.
            let (note, color, sub) = match &self.reading {
                Some(r) => (
                    tuning::note_name(r.target_midi),
                    zone_color(r.cents),
                    format!("{:.1} Hz → {:.1} Hz", r.freq, r.target_freq),
                ),
                None => ("—".to_string(), DIM, "listening…".to_string()),
            };
            ui.allocate_ui(vec2(170.0, 110.0), |ui| {
                ui.vertical(|ui| {
                    ui.label(RichText::new(note).size(64.0).strong().color(color));
                    ui.label(RichText::new(sub).color(DIM).size(12.0));
                });
            });

            // Cents bar + verdict + strings on the right.
            ui.vertical(|ui| {
                self.cents_bar(ui);
                ui.add_space(8.0);
                self.strings_row(ui);
            });
        });
    }

    fn cents_bar(&self, ui: &mut egui::Ui) {
        let width = ui.available_width();
        let (response, painter) = ui.allocate_painter(vec2(width, 56.0), Sense::hover());
        let rect = response.rect;
        let track = Rect::from_min_size(pos2(rect.left(), rect.top() + 14.0), vec2(width, 22.0));

        painter.rect_filled(track, CornerRadius::same(11), TRACK);

        let x_of =
            |cents: f32| track.left() + (cents + SPAN_CENTS) / (2.0 * SPAN_CENTS) * track.width();

        // In-tune zone, subtle.
        let zone = Rect::from_min_max(
            pos2(x_of(-IN_TUNE_CENTS), track.top()),
            pos2(x_of(IN_TUNE_CENTS), track.bottom()),
        );
        painter.rect_filled(zone, CornerRadius::ZERO, Color32::from_rgb(28, 62, 42));

        // Scale ticks every 10 cents.
        for t in (-50..=50).step_by(10) {
            let x = x_of(t as f32);
            let h = if t == 0 { track.height() } else { 6.0 };
            painter.line_segment(
                [pos2(x, track.bottom() - h), pos2(x, track.bottom())],
                Stroke::new(1.0, Color32::from_rgb(70, 62, 56)),
            );
            if t % 50 == 0 || t == 0 {
                painter.text(
                    pos2(x, track.bottom() + 10.0),
                    Align2::CENTER_CENTER,
                    format!("{t:+}").replace("+0", "0"),
                    FontId::proportional(10.0),
                    DIM,
                );
            }
        }

        // Indicator.
        if self.reading.is_some() {
            let c = self.bar_cents.clamp(-SPAN_CENTS, SPAN_CENTS);
            let x = x_of(c);
            let color = zone_color(c);
            painter.rect_filled(
                Rect::from_center_size(pos2(x, track.center().y), vec2(5.0, track.height() + 10.0)),
                CornerRadius::same(2),
                color,
            );
        }

        // Verdict line.
        let verdict_pos = pos2(track.center().x, rect.top() + 2.0);
        match &self.reading {
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

    fn strings_row(&self, ui: &mut egui::Ui) {
        let active = self.reading.as_ref().map(|r| r.string_no);
        let zone = self.reading.as_ref().map_or(GREEN, |r| zone_color(r.cents));

        ui.horizontal(|ui| {
            let badge = vec2(
                ((ui.available_width() - 5.0 * 6.0) / 6.0).clamp(48.0, 78.0),
                30.0,
            );
            for (i, &open) in self.tuning.open_strings().iter().enumerate() {
                let no = 6 - i;
                let is_active = Some(no) == active;
                let (rect, _) = ui.allocate_exact_size(badge, Sense::hover());
                let fill = if is_active { zone } else { PANEL };
                let text_color = if is_active { Color32::BLACK } else { DIM };
                ui.painter().rect_filled(rect, CornerRadius::same(6), fill);
                ui.painter().text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    format!("{no} · {}", tuning::note_name(open + self.capo)),
                    FontId::proportional(13.0),
                    text_color,
                );
                ui.add_space(6.0);
            }
        });
    }

    // ── Graphs ───────────────────────────────────────────────────────────

    fn spectrum_section(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("SPECTRUM").color(DIM).size(11.0));
        let target = self.reading.as_ref().map(|r| r.target_freq as f64);
        let detected = self.reading.as_ref().map(|r| r.freq as f64);
        let points: PlotPoints = self.spectrum.iter().copied().collect();

        Plot::new("spectrum")
            .height(190.0)
            .include_x(0.0)
            .include_x(SPECTRUM_MAX_HZ)
            .include_y(-90.0)
            .include_y(0.0)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .allow_boxed_zoom(false)
            .show_x(false)
            .show_y(false)
            .x_axis_label("Hz")
            .y_axis_label("dB")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("spectrum", points)
                        .color(FLAMENCO_GOLD)
                        .fill(-90.0)
                        .width(1.5),
                );
                if let Some(f) = target {
                    plot_ui.vline(
                        VLine::new("target", f)
                            .color(GREEN)
                            .style(egui_plot::LineStyle::dashed_loose()),
                    );
                }
                if let Some(f) = detected {
                    plot_ui.vline(VLine::new("detected", f).color(FLAMENCO_RED));
                }
            });
    }

    fn bottom_graphs(&self, ui: &mut egui::Ui) {
        let half = (ui.available_width() - 12.0) / 2.0;
        let height = ui.available_height().clamp(120.0, 220.0);
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.set_width(half);
                ui.label(RichText::new("PITCH HISTORY (¢)").color(DIM).size(11.0));
                self.history_plot(ui, height);
            });
            ui.add_space(12.0);
            ui.vertical(|ui| {
                ui.set_width(half);
                ui.label(RichText::new("WAVEFORM").color(DIM).size(11.0));
                self.waveform_plot(ui, height);
            });
        });
    }

    fn history_plot(&self, ui: &mut egui::Ui, height: f32) {
        let now = self.started.elapsed().as_secs_f64();
        let points: PlotPoints = self.history.iter().copied().collect();

        Plot::new("history")
            .height(height)
            .include_x(now - HISTORY_SECS)
            .include_x(now)
            .include_y(-50.0)
            .include_y(50.0)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .allow_boxed_zoom(false)
            .show_x(false)
            .show_y(false)
            .show_axes([false, true])
            .show(ui, |plot_ui| {
                plot_ui.hline(
                    HLine::new("sharp edge", IN_TUNE_CENTS as f64)
                        .color(Color32::from_rgb(40, 90, 60))
                        .style(egui_plot::LineStyle::dashed_dense()),
                );
                plot_ui.hline(
                    HLine::new("flat edge", -IN_TUNE_CENTS as f64)
                        .color(Color32::from_rgb(40, 90, 60))
                        .style(egui_plot::LineStyle::dashed_dense()),
                );
                plot_ui.line(Line::new("cents", points).color(FLAMENCO_GOLD).width(1.8));
            });
    }

    fn waveform_plot(&self, ui: &mut egui::Ui, height: f32) {
        let step = 8;
        let ms_per_sample = 1000.0 / self.sample_rate as f64;
        let points: PlotPoints = self
            .window
            .iter()
            .step_by(step)
            .enumerate()
            .map(|(i, &s)| [(i * step) as f64 * ms_per_sample, s as f64])
            .collect();

        Plot::new("waveform")
            .height(height)
            .include_y(-0.5)
            .include_y(0.5)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .allow_boxed_zoom(false)
            .show_x(false)
            .show_y(false)
            .show_axes([true, false])
            .x_axis_label("ms")
            .show(ui, |plot_ui| {
                plot_ui.line(Line::new("wave", points).color(FLAMENCO_RED).width(1.0));
            });
    }

    fn level_pill(&self, ui: &mut egui::Ui) {
        let (response, painter) = ui.allocate_painter(vec2(120.0, 12.0), Sense::hover());
        let rect = response.rect;
        painter.rect_filled(rect, CornerRadius::same(6), TRACK);
        // ~0.25 RMS reads as full scale.
        let ratio = (self.level * 4.0).clamp(0.0, 1.0);
        if ratio > 0.0 {
            let fill =
                Rect::from_min_size(rect.min, Vec2::new(rect.width() * ratio, rect.height()));
            painter.rect_filled(fill, CornerRadius::same(6), FLAMENCO_RED);
        }
        ui.label(RichText::new("mic").color(DIM).size(11.0));
    }
}

/// Green within the in-tune window, amber up to 15¢, red beyond.
fn zone_color(cents: f32) -> Color32 {
    let c = cents.abs();
    if c <= IN_TUNE_CENTS {
        GREEN
    } else if c <= 15.0 {
        FLAMENCO_GOLD
    } else {
        FLAMENCO_RED
    }
}
