mod analysis;
mod draw;
// The vector scope pairs the Mac mic with a phone mic — native only; a
// static page has no server for phones to stream to.
#[cfg(not(target_arch = "wasm32"))]
mod scope;
mod spectrogram;
mod spectrum;
mod tuner;

use std::collections::{HashMap, VecDeque};

// std::time::Instant panics on wasm; web-time is std on native targets.
use web_time::{Duration, Instant};

use eframe::egui::{self, Color32, RichText};

// `audio::play_chirp` is only wired to the (native-only) calibrate button.
#[cfg(not(target_arch = "wasm32"))]
use crate::audio;
use crate::audio::{AudioEngine, FilterConfig, RING_LEN};
use crate::dsp::align::{self, Correlator, Servo};
use crate::dsp::chroma::{self, Chord, KeyEstimator};
use crate::dsp::cqt::Cqt;
use crate::dsp::peaks::{self, Tracker};
use crate::dsp::separate;
use crate::dsp::visibility::{self, VisMetrics};
use crate::pitch::{self, FFT_LEN, GUITAR_RANGE, VOICE_RANGE};
use crate::remote::RemoteMics;
use crate::tuning::{self, Tuning};
use crate::{HOLD_FRAMES, IN_TUNE_CENTS, RMS_GATE};

pub(crate) const BG: Color32 = Color32::from_rgb(16, 14, 13);
pub(crate) const PANEL: Color32 = Color32::from_rgb(30, 26, 24);
pub(crate) const TRACK: Color32 = Color32::from_rgb(44, 38, 34);
pub(crate) const TEXT: Color32 = Color32::from_rgb(228, 218, 208);
pub(crate) const FLAMENCO_RED: Color32 = Color32::from_rgb(206, 36, 62);
pub(crate) const FLAMENCO_GOLD: Color32 = Color32::from_rgb(255, 196, 0);
pub(crate) const GREEN: Color32 = Color32::from_rgb(80, 220, 120);
pub(crate) const DIM: Color32 = Color32::from_rgb(130, 118, 108);

pub(crate) const HISTORY_SECS: f64 = 12.0;
/// Envelope points kept for the visibility graph (~13 s at 20 fps).
pub(crate) const ENVELOPE_LEN: usize = 256;

#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Guitar,
    Voice,
}

pub struct Reading {
    pub freq: f32,
    /// 1–6 in guitar mode; 0 in voice mode (no string).
    pub string_no: usize,
    pub target_midi: i32,
    pub target_freq: f32,
    pub cents: f32,
}

pub struct TunerApp {
    engine: AudioEngine,
    pub(crate) remote: RemoteMics,
    window: Vec<f32>,
    detector: pitch::Detector,
    cqt: Cqt,
    pub(crate) tracker: Tracker,
    key_est: KeyEstimator,
    pub(crate) chroma: [f32; 12],
    pub(crate) chord: Option<Chord>,
    pub(crate) key: Option<String>,
    envelope: VecDeque<f32>,
    pub(crate) vis: VisMetrics,

    /// Kalman oscillator bank: two-string decomposition (guitar mode).
    separator: separate::Separator,
    pub(crate) duet: Vec<separate::SourceReading>,
    pub(crate) sep_residual: f32,
    /// Ring `total` already fed to the separator.
    sep_total: u64,

    pub(crate) mode: Mode,
    pub(crate) tuning: Tuning,
    pub(crate) capo: i32,
    pub(crate) a4: f32,
    filters: FilterConfig,

    pub(crate) reading: Option<Reading>,
    pub(crate) level: f32,
    /// Mics fused into the analysis spectrum this frame (1 = Mac only).
    pub(crate) fused_mics: usize,
    /// Mics coherently combined in the beam (0 = no beam this frame).
    pub(crate) beam_mics: usize,
    /// Delay-and-sum of the Mac window with every aligned phone.
    beam: Option<Vec<f32>>,
    /// Alignment servo per phone, keyed by client id.
    servos: HashMap<u64, Servo>,
    correlator: Correlator,
    hold: u32,
    pub(crate) bar_cents: f32,

    /// Spectrum display points (Hz, dB) with attack/decay smoothing.
    pub(crate) spectrum: Vec<[f64; 2]>,
    pub(crate) history: VecDeque<[f64; 2]>,
    pub(crate) started: Instant,
    pub(crate) spectrogram: spectrogram::Spectrogram,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) qr_texture: Option<egui::TextureHandle>,
    styled: bool,
}

impl TunerApp {
    pub fn new(engine: AudioEngine, remote: RemoteMics) -> Self {
        let sample_rate = engine.sample_rate;
        let filters = engine.filter_config();
        Self {
            remote,
            window: vec![0.0; RING_LEN],
            detector: pitch::Detector::new(sample_rate),
            cqt: Cqt::new(sample_rate, 440.0),
            tracker: Tracker::new(),
            key_est: KeyEstimator::new(),
            chroma: [0.0; 12],
            chord: None,
            key: None,
            envelope: VecDeque::with_capacity(ENVELOPE_LEN + 1),
            vis: VisMetrics::default(),
            separator: separate::Separator::new(sample_rate),
            duet: Vec::new(),
            sep_residual: 0.0,
            sep_total: 0,
            mode: Mode::Guitar,
            tuning: Tuning::Standard,
            capo: 0,
            a4: 440.0,
            filters,
            reading: None,
            level: 0.0,
            fused_mics: 1,
            beam_mics: 0,
            beam: None,
            servos: HashMap::new(),
            correlator: Correlator::new(),
            hold: 0,
            bar_cents: 0.0,
            spectrum: Vec::new(),
            history: VecDeque::new(),
            started: Instant::now(),
            spectrogram: spectrogram::Spectrogram::new(),
            #[cfg(not(target_arch = "wasm32"))]
            qr_texture: None,
            styled: false,
            engine,
        }
    }

    fn process_audio(&mut self) {
        let (filled, mac_total) = {
            let ring = self.engine.ring.lock().unwrap();
            if ring.buf.len() < RING_LEN {
                (false, ring.total)
            } else {
                for (dst, &src) in self.window.iter_mut().zip(ring.buf.iter()) {
                    *dst = src;
                }
                (true, ring.total)
            }
        };

        self.level = if filled {
            pitch::rms(&self.window[RING_LEN - FFT_LEN..])
        } else {
            0.0
        };

        // Phone mics: run the alignment servos and build the beam; phones
        // without a lock fall back to incoherent spectral fusion.
        let extra_tails = self.align_and_beam(filled, mac_total);
        let drive = self
            .remote
            .registry
            .clients()
            .iter()
            .map(|c| c.level())
            .fold(self.level, f32::max);

        let audible = filled && drive > RMS_GATE;

        if filled {
            let streams = analysis_streams(&self.beam, &self.window, &extra_tails);
            self.detector.analyze_streams(&streams);
            self.fused_mics = streams.len();
            self.update_spectrum_display();
            self.cqt.set_a4(self.a4);
            self.cqt.process(&self.window);
            self.spectrogram.push(&self.cqt.bands, audible);

            // Peak tracking over the analysis spectrum.
            let picks = peaks::pick(self.detector.mags(), self.detector.bin_hz(), 1500.0);
            self.tracker.update(if audible { &picks } else { &[] });
        }

        // Chroma / chord / key only while something is sounding. The
        // displayed chroma is EMA-smoothed so the bars glide instead of
        // flickering frame to frame.
        if audible {
            let fresh = chroma::fold(&self.cqt.bands);
            for (c, f) in self.chroma.iter_mut().zip(&fresh) {
                *c = *c * 0.65 + f * 0.35;
            }
            self.chord = chroma::detect_chord(&self.chroma);
            self.key_est.update(&fresh);
            self.key = self.key_est.estimate();
        } else {
            for c in &mut self.chroma {
                *c *= 0.88;
            }
            self.chord = None;
        }

        // Envelope for the visibility graph — driven by the hottest mic.
        self.envelope.push_back(drive);
        while self.envelope.len() > ENVELOPE_LEN {
            self.envelope.pop_front();
        }
        let env: Vec<f32> = self.envelope.iter().copied().collect();
        self.vis = visibility::horizontal_visibility(&env);

        // Pitch tracking.
        let now = self.started.elapsed().as_secs_f64();
        let mut detected = false;
        if audible {
            let streams = analysis_streams(&self.beam, &self.window, &extra_tails);
            if let Some(freq) = self.detector.track_streams(&streams) {
                let m = match self.mode {
                    Mode::Guitar => tuning::nearest_string(freq, self.tuning, self.capo, self.a4),
                    Mode::Voice => tuning::nearest_note(freq, self.a4),
                };
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

        // Two-string Kalman decomposition: seeded by the detector's lock
        // plus a non-harmonic tracked peak; fed only the samples that
        // arrived since the last frame so the filter state is continuous.
        if self.mode == Mode::Guitar
            && audible
            && let Some(primary) = self.reading.as_ref().map(|r| r.freq)
        {
            let mut seeds = vec![primary];
            if let Some(second) =
                separate::pick_second(primary, self.tracker.tracks(), GUITAR_RANGE)
            {
                seeds.push(second);
            }
            self.separator.set_sources(&seeds);
            let fresh = (mac_total - self.sep_total).min(RING_LEN as u64) as usize;
            if fresh > 0 {
                self.separator.process(&self.window[RING_LEN - fresh..]);
            }
            self.duet = self.separator.readings();
            self.sep_residual = self.separator.residual_rms();
        } else {
            self.separator.clear();
            self.duet.clear();
            self.sep_residual = 0.0;
        }
        self.sep_total = mac_total;
    }

    /// Run one alignment-servo step per phone, build the delay-and-sum beam
    /// from every locked phone, and return the analysis tails of phones that
    /// could not be aligned this frame (they still fuse incoherently).
    fn align_and_beam(&mut self, filled: bool, mac_total: u64) -> Vec<Vec<f32>> {
        self.beam = None;
        self.beam_mics = 0;
        let mut extras: Vec<Vec<f32>> = Vec::new();
        if !filled {
            return extras;
        }

        let now = self.started.elapsed().as_secs_f64();
        let clients = self.remote.registry.clients();
        self.servos
            .retain(|id, _| clients.iter().any(|c| c.id == *id));

        // Mac reference window, taken REF_AGE back so the unlocked search
        // range covers phone latencies up to ~250 ms.
        let x_hi = RING_LEN - align::REF_AGE;
        let mac_ref = &self.window[x_hi - align::CORR_LEN..x_hi];
        let mac_rms = pitch::rms(mac_ref);

        // Pass 1: measure + update each servo; keep snapshots for pass 2.
        struct Frame {
            snap: Vec<f32>,
            snap_total: u64,
            offset: Option<f64>,
        }
        let mut frames: Vec<Frame> = Vec::new();
        for c in &clients {
            let servo = self.servos.entry(c.id).or_default();
            let (snap, snap_total) = c.snapshot();
            if snap.len() < align::CORR_LEN {
                servo.miss();
                continue;
            }
            let tail = &snap[snap.len() - align::CORR_LEN..];
            let dtot = snap_total as i64 - mac_total as i64;
            if mac_rms > RMS_GATE && pitch::rms(tail) > RMS_GATE {
                let (lo, hi) = match servo.offset_at(now).filter(|_| servo.locked()) {
                    Some(o) => {
                        let p = (o - dtot as f64 - align::REF_AGE as f64) as f32;
                        (p - align::TRACK_WINDOW, p + align::TRACK_WINDOW)
                    }
                    None => align::SEARCH,
                };
                match self.correlator.estimate(mac_ref, tail, lo, hi) {
                    Some(est) if est.confidence >= align::MIN_CONFIDENCE => {
                        servo.update(now, dtot as f64 + align::REF_AGE as f64 + est.lag as f64);
                    }
                    _ => servo.miss(),
                }
            } else {
                servo.miss();
            }
            let offset = servo.locked().then(|| servo.offset_at(now)).flatten();
            frames.push(Frame {
                snap,
                snap_total,
                offset,
            });
        }

        // Pass 2: the beam window must be old enough that every locked
        // phone has already received it.
        let max_lag = frames
            .iter()
            .filter_map(|f| {
                f.offset
                    .map(|o| o - (f.snap_total as i64 - mac_total as i64) as f64)
            })
            .fold(0.0f64, f64::max);
        let beam_age = ((max_lag as usize) + 128).clamp(256, RING_LEN - FFT_LEN);

        let mut aligned: Vec<Vec<f32>> = Vec::new();
        for f in frames {
            let win = f.offset.and_then(|o| {
                align::read_aligned(&f.snap, f.snap_total, mac_total, o, beam_age, FFT_LEN)
            });
            match win {
                Some(w) => aligned.push(w),
                None => extras.push(f.snap[f.snap.len() - FFT_LEN..].to_vec()),
            }
        }

        if !aligned.is_empty() {
            let lo = RING_LEN - beam_age - FFT_LEN;
            let seg = &self.window[lo..lo + FFT_LEN];
            let scale = 1.0 / (aligned.len() + 1) as f32;
            let mut beam: Vec<f32> = seg.iter().map(|&v| v * scale).collect();
            for w in &aligned {
                for (b, &v) in beam.iter_mut().zip(w) {
                    *b += v * scale;
                }
            }
            self.beam_mics = aligned.len() + 1;
            self.beam = Some(beam);
        }
        extras
    }

    fn update_spectrum_display(&mut self) {
        let bin_hz = self.detector.bin_hz() as f64;
        let mags = self.detector.mags();
        let n_bins = ((spectrum::SPECTRUM_MAX_HZ / bin_hz) as usize).min(mags.len());
        if self.spectrum.len() != n_bins {
            self.spectrum = (0..n_bins).map(|i| [i as f64 * bin_hz, -90.0]).collect();
        }
        for (point, &mag) in self.spectrum.iter_mut().zip(mags) {
            let db = (20.0 * (mag + 1e-9).log10()).clamp(-90.0, 0.0) as f64;
            point[1] = if db > point[1] {
                db
            } else {
                point[1] * 0.82 + db * 0.18
            };
        }
    }

    fn set_mode(&mut self, mode: Mode) {
        if self.mode != mode {
            self.mode = mode;
            self.detector.set_range(match mode {
                Mode::Guitar => GUITAR_RANGE,
                Mode::Voice => VOICE_RANGE,
            });
            self.history.clear();
            self.reading = None;
        }
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        ctx.input(|i| {
            if i.key_pressed(egui::Key::T) && self.mode == Mode::Guitar {
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

    fn controls(&mut self, ui: &mut egui::Ui) {
        // Phone portrait widths: wrap rows, shrink sliders, drop the
        // right-aligned status cluster.
        let narrow = ui.available_width() < 700.0;
        if narrow {
            ui.spacing_mut().slider_width = 100.0;
        }
        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new("♪ Flamenco Tuner Pro")
                    .color(FLAMENCO_RED)
                    .strong()
                    .size(16.0),
            );
            ui.add_space(10.0);

            ui.label(RichText::new("Mode").color(DIM));
            if ui
                .selectable_label(self.mode == Mode::Guitar, "Guitar")
                .clicked()
            {
                self.set_mode(Mode::Guitar);
            }
            if ui
                .selectable_label(self.mode == Mode::Voice, "Voice")
                .clicked()
            {
                self.set_mode(Mode::Voice);
            }

            if self.mode == Mode::Guitar {
                ui.add_space(10.0);
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
                ui.add_space(10.0);
                ui.label(RichText::new("Cejilla").color(DIM));
                ui.add(egui::Slider::new(&mut self.capo, 0..=9).integer());
            }

            ui.add_space(10.0);
            ui.label(RichText::new("A4").color(DIM));
            ui.add(
                egui::DragValue::new(&mut self.a4)
                    .speed(0.5)
                    .range(415.0..=466.0)
                    .suffix(" Hz"),
            );

            if narrow {
                tuner::level_pill(ui, self.level, "mic");
            } else {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    tuner::level_pill(ui, self.level, "mic");
                    ui.label(
                        RichText::new(&self.engine.device_name)
                            .color(DIM)
                            .size(11.0),
                    );
                });
            }
        });

        ui.add_space(4.0);

        // Filter row — sliders don't wrap mid-widget, so narrow screens
        // get one explicit row per filter instead.
        let mut cfg = self.filters;
        if narrow {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Filters").color(DIM));
                ui.checkbox(&mut cfg.hp_enabled, "High-pass");
                ui.add_enabled(
                    cfg.hp_enabled,
                    egui::Slider::new(&mut cfg.hp_cutoff, 20.0..=400.0)
                        .logarithmic(true)
                        .suffix(" Hz"),
                );
            });
            ui.horizontal(|ui| {
                ui.add_space(42.0);
                ui.checkbox(&mut cfg.lp_enabled, "Low-pass");
                ui.add_enabled(
                    cfg.lp_enabled,
                    egui::Slider::new(&mut cfg.lp_cutoff, 500.0..=20000.0)
                        .logarithmic(true)
                        .suffix(" Hz"),
                );
            });
        }
        if !narrow {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Filters").color(DIM));
                ui.checkbox(&mut cfg.hp_enabled, "High-pass");
                ui.add_enabled(
                    cfg.hp_enabled,
                    egui::Slider::new(&mut cfg.hp_cutoff, 20.0..=400.0)
                        .logarithmic(true)
                        .suffix(" Hz"),
                );
                ui.add_space(14.0);
                ui.checkbox(&mut cfg.lp_enabled, "Low-pass");
                ui.add_enabled(
                    cfg.lp_enabled,
                    egui::Slider::new(&mut cfg.lp_cutoff, 500.0..=20000.0)
                        .logarithmic(true)
                        .suffix(" Hz"),
                );

                // Phone mics exist only in the native app; never mention
                // them in the browser build.
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let clients = self.remote.registry.clients();
                    if !clients.is_empty() {
                        ui.add_space(14.0);
                        if ui
                            .button("🔊 calibrate")
                            .on_hover_text(
                                "Play a chirp from the Mac speakers so every phone \
                                 locks its time alignment (or just play a few notes)",
                            )
                            .clicked()
                        {
                            audio::play_chirp();
                        }
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if clients.is_empty() {
                            ui.label(
                                RichText::new("phone mics: scan QR in the Scope panel")
                                    .color(DIM)
                                    .size(11.0),
                            );
                        } else {
                            if clients.len() > 4 {
                                ui.label(
                                    RichText::new(format!("+{} more", clients.len() - 4))
                                        .color(DIM)
                                        .size(11.0),
                                );
                            }
                            for c in clients.iter().take(4).rev() {
                                let name = match self.servos.get(&c.id).filter(|s| s.locked()) {
                                    Some(s) => format!(
                                        "⊕ {} {:+.0}ppm",
                                        c.name(),
                                        s.drift_ppm(self.engine.sample_rate)
                                    ),
                                    None => c.name(),
                                };
                                tuner::level_pill(ui, c.level(), &name);
                            }
                            let (text, color) = if self.beam_mics >= 2 {
                                (format!("beamforming ×{}", self.beam_mics), GREEN)
                            } else {
                                (
                                    format!("{} phone mic(s) · seeking lock", clients.len()),
                                    FLAMENCO_GOLD,
                                )
                            };
                            ui.label(RichText::new(text).color(color).size(11.0));
                        }
                    });
                }
            });
        }

        if cfg != self.filters {
            self.filters = cfg;
            self.engine.set_filter_config(cfg);
        }
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
                egui::ScrollArea::vertical().show(ui, |ui| {
                    // Below this width (phones, narrow windows) every card
                    // stacks full-width in a single column.
                    let narrow = ui.available_width() < 700.0;
                    tuner::tuner_strip(ui, self, narrow);
                    ui.add_space(12.0);

                    // Every card below is an exact-size, clipped canvas —
                    // the dashboard geometry is fully pinned.
                    if narrow {
                        let full = ui.available_width();
                        spectrum::section(ui, self, egui::vec2(full, 310.0));
                        ui.add_space(12.0);
                        spectrogram::section(ui, self, egui::vec2(full, 310.0));
                        ui.add_space(12.0);
                        analysis::chroma_section(ui, self, egui::vec2(full, 230.0));
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            ui.add_space(12.0);
                            scope::section(ui, self, egui::vec2(full, 230.0));
                        }
                        ui.add_space(12.0);
                        analysis::history_section(ui, self, egui::vec2(full, 180.0));
                        ui.add_space(12.0);
                        analysis::waveform_section(ui, self, egui::vec2(full, 180.0));
                        ui.add_space(12.0);
                        analysis::visibility_section(ui, self, egui::vec2(full, 180.0));
                    } else {
                        let half = (ui.available_width() - 12.0) / 2.0;
                        ui.horizontal(|ui| {
                            spectrum::section(ui, self, egui::vec2(half, 310.0));
                            ui.add_space(12.0);
                            spectrogram::section(ui, self, egui::vec2(half, 310.0));
                        });
                        ui.add_space(12.0);

                        ui.horizontal(|ui| {
                            // Native pairs the chromagram with the vector
                            // scope; the web build gives it the full row.
                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                analysis::chroma_section(ui, self, egui::vec2(half, 230.0));
                                ui.add_space(12.0);
                                scope::section(ui, self, egui::vec2(half, 230.0));
                            }
                            #[cfg(target_arch = "wasm32")]
                            analysis::chroma_section(
                                ui,
                                self,
                                egui::vec2(half * 2.0 + 12.0, 230.0),
                            );
                        });
                        ui.add_space(12.0);

                        let third = (ui.available_width() - 24.0) / 3.0;
                        ui.horizontal(|ui| {
                            analysis::history_section(ui, self, egui::vec2(third, 180.0));
                            ui.add_space(12.0);
                            analysis::waveform_section(ui, self, egui::vec2(third, 180.0));
                            ui.add_space(12.0);
                            analysis::visibility_section(ui, self, egui::vec2(third, 180.0));
                        });
                    }
                });
            });

        ctx.request_repaint_after(Duration::from_millis(50));
    }
}

impl TunerApp {
    pub(crate) fn window_samples(&self) -> &[f32] {
        &self.window
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn stereo_pairs(&self) -> Option<Vec<(f32, f32)>> {
        // Phones connected: pair local mono with the loudest phone.
        if let Some(client) = self
            .remote
            .registry
            .clients()
            .into_iter()
            .max_by(|a, b| a.level().total_cmp(&b.level()))
        {
            let remote = client.ring.lock().unwrap();
            let n = remote.buf.len().min(2048);
            if n > 256 {
                let local = &self.window[RING_LEN - n..];
                return Some(
                    remote
                        .buf
                        .iter()
                        .rev()
                        .take(n)
                        .rev()
                        .zip(local)
                        .map(|(&r, &l)| (l, r))
                        .collect(),
                );
            }
        }
        // Otherwise a true stereo input device, if present.
        if let Some(stereo) = &self.engine.stereo_ring {
            let buf = stereo.lock().unwrap();
            if buf.len() > 256 {
                return Some(buf.iter().rev().take(2048).rev().copied().collect());
            }
        }
        None
    }
}

/// Analysis windows: the coherent beam when one exists (it already contains
/// the Mac mic), otherwise the raw Mac window; plus the tails of any phones
/// not in the beam, fused incoherently.
fn analysis_streams<'a>(
    beam: &'a Option<Vec<f32>>,
    window: &'a [f32],
    extra_tails: &'a [Vec<f32>],
) -> Vec<&'a [f32]> {
    let primary: &[f32] = beam.as_deref().unwrap_or(&window[RING_LEN - FFT_LEN..]);
    let mut streams = vec![primary];
    streams.extend(extra_tails.iter().map(|t| t.as_slice()));
    streams
}

/// Green within the in-tune window, amber up to 15¢, red beyond.
pub(crate) fn zone_color(cents: f32) -> Color32 {
    let c = cents.abs();
    if c <= IN_TUNE_CENTS {
        GREEN
    } else if c <= 15.0 {
        FLAMENCO_GOLD
    } else {
        FLAMENCO_RED
    }
}
