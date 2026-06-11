//! Cross-stream time alignment for beamforming.
//!
//! Each phone free-runs on its own ADC clock and reaches us through a
//! jittery browser/WebSocket path, so coherent (delay-and-sum) combination
//! needs continuous alignment, not one-shot calibration:
//!
//! 1. Both the Mac ring and every phone ring carry an absolute sample
//!    counter. The offset `O = phone_pos − mac_pos` of the *same audio* in
//!    those coordinates is constant up to clock drift — unlike lag measured
//!    from ring ends, it does not jump with WebSocket burst arrivals.
//! 2. `Correlator` (GCC-PHAT) measures `O` from live signal. Sustained
//!    guitar tones are periodic, so their correlation is comb-ambiguous;
//!    the confidence gate only accepts transient-rich frames (plucks,
//!    rasgueado, the calibration chirp).
//! 3. `Servo` — an alpha-beta tracker per phone — smooths `O` and estimates
//!    its rate (the relative clock drift, samples/sec), extrapolating
//!    through silence. A software PLL on the offset.
//! 4. `read_aligned` extracts a phone window at fractional-sample offset
//!    for delay-and-sum with the Mac window.

use std::f32::consts::PI;
use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

/// Samples of each stream compared per measurement (~171 ms at 48 kHz).
pub const CORR_LEN: usize = 8192;
/// FFT size: 2× window so circular correlation acts linear.
const CFFT: usize = 2 * CORR_LEN;
/// The Mac reference window is taken this many samples back from "now" so
/// one search range covers phone latencies from ~0 up to ~250 ms.
pub const REF_AGE: usize = 4096;
/// Lag search range relative to the reference window when unlocked.
pub const SEARCH: (f32, f32) = (-4090.0, 8000.0);
/// Lag search half-width around the prediction once locked.
pub const TRACK_WINDOW: f32 = 900.0;
/// Peak must beat the runner-up by this factor — rejects the comb
/// ambiguity of sustained periodic tones.
pub const MIN_CONFIDENCE: f32 = 1.3;

pub struct DelayEstimate {
    /// Lag of the delayed stream behind the reference window, in samples
    /// (sub-sample, via parabolic interpolation).
    pub lag: f32,
    /// Peak height over the strongest competing peak.
    pub confidence: f32,
}

/// GCC-PHAT cross-correlator. PHAT whitening keeps only phase, which
/// sharpens the peak and makes it robust to spectral coloration.
pub struct Correlator {
    fwd: Arc<dyn Fft<f32>>,
    inv: Arc<dyn Fft<f32>>,
    hann: Vec<f32>,
}

impl Correlator {
    pub fn new() -> Self {
        let mut planner = FftPlanner::new();
        let hann = (0..CORR_LEN)
            .map(|i| {
                let x = i as f32 / (CORR_LEN - 1) as f32;
                0.5 - 0.5 * (2.0 * PI * x).cos()
            })
            .collect();
        Self {
            fwd: planner.plan_fft_forward(CFFT),
            inv: planner.plan_fft_inverse(CFFT),
            hann,
        }
    }

    /// Find where `delayed` lags `reference` (both CORR_LEN samples),
    /// searching lags in [lo, hi]. Positive lag = `delayed` is older.
    pub fn estimate(&self, reference: &[f32], delayed: &[f32], lo: f32, hi: f32) -> Option<DelayEstimate> {
        debug_assert_eq!(reference.len(), CORR_LEN);
        debug_assert_eq!(delayed.len(), CORR_LEN);
        // Hann window: without it the slice edges act like clicks and the
        // rectangular envelope biases the peak toward zero lag.
        let pad = |s: &[f32]| -> Vec<Complex<f32>> {
            s.iter()
                .zip(&self.hann)
                .map(|(&v, &w)| Complex::new(v * w, 0.0))
                .chain(std::iter::repeat_n(Complex::new(0.0, 0.0), CFFT - CORR_LEN))
                .collect()
        };
        let mut x = pad(reference);
        let mut y = pad(delayed);
        self.fwd.process(&mut x);
        self.fwd.process(&mut y);

        // Whitened cross-spectrum (GCC-PHAT), restricted to bins where BOTH
        // streams carry real energy. Whitening empty bins would flood the
        // correlation with random phase noise; including them only in one
        // stream adds nothing. With this gate a sustained periodic tone
        // yields near-equal comb peaks — low confidence — instead of a
        // confidently wrong one, while broadband transients stay sharp.
        let floor = |s: &[Complex<f32>]| {
            3e-3 * s.iter().map(|c| c.norm()).fold(0.0f32, f32::max)
        };
        let (fx, fy) = (floor(&x), floor(&y));
        let mut r: Vec<Complex<f32>> = x
            .iter()
            .zip(&y)
            .map(|(a, b)| {
                if a.norm() < fx || b.norm() < fy {
                    return Complex::new(0.0, 0.0);
                }
                let p = a.conj() * b;
                let n = p.norm();
                if n > 1e-12 { p / n } else { Complex::new(0.0, 0.0) }
            })
            .collect();
        self.inv.process(&mut r);
        let c = |tau: isize| r[tau.rem_euclid(CFFT as isize) as usize].re;

        let lo = lo.max(-(CORR_LEN as f32) + 2.0).ceil() as isize;
        let hi = hi.min(CORR_LEN as f32 - 2.0).floor() as isize;
        if lo >= hi {
            return None;
        }
        let best = (lo..=hi).max_by(|&a, &b| c(a).total_cmp(&c(b)))?;
        let peak = c(best);
        if peak <= 0.0 {
            return None;
        }
        let runner = (lo..=hi)
            .filter(|t| (t - best).abs() > 64)
            .map(c)
            .fold(f32::MIN, f32::max);
        let confidence = if runner > 1e-12 { peak / runner } else { 99.0 };

        // Sub-sample refinement.
        let (a, b, d) = (c(best - 1), peak, c(best + 1));
        let denom = a - 2.0 * b + d;
        let delta = if denom.abs() > f32::EPSILON {
            (0.5 * (a - d) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        Some(DelayEstimate {
            lag: best as f32 + delta,
            confidence: confidence.min(99.0),
        })
    }
}

/// Alpha-beta tracker on the absolute offset of one phone: smooths the
/// GCC-PHAT measurements and learns the relative clock drift so the offset
/// can be extrapolated through silence.
#[derive(Default)]
pub struct Servo {
    /// Offset O = phone_pos − mac_pos (samples) at `last`.
    offset: f64,
    /// dO/dt — relative clock drift, samples per second.
    rate: f64,
    last: Option<f64>,
    quality: f32,
    outliers: u32,
}

const ALPHA: f64 = 0.35;
const BETA: f64 = 0.05;
const LOCK_QUALITY: f32 = 0.3;
/// Prediction error beyond this is treated as a false (comb) peak unless
/// it persists.
const OUTLIER_SAMPLES: f64 = 400.0;

impl Servo {
    pub fn locked(&self) -> bool {
        self.last.is_some() && self.quality >= LOCK_QUALITY
    }

    /// Best offset estimate at time `t` (seconds, same clock as `update`).
    pub fn offset_at(&self, t: f64) -> Option<f64> {
        self.last.map(|t0| self.offset + self.rate * (t - t0))
    }

    /// Relative clock drift in parts per million. Shown in the native
    /// phone-mic pills; wasm builds have no phones.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn drift_ppm(&self, sample_rate: f32) -> f64 {
        self.rate / sample_rate as f64 * 1e6
    }

    pub fn update(&mut self, t: f64, measured: f64) {
        let Some(t0) = self.last else {
            self.offset = measured;
            self.last = Some(t);
            self.quality = 0.4;
            return;
        };
        let dt = (t - t0).max(1e-3);
        let predicted = self.offset + self.rate * dt;
        let e = measured - predicted;
        if e.abs() > OUTLIER_SAMPLES && self.locked() {
            self.outliers += 1;
            if self.outliers >= 4 {
                // The "outlier" is consistent — our lock was the false peak.
                *self = Servo {
                    offset: measured,
                    last: Some(t),
                    quality: 0.4,
                    ..Servo::default()
                };
            }
            return;
        }
        self.outliers = 0;
        self.offset = predicted + ALPHA * e;
        if dt < 2.0 {
            // Rate only learns from closely spaced measurements; across a
            // long gap the offset innovation says little about drift.
            self.rate = (self.rate + BETA * e / dt).clamp(-25.0, 25.0);
        }
        self.last = Some(t);
        self.quality = (self.quality + 0.15).min(1.0);
    }

    /// No usable measurement this frame: decay the lock slowly — drift
    /// extrapolation stays good for tens of seconds.
    pub fn miss(&mut self) {
        self.quality = (self.quality - 0.002).max(0.0);
    }
}

/// Read `len` phone samples aligned with the Mac window covering ages
/// [age_end, age_end + len) (samples before the Mac ring end), oldest
/// first, linearly interpolating the fractional offset. `None` if the
/// required audio is outside the snapshot.
pub fn read_aligned(
    snapshot: &[f32],
    snap_total: u64,
    mac_total: u64,
    offset: f64,
    age_end: usize,
    len: usize,
) -> Option<Vec<f32>> {
    let snap_len = snapshot.len();
    // Index of Mac age `a` in the snapshot: the audio at Mac position
    // (mac_total − 1 − a) sits at phone position (+ offset), i.e. snapshot
    // index (phone position − snap_total + snap_len).
    let base = mac_total as f64 + offset - snap_total as f64 + snap_len as f64 - 1.0;
    let mut out = Vec::with_capacity(len);
    for j in 0..len {
        let a = (age_end + len - 1 - j) as f64;
        let i = base - a;
        let i0 = i.floor();
        if i0 < 0.0 || i0 as usize + 1 >= snap_len {
            return None;
        }
        let f = (i - i0) as f32;
        let i0 = i0 as usize;
        out.push(snapshot[i0] * (1.0 - f) + snapshot[i0 + 1] * f);
    }
    Some(out)
}

/// Calibration chirp: 0.8 s logarithmic sweep with faded edges. Wideband,
/// aperiodic — an unambiguous GCC-PHAT bootstrap even in a live room.
/// Only the native calibrate button plays it; wasm has no phones to align.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub fn chirp(sample_rate: f32) -> Vec<f32> {
    const DUR: f32 = 0.8;
    const F0: f32 = 150.0;
    const F1: f32 = 4500.0;
    const FADE: f32 = 0.02;
    let n = (DUR * sample_rate) as usize;
    let k = (F1 / F0).ln() / DUR;
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate;
            let phase = 2.0 * PI * F0 * ((k * t).exp() - 1.0) / k;
            let edge = (t / FADE).min((DUR - t) / FADE).clamp(0.0, 1.0);
            let fade = 0.5 - 0.5 * (PI * edge).cos();
            0.4 * phase.sin() * fade
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic broadband-ish test signal: LCG noise smoothed a touch
    /// so fractional-delay interpolation is meaningful.
    fn noise(seed: u32, n: usize) -> Vec<f32> {
        let mut x = seed;
        let raw: Vec<f32> = (0..n + 2)
            .map(|_| {
                x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                (x as f32 / u32::MAX as f32) - 0.5
            })
            .collect();
        raw.windows(3).map(|w| (w[0] + w[1] + w[2]) / 3.0).collect()
    }

    /// `n` samples of `source` read at fractional positions start, start+1, …
    fn frac_window(source: &[f32], start: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let p = start + i as f32;
                let p0 = p.floor();
                let f = p - p0;
                let p0 = p0 as usize;
                source[p0] * (1.0 - f) + source[p0 + 1] * f
            })
            .collect()
    }

    #[test]
    fn correlator_finds_known_delay() {
        let src = noise(7, 40000);
        let d = 1234.4f32;
        let x = &src[20000..20000 + CORR_LEN];
        // y(i) = src(20000 + i − d) = x(i − d) → lag = d.
        let y = frac_window(&src, 20000.0 - d, CORR_LEN);
        let est = Correlator::new()
            .estimate(x, &y, -200.0, 4000.0)
            .expect("no estimate");
        assert!((est.lag - d).abs() < 0.3, "lag {} expected {}", est.lag, d);
        assert!(est.confidence > MIN_CONFIDENCE, "confidence {}", est.confidence);
    }

    #[test]
    fn correlator_is_never_confidently_wrong_on_periodic_tones() {
        // A sustained tone correlates at every period (comb ambiguity).
        // The gate's contract: if confidence clears the threshold, the lag
        // must be the true one — a wrong comb peak must score below it.
        let sr = 48000.0;
        for true_lag in [300usize, 1100, 2500] {
            let tone: Vec<f32> = (0..CORR_LEN + 3000)
                .map(|i| (2.0 * PI * 110.0 * i as f32 / sr).sin())
                .collect();
            let x = &tone[2800..2800 + CORR_LEN];
            let y = &tone[2800 - true_lag..2800 - true_lag + CORR_LEN];
            if let Some(est) = Correlator::new().estimate(x, y, -200.0, 4000.0)
                && est.confidence >= MIN_CONFIDENCE
            {
                assert!(
                    (est.lag - true_lag as f32).abs() < 2.0,
                    "confident ({:.2}) but wrong: lag {} expected {}",
                    est.confidence,
                    est.lag,
                    true_lag
                );
            }
        }
    }

    #[test]
    fn servo_tracks_offset_and_drift() {
        // O(t) = 100 + 2.4t (50 ppm at 48 kHz) with ±0.5-sample noise,
        // measured 4×/s for 10 s, then 15 s of silence.
        let mut servo = Servo::default();
        let mut x = 99u32;
        let mut rand = move || {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            (x as f64 / u32::MAX as f64) - 0.5
        };
        let mut t = 0.0;
        while t < 10.0 {
            servo.update(t, 100.0 + 2.4 * t + rand());
            t += 0.25;
        }
        assert!(servo.locked());
        let p10 = servo.offset_at(10.0).unwrap();
        assert!((p10 - 124.0).abs() < 2.0, "offset at t=10: {p10}");
        let drift = servo.drift_ppm(48000.0);
        assert!((drift - 50.0).abs() < 15.0, "drift {drift} ppm");
        // Extrapolation through silence.
        let p25 = servo.offset_at(25.0).unwrap();
        assert!((p25 - 160.0).abs() < 10.0, "offset at t=25: {p25}");
    }

    #[test]
    fn servo_survives_isolated_false_peak() {
        let mut servo = Servo::default();
        for i in 0..20 {
            servo.update(i as f64 * 0.25, 500.0);
        }
        servo.update(5.25, 1200.0); // comb peak one period off
        let p = servo.offset_at(5.5).unwrap();
        assert!((p - 500.0).abs() < 2.0, "false peak moved the lock: {p}");
    }

    #[test]
    fn aligned_delay_and_sum_improves_snr() {
        // Mac hears the source clean-ish; three phones hear it with
        // different latencies/offsets and independent noise. After
        // measurement + aligned summation, the residual error vs the clean
        // source must drop well below a single noisy stream's.
        let n_src = 60000;
        let src = noise(3, n_src);
        let mac_total = 50000u64;
        let mac_ring = &src[50000 - 16384..50000];

        let correlator = Correlator::new();
        let mut aligned: Vec<Vec<f32>> = Vec::new();
        let age_end = 5000;
        let want = 4096;
        let clean = &mac_ring[16384 - age_end - want..16384 - age_end];

        for (i, true_delay) in [803.3f32, 2241.7, 411.9].into_iter().enumerate() {
            let noise_i = noise(100 + i as u32, 16384);
            // Phone stream: 49000 samples received; sample p holds
            // src(p − delay) → offset O = +delay. Only the ring snapshot
            // (last 16384 samples) is materialized, plus the phone's own noise.
            let snap_total = 49000u64;
            let snap: Vec<f32> =
                frac_window(&src, (49000 - 16384) as f32 - true_delay, 16384)
                    .iter()
                    .zip(&noise_i)
                    .map(|(&s, &m)| s + 0.5 * m)
                    .collect();

            // Measure: Mac window at REF_AGE vs phone tail.
            let x = &mac_ring[16384 - REF_AGE - CORR_LEN..16384 - REF_AGE];
            let y = &snap[16384 - CORR_LEN..];
            let est = correlator
                .estimate(x, y, SEARCH.0, SEARCH.1)
                .expect("no estimate");
            assert!(est.confidence > MIN_CONFIDENCE);
            let dtot = snap_total as i64 - mac_total as i64;
            let offset = dtot as f64 + REF_AGE as f64 + est.lag as f64;
            let w = read_aligned(&snap, snap_total, mac_total, offset, age_end, want)
                .expect("aligned read out of range");
            aligned.push(w);
        }

        let single_err = rms_diff(&aligned[0], clean);
        let beam: Vec<f32> = (0..want)
            .map(|i| aligned.iter().map(|w| w[i]).sum::<f32>() / aligned.len() as f32)
            .collect();
        let beam_err = rms_diff(&beam, clean);
        assert!(
            beam_err < single_err * 0.75,
            "beam err {beam_err} vs single {single_err}"
        );
    }

    fn rms_diff(a: &[f32], b: &[f32]) -> f32 {
        let s: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        (s / a.len() as f32).sqrt()
    }

    #[test]
    fn chirp_is_bounded_and_fades() {
        let c = chirp(48000.0);
        assert_eq!(c.len(), 38400);
        assert!(c.iter().all(|v| v.abs() <= 0.401));
        assert!(c[0].abs() < 1e-3 && c[c.len() - 1].abs() < 0.02);
    }
}
