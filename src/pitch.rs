//! Pitch detection robust to sympathetic resonance.
//!
//! Plain YIN (de Cheveigné & Kawahara, 2002) picks the first CMNDF dip
//! below an absolute threshold. On a real guitar that fails when other
//! strings ring sympathetically — pluck A2 and the open E2 sings along,
//! polluting the dip at A2's period and creating competing dips.
//!
//! This detector instead:
//!  1. collects ALL CMNDF local minima as candidates (looser threshold),
//!  2. scores each candidate by the energy of its harmonic series in the
//!     FFT spectrum — the plucked string's series dominates the
//!     sympathetic one's, so the right fundamental wins,
//!  3. refines the winner by magnitude-weighted parabolic interpolation
//!     over harmonics 1–5 (an 8192-point FFT gives 5.9 Hz bins at 48 kHz,
//!     enough to resolve E2 from A2, 27.6 Hz apart),
//!  4. stabilizes across frames with a median filter: 3 of the last 5
//!     estimates must agree within 25 cents before a pitch is reported.

use std::collections::VecDeque;
use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

/// Spectrum window: ~171 ms at 48 kHz, 5.9 Hz bins.
pub const FFT_LEN: usize = 8192;
/// YIN window (most recent samples): ~85 ms at 48 kHz, keeps tracking responsive.
const YIN_LEN: usize = 4096;

/// Guitar fundamentals: low D2 (73.4 Hz, rondeña) down to 55 for margin,
/// up past a high E string with a cejilla at the 9th fret.
const FMIN: f32 = 55.0;
const FMAX: f32 = 600.0;

/// CMNDF ceiling for candidate dips. Loose on purpose: resonance from
/// other strings raises the dip of the true pitch well above the classic
/// 0.1–0.15 threshold.
const CAND_THRESHOLD: f32 = 0.35;
/// Frames must agree within this many cents to count as stable.
const STABLE_CENTS: f32 = 25.0;
const RECENT_FRAMES: usize = 5;
const STABLE_QUORUM: usize = 3;

pub struct Detector {
    sample_rate: f32,
    fft: Arc<dyn Fft<f32>>,
    hann: Vec<f32>,
    /// Linear FFT magnitudes (amplitude-normalized), FFT_LEN/2 bins.
    mags: Vec<f32>,
    /// Recent raw frame estimates, for the median stabilizer.
    recent: VecDeque<f32>,
    /// Last reported pitch, for the continuity bonus.
    last: Option<f32>,
}

impl Detector {
    pub fn new(sample_rate: f32) -> Self {
        let fft = FftPlanner::new().plan_fft_forward(FFT_LEN);
        let hann: Vec<f32> = (0..FFT_LEN)
            .map(|i| {
                let x = i as f32 / (FFT_LEN - 1) as f32;
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * x).cos()
            })
            .collect();
        Self {
            sample_rate,
            fft,
            hann,
            mags: vec![0.0; FFT_LEN / 2],
            recent: VecDeque::with_capacity(RECENT_FRAMES + 1),
            last: None,
        }
    }

    pub fn bin_hz(&self) -> f32 {
        self.sample_rate / FFT_LEN as f32
    }

    /// Latest linear FFT magnitudes (for display).
    pub fn mags(&self) -> &[f32] {
        &self.mags
    }

    /// Update the spectrum from a full FFT_LEN window of samples.
    pub fn analyze_spectrum(&mut self, samples: &[f32]) {
        debug_assert_eq!(samples.len(), FFT_LEN);
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let mut buf: Vec<Complex<f32>> = samples
            .iter()
            .zip(&self.hann)
            .map(|(&s, &w)| Complex::new((s - mean) * w, 0.0))
            .collect();
        self.fft.process(&mut buf);
        // Amplitude normalization: 2/N for the one-sided spectrum, /0.5 for
        // the Hann window's coherent gain.
        let norm = 4.0 / FFT_LEN as f32;
        for (m, b) in self.mags.iter_mut().zip(&buf) {
            *m = b.norm() * norm;
        }
    }

    /// Track the pitch in this window. Call `analyze_spectrum` first.
    /// Returns a stabilized frequency once enough frames agree.
    pub fn track(&mut self, samples: &[f32]) -> Option<f32> {
        let raw = self.estimate(&samples[samples.len() - YIN_LEN..]);
        match raw {
            Some(f) => {
                self.recent.push_back(f);
                if self.recent.len() > RECENT_FRAMES {
                    self.recent.pop_front();
                }
            }
            None => {
                self.recent.pop_front();
            }
        }
        let stable = self.stabilized();
        if stable.is_some() {
            self.last = stable;
        }
        stable
    }

    /// Decay the stabilizer during near-silence (gated frames).
    pub fn relax(&mut self) {
        self.recent.pop_front();
    }

    pub fn reset(&mut self) {
        self.recent.clear();
        self.last = None;
    }

    /// Median-based agreement: at least STABLE_QUORUM of the recent raw
    /// estimates within STABLE_CENTS of their median.
    fn stabilized(&self) -> Option<f32> {
        if self.recent.len() < STABLE_QUORUM {
            return None;
        }
        let mut sorted: Vec<f32> = self.recent.iter().copied().collect();
        sorted.sort_by(f32::total_cmp);
        let median = sorted[sorted.len() / 2];
        let close: Vec<f32> = sorted
            .into_iter()
            .filter(|&f| cents_between(f, median).abs() < STABLE_CENTS)
            .collect();
        if close.len() >= STABLE_QUORUM {
            Some(close.iter().sum::<f32>() / close.len() as f32)
        } else {
            None
        }
    }

    /// Single-frame estimate: YIN candidates → spectral scoring → refinement.
    fn estimate(&self, samples: &[f32]) -> Option<f32> {
        let candidates = yin_candidates(samples, self.sample_rate);
        let (best, score) = candidates
            .into_iter()
            .map(|c| {
                // Harmonic energy, discounted by dip quality; a continuity
                // bonus keeps the lock through brief interference.
                let mut score = self.harmonic_score(c.freq) * (1.2 - c.cmndf);
                if let Some(last) = self.last
                    && cents_between(c.freq, last).abs() < 80.0
                {
                    score *= 1.3;
                }
                (c, score)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))?;
        if score <= 0.0 {
            return None;
        }

        let refined = self.refine(best.freq).unwrap_or(best.freq);
        // Trust the spectral refinement only if it agrees with YIN.
        Some(if cents_between(refined, best.freq).abs() < 40.0 {
            refined
        } else {
            best.freq
        })
    }

    /// Peak magnitude within ±1.5 bins of `f` (tolerates string
    /// inharmonicity pushing harmonics slightly sharp).
    fn mag_near(&self, f: f32) -> f32 {
        let bin = f / self.bin_hz();
        let lo = (bin - 1.5).floor().max(0.0) as usize;
        let hi = ((bin + 1.5).ceil() as usize).min(self.mags.len() - 1);
        self.mags[lo..=hi].iter().fold(0.0f32, |m, &x| m.max(x))
    }

    /// Energy of the harmonic series of `f`, weighted 1/k so a true
    /// fundamental beats its own subharmonic (which only hits every
    /// second partial).
    fn harmonic_score(&self, f: f32) -> f32 {
        let limit = self.sample_rate * 0.45;
        (1..=6)
            .map(|k| {
                let fk = f * k as f32;
                if fk < limit {
                    self.mag_near(fk) / k as f32
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Refine `f0` by parabolic interpolation of the log-magnitude peaks
    /// at harmonics 1–5, weighted by peak magnitude. Each harmonic k
    /// estimates the fundamental with k× the bin resolution.
    fn refine(&self, f0: f32) -> Option<f32> {
        let bin_hz = self.bin_hz();
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for k in 1..=5u32 {
            let fk = f0 * k as f32;
            let center = (fk / bin_hz).round() as isize;
            let lo = (center - 2).max(1) as usize;
            let hi = ((center + 2) as usize).min(self.mags.len() - 2);
            if lo >= hi {
                break;
            }
            let (mut pi, mut pm) = (lo, 0.0f32);
            for i in lo..=hi {
                if self.mags[i] > pm {
                    pm = self.mags[i];
                    pi = i;
                }
            }
            // Must be a genuine local peak with usable energy.
            if pm <= 1e-7 || self.mags[pi - 1] > pm || self.mags[pi + 1] > pm {
                continue;
            }
            let (a, b, c) = (
                self.mags[pi - 1].max(1e-12).ln(),
                pm.ln(),
                self.mags[pi + 1].max(1e-12).ln(),
            );
            let denom = a - 2.0 * b + c;
            let delta = if denom.abs() > f32::EPSILON {
                (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            let est = (pi as f32 + delta) * bin_hz / k as f32;
            // Reject harmonics captured by another string's partial.
            if cents_between(est, f0).abs() < 30.0 {
                num += pm * est;
                den += pm;
            }
        }
        (den > 0.0).then(|| num / den)
    }
}

struct Candidate {
    freq: f32,
    cmndf: f32,
}

/// All local minima of the cumulative mean normalized difference function
/// below CAND_THRESHOLD, refined to sub-sample precision.
fn yin_candidates(samples: &[f32], sample_rate: f32) -> Vec<Candidate> {
    let n = samples.len();
    let w = n / 2;
    let tau_max = ((sample_rate / FMIN) as usize).min(w - 1);
    let tau_min = ((sample_rate / FMAX) as usize).max(2);
    if tau_max <= tau_min + 2 {
        return Vec::new();
    }

    let mean = samples.iter().sum::<f32>() / n as f32;
    let s: Vec<f32> = samples.iter().map(|&x| x - mean).collect();

    // Difference function d(tau) over a window of w samples.
    let mut d = vec![0.0f32; tau_max + 1];
    for (tau, dt) in d.iter_mut().enumerate().skip(1) {
        let mut sum = 0.0f32;
        for i in 0..w {
            let delta = s[i] - s[i + tau];
            sum += delta * delta;
        }
        *dt = sum;
    }

    // Cumulative mean normalized difference function.
    let mut cmndf = vec![1.0f32; tau_max + 1];
    let mut running = 0.0f32;
    for tau in 1..=tau_max {
        running += d[tau];
        if running > 0.0 {
            cmndf[tau] = d[tau] * tau as f32 / running;
        }
    }

    // Every local minimum below the candidate threshold.
    let mut candidates = Vec::new();
    for tau in tau_min..tau_max {
        if cmndf[tau] < CAND_THRESHOLD
            && cmndf[tau] <= cmndf[tau - 1]
            && cmndf[tau] <= cmndf[tau + 1]
        {
            // Parabolic interpolation for sub-sample precision.
            let (a, b, c) = (cmndf[tau - 1], cmndf[tau], cmndf[tau + 1]);
            let denom = a - 2.0 * b + c;
            let delta = if denom.abs() > f32::EPSILON {
                (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            let freq = sample_rate / (tau as f32 + delta);
            if (FMIN..=FMAX).contains(&freq) {
                candidates.push(Candidate {
                    freq,
                    cmndf: cmndf[tau],
                });
            }
        }
    }

    // Keep the strongest few; drop near-duplicates (within 10 cents).
    candidates.sort_by(|a, b| a.cmndf.total_cmp(&b.cmndf));
    let mut kept: Vec<Candidate> = Vec::new();
    for c in candidates {
        if kept.len() >= 6 {
            break;
        }
        if kept
            .iter()
            .all(|k| cents_between(c.freq, k.freq).abs() > 10.0)
        {
            kept.push(c);
        }
    }
    kept
}

fn cents_between(a: f32, b: f32) -> f32 {
    1200.0 * (a / b).log2()
}

/// Root-mean-square level, used as a noise gate.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// A plucked-string-like tone: fundamental plus decaying harmonics.
    pub fn string_tone(freq: f32, amp: f32, phase: f32, sr: f32, n: usize) -> Vec<f32> {
        let harmonics = [1.0f32, 0.45, 0.25, 0.12, 0.06];
        (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                harmonics
                    .iter()
                    .enumerate()
                    .map(|(k, &h)| {
                        amp * h
                            * (2.0 * std::f32::consts::PI * freq * (k + 1) as f32 * t + phase).sin()
                    })
                    .sum()
            })
            .collect()
    }

    pub fn mix(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    /// Feed the same frame until the stabilizer reports.
    pub fn detect(samples: &[f32]) -> Option<f32> {
        let mut det = Detector::new(48000.0);
        let mut out = None;
        for _ in 0..RECENT_FRAMES {
            det.analyze_spectrum(samples);
            out = det.track(samples);
        }
        out
    }

    fn assert_cents(detected: f32, target: f32, tol: f32, label: &str) {
        let cents = cents_between(detected, target);
        assert!(
            cents.abs() < tol,
            "{label}: detected {detected:.2} Hz, target {target:.2} Hz ({cents:+.1} cents, tol ±{tol})"
        );
    }

    #[test]
    fn pure_strings_within_a_cent_and_a_half() {
        // E2, A2, D3, G3, B3, E4, plus rondeña low D2
        for target in [82.41, 110.0, 146.83, 196.0, 246.94, 329.63, 73.42] {
            let samples = string_tone(target, 0.5, 0.0, 48000.0, FFT_LEN);
            let f = detect(&samples).expect("no pitch detected");
            assert_cents(f, target, 1.5, "pure string");
        }
    }

    #[test]
    fn a_string_with_sympathetic_low_e() {
        // Pluck A2; the open E2 rings along at 30% amplitude.
        let a2 = string_tone(110.0, 0.5, 0.0, 48000.0, FFT_LEN);
        let e2 = string_tone(82.41, 0.15, 1.1, 48000.0, FFT_LEN);
        let f = detect(&mix(&a2, &e2)).expect("no pitch detected");
        assert_cents(f, 110.0, 3.0, "A2 with E2 resonance");
    }

    #[test]
    fn low_e_with_sympathetic_a() {
        // Pluck E2; the A2 string rings along.
        let e2 = string_tone(82.41, 0.5, 0.3, 48000.0, FFT_LEN);
        let a2 = string_tone(110.0, 0.15, 2.0, 48000.0, FFT_LEN);
        let f = detect(&mix(&e2, &a2)).expect("no pitch detected");
        assert_cents(f, 82.41, 3.0, "E2 with A2 resonance");
    }

    #[test]
    fn detuned_a_string_with_resonance_reads_the_detune() {
        // A2 tuned 25 cents flat, E2 ringing along: the tuner must report
        // the detuned pitch, not snap to the nominal note.
        let flat_a = 110.0 * 2.0f32.powf(-25.0 / 1200.0);
        let a2 = string_tone(flat_a, 0.5, 0.0, 48000.0, FFT_LEN);
        let e2 = string_tone(82.41, 0.15, 0.8, 48000.0, FFT_LEN);
        let f = detect(&mix(&a2, &e2)).expect("no pitch detected");
        assert_cents(f, flat_a, 4.0, "detuned A2 with E2 resonance");
    }

    #[test]
    fn strong_second_harmonic_is_not_an_octave_error() {
        // Some plucks ring with the 2nd harmonic louder than the fundamental.
        let n = FFT_LEN;
        let sr = 48000.0;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                let w = 2.0 * std::f32::consts::PI * 110.0 * t;
                0.3 * w.sin() + 0.5 * (2.0 * w).sin() + 0.2 * (3.0 * w).sin()
            })
            .collect();
        let f = detect(&samples).expect("no pitch detected");
        assert_cents(f, 110.0, 3.0, "dominant 2nd harmonic");
    }

    #[test]
    fn rejects_silence_and_noise() {
        assert!(detect(&vec![0.0; FFT_LEN]).is_none());
        // Deterministic pseudo-noise
        let mut x = 12345u32;
        let noise: Vec<f32> = (0..FFT_LEN)
            .map(|_| {
                x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                (x as f32 / u32::MAX as f32) - 0.5
            })
            .collect();
        assert!(detect(&noise).is_none());
    }
}

#[cfg(test)]
mod accuracy_report {
    use super::tests as t;
    use super::*;

    // cargo test report -- --ignored --nocapture
    #[test]
    #[ignore]
    fn report() {
        let sr = 48000.0;
        let mk = |f, a, p| t::string_tone(f, a, p, sr, FFT_LEN);
        let cases: Vec<(&str, f32, Vec<f32>)> = vec![
            ("pure E2", 82.41, mk(82.41, 0.5, 0.0)),
            ("pure A2", 110.0, mk(110.0, 0.5, 0.0)),
            ("pure G3", 196.0, mk(196.0, 0.5, 0.0)),
            ("pure E4", 329.63, mk(329.63, 0.5, 0.0)),
            (
                "A2 + E2 res 30%",
                110.0,
                t::mix(&mk(110.0, 0.5, 0.0), &mk(82.41, 0.15, 1.1)),
            ),
            (
                "A2 + E2 res 50%",
                110.0,
                t::mix(&mk(110.0, 0.5, 0.0), &mk(82.41, 0.25, 1.1)),
            ),
            (
                "E2 + A2 res 30%",
                82.41,
                t::mix(&mk(82.41, 0.5, 0.3), &mk(110.0, 0.15, 2.0)),
            ),
            (
                "A2 -25c + E2 res",
                110.0 * 2.0f32.powf(-25.0 / 1200.0),
                t::mix(
                    &mk(110.0 * 2.0f32.powf(-25.0 / 1200.0), 0.5, 0.0),
                    &mk(82.41, 0.15, 0.8),
                ),
            ),
            (
                "D2 rondeña + A2 res",
                73.42,
                t::mix(&mk(73.42, 0.5, 0.0), &mk(110.0, 0.15, 1.7)),
            ),
        ];
        for (label, target, samples) in cases {
            match t::detect(&samples) {
                Some(f) => println!(
                    "  {label:<22} target {target:>7.2} Hz  detected {f:>7.2} Hz  err {:+.2} cents",
                    cents_between(f, target)
                ),
                None => println!("  {label:<22} NOT DETECTED"),
            }
        }
    }
}
