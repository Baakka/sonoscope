//! Single-channel two-string separation: a physics-informed Kalman filter
//! over a bank of damped harmonic oscillators.
//!
//! A plucked string is, to good approximation, a sum of exponentially
//! decaying sinusoids at near-integer multiples of its fundamental. With
//! the fundamentals known (the pitch detector supplies the primary, a
//! non-harmonic tracked spectral peak the secondary), the mixture is a
//! LINEAR state-space model
//!
//! ```text
//!   x[t+1] = A x[t] + w      A = blockdiag of damped 2-D rotations
//!   y[t]   = H x[t] + v      H sums the cosine component of every phasor
//! ```
//!
//! so the optimal estimator is a plain Kalman filter — no training data,
//! no linearization. Each harmonic of each string is one rotating phasor
//! (a·cos φ, a·sin φ); the joint covariance is what disentangles partials
//! that land near each other. A model frequency error of Δω shows up as a
//! steady phase drift of k·Δω at harmonic k, so a phase-slope servo
//! refines each source's fundamental between blocks — the frequency
//! refinement an EKF would do, without the linearization bookkeeping.

use std::f32::consts::{PI, TAU};

use crate::dsp::peaks::Track;

pub const HARMONICS: usize = 5;
pub const MAX_SOURCES: usize = 2;

/// Per-sample process noise. Together with R this sets each oscillator's
/// tracking bandwidth (~8 Hz): wide enough to follow pluck dynamics,
/// narrow enough that the closest guitar fundamentals (E2/A2, 27 Hz
/// apart) don't bleed into each other.
const Q: f32 = 1e-10;
const R: f32 = 1e-4;
const INIT_VAR: f32 = 0.25;
/// Sub-block length for the frequency servo. Must be short enough that a
/// worst-case detune keeps the highest harmonic's phase drift under ±π.
const SUB: usize = 256;
/// Servo correction per sub-block (time constant ≈ 70 ms).
const FREQ_GAIN: f32 = 0.08;
/// The refined fundamental stays within this many cents of its seed.
const MAX_PULL_CENTS: f32 = 80.0;
/// A new seed within this of a tracked source continues that source.
const MATCH_CENTS: f32 = 30.0;
/// Secondary candidates this close to a harmonic of the primary are
/// rejected. Adjacent guitar strings are ≥ 400 cents apart, so 90 is safe.
const REJECT_CENTS: f32 = 90.0;
/// Innovation power EMA (~20 ms at 48 kHz).
const RES_BETA: f32 = 0.999;
/// Slow state leak (τ = 2 s) so phasors stay bounded through silence;
/// actual string decay is tracked by the filter itself via Q.
const DECAY_TAU_S: f32 = 2.0;

#[derive(Clone, Copy, Debug)]
pub struct SourceReading {
    /// Servo-refined fundamental (Hz).
    pub freq: f32,
    /// Amplitude of each harmonic phasor (exercised by tests; the UI
    /// shows only freq and level).
    #[allow(dead_code)]
    pub amps: [f32; HARMONICS],
    /// RMS of the reconstructed source.
    pub level: f32,
}

struct Source {
    seed: f32,
    freq: f32,
}

pub struct Separator {
    sample_rate: f32,
    decay: f32,
    sources: Vec<Source>,
    /// Interleaved phasors [c, s] per oscillator, source-major.
    x: Vec<f32>,
    /// Row-major n×n covariance.
    p: Vec<f32>,
    /// Per-oscillator per-sample rotation (cos θ, sin θ) and θ itself.
    rot: Vec<(f32, f32)>,
    theta: Vec<f32>,
    /// Predicted phase of each phasor at the next sub-block boundary;
    /// the measured shortfall is the frequency-servo error signal.
    ref_phase: Vec<f32>,
    sub_fill: usize,
    /// Scratch: P·Hᵀ.
    hp: Vec<f32>,
    res2: f32,
}

impl Separator {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            decay: (-1.0 / (DECAY_TAU_S * sample_rate)).exp(),
            sources: Vec::new(),
            x: Vec::new(),
            p: Vec::new(),
            rot: Vec::new(),
            theta: Vec::new(),
            ref_phase: Vec::new(),
            sub_fill: 0,
            hp: Vec::new(),
            res2: 0.0,
        }
    }

    /// Seed (or re-seed) the source set. A seed within MATCH_CENTS of a
    /// source already being tracked continues that source — its phasor
    /// state and refined frequency survive, so per-frame detector jitter
    /// never resets the filter.
    pub fn set_sources(&mut self, freqs: &[f32]) {
        let mut seeds: Vec<f32> = Vec::new();
        for &f in freqs {
            if seeds.len() >= MAX_SOURCES {
                break;
            }
            if f > 0.0
                && seeds
                    .iter()
                    .all(|&s| cents_between(f, s).abs() > 2.0 * MATCH_CENTS)
            {
                seeds.push(f);
            }
        }

        let matches: Vec<Option<usize>> = seeds
            .iter()
            .map(|&f| {
                self.sources
                    .iter()
                    .position(|s| cents_between(f, s.freq).abs() < MATCH_CENTS)
            })
            .collect();
        let all_matched = matches.iter().all(Option::is_some);
        let distinct = matches
            .iter()
            .flatten()
            .collect::<std::collections::HashSet<_>>()
            .len()
            == seeds.len();
        if all_matched && distinct && self.sources.len() == seeds.len() {
            for (i, m) in matches.iter().enumerate() {
                self.sources[m.unwrap()].seed = seeds[i];
            }
            return;
        }

        // Topology changed: rebuild, carrying the phasor block of every
        // matched source; cross-covariances restart from scratch.
        let old_x = std::mem::take(&mut self.x);
        let mut sources = Vec::new();
        let mut x = Vec::new();
        for (i, &f) in seeds.iter().enumerate() {
            match matches[i].filter(|_| distinct) {
                Some(j) => {
                    sources.push(Source {
                        seed: f,
                        freq: self.sources[j].freq,
                    });
                    x.extend_from_slice(&old_x[2 * j * HARMONICS..2 * (j + 1) * HARMONICS]);
                }
                None => {
                    sources.push(Source { seed: f, freq: f });
                    x.extend(std::iter::repeat_n(0.0, 2 * HARMONICS));
                }
            }
        }
        self.sources = sources;
        self.x = x;
        let n = self.x.len();
        self.p = vec![0.0; n * n];
        for i in 0..n {
            self.p[i * n + i] = INIT_VAR;
        }
        self.hp = vec![0.0; n];
        self.rot = vec![(1.0, 0.0); n / 2];
        self.theta = vec![0.0; n / 2];
        self.ref_phase = vec![0.0; n / 2];
        for si in 0..self.sources.len() {
            self.rebuild_rot(si);
        }
        self.reset_ref_phases();
        self.sub_fill = 0;
    }

    pub fn clear(&mut self) {
        self.sources.clear();
        self.x.clear();
        self.p.clear();
        self.rot.clear();
        self.theta.clear();
        self.ref_phase.clear();
        self.hp.clear();
        self.sub_fill = 0;
        self.res2 = 0.0;
    }

    pub fn process(&mut self, samples: &[f32]) {
        if self.sources.is_empty() {
            return;
        }
        for &y in samples {
            self.predict();
            self.update(y);
            self.sub_fill += 1;
            if self.sub_fill >= SUB {
                self.sub_fill = 0;
                self.servo();
                self.symmetrize();
            }
        }
    }

    pub fn readings(&self) -> Vec<SourceReading> {
        self.sources
            .iter()
            .enumerate()
            .map(|(si, s)| {
                let mut amps = [0.0; HARMONICS];
                for (m, amp) in amps.iter_mut().enumerate() {
                    let o = si * HARMONICS + m;
                    *amp = (self.x[2 * o].powi(2) + self.x[2 * o + 1].powi(2)).sqrt();
                }
                let level = (amps.iter().map(|a| a * a).sum::<f32>() / 2.0).sqrt();
                SourceReading {
                    freq: s.freq,
                    amps,
                    level,
                }
            })
            .collect()
    }

    /// RMS of the innovation — what the oscillator-bank model does NOT
    /// explain about the input.
    pub fn residual_rms(&self) -> f32 {
        self.res2.sqrt()
    }

    fn rebuild_rot(&mut self, si: usize) {
        for m in 0..HARMONICS {
            let o = si * HARMONICS + m;
            let th = TAU * self.sources[si].freq * (m as f32 + 1.0) / self.sample_rate;
            self.theta[o] = th;
            self.rot[o] = (th.cos(), th.sin());
        }
    }

    fn reset_ref_phases(&mut self) {
        for o in 0..self.theta.len() {
            let phase = self.x[2 * o + 1].atan2(self.x[2 * o]);
            self.ref_phase[o] = wrap(phase + (SUB as f32 * self.theta[o]) % TAU);
        }
    }

    /// x ← A x,  P ← A P Aᵀ + Q, exploiting A's 2×2 block structure.
    fn predict(&mut self) {
        let n = self.x.len();
        let osc = n / 2;
        let d = self.decay;
        for o in 0..osc {
            let (c, s) = self.rot[o];
            let (a, b) = (self.x[2 * o], self.x[2 * o + 1]);
            self.x[2 * o] = d * (c * a - s * b);
            self.x[2 * o + 1] = d * (s * a + c * b);
        }
        for o in 0..osc {
            let (c, s) = self.rot[o];
            let (r0, r1) = (2 * o * n, (2 * o + 1) * n);
            for j in 0..n {
                let (a, b) = (self.p[r0 + j], self.p[r1 + j]);
                self.p[r0 + j] = d * (c * a - s * b);
                self.p[r1 + j] = d * (s * a + c * b);
            }
        }
        for o in 0..osc {
            let (c, s) = self.rot[o];
            for i in 0..n {
                let row = i * n;
                let (a, b) = (self.p[row + 2 * o], self.p[row + 2 * o + 1]);
                self.p[row + 2 * o] = d * (c * a - s * b);
                self.p[row + 2 * o + 1] = d * (s * a + c * b);
            }
        }
        for i in 0..n {
            self.p[i * n + i] += Q;
        }
    }

    /// Scalar measurement update: y = Σ cosine components + v.
    fn update(&mut self, y: f32) {
        let n = self.x.len();
        let osc = n / 2;
        for i in 0..n {
            let row = i * n;
            let mut acc = 0.0;
            for m in 0..osc {
                acc += self.p[row + 2 * m];
            }
            self.hp[i] = acc;
        }
        let mut s_cov = R;
        let mut pred = 0.0;
        for m in 0..osc {
            s_cov += self.hp[2 * m];
            pred += self.x[2 * m];
        }
        let innov = y - pred;
        let inv_s = 1.0 / s_cov;
        for i in 0..n {
            self.x[i] += self.hp[i] * inv_s * innov;
        }
        for i in 0..n {
            let k = self.hp[i] * inv_s;
            let row = i * n;
            for j in 0..n {
                self.p[row + j] -= k * self.hp[j];
            }
        }
        self.res2 = self.res2 * RES_BETA + innov * innov * (1.0 - RES_BETA);
    }

    /// Frequency servo: the amplitude-weighted average of (phase drift /
    /// harmonic number) over one sub-block measures the fundamental's
    /// detune directly.
    fn servo(&mut self) {
        let dt = SUB as f32 / self.sample_rate;
        for si in 0..self.sources.len() {
            let (mut num, mut den) = (0.0f32, 0.0f32);
            for m in 0..HARMONICS {
                let o = si * HARMONICS + m;
                let (c, s) = (self.x[2 * o], self.x[2 * o + 1]);
                let a2 = c * c + s * s;
                if a2 < 1e-9 {
                    continue;
                }
                let drift = wrap(s.atan2(c) - self.ref_phase[o]);
                num += a2 * drift / (m as f32 + 1.0);
                den += a2;
            }
            if den > 1e-9 {
                let df = FREQ_GAIN * (num / den) / (TAU * dt);
                let seed = self.sources[si].seed;
                let lo = seed * 2.0f32.powf(-MAX_PULL_CENTS / 1200.0);
                let hi = seed * 2.0f32.powf(MAX_PULL_CENTS / 1200.0);
                self.sources[si].freq = (self.sources[si].freq + df).clamp(lo, hi);
                self.rebuild_rot(si);
            }
        }
        self.reset_ref_phases();
    }

    /// Float drift makes P slightly asymmetric over time; re-balance at
    /// sub-block boundaries.
    fn symmetrize(&mut self) {
        let n = self.x.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = 0.5 * (self.p[i * n + j] + self.p[j * n + i]);
                self.p[i * n + j] = avg;
                self.p[j * n + i] = avg;
            }
            self.p[i * n + i] = self.p[i * n + i].max(0.0);
        }
    }
}

/// Choose a second fundamental from the tracked spectral peaks: the
/// strongest aged track that is neither a harmonic of `primary` nor the
/// octave partial of a lower qualifying track. Known limitation: a second
/// string whose fundamental coincides with a harmonic of the first (E4
/// over A2's 3rd partial) is indistinguishable by frequency alone and is
/// rejected as well.
pub fn pick_second(primary: f32, tracks: &[Track], range: (f32, f32)) -> Option<f32> {
    let strongest = tracks
        .iter()
        .map(|t| t.db)
        .fold(f32::NEG_INFINITY, f32::max);
    let qualifying: Vec<&Track> = tracks
        .iter()
        .filter(|t| {
            t.age >= 3
                && t.freq >= range.0
                && t.freq <= range.1
                && t.db > strongest - 30.0
                && (1..=10).all(|k| cents_between(t.freq, primary * k as f32).abs() > REJECT_CENTS)
        })
        .collect();
    qualifying
        .iter()
        .filter(|t| {
            // Drop tracks that look like the 2nd harmonic of another
            // qualifying track — prefer the fundamental below.
            !qualifying.iter().any(|u| {
                u.freq < t.freq * 0.9 && cents_between(t.freq, u.freq * 2.0).abs() < REJECT_CENTS
            })
        })
        .max_by(|a, b| a.db.total_cmp(&b.db))
        .map(|t| t.freq)
}

fn cents_between(a: f32, b: f32) -> f32 {
    1200.0 * (a / b).log2()
}

fn wrap(a: f32) -> f32 {
    let mut a = a % TAU;
    if a > PI {
        a -= TAU;
    } else if a < -PI {
        a += TAU;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::tests::{mix, string_tone};

    const SR: f32 = 48000.0;

    #[test]
    fn separates_two_strings() {
        // A2 + D3 ringing together (their 4th/3rd partials nearly collide
        // at 440 Hz). string_tone's harmonic profile is [1, .45, .25, ...].
        let n = 24000;
        let a2 = string_tone(110.0, 0.5, 0.0, SR, n);
        let d3 = string_tone(146.83, 0.35, 1.2, SR, n);
        let mixed = mix(&a2, &d3);

        let mut sep = Separator::new(SR);
        sep.set_sources(&[110.0, 146.83]);
        sep.process(&mixed);

        let r = sep.readings();
        assert_eq!(r.len(), 2);
        assert!(
            (r[0].amps[0] - 0.5).abs() < 0.05,
            "A2 fundamental {} vs 0.5",
            r[0].amps[0]
        );
        assert!(
            (r[1].amps[0] - 0.35).abs() < 0.035,
            "D3 fundamental {} vs 0.35",
            r[1].amps[0]
        );
        assert!(
            (r[0].amps[1] - 0.5 * 0.45).abs() < 0.04,
            "A2 2nd harmonic {} vs {}",
            r[0].amps[1],
            0.5 * 0.45
        );

        let in_rms = crate::pitch::rms(&mixed[n - 4096..]);
        assert!(
            sep.residual_rms() < 0.1 * in_rms,
            "residual {} vs input {}",
            sep.residual_rms(),
            in_rms
        );
    }

    #[test]
    fn refines_a_detuned_seed_sub_cent() {
        // True string 6 cents sharp of the seed; the phase servo must
        // converge to it within a cent.
        let truth = 110.0 * 2.0f32.powf(6.0 / 1200.0);
        let tone = string_tone(truth, 0.4, 0.7, SR, 48000);
        let mut sep = Separator::new(SR);
        sep.set_sources(&[110.0]);
        sep.process(&tone);
        let f = sep.readings()[0].freq;
        let err = cents_between(f, truth);
        assert!(
            err.abs() < 1.0,
            "refined {f} Hz, truth {truth} ({err:+.2}¢)"
        );
    }

    #[test]
    fn silent_source_stays_silent() {
        // Two sources seeded, only one sounding: the filter must not
        // hallucinate energy into the silent one.
        let a2 = string_tone(110.0, 0.5, 0.0, SR, 16384);
        let mut sep = Separator::new(SR);
        sep.set_sources(&[110.0, 146.83]);
        sep.process(&a2);
        let r = sep.readings();
        assert!(
            r[1].level < 0.1 * r[0].level,
            "silent source level {} vs active {}",
            r[1].level,
            r[0].level
        );
    }

    #[test]
    fn tracks_a_decaying_string() {
        let tone = string_tone(110.0, 0.5, 0.0, SR, 48000);
        let decaying: Vec<f32> = tone
            .iter()
            .enumerate()
            .map(|(i, &v)| v * (-2.0 * i as f32 / SR).exp())
            .collect();
        let mut sep = Separator::new(SR);
        sep.set_sources(&[110.0]);
        sep.process(&decaying[..24000]);
        let mid = sep.readings()[0].level;
        sep.process(&decaying[24000..]);
        let end = sep.readings()[0].level;
        // True amplitude ratio over the second half is e^-1 ≈ 0.37.
        assert!(
            end < 0.55 * mid && end > 0.2 * mid,
            "decay not tracked: mid {mid}, end {end}"
        );
    }

    #[test]
    fn reseeding_with_jitter_keeps_state() {
        let tone = string_tone(110.0, 0.5, 0.0, SR, 16384);
        let mut sep = Separator::new(SR);
        sep.set_sources(&[110.0]);
        sep.process(&tone[..8192]);
        let before = sep.readings()[0].amps[0];
        // Detector jitter: ±a few cents must not reset the filter.
        sep.set_sources(&[110.0 * 2.0f32.powf(4.0 / 1200.0)]);
        let after = sep.readings()[0].amps[0];
        assert!(
            (before - after).abs() < 1e-6,
            "state lost on jittered reseed: {before} vs {after}"
        );
    }

    #[test]
    fn second_string_picker_rejects_harmonics() {
        let range = (55.0, 600.0);
        // 220 is A2's 2nd harmonic; 146.8 is a genuine D3.
        let tracks = [Track::synth(220.0, -18.0, 5), Track::synth(146.8, -24.0, 5)];
        let pick = pick_second(110.0, &tracks, range).expect("no second string");
        assert!((pick - 146.8).abs() < 0.1, "picked {pick}");

        // Only harmonics of the primary → nothing to pick.
        let harmonics_only = [Track::synth(220.0, -18.0, 5), Track::synth(330.0, -22.0, 5)];
        assert!(pick_second(110.0, &harmonics_only, range).is_none());

        // The secondary's own octave partial is stronger than its
        // fundamental — the picker must still return the fundamental.
        let with_partial = [
            Track::synth(293.66, -15.0, 5),
            Track::synth(146.83, -20.0, 5),
        ];
        let pick = pick_second(110.0, &with_partial, range).expect("no second string");
        assert!((pick - 146.83).abs() < 0.1, "picked {pick}");

        // Young tracks don't qualify.
        let young = [Track::synth(146.8, -20.0, 1)];
        assert!(pick_second(110.0, &young, range).is_none());
    }
}
