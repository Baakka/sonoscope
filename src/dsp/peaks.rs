//! Spectral peak picking and frame-to-frame tracking.
//!
//! Picking finds local maxima above an adaptive threshold (median noise
//! floor + margin) and refines each to sub-bin precision with parabolic
//! interpolation of log magnitudes. The tracker associates picks across
//! frames by nearest-neighbor in cents, smoothing each track's frequency
//! with an EMA — the location of a tracked peak is far more accurate and
//! stable than any single frame's pick.

#[derive(Clone, Copy, Debug)]
pub struct Peak {
    pub freq: f32,
    pub db: f32,
}

/// Local maxima above `median + MARGIN_DB`, strongest first, capped.
const MARGIN_DB: f32 = 12.0;
const MAX_PEAKS: usize = 16;

pub fn pick(mags: &[f32], bin_hz: f32, max_freq: f32) -> Vec<Peak> {
    let n = ((max_freq / bin_hz) as usize).min(mags.len().saturating_sub(2));
    if n < 4 {
        return Vec::new();
    }

    let db = |m: f32| 20.0 * (m + 1e-12).log10();
    let mut floor: Vec<f32> = mags[1..n].iter().map(|&m| db(m)).collect();
    floor.sort_by(f32::total_cmp);
    let median_db = floor[floor.len() / 2];
    let threshold = median_db + MARGIN_DB;

    let mut peaks = Vec::new();
    for i in 2..n {
        let m = mags[i];
        if m <= mags[i - 1] || m < mags[i + 1] {
            continue;
        }
        let peak_db = db(m);
        if peak_db < threshold {
            continue;
        }
        // Parabolic interpolation on log magnitudes.
        let (a, b, c) = (
            (mags[i - 1] + 1e-12).ln(),
            (m + 1e-12).ln(),
            (mags[i + 1] + 1e-12).ln(),
        );
        let denom = a - 2.0 * b + c;
        let delta = if denom.abs() > f32::EPSILON {
            (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        peaks.push(Peak {
            freq: (i as f32 + delta) * bin_hz,
            db: peak_db,
        });
    }
    peaks.sort_by(|a, b| b.db.total_cmp(&a.db));
    peaks.truncate(MAX_PEAKS);
    peaks
}

#[derive(Clone, Copy, Debug)]
pub struct Track {
    /// Stable identity across frames (exercised by tests; not displayed).
    #[allow(dead_code)]
    pub id: u64,
    /// EMA-smoothed frequency — the accurate "location" of this peak.
    pub freq: f32,
    pub db: f32,
    /// Frames this track has been alive.
    pub age: u32,
    missed: u32,
}

impl Track {
    /// Test-only constructor (`missed` is private to this module).
    #[cfg(test)]
    pub fn synth(freq: f32, db: f32, age: u32) -> Self {
        Self {
            id: 0,
            freq,
            db,
            age,
            missed: 0,
        }
    }
}

/// Frame-to-frame peak association by nearest neighbor in cents.
pub struct Tracker {
    tracks: Vec<Track>,
    next_id: u64,
}

const ASSOC_CENTS: f32 = 35.0;
const MAX_MISSED: u32 = 3;
const FREQ_SMOOTH: f32 = 0.35; // EMA weight of the new observation

impl Tracker {
    pub fn new() -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 0,
        }
    }

    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    pub fn update(&mut self, peaks: &[Peak]) {
        let mut claimed = vec![false; peaks.len()];

        for track in &mut self.tracks {
            let nearest = peaks
                .iter()
                .enumerate()
                .filter(|(i, p)| {
                    !claimed[*i] && cents_between(p.freq, track.freq).abs() < ASSOC_CENTS
                })
                .min_by(|(_, a), (_, b)| {
                    cents_between(a.freq, track.freq)
                        .abs()
                        .total_cmp(&cents_between(b.freq, track.freq).abs())
                });
            match nearest {
                Some((i, p)) => {
                    claimed[i] = true;
                    track.freq = track.freq * (1.0 - FREQ_SMOOTH) + p.freq * FREQ_SMOOTH;
                    track.db = p.db;
                    track.age += 1;
                    track.missed = 0;
                }
                None => track.missed += 1,
            }
        }
        self.tracks.retain(|t| t.missed <= MAX_MISSED);

        for (i, p) in peaks.iter().enumerate() {
            if !claimed[i] {
                self.tracks.push(Track {
                    id: self.next_id,
                    freq: p.freq,
                    db: p.db,
                    age: 1,
                    missed: 0,
                });
                self.next_id += 1;
            }
        }
        // Strongest first for display.
        self.tracks.sort_by(|a, b| b.db.total_cmp(&a.db));
    }
}

fn cents_between(a: f32, b: f32) -> f32 {
    1200.0 * (a / b).log2()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hann-windowed FFT magnitudes of a sum of sines.
    fn spectrum_of(freqs: &[(f32, f32)], sr: f32, n: usize) -> (Vec<f32>, f32) {
        use rustfft::{FftPlanner, num_complex::Complex};
        let mut buf: Vec<Complex<f32>> = (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos();
                let s: f32 = freqs
                    .iter()
                    .map(|&(f, a)| a * (2.0 * std::f32::consts::PI * f * t).sin())
                    .sum();
                Complex::new(s * w, 0.0)
            })
            .collect();
        FftPlanner::new().plan_fft_forward(n).process(&mut buf);
        let mags: Vec<f32> = buf[..n / 2]
            .iter()
            .map(|c| c.norm() * 4.0 / n as f32)
            .collect();
        (mags, sr / n as f32)
    }

    #[test]
    fn picks_two_tones() {
        let (mags, bin_hz) = spectrum_of(&[(440.0, 0.5), (660.0, 0.3)], 48000.0, 8192);
        let peaks = pick(&mags, bin_hz, 1500.0);
        assert!(peaks.len() >= 2, "found {} peaks", peaks.len());
        assert!((peaks[0].freq - 440.0).abs() < 1.0, "got {}", peaks[0].freq);
        assert!((peaks[1].freq - 660.0).abs() < 1.0, "got {}", peaks[1].freq);
    }

    #[test]
    fn tracker_follows_a_glide() {
        let mut tracker = Tracker::new();
        let mut freq = 440.0;
        let mut id = None;
        for _ in 0..20 {
            freq *= 1.005; // ~8.6 cents per frame, well within association range
            let (mags, bin_hz) = spectrum_of(&[(freq, 0.5)], 48000.0, 8192);
            tracker.update(&pick(&mags, bin_hz, 1500.0));
            let t = tracker.tracks().first().expect("track lost");
            match id {
                None => id = Some(t.id),
                Some(id) => assert_eq!(t.id, id, "track identity changed mid-glide"),
            }
        }
        let t = tracker.tracks()[0];
        // EMA lags a moving target; it must stay within ~25 cents of truth.
        assert!(
            cents_between(t.freq, freq).abs() < 25.0,
            "track at {} vs true {freq}",
            t.freq
        );
        assert!(t.age >= 19);
    }

    #[test]
    fn track_survives_dropout() {
        let mut tracker = Tracker::new();
        let (mags, bin_hz) = spectrum_of(&[(440.0, 0.5)], 48000.0, 8192);
        let peaks = pick(&mags, bin_hz, 1500.0);
        for _ in 0..5 {
            tracker.update(&peaks);
        }
        let id = tracker.tracks()[0].id;
        tracker.update(&[]); // 2-frame dropout
        tracker.update(&[]);
        tracker.update(&peaks);
        assert_eq!(tracker.tracks()[0].id, id, "track did not survive dropout");
        // 4 more missed frames kills it.
        for _ in 0..5 {
            tracker.update(&[]);
        }
        assert!(tracker.tracks().is_empty());
    }
}
