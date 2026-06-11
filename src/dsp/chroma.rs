//! Chromagram and chord/key recognition.
//!
//! Folds CQT semitone bands into 12 pitch classes (octave-independent),
//! matches the result against 24 major/minor triad templates, and keeps a
//! long-term average for a Krumhansl-Schmuckler key estimate.

use super::cqt::{MIDI_LO, N_BANDS};

pub const PITCH_CLASSES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Fold CQT band energies into 12 normalized pitch classes (index 0 = C).
pub fn fold(bands: &[f32; N_BANDS]) -> [f32; 12] {
    let mut chroma = [0.0f32; 12];
    for (i, &b) in bands.iter().enumerate() {
        let pc = (MIDI_LO + i as i32).rem_euclid(12) as usize;
        chroma[pc] += b * b; // energy domain
    }
    let max = chroma.iter().fold(0.0f32, |m, &x| m.max(x));
    if max > 0.0 {
        for c in &mut chroma {
            *c = (*c / max).sqrt(); // back to amplitude-ish, normalized to 1
        }
    }
    chroma
}

#[derive(Clone, Debug, PartialEq)]
pub struct Chord {
    /// e.g. "A" (major) or "Am" (minor).
    pub name: String,
    /// Correlation score 0..1.
    pub confidence: f32,
}

/// Template-match against 24 triads. Requires both a decent absolute score
/// and a margin over the runner-up of a *different* root, so a single
/// ambiguous note doesn't get labeled a chord.
pub fn detect_chord(chroma: &[f32; 12]) -> Option<Chord> {
    let norm = |v: &[f32]| -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() };
    let cn = norm(chroma);
    if cn < 1e-6 {
        return None;
    }

    // (score, root, is_minor)
    let mut scores: Vec<(f32, usize, bool)> = Vec::with_capacity(24);
    for root in 0..12 {
        for (is_minor, third) in [(false, 4usize), (true, 3usize)] {
            let mut template = [0.0f32; 12];
            template[root] = 1.0;
            template[(root + third) % 12] = 1.0;
            template[(root + 7) % 12] = 1.0;
            let dot: f32 = chroma.iter().zip(&template).map(|(a, b)| a * b).sum();
            let score = dot / (cn * 3.0f32.sqrt());
            scores.push((score, root, is_minor));
        }
    }
    scores.sort_by(|a, b| b.0.total_cmp(&a.0));
    let best = scores[0];
    let runner_up_other_root = scores[1..]
        .iter()
        .find(|s| s.1 != best.1)
        .map_or(0.0, |s| s.0);

    if best.0 < 0.62 || best.0 - runner_up_other_root < 0.05 {
        return None;
    }
    Some(Chord {
        name: format!("{}{}", PITCH_CLASSES[best.1], if best.2 { "m" } else { "" }),
        confidence: best.0,
    })
}

/// Krumhansl-Schmuckler key profiles (probe-tone ratings).
const MAJOR_PROFILE: [f32; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];
const MINOR_PROFILE: [f32; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

/// Long-term chroma average → key estimate.
pub struct KeyEstimator {
    avg: [f32; 12],
    frames: u32,
}

impl KeyEstimator {
    pub fn new() -> Self {
        Self {
            avg: [0.0; 12],
            frames: 0,
        }
    }

    pub fn update(&mut self, chroma: &[f32; 12]) {
        // ~10 s horizon at 20 fps.
        const ALPHA: f32 = 1.0 / 200.0;
        for (a, &c) in self.avg.iter_mut().zip(chroma) {
            *a = *a * (1.0 - ALPHA) + c * ALPHA;
        }
        self.frames += 1;
    }

    /// e.g. "A major" / "F# minor"; None until enough evidence accumulates.
    pub fn estimate(&self) -> Option<String> {
        if self.frames < 40 {
            return None;
        }
        let mut best: Option<(f32, usize, bool)> = None;
        for root in 0..12 {
            for (is_minor, profile) in [(false, &MAJOR_PROFILE), (true, &MINOR_PROFILE)] {
                let r = correlation(&self.avg, profile, root);
                if best.is_none_or(|b| r > b.0) {
                    best = Some((r, root, is_minor));
                }
            }
        }
        let (r, root, is_minor) = best?;
        (r > 0.5).then(|| {
            format!(
                "{} {}",
                PITCH_CLASSES[root],
                if is_minor { "minor" } else { "major" }
            )
        })
    }
}

/// Pearson correlation between chroma and a profile rotated to `root`.
fn correlation(chroma: &[f32; 12], profile: &[f32; 12], root: usize) -> f32 {
    let cm = chroma.iter().sum::<f32>() / 12.0;
    let pm = profile.iter().sum::<f32>() / 12.0;
    let (mut num, mut cd, mut pd) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..12 {
        let c = chroma[(root + i) % 12] - cm;
        let p = profile[i] - pm;
        num += c * p;
        cd += c * c;
        pd += p * p;
    }
    if cd <= 0.0 || pd <= 0.0 {
        0.0
    } else {
        num / (cd * pd).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::cqt::{CQT_LEN, Cqt};

    fn chroma_of(freqs: &[f32]) -> [f32; 12] {
        let sr = 48000.0;
        let samples: Vec<f32> = (0..CQT_LEN)
            .map(|i| {
                let t = i as f32 / sr;
                freqs
                    .iter()
                    .map(|&f| 0.3 * (2.0 * std::f32::consts::PI * f * t).sin())
                    .sum()
            })
            .collect();
        let mut cqt = Cqt::new(sr, 440.0);
        cqt.process(&samples);
        fold(&cqt.bands)
    }

    #[test]
    fn a_major_triad_recognized() {
        // A3, C#4, E4 — pure sines so harmonics don't blur the classes.
        let chroma = chroma_of(&[220.0, 277.18, 329.63]);
        assert!(chroma[9] > 0.7, "A class weak: {chroma:?}");
        assert!(chroma[1] > 0.7, "C# class weak: {chroma:?}");
        assert!(chroma[4] > 0.7, "E class weak: {chroma:?}");
        let chord = detect_chord(&chroma).expect("no chord detected");
        assert_eq!(chord.name, "A");
    }

    #[test]
    fn a_minor_triad_recognized() {
        // A3, C4, E4.
        let chroma = chroma_of(&[220.0, 261.63, 329.63]);
        let chord = detect_chord(&chroma).expect("no chord detected");
        assert_eq!(chord.name, "Am");
    }

    #[test]
    fn single_note_is_not_a_chord() {
        let chroma = chroma_of(&[220.0]);
        assert!(
            detect_chord(&chroma).is_none(),
            "single note misread as chord"
        );
    }

    #[test]
    fn key_estimator_finds_c_major() {
        let mut est = KeyEstimator::new();
        // Alternate C-major scale-degree triads for a while.
        let c = chroma_of(&[261.63, 329.63, 392.0]); // C E G
        let f = chroma_of(&[349.23, 440.0, 523.25]); // F A C
        let g = chroma_of(&[392.0, 493.88, 587.33]); // G B D
        for _ in 0..40 {
            est.update(&c);
            est.update(&f);
            est.update(&g);
        }
        let key = est.estimate().expect("no key estimate");
        assert_eq!(key, "C major");
    }
}
