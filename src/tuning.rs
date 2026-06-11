/// Tunings used in flamenco guitar.
///
/// Standard (E A D G B E) covers the vast majority of flamenco playing —
/// "por medio" and "por arriba" are left-hand positions, not tunings.
/// Rondeña is the one true scordatura of the repertoire: 6th down to D,
/// 3rd down to F#.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tuning {
    Standard,
    Rondena,
}

impl Tuning {
    /// Open-string MIDI note numbers, 6th string first.
    pub fn open_strings(self) -> [i32; 6] {
        match self {
            // E2 A2 D3 G3 B3 E4
            Tuning::Standard => [40, 45, 50, 55, 59, 64],
            // D2 A2 D3 F#3 B3 E4
            Tuning::Rondena => [38, 45, 50, 54, 59, 64],
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            Tuning::Standard => Tuning::Rondena,
            Tuning::Rondena => Tuning::Standard,
        }
    }
}

const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

pub fn note_name(midi: i32) -> String {
    let name = NOTE_NAMES[midi.rem_euclid(12) as usize];
    let octave = midi / 12 - 1;
    format!("{name}{octave}")
}

pub fn midi_to_freq(midi: i32, a4: f32) -> f32 {
    a4 * 2.0f32.powf((midi - 69) as f32 / 12.0)
}

pub struct Match {
    /// String number, 1 (high E) to 6 (low E/D).
    pub string_no: usize,
    /// Target note (after capo shift).
    pub target_midi: i32,
    pub target_freq: f32,
    /// Deviation of the played pitch from the target, in cents.
    pub cents: f32,
}

/// Finds the string (with `capo` semitones added per the cejilla) whose
/// target pitch is closest to `freq`.
pub fn nearest_string(freq: f32, tuning: Tuning, capo: i32, a4: f32) -> Match {
    let mut best: Option<Match> = None;
    for (i, &open) in tuning.open_strings().iter().enumerate() {
        let target_midi = open + capo;
        let target_freq = midi_to_freq(target_midi, a4);
        let cents = 1200.0 * (freq / target_freq).log2();
        if best.as_ref().is_none_or(|b| cents.abs() < b.cents.abs()) {
            best = Some(Match {
                string_no: 6 - i,
                target_midi,
                target_freq,
                cents,
            });
        }
    }
    best.unwrap()
}
