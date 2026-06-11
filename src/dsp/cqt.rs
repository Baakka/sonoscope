//! Pseudo constant-Q transform: per-semitone band energies computed from a
//! long FFT. Each band is a Gaussian-weighted sum of FFT bins centered on a
//! semitone frequency with bandwidth proportional to frequency (constant Q),
//! so the frequency axis aligns with musical pitch instead of linear hertz.

use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

/// CQT analysis window: 16384 samples ≈ 341 ms at 48 kHz, 2.9 Hz bins.
pub const CQT_LEN: usize = 16384;
/// Semitone bands C2..C7 inclusive.
pub const MIDI_LO: i32 = 36;
pub const MIDI_HI: i32 = 96;
pub const N_BANDS: usize = (MIDI_HI - MIDI_LO + 1) as usize;

/// One semitone's bandwidth as a fraction of its center frequency
/// (2^(1/12) − 1 ≈ 0.0595 → Q ≈ 16.8).
const SEMITONE_BW: f32 = 0.059_463;

pub struct Cqt {
    fft: Arc<dyn Fft<f32>>,
    hann: Vec<f32>,
    /// Per band: sparse (bin index, weight) pairs, weights summing to 1.
    kernels: Vec<Vec<(usize, f32)>>,
    /// Band energies (linear amplitude), index 0 = MIDI_LO.
    pub bands: [f32; N_BANDS],
    a4: f32,
    sample_rate: f32,
}

impl Cqt {
    pub fn new(sample_rate: f32, a4: f32) -> Self {
        let fft = FftPlanner::new().plan_fft_forward(CQT_LEN);
        let hann = (0..CQT_LEN)
            .map(|i| {
                let x = i as f32 / (CQT_LEN - 1) as f32;
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * x).cos()
            })
            .collect();
        let kernels = build_kernels(sample_rate, a4);
        Self {
            fft,
            hann,
            kernels,
            bands: [0.0; N_BANDS],
            a4,
            sample_rate,
        }
    }

    /// Rebuild kernels if the reference pitch changed.
    pub fn set_a4(&mut self, a4: f32) {
        if (self.a4 - a4).abs() > 0.01 {
            self.a4 = a4;
            self.kernels = build_kernels(self.sample_rate, a4);
        }
    }

    pub fn process(&mut self, samples: &[f32]) {
        debug_assert_eq!(samples.len(), CQT_LEN);
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let mut buf: Vec<Complex<f32>> = samples
            .iter()
            .zip(&self.hann)
            .map(|(&s, &w)| Complex::new((s - mean) * w, 0.0))
            .collect();
        self.fft.process(&mut buf);
        let norm = 4.0 / CQT_LEN as f32;

        for (band, kernel) in self.bands.iter_mut().zip(&self.kernels) {
            // Energy sum under the Gaussian kernel.
            let energy: f32 = kernel
                .iter()
                .map(|&(bin, w)| {
                    let m = buf[bin].norm() * norm;
                    w * m * m
                })
                .sum();
            *band = energy.sqrt();
        }
    }
}

fn build_kernels(sample_rate: f32, a4: f32) -> Vec<Vec<(usize, f32)>> {
    let bin_hz = sample_rate / CQT_LEN as f32;
    let n_bins = CQT_LEN / 2;
    (0..N_BANDS)
        .map(|i| {
            let midi = MIDI_LO + i as i32;
            let fc = a4 * 2.0f32.powf((midi - 69) as f32 / 12.0);
            // FWHM = one semitone bandwidth; σ = FWHM / 2.355. Never narrower
            // than one FFT bin (the low-octave resolution limit).
            let sigma = ((fc * SEMITONE_BW) / 2.355).max(bin_hz * 0.8);
            let lo = (((fc - 3.0 * sigma) / bin_hz).floor().max(1.0)) as usize;
            let hi = (((fc + 3.0 * sigma) / bin_hz).ceil() as usize).min(n_bins - 1);
            let mut kernel: Vec<(usize, f32)> = (lo..=hi)
                .map(|bin| {
                    let d = (bin as f32 * bin_hz - fc) / sigma;
                    (bin, (-0.5 * d * d).exp())
                })
                .collect();
            let sum: f32 = kernel.iter().map(|&(_, w)| w).sum();
            if sum > 0.0 {
                for (_, w) in &mut kernel {
                    *w /= sum;
                }
            }
            kernel
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f32, sr: f32) -> Vec<f32> {
        (0..CQT_LEN)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin() * 0.5)
            .collect()
    }

    fn band_of(midi: i32) -> usize {
        (midi - MIDI_LO) as usize
    }

    fn dominant(cqt: &Cqt) -> usize {
        cqt.bands
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    }

    #[test]
    fn a4_lands_in_band_69() {
        let mut cqt = Cqt::new(48000.0, 440.0);
        cqt.process(&tone(440.0, 48000.0));
        assert_eq!(dominant(&cqt), band_of(69));
    }

    #[test]
    fn octaves_are_twelve_bands_apart() {
        let mut cqt = Cqt::new(48000.0, 440.0);
        cqt.process(&tone(220.0, 48000.0));
        let a3 = dominant(&cqt);
        cqt.process(&tone(440.0, 48000.0));
        let a4 = dominant(&cqt);
        assert_eq!(a4 - a3, 12);
        assert_eq!(a3, band_of(57));
    }

    #[test]
    fn semitone_neighbors_separate() {
        let mut cqt = Cqt::new(48000.0, 440.0);
        cqt.process(&tone(440.0, 48000.0));
        let a4 = cqt.bands[band_of(69)];
        let a_sharp = cqt.bands[band_of(70)];
        let ratio_db = 20.0 * (a4 / a_sharp.max(1e-12)).log10();
        assert!(ratio_db > 6.0, "only {ratio_db:.1} dB separation");
    }
}
