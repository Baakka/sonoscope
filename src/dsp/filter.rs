//! RBJ cookbook biquad filters (Audio EQ Cookbook, Robert Bristow-Johnson).
//! Butterworth response at Q = 1/√2.

pub const BUTTERWORTH_Q: f32 = std::f32::consts::FRAC_1_SQRT_2;

#[derive(Clone, Copy)]
pub struct Biquad {
    // Normalized coefficients (a0 = 1).
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    // Direct form 1 state.
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    pub fn lowpass(sample_rate: f32, cutoff: f32, q: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * (cutoff / sample_rate).clamp(1e-5, 0.49);
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self::normalized(
            (1.0 - cos) / 2.0,
            1.0 - cos,
            (1.0 - cos) / 2.0,
            a0,
            -2.0 * cos,
            1.0 - alpha,
        )
    }

    pub fn highpass(sample_rate: f32, cutoff: f32, q: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * (cutoff / sample_rate).clamp(1e-5, 0.49);
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self::normalized(
            (1.0 + cos) / 2.0,
            -(1.0 + cos),
            (1.0 + cos) / 2.0,
            a0,
            -2.0 * cos,
            1.0 - alpha,
        )
    }

    fn normalized(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Steady-state gain of a filter at `freq`: run a sine through and
    /// compare RMS after the transient settles.
    fn gain_at(mut filter: Biquad, freq: f32, sr: f32) -> f32 {
        let n = 48000;
        let settle = n / 2;
        let mut in_sq = 0.0f64;
        let mut out_sq = 0.0f64;
        for i in 0..n {
            let x = (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin();
            let y = filter.process(x);
            if i >= settle {
                in_sq += (x as f64) * (x as f64);
                out_sq += (y as f64) * (y as f64);
            }
        }
        (out_sq / in_sq).sqrt() as f32
    }

    fn db(gain: f32) -> f32 {
        20.0 * gain.log10()
    }

    #[test]
    fn lowpass_response() {
        let sr = 48000.0;
        let fc = 1000.0;
        let make = || Biquad::lowpass(sr, fc, BUTTERWORTH_Q);
        // Passband flat within 0.5 dB.
        assert!(db(gain_at(make(), 100.0, sr)).abs() < 0.5);
        // −3 dB at cutoff (±0.5 dB).
        assert!((db(gain_at(make(), fc, sr)) + 3.0).abs() < 0.5);
        // ~−12 dB/octave rolloff: one octave above ≈ −12 dB, two ≈ −24 dB.
        assert!(db(gain_at(make(), 2.0 * fc, sr)) < -10.0);
        assert!(db(gain_at(make(), 4.0 * fc, sr)) < -22.0);
    }

    #[test]
    fn highpass_response() {
        let sr = 48000.0;
        let fc = 200.0;
        let make = || Biquad::highpass(sr, fc, BUTTERWORTH_Q);
        assert!(db(gain_at(make(), 2000.0, sr)).abs() < 0.5);
        assert!((db(gain_at(make(), fc, sr)) + 3.0).abs() < 0.5);
        assert!(db(gain_at(make(), fc / 2.0, sr)) < -10.0);
        assert!(db(gain_at(make(), fc / 4.0, sr)) < -22.0);
    }
}
