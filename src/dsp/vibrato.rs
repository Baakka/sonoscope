//! Vibrato analysis over the pitch-deviation history.
//!
//! Works on (time s, cents) samples of the recent sung note: detrend by the
//! mean, measure rate from zero crossings and depth from the percentile
//! span (robust to a stray frame).

#[derive(Clone, Copy, Debug)]
pub struct Vibrato {
    /// Oscillation rate in Hz (typical singing vibrato: 4–8 Hz).
    pub rate_hz: f32,
    /// Half the peak-to-peak excursion, in cents.
    pub depth_cents: f32,
    /// Mean pitch offset from the target note, in cents.
    pub mean_cents: f32,
}

/// Analyze the last `window_secs` of (t, cents) history (NaN rows are gaps).
/// Returns None until there is enough contiguous voiced data or when the
/// modulation is too small/slow to be vibrato.
pub fn analyze(history: &[[f64; 2]], window_secs: f64) -> Option<Vibrato> {
    let t_end = history.iter().rev().find(|p| !p[1].is_nan())?[0];
    let pts: Vec<[f64; 2]> = history
        .iter()
        .filter(|p| !p[1].is_nan() && p[0] >= t_end - window_secs)
        .copied()
        .collect();
    if pts.len() < 12 {
        return None;
    }
    let duration = (pts.last()?[0] - pts.first()?[0]) as f32;
    if duration < 0.5 {
        return None;
    }

    let mean = pts.iter().map(|p| p[1]).sum::<f64>() / pts.len() as f64;
    let detrended: Vec<f32> = pts.iter().map(|p| (p[1] - mean) as f32).collect();

    // Depth from the 5th–95th percentile span (robust peak-to-peak / 2).
    let mut sorted = detrended.clone();
    sorted.sort_by(f32::total_cmp);
    let lo = sorted[sorted.len() * 5 / 100];
    let hi = sorted[sorted.len() * 95 / 100];
    let depth = (hi - lo) / 2.0;
    if depth < 5.0 {
        return None; // steady tone, not vibrato
    }

    // Rate from zero crossings of the detrended signal.
    let crossings = detrended
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    let rate = crossings as f32 / 2.0 / duration;
    if rate < 1.0 {
        return None;
    }

    Some(Vibrato {
        rate_hz: rate,
        depth_cents: depth,
        mean_cents: mean as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_synthetic_vibrato() {
        // 6 Hz, ±40 cents, sampled at 20 fps for 3 s, centered +5 cents.
        let history: Vec<[f64; 2]> = (0..60)
            .map(|i| {
                let t = i as f64 / 20.0;
                [t, 5.0 + 40.0 * (2.0 * std::f64::consts::PI * 6.0 * t).sin()]
            })
            .collect();
        let v = analyze(&history, 3.0).expect("vibrato not detected");
        assert!((v.rate_hz - 6.0).abs() < 0.5, "rate {}", v.rate_hz);
        assert!(
            (v.depth_cents - 40.0).abs() / 40.0 < 0.15,
            "depth {}",
            v.depth_cents
        );
        assert!((v.mean_cents - 5.0).abs() < 3.0, "mean {}", v.mean_cents);
    }

    #[test]
    fn steady_tone_is_not_vibrato() {
        let history: Vec<[f64; 2]> = (0..60).map(|i| [i as f64 / 20.0, 2.0]).collect();
        assert!(analyze(&history, 3.0).is_none());
    }

    #[test]
    fn gaps_are_ignored() {
        let mut history: Vec<[f64; 2]> = (0..60)
            .map(|i| {
                let t = i as f64 / 20.0;
                [t, 30.0 * (2.0 * std::f64::consts::PI * 5.0 * t).sin()]
            })
            .collect();
        history.insert(30, [1.5, f64::NAN]);
        let v = analyze(&history, 3.0).expect("vibrato not detected through gap");
        assert!((v.rate_hz - 5.0).abs() < 0.6);
    }
}
