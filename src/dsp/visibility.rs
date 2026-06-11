//! Horizontal visibility graph (HVG) complexity metrics.
//!
//! Maps a time series to a network: nodes are samples, and two samples are
//! connected if every sample strictly between them is lower than both
//! (they can "see" each other horizontally). Network statistics then act
//! as a texture/complexity signature of the audio envelope: a sustained
//! tone yields a near-chain graph (mean degree → 2), while percussive or
//! noisy textures yield hubs and heavier-tailed degree distributions.
//! Construction is O(n) amortized via a monotonic stack (Luque et al. 2009).

pub const HIST_BINS: usize = 10;

#[derive(Clone, Debug, Default)]
pub struct VisMetrics {
    pub nodes: usize,
    pub edges: usize,
    pub mean_degree: f32,
    /// 2E / n(n−1).
    pub density: f32,
    pub max_degree: usize,
    /// Degree counts: index d = nodes with degree d (last bin = ≥ HIST_BINS−1).
    pub histogram: [u32; HIST_BINS],
}

pub fn horizontal_visibility(series: &[f32]) -> VisMetrics {
    let n = series.len();
    if n < 2 {
        return VisMetrics::default();
    }

    let mut degree = vec![0u32; n];
    let mut edges = 0usize;
    // Monotonic decreasing stack of indices.
    let mut stack: Vec<usize> = Vec::with_capacity(n);

    let connect = |a: usize, b: usize, degree: &mut [u32], edges: &mut usize| {
        degree[a] += 1;
        degree[b] += 1;
        *edges += 1;
    };

    for i in 0..n {
        // Everything on the stack lower than series[i] is visible from i
        // and then gets blocked by i.
        while let Some(&top) = stack.last() {
            if series[top] < series[i] {
                connect(top, i, &mut degree, &mut edges);
                stack.pop();
            } else {
                break;
            }
        }
        // The first element ≥ series[i] (if any) also sees i.
        if let Some(&top) = stack.last() {
            connect(top, i, &mut degree, &mut edges);
            // Equal heights block each other: pop the equal one.
            if series[top] == series[i] {
                stack.pop();
            }
        }
        stack.push(i);
    }

    let mut histogram = [0u32; HIST_BINS];
    let mut max_degree = 0usize;
    for &d in &degree {
        let d = d as usize;
        max_degree = max_degree.max(d);
        histogram[d.min(HIST_BINS - 1)] += 1;
    }

    VisMetrics {
        nodes: n,
        edges,
        mean_degree: 2.0 * edges as f32 / n as f32,
        density: 2.0 * edges as f32 / (n as f32 * (n - 1) as f32),
        max_degree,
        histogram,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_series_is_a_chain() {
        let m = horizontal_visibility(&[1.0; 64]);
        // Equal neighbors see only each other: n−1 edges, mean degree → 2.
        assert_eq!(m.edges, 63);
        assert!((m.mean_degree - 2.0).abs() < 0.1, "{}", m.mean_degree);
    }

    #[test]
    fn alternating_series_known_structure() {
        // 1,2,1,2,... high nodes see both neighbors and over the lows:
        // every adjacent pair connects (n−1) plus each pair of consecutive
        // highs connects over the low between them.
        let series: Vec<f32> = (0..64)
            .map(|i| if i % 2 == 0 { 1.0 } else { 2.0 })
            .collect();
        let m = horizontal_visibility(&series);
        let highs = 32;
        assert_eq!(m.edges, 63 + (highs - 1));
        assert!(m.max_degree >= 4);
    }

    #[test]
    fn noise_is_more_complex_than_constant() {
        let mut x = 99u32;
        let noise: Vec<f32> = (0..256)
            .map(|_| {
                x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                x as f32 / u32::MAX as f32
            })
            .collect();
        let noisy = horizontal_visibility(&noise);
        let constant = horizontal_visibility(&[0.5; 256]);
        assert!(
            noisy.mean_degree > constant.mean_degree + 0.5,
            "noise {} vs constant {}",
            noisy.mean_degree,
            constant.mean_degree
        );
    }
}
