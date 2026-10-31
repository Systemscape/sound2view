use std::ops::RangeInclusive;

use nalgebra::{SMatrix, SVector};
use rayon::prelude::*;

use crate::{
    Complex64,
    dsp::beamformer::{Beamformer, Steering},
};

/// CLEAN-SC (Sijtsma 2007): removes sources one by one from R, using the coherence of the
/// strongest source with the rest of the array instead of a modeled point spread function.
///
/// Per bin, with true-location weights w_j = a_j / (|a_j| sqrt(M)) (same levels as DAS
/// `TrueLocation`):
/// - Dirty map P_j = w_j^H R w_j.
/// - Iteration: maximum P_max at j_max, coherent part h = R w_max / P_max, add gain * P_max to
///   the clean map at j_max, R <- R - gain * P_max h h^H. The dirty map is updated with
///   P_j <- P_j - gain * P_max |w_j^H h|^2, which costs O(points * M) instead of a new map.
/// - Stops after `max_iter` sources or when P_max falls `stop_db` below the first maximum.
///
/// The clean maps of all bins are summed (bins in parallel) and convolved with a Gaussian clean
/// beam. No diagonal removal: mic noise sigma^2 I is a floor in the dirty map, which the stop
/// criterion keeps out of the clean map.
pub struct CleanSc<const M: usize> {
    steering: Steering<M>,
    /// Loop gain in (0, 1]
    gain: f64,
    /// Maximum number of removed sources per bin
    max_iter: usize,
    /// Stop when P_max < stop_ratio * first P_max of the bin
    stop_ratio: f64,
    /// Standard deviation of the Gaussian clean beam in grid cells; 0 gives single points
    beam_sigma: f64,
    /// 1 / (|a| sqrt(M)) per grid point, scales a to w
    weight_scale: Vec<f64>,
}

impl<const M: usize> CleanSc<M> {
    /// - `gain`: loop gain in (0, 1], typically 0.9
    /// - `max_iter`: maximum number of removed sources per bin, at least 1
    /// - `stop_db`: stop when the maximum is this many dB below the first one, >= 0
    /// - `beam_sigma`: clean beam standard deviation in grid cells, >= 0
    ///
    /// Panics on values outside these ranges.
    pub fn new(
        steering: Steering<M>,
        gain: f64,
        max_iter: usize,
        stop_db: f64,
        beam_sigma: f64,
    ) -> Self {
        assert!(gain > 0.0 && gain <= 1.0, "gain must be in (0, 1]");
        assert!(max_iter > 0, "max_iter must be at least 1");
        assert!(stop_db >= 0.0, "stop_db must be >= 0");
        assert!(beam_sigma >= 0.0, "beam_sigma must be >= 0");

        let weight_scale = (0..steering.n_points())
            .map(|j| {
                let norm2: f64 = steering.gains(j).iter().sum();
                1.0 / (norm2 * M as f64).sqrt()
            })
            .collect();

        Self {
            steering,
            gain,
            max_iter,
            stop_ratio: 10f64.powf(-stop_db / 10.0),
            beam_sigma,
            weight_scale,
        }
    }

    /// Run CLEAN-SC on bin `k` and add its clean map to `clean`.
    fn clean_bin(&self, cov_k: &SMatrix<Complex64, M, M>, k: usize, clean: &mut [f64]) {
        let weights: Vec<SVector<Complex64, M>> = self
            .weight_scale
            .iter()
            .enumerate()
            .map(|(j, &scale)| self.steering.vector(j, k).scale(scale))
            .collect();

        let mut cov = *cov_k;
        let mut dirty: Vec<f64> = weights.iter().map(|w| w.dotc(&(cov * w)).re).collect();
        let mut first_max = None;

        for _ in 0..self.max_iter {
            let (j_max, p_max) = dirty
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("scan grid has points");
            let first = *first_max.get_or_insert(p_max);
            if p_max <= 0.0 || p_max < first * self.stop_ratio {
                break;
            }

            // Coherent part of the strongest source, from the degraded R
            let h = (cov * weights[j_max]).unscale(p_max);
            let removed = self.gain * p_max;

            clean[j_max] += removed;
            cov -= (h * h.adjoint()).scale(removed);
            for (p_j, w_j) in dirty.iter_mut().zip(&weights) {
                *p_j -= removed * w_j.dotc(&h).norm_sqr();
            }
        }
    }
}

impl<const M: usize> Beamformer<M> for CleanSc<M> {
    fn n_points(&self) -> usize {
        self.steering.n_points()
    }

    /// Levels as DAS `TrueLocation`: a single source with power S and no noise gives
    /// P = S (a_0^H a_0) / M at the source (before the clean beam, with gain 1).
    fn power_map(
        &self,
        cov: &[SMatrix<Complex64, M, M>],
        bins: RangeInclusive<usize>,
        out: &mut [f64],
    ) {
        let n = self.steering.n_points();
        assert_eq!(
            out.len(),
            n,
            "length of `out` must match number of grid points"
        );
        assert!(
            *bins.end() < cov.len(),
            "`bins` range must not exceed `cov` length"
        );

        // Bins are independent: each thread cleans its bins into its own map, then the maps
        // are summed
        let clean = bins
            .into_par_iter()
            .fold(
                || vec![0.0; n],
                |mut clean, k| {
                    self.clean_bin(&cov[k], k, &mut clean);
                    clean
                },
            )
            .reduce(
                || vec![0.0; n],
                |mut a, b| {
                    a.iter_mut().zip(&b).for_each(|(a_j, b_j)| *a_j += b_j);
                    a
                },
            );

        gaussian_blur(&clean, self.steering.grid_size(), self.beam_sigma, out);
    }
}

/// Separable Gaussian with peak 1 (a point keeps its value at the center); `sigma` in cells.
/// `sigma = 0` copies `input`. Cells outside the grid are left out.
fn gaussian_blur(input: &[f64], [width, height]: [usize; 2], sigma: f64, out: &mut [f64]) {
    if sigma == 0.0 {
        out.copy_from_slice(input);
        return;
    }

    let radius = (3.0 * sigma).ceil() as isize;
    let kernel: Vec<f64> = (-radius..=radius)
        .map(|d| (-(d * d) as f64 / (2.0 * sigma * sigma)).exp())
        .collect();

    // Sum of kernel[d] * line[i + d] over the cells that exist
    let convolve = |line: &dyn Fn(isize) -> Option<f64>, i: isize| {
        (-radius..=radius)
            .filter_map(|d| line(i + d).map(|v| v * kernel[(d + radius) as usize]))
            .sum::<f64>()
    };

    let mut rows = vec![0.0; input.len()];
    for r in 0..height {
        let row = |c: isize| {
            (0..width as isize)
                .contains(&c)
                .then(|| input[r * width + c as usize])
        };
        for c in 0..width {
            rows[r * width + c] = convolve(&row, c as isize);
        }
    }
    for c in 0..width {
        let col = |r: isize| {
            (0..height as isize)
                .contains(&r)
                .then(|| rows[r as usize * width + c])
        };
        for r in 0..height {
            out[r * width + c] = convolve(&col, r as isize);
        }
    }
}

#[cfg(test)]
mod test {
    use nalgebra::SMatrix;

    use super::gaussian_blur;
    use crate::{
        dsp::beamformer::{Beamformer as _, Steering, clean_sc::CleanSc},
        geometry::mic_array::MicArray,
        test_util::{FFT_LEN, SAMPLE_RATE, test_scan_grid},
    };

    /// Bin with a main lobe narrow enough to separate sources about 90 degrees apart
    const BIN: usize = 80;

    /// Single hand-built source on grid point p_0, gain 1, no clean beam: one iteration removes
    /// it completely, so the clean map is exactly |a_0|^2 / M at p_0 and 0 elsewhere.
    #[test]
    fn single_source_is_one_point() {
        let scan_grid = test_scan_grid();
        let mic_array = MicArray::uma16();
        let steering = Steering::new(&scan_grid, &mic_array, SAMPLE_RATE, FFT_LEN);
        let clean_sc = CleanSc::new(steering, 1.0, 5, 20.0, 0.0);

        let i_0 = scan_grid.index(40, 100);
        let f = BIN as f64 * SAMPLE_RATE / FFT_LEN as f64;
        let a_0 = mic_array.steering(&scan_grid.points()[i_0], f);
        let mut cov = vec![SMatrix::zeros(); FFT_LEN / 2 + 1];
        cov[BIN] = a_0 * a_0.adjoint();

        let mut power = vec![0.0; scan_grid.points().len()];
        clean_sc.power_map(&cov, BIN..=BIN, &mut power);

        let expected = a_0.norm_squared() / 16.0;
        assert!(
            (power[i_0] - expected).abs() / expected < 1e-9,
            "P(p_0) = {}",
            power[i_0]
        );
        let rest: f64 = power
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != i_0)
            .map(|(_, p)| p.abs())
            .sum();
        assert!(rest < 1e-9 * expected, "clean map outside p_0: {rest}");
    }

    /// Two incoherent sources far apart: the strongest clean point of each image half is at
    /// (or next to) its source.
    #[test]
    fn two_sources_separated() {
        let scan_grid = test_scan_grid();
        let mic_array = MicArray::uma16();
        let steering = Steering::new(&scan_grid, &mic_array, SAMPLE_RATE, FFT_LEN);
        let clean_sc = CleanSc::new(steering, 0.9, 30, 30.0, 0.0);

        let row = scan_grid.height() / 2;
        let (col_1, col_2) = (scan_grid.width() / 6, 5 * scan_grid.width() / 6);
        let f = BIN as f64 * SAMPLE_RATE / FFT_LEN as f64;
        let a_1 = mic_array.steering(&scan_grid.points()[scan_grid.index(row, col_1)], f);
        let a_2 = mic_array.steering(&scan_grid.points()[scan_grid.index(row, col_2)], f);
        let mut cov = vec![SMatrix::zeros(); FFT_LEN / 2 + 1];
        cov[BIN] = a_1 * a_1.adjoint() + (a_2 * a_2.adjoint()).scale(0.5);

        let mut power = vec![0.0; scan_grid.points().len()];
        clean_sc.power_map(&cov, BIN..=BIN, &mut power);

        let width = scan_grid.width();
        let half_max = |cols: std::ops::Range<usize>| {
            (0..scan_grid.height())
                .flat_map(|r| cols.clone().map(move |c| (r, c)))
                .max_by(|&(r1, c1), &(r2, c2)| {
                    power[r1 * width + c1].total_cmp(&power[r2 * width + c2])
                })
                .unwrap()
        };
        for ((r, c), col) in [
            (half_max(0..width / 2), col_1),
            (half_max(width / 2..width), col_2),
        ] {
            assert!(
                r.abs_diff(row) <= 1 && c.abs_diff(col) <= 1,
                "maximum at ({r}, {c}), source at ({row}, {col})"
            );
        }
    }

    /// The clean beam keeps a single point's value at its center and spreads symmetrically.
    #[test]
    fn blur_keeps_peak_and_is_symmetric() {
        let (width, height) = (9, 7);
        let mut input = vec![0.0; width * height];
        input[3 * width + 4] = 2.0;
        let mut out = vec![0.0; width * height];
        gaussian_blur(&input, [width, height], 1.0, &mut out);

        assert!((out[3 * width + 4] - 2.0).abs() < 1e-12);
        assert!((out[3 * width + 3] - out[3 * width + 5]).abs() < 1e-12);
        assert!((out[2 * width + 4] - out[4 * width + 4]).abs() < 1e-12);
        assert!(out[3 * width + 5] < out[3 * width + 4]);
    }
}
