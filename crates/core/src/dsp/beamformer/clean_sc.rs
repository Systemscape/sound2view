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
/// Per bin, with true-location weights w_j = a_j / (|a_j| sqrt(M)), or sqrt(M - 1) with diagonal
/// removal (same levels as DAS `TrueLocation`):
/// - Dirty map P_j = w_j^H G w_j, with G = R, or R with zeroed diagonal.
/// - Iteration: maximum P_max at j_max, coherent source vector h with
///   P_max (h h^H - H) w_max = G w_max, where H = diag(|h_m|^2) with diagonal removal and 0
///   without (then simply h = R w_max / P_max); with diagonal removal h is found by fixed-point
///   iteration h = (G w / P_max + H w) / sqrt(1 + w^H H w), as in Sijtsma 2007 and Acoular.
///   Add gain * P_max to the clean map at j_max, G <- G - gain * P_max (h h^H - H). The dirty
///   map is updated with P_j <- P_j - gain * P_max (|w_j^H h|^2 - w_j^H H w_j), which costs
///   O(points * M) instead of a new map.
/// - Stops after `max_iter` sources or when P_max falls `stop_db` below the largest dirty-map
///   value of all bins. A bin without a source therefore adds nothing; with a threshold per bin
///   it would add its noise or reflection maximum.
///
/// The clean maps of all bins are summed (bins in parallel) and convolved with a Gaussian clean
/// beam. Without diagonal removal, mic noise sigma^2 I is a floor in every dirty map.
pub struct CleanSc<const M: usize> {
    steering: Steering<M>,
    /// Loop gain in (0, 1]
    gain: f64,
    /// Maximum number of removed sources per bin
    max_iter: usize,
    /// Stop when P_max < stop_ratio * largest dirty-map value of all bins
    stop_ratio: f64,
    /// Standard deviation of the Gaussian clean beam in grid cells; 0 gives single points
    beam_sigma: f64,
    /// Zero the diagonal of R (removes mic self-noise)
    remove_diag: bool,
    /// 1 / (|a| sqrt(M)), or sqrt(M - 1) with diagonal removal, per grid point; scales a to w
    weight_scale: Vec<f64>,
}

/// Fixed-point iterations for h with diagonal removal (Acoular uses 20)
const H_ITER: usize = 20;

/// Bin state after the dirty map, before cleaning
struct DirtyBin<const M: usize> {
    /// Bin index k
    bin: usize,
    /// G: R, or R with zeroed diagonal
    cov: SMatrix<Complex64, M, M>,
    dirty: Vec<f64>,
}

impl<const M: usize> CleanSc<M> {
    /// - `gain`: loop gain in (0, 1], typically 0.9
    /// - `max_iter`: maximum number of removed sources per bin, at least 1
    /// - `stop_db`: stop when the maximum is this many dB below the largest one of all bins, >= 0
    /// - `beam_sigma`: clean beam standard deviation in grid cells, >= 0
    /// - `remove_diag`: zero the diagonal of R, removes mic self-noise
    ///
    /// Panics on values outside these ranges.
    pub fn new(
        steering: Steering<M>,
        gain: f64,
        max_iter: usize,
        stop_db: f64,
        beam_sigma: f64,
        remove_diag: bool,
    ) -> Self {
        assert!(gain > 0.0 && gain <= 1.0, "gain must be in (0, 1]");
        assert!(max_iter > 0, "max_iter must be at least 1");
        assert!(stop_db >= 0.0, "stop_db must be >= 0");
        assert!(beam_sigma >= 0.0, "beam_sigma must be >= 0");

        let n_mics = if remove_diag { M - 1 } else { M } as f64;
        let weight_scale = (0..steering.n_points())
            .map(|j| {
                let norm2: f64 = steering.gains(j).iter().sum();
                1.0 / (norm2 * n_mics).sqrt()
            })
            .collect();

        Self {
            steering,
            gain,
            max_iter,
            stop_ratio: 10f64.powf(-stop_db / 10.0),
            beam_sigma,
            remove_diag,
            weight_scale,
        }
    }

    /// w_j for all grid points at bin `k`
    fn weights(&self, k: usize) -> Vec<SVector<Complex64, M>> {
        self.weight_scale
            .iter()
            .enumerate()
            .map(|(j, &scale)| self.steering.vector(j, k).scale(scale))
            .collect()
    }

    /// G and the dirty map of bin `k`
    fn dirty_bin(&self, cov_k: &SMatrix<Complex64, M, M>, k: usize) -> DirtyBin<M> {
        let mut cov = *cov_k;
        if self.remove_diag {
            cov.fill_diagonal(Complex64::from(0.0));
        }
        let dirty = self
            .weights(k)
            .iter()
            .map(|w| w.dotc(&(cov * w)).re)
            .collect();
        DirtyBin { bin: k, cov, dirty }
    }

    /// Clean one bin down to `threshold` and add its clean map to `clean`.
    fn clean_bin(&self, bin: DirtyBin<M>, threshold: f64, clean: &mut [f64]) {
        let DirtyBin {
            bin: k,
            mut cov,
            mut dirty,
        } = bin;
        let weights = self.weights(k);

        for _ in 0..self.max_iter {
            let (j_max, p_max) = dirty
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("scan grid has points");
            if p_max <= 0.0 || p_max < threshold {
                break;
            }

            let w_max = &weights[j_max];
            let (h, h_diag) = self.source_vector(&cov, w_max, p_max);
            let removed = self.gain * p_max;

            clean[j_max] += removed;
            cov -= (h * h.adjoint() - SMatrix::from_diagonal(&h_diag.map(Complex64::from)))
                .scale(removed);
            for (p_j, w_j) in dirty.iter_mut().zip(&weights) {
                let diag_part: f64 = w_j
                    .iter()
                    .zip(h_diag.iter())
                    .map(|(w_m, h2_m)| w_m.norm_sqr() * h2_m)
                    .sum();
                *p_j -= removed * (w_j.dotc(&h).norm_sqr() - diag_part);
            }
        }
    }

    /// Coherent source vector h for the maximum at `w_max`, and diag(|h_m|^2) as removed from G
    /// (zero without diagonal removal)
    fn source_vector(
        &self,
        cov: &SMatrix<Complex64, M, M>,
        w_max: &SVector<Complex64, M>,
        p_max: f64,
    ) -> (SVector<Complex64, M>, SVector<f64, M>) {
        let g_w = (cov * w_max).unscale(p_max);
        if !self.remove_diag {
            return (g_w, SVector::zeros());
        }

        let mut h = *w_max;
        for _ in 0..H_ITER {
            let h_diag = h.map(|h_m| h_m.norm_sqr());
            let h_w = w_max.zip_map(&h_diag, |w_m, h2_m| w_m.scale(h2_m));
            let w_h_w: f64 = w_max
                .zip_map(&h_diag, |w_m, h2_m| w_m.norm_sqr() * h2_m)
                .sum();
            h = (g_w + h_w).unscale((1.0 + w_h_w).sqrt());
        }
        let h_diag = h.map(|h_m| h_m.norm_sqr());
        (h, h_diag)
    }
}

impl<const M: usize> Beamformer<M> for CleanSc<M> {
    fn n_points(&self) -> usize {
        self.steering.n_points()
    }

    /// Levels as DAS `TrueLocation`: a single source with power S and no noise gives
    /// P = S (a_0^H a_0) / M at the source without diagonal removal and
    /// P = S ((a_0^H a_0)^2 - sum |a_0m|^4) / ((M - 1) a_0^H a_0) with it (before the clean beam,
    /// with gain 1).
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

        // Dirty maps first: the stop threshold is relative to the strongest bin
        let dirty_bins: Vec<DirtyBin<M>> = bins
            .into_par_iter()
            .map(|k| self.dirty_bin(&cov[k], k))
            .collect();
        let p_max_all = dirty_bins
            .iter()
            .flat_map(|bin| bin.dirty.iter().copied())
            .fold(f64::NEG_INFINITY, f64::max);
        let threshold = p_max_all * self.stop_ratio;

        // Bins are independent: each thread cleans its bins into its own map, then the maps
        // are summed
        let clean = dirty_bins
            .into_par_iter()
            .fold(
                || vec![0.0; n],
                |mut clean, bin| {
                    self.clean_bin(bin, threshold, &mut clean);
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
    use nalgebra::{SMatrix, SVector};

    use super::gaussian_blur;
    use crate::{
        Complex64,
        dsp::beamformer::{Beamformer as _, Steering, clean_sc::CleanSc},
        geometry::{mic_array::MicArray, scan_grid::ScanGrid},
        test_util::{FFT_LEN, SAMPLE_RATE, test_scan_grid},
    };

    /// Bin with a main lobe narrow enough to separate sources about 90 degrees apart
    const BIN: usize = 80;

    /// Steering vector of grid point `i` at bin `k`
    fn steering_at(scan_grid: &ScanGrid, i: usize, k: usize) -> SVector<Complex64, 16> {
        let f = k as f64 * SAMPLE_RATE / FFT_LEN as f64;
        MicArray::uma16().steering(&scan_grid.points()[i], f)
    }

    /// Clean map of `cov` at bins `bins`, no clean beam
    fn clean_map(
        cov: &[SMatrix<Complex64, 16, 16>],
        bins: std::ops::RangeInclusive<usize>,
        gain: f64,
        stop_db: f64,
        remove_diag: bool,
    ) -> Vec<f64> {
        let scan_grid = test_scan_grid();
        let steering = Steering::new(&scan_grid, &MicArray::uma16(), SAMPLE_RATE, FFT_LEN);
        let clean_sc = CleanSc::new(steering, gain, 30, stop_db, 0.0, remove_diag);
        let mut power = vec![0.0; scan_grid.points().len()];
        clean_sc.power_map(cov, bins, &mut power);
        power
    }

    /// Expected clean value of a single source a_0 (power 1), gain 1: P_max of the dirty map
    fn single_source_level(a_0: &SVector<Complex64, 16>, remove_diag: bool) -> f64 {
        let norm2 = a_0.norm_squared();
        if remove_diag {
            let sum4: f64 = a_0.iter().map(|a_m| a_m.norm_sqr().powi(2)).sum();
            (norm2 * norm2 - sum4) / (15.0 * norm2)
        } else {
            norm2 / 16.0
        }
    }

    /// Sum of |P| outside grid point `i_0`
    fn rest_outside(power: &[f64], i_0: usize) -> f64 {
        power
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != i_0)
            .map(|(_, p)| p.abs())
            .sum()
    }

    /// Single hand-built source on grid point p_0, gain 1, no clean beam: one iteration removes
    /// it completely, so the clean map is exactly the dirty maximum at p_0 and 0 elsewhere. With
    /// diagonal removal this checks the fixed-point iteration for h.
    #[test]
    fn single_source_is_one_point() {
        let scan_grid = test_scan_grid();
        let i_0 = scan_grid.index(40, 100);
        let a_0 = steering_at(&scan_grid, i_0, BIN);
        let mut cov = vec![SMatrix::zeros(); FFT_LEN / 2 + 1];
        cov[BIN] = a_0 * a_0.adjoint();

        for remove_diag in [false, true] {
            let power = clean_map(&cov, BIN..=BIN, 1.0, 20.0, remove_diag);
            let expected = single_source_level(&a_0, remove_diag);
            assert!(
                (power[i_0] - expected).abs() / expected < 1e-9,
                "remove_diag {remove_diag}: P(p_0) = {}, expected {expected}",
                power[i_0]
            );
            let rest = rest_outside(&power, i_0);
            assert!(
                rest < 1e-9 * expected,
                "remove_diag {remove_diag}: clean map outside p_0: {rest}"
            );
        }
    }

    /// Mic noise sigma^2 I 10 dB above the source per mic: with diagonal removal the clean map
    /// is the same single point as without noise; without it the noise floor adds points.
    #[test]
    fn diag_removal_ignores_mic_noise() {
        let scan_grid = test_scan_grid();
        let i_0 = scan_grid.index(40, 100);
        let a_0 = steering_at(&scan_grid, i_0, BIN);
        let sigma2 = 10.0 * a_0.norm_squared() / 16.0;
        let mut cov = vec![SMatrix::zeros(); FFT_LEN / 2 + 1];
        cov[BIN] = a_0 * a_0.adjoint()
            + SMatrix::<Complex64, 16, 16>::identity() * Complex64::from(sigma2);

        let power = clean_map(&cov, BIN..=BIN, 1.0, 20.0, true);
        let expected = single_source_level(&a_0, true);
        assert!(
            (power[i_0] - expected).abs() / expected < 1e-9,
            "P(p_0) = {}, expected {expected}",
            power[i_0]
        );
        assert!(rest_outside(&power, i_0) < 1e-9 * expected);

        let power = clean_map(&cov, BIN..=BIN, 1.0, 20.0, false);
        assert!(rest_outside(&power, i_0) > 1e-3 * expected);
    }

    /// A bin whose strongest source is below the threshold of the strongest bin adds nothing,
    /// even though it is the maximum of its own bin.
    #[test]
    fn weak_bin_below_threshold_adds_nothing() {
        let scan_grid = test_scan_grid();
        let (i_0, i_1) = (scan_grid.index(40, 100), scan_grid.index(20, 30));
        let weak_bin = BIN + 10;
        let a_0 = steering_at(&scan_grid, i_0, BIN);
        let a_1 = steering_at(&scan_grid, i_1, weak_bin);
        let mut cov = vec![SMatrix::zeros(); FFT_LEN / 2 + 1];
        cov[BIN] = a_0 * a_0.adjoint();
        // 30 dB below the source in BIN
        cov[weak_bin] = (a_1 * a_1.adjoint()).scale(1e-3);

        let power = clean_map(&cov, BIN..=weak_bin, 1.0, 20.0, true);
        assert!(power[i_0] > 0.0);
        assert_eq!(power[i_1], 0.0, "weak bin cleaned at 20 dB");

        let power = clean_map(&cov, BIN..=weak_bin, 1.0, 40.0, true);
        assert!(power[i_1] > 0.0, "weak bin not cleaned at 40 dB");
    }

    /// Two incoherent sources far apart: the strongest clean point of each image half is at
    /// (or next to) its source.
    #[test]
    fn two_sources_separated() {
        let scan_grid = test_scan_grid();
        let mic_array = MicArray::uma16();
        let steering = Steering::new(&scan_grid, &mic_array, SAMPLE_RATE, FFT_LEN);
        let clean_sc = CleanSc::new(steering, 0.9, 30, 30.0, 0.0, true);

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
