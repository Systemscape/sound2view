use std::ops::RangeInclusive;

use nalgebra::SMatrix;

use crate::{Complex64, mic_array::MicArray, scan_grid::ScanGrid};

pub trait Beamformer {}

/// Normalization c in P = c * a^H R a (Sarradj 2012, steering formulations III and IV)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Normalization {
    /// c = 1/(a^H a)^2: P = S at the source, but the peak drifts toward farther points
    TrueLevel,
    /// c = 1/(M a^H a): peak exactly at the source, P = S * (a^H a)/M there
    TrueLocation,
}

pub struct Das<const M: usize> {
    /// Precomputed distances per grid point
    distances: Vec<[f64; M]>,
    /// Normalization factor per grid point
    normalization: Vec<f64>,
    /// FFT bin spacing in Hz
    df: f64,
    /// Diagonal removal enabled
    remove_diag: bool,
}

impl<const M: usize> Das<M> {
    /// Create a new DAS (delay and sum) beamformer
    ///
    /// ## Parameters
    /// - `scan_grid`: points the beamformer scans, only used for construction
    /// - `mic_array`: microphone array configuration, only used for construction
    /// - `f_s`: sample rate in Hz
    /// - `fft_len`: FFT length N
    /// - `remove_diag`: turn on diagonal removal
    /// - `normalization`: see [`Normalization`]
    pub fn new(
        scan_grid: &ScanGrid,
        mic_array: &MicArray<M>,
        f_s: f64,
        fft_len: usize,
        remove_diag: bool,
        normalization: Normalization,
    ) -> Self {
        // Distances from each mic to each grid point
        let distances = Vec::from_iter(
            scan_grid
                .points()
                .iter()
                .map(|pos| mic_array.get_distance_from(pos)),
        );

        // With g_m = |a_m|^2 = 1/r_m^2 and sum(g_m) = a^H a:
        // - TrueLevel: `1/sum(g_m)^2`, with diagonal removal `1/(sum(g_m)^2 - sum(g_m^2))`
        // - TrueLocation: `1/(M sum(g_m))`, with diagonal removal `1/((M - 1) sum(g_m))`
        let normalization = Vec::from_iter(distances.iter().map(|r| {
            let g = r.map(|r_m| r_m.powi(-2));
            let sum_g: f64 = g.iter().sum();
            let denom = match (normalization, remove_diag) {
                (Normalization::TrueLevel, false) => sum_g.powi(2),
                (Normalization::TrueLevel, true) => {
                    sum_g.powi(2) - g.iter().map(|g_m| g_m * g_m).sum::<f64>()
                }
                (Normalization::TrueLocation, false) => M as f64 * sum_g,
                (Normalization::TrueLocation, true) => (M - 1) as f64 * sum_g,
            };
            1.0 / denom
        }));

        Self {
            distances,
            normalization,
            df: f_s / fft_len as f64,
            remove_diag,
        }
    }

    /// Get the number of points in the scanning grid
    pub fn n_points(&self) -> usize {
        self.distances.len()
    }

    /// Compute the power map
    ///
    /// P = Σ_k aᴴR_k a * c, with optional diagonal removal
    ///
    /// c is precomputed per grid point from [`Normalization`] and diagonal removal
    ///
    /// - `cov`: Covariance matrices for the whole spectrum (N/2 + 1 bins)
    /// - `bins`: absolute bin indices
    /// - Sums power over `bins` of the fft
    /// - `out`: row-major like the [`ScanGrid`], overwritten
    pub fn power_map(
        &self,
        cov: &[SMatrix<Complex64, M, M>],
        bins: RangeInclusive<usize>,
        out: &mut [f64],
    ) {
        assert_eq!(
            out.len(),
            self.distances.len(),
            "length of `out` must match number of grid points"
        );
        assert!(
            *bins.end() < cov.len(),
            "`bins` range must not exceed `cov` length"
        );

        // Ensure `out` is all zeros
        out.fill(0.0);

        for k in bins {
            // frequency of this bin
            let f_k = k as f64 * self.df;

            let cov_k = &cov[k];
            for ((r_p, normalization), out_p) in self
                .distances
                .iter()
                .zip(self.normalization.iter())
                .zip(out.iter_mut())
            {
                let a = MicArray::steering_from_distances(r_p, f_k);

                let mut p = a.dotc(&(cov_k * a)).re;

                if self.remove_diag {
                    let diag = r_p
                        .iter()
                        .enumerate()
                        .map(|(m, r_m)| r_m.powi(-2) * cov_k[(m, m)].re)
                        .sum::<f64>();
                    p -= diag;
                }
                *out_p += p * normalization;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use nalgebra::SMatrix;

    use crate::{
        beamformer::{Das, Normalization},
        mic_array::MicArray,
        reference::reference_covariance,
        stft::Stft,
        test_util::{
            BIN, F_SINE, FFT_LEN, SAMPLE_RATE, sine_bin_power, stream_covariance, test_scan_grid,
            test_scene,
        },
    };

    /// Hand-built R = a_0 a_0^H for a source exactly on grid point p_0, one bin.
    /// Expect P = 1 at p_0 and P * |a|^2 <= |a_0|^2 everywhere (Cauchy-Schwarz).
    /// P > 1 is possible at points farther from the array than p_0.
    #[test]
    fn hand_built_r_peak_at_source() {
        let row = 10;
        let col = 20;

        let scan_grid = test_scan_grid();
        let mic_array = MicArray::uma16();
        let das_beamformer = Das::new(&scan_grid, &mic_array, SAMPLE_RATE, FFT_LEN, false, Normalization::TrueLevel);

        // 1. Camera + ScanGrid, pick p_0 = grid.points()[grid.index(row, col)]
        let p_0 = scan_grid.points()[scan_grid.index(row, col)];

        // 2. a_0 = steering at p_0 for bin BIN, R_k = a_0 a_0^H in bin BIN, zeros elsewhere
        let a_0 = mic_array.steering(&p_0, F_SINE);
        let mut cov = vec![SMatrix::zeros(); FFT_LEN / 2 + 1];
        cov[BIN] = a_0 * a_0.adjoint();

        // 3. power_map over BIN..=BIN
        let mut power = vec![0.0; scan_grid.points().len()];
        das_beamformer.power_map(&cov, BIN..=BIN, &mut power);

        // 4. assert P(p_0) = 1
        assert!(
            (power[scan_grid.index(row, col)] - 1.0).abs() < 1e-12,
            "P(p_0) should be 1"
        );

        // 5. Check Cauchy Schwarz
        let a_0_norm_squared = a_0.norm_squared();

        for (power, r) in power.iter().zip(scan_grid.points()) {
            let a_i_norm_squared: f64 = mic_array
                .get_distance_from(&r)
                .iter()
                .map(|r| r.powi(-2))
                .sum();

            assert!(
                power * a_i_norm_squared <= a_0_norm_squared * (1.0 + 1e-12),
                "Cauchy-Schwarz violated at {r}: P = {power}, bound = {}",
                a_0_norm_squared / a_i_norm_squared
            );
        }
    }

    /// Oracle R from `reference_covariance` for a sine on a grid point.
    /// Expect the peak at that grid point.
    #[test]
    fn oracle_r_peak_at_source() {
        // 1. Scene with the sine source at a grid point
        // Duration does not matter here
        const DURATION_S: f64 = 0.1;
        let scan_grid = test_scan_grid();
        let mut scene = test_scene(-1000.0, DURATION_S);
        let idx_source = scan_grid.index(scan_grid.height() / 2, scan_grid.width() / 2);
        scene.sources[0].pos = scan_grid.points()[idx_source];
        let mic_array = MicArray::uma16();
        let stft = Stft::new(FFT_LEN);

        // 2. R = reference_covariance(...)
        let cov_ref = reference_covariance(&scene, &mic_array, &stft);

        // 3. power_map over BIN..=BIN
        let das_beamformer = Das::new(&scan_grid, &mic_array, SAMPLE_RATE, FFT_LEN, false, Normalization::TrueLevel);
        let mut power = vec![0.0; scan_grid.points().len()];
        das_beamformer.power_map(&cov_ref, BIN..=BIN, &mut power);

        // 4. assert argmax == index of the source grid point
        let idx_max = power
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(
            idx_max, idx_source,
            "Index of maximum should be index of source point"
        );

        let expected_power =
            sine_bin_power(scene.sources[0].signal.components()[0].amplitude, FFT_LEN);
        assert!(
            (power[idx_source] - expected_power).abs() < 1e-9 * expected_power,
            "expected power: {expected_power}, power[idx] = {}",
            power[idx_source]
        );
    }

    /// Stream R_est (simulator -> STFT -> Covariance) for the same scene.
    /// Expect the peak at the same grid point as with the oracle R.
    #[test]
    fn stream_r_peak_at_source() {
        const DURATION_S: f64 = 0.5;
        let scan_grid = test_scan_grid();
        let mut scene = test_scene(90.0, DURATION_S);
        let idx_source = scan_grid.index(scan_grid.height() / 2, scan_grid.width() / 2);
        scene.sources[0].pos = scan_grid.points()[idx_source];
        let mic_array = MicArray::uma16();

        let cov_ref = reference_covariance(&scene, &mic_array, &Stft::new(FFT_LEN));
        let (cov_est, l) = stream_covariance(&scene, &mic_array);
        assert!(l > 0, "no STFT frames");

        let das_beamformer = Das::new(&scan_grid, &mic_array, SAMPLE_RATE, FFT_LEN, false, Normalization::TrueLevel);
        let mut power_ref = vec![0.0; scan_grid.points().len()];
        let mut power_est = vec![0.0; scan_grid.points().len()];
        das_beamformer.power_map(&cov_ref, BIN..=BIN, &mut power_ref);
        das_beamformer.power_map(&cov_est, BIN..=BIN, &mut power_est);

        let idx_max = power_est
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(
            idx_max, idx_source,
            "index of maximum should be index of source point"
        );

        // Relative error of P at the source is about sqrt(2 sigma^2 / (|a|^2 S L)), ~0.3 % here
        let p_ref = power_ref[idx_source];
        let p_est = power_est[idx_source];
        assert!(
            (p_est - p_ref).abs() < 0.02 * p_ref,
            "P at source: expected {p_ref}, got {p_est}"
        );
    }

    /// Hand-built R = a_0 a_0^H on grid point p_0, diagonal removal on.
    /// Expect P(p_0) = 1: checks the new normalization and the diagonal term together.
    #[test]
    fn diag_removal_hand_built_r_at_source() {
        // 1. Same setup as `hand_built_r_peak_at_source`, but `Das::new(..., true)`
        // 2. power_map over BIN..=BIN
        // 3. assert P(p_0) = 1
        todo!()
    }

    /// R = sigma^2 I in one bin, nothing else.
    /// Expect P = 0 everywhere with diagonal removal, sigma^2 / (a^H a) without.
    #[test]
    fn diag_removal_noise_only_is_zero() {
        // 1. cov[BIN] = sigma^2 * I, zeros elsewhere
        // 2. power_map with `true`: assert |P| < tol everywhere (tol relative to sigma^2)
        // 3. power_map with `false`: assert P = sigma^2 / sum(1/r_m^2) per point
        todo!()
    }

    /// Oracle R vs stream R_est with strong mic noise, diagonal removal on.
    /// Expect the peak at the source for both, and P_est close to P_ref = S.
    #[test]
    fn diag_removal_stream_r_peak_at_source() {
        // 1. Source on the center grid point, 110 dB mic noise
        // 2. cov_ref = reference_covariance(...), (cov_est, l) = stream_covariance(...)
        // 3. power_map for both with `true`
        // 4. assert P_ref(source) = sine_bin_power (sigma^2 I is gone)
        // 5. assert argmax of P_est == source index
        // 6. assert P_est(source) close to P_ref(source)
        // 7. optional: compare with `false`, |P_est - P_ref| should be smaller with removal
        todo!()
    }
}
