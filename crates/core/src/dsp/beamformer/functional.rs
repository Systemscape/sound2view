use std::ops::RangeInclusive;

use nalgebra::{Const, DefaultAllocator, DimDiff, DimSub, SMatrix, U1, allocator::Allocator};
use rayon::prelude::*;

use crate::{
    Complex64,
    dsp::beamformer::{Beamformer, Steering},
};

/// Functional beamforming (Dougherty 2014): P = (w^H R^(1/nu) w)^nu / M, with w = a / |a|
///
/// R^(1/nu) = U diag(lambda^(1/nu)) U^H from the eigendecomposition of R per bin. The exponent
/// nu narrows the main lobe and suppresses sidelobes; nu = 1 is DAS with true-location
/// normalization and without diagonal removal. Larger nu is more sensitive to model errors
/// (mic gain and phase, focus distance, camera field of view).
///
/// Two options:
/// - `noise_removal`: subtract the median eigenvalue (noise estimate, assumes fewer than M/2
///   sources per bin) from all eigenvalues and clip at 0. The 1/nu power lifts small eigenvalues
///   close to the large ones, so uncorrelated noise would otherwise form a floor that widens the
///   main lobe. The median instead of the smallest eigenvalue, because the noise eigenvalues of
///   an estimate from L frames spread by about (1 +- sqrt(M/L))^2.
/// - `equal_bins`: normalize each bin's map to its maximum before summing, so the broad low
///   bins, where speech has most of its power, do not outweigh the sharp high ones. Bins whose
///   maximum is more than `EQUAL_BINS_GATE_DB` below the strongest bin are left out (they hold
///   no source and would add a flat map). Levels are then relative only.
pub struct Functional<const M: usize> {
    steering: Steering<M>,
    /// Exponent nu, at least 1
    nu: f64,
    /// Subtract the median eigenvalue per bin
    noise_removal: bool,
    /// Normalize each bin's map to its maximum before summing
    equal_bins: bool,
    /// 1 / (a^H a) per grid point, normalizes a to w
    inv_norm2: Vec<f64>,
}

/// With `equal_bins`: bins whose maximum is more than this below the strongest bin are left out
const EQUAL_BINS_GATE_DB: f64 = 20.0;
/// Eigenvalues below this fraction of the largest count as 0
const EIGEN_ROUNDING: f64 = 1e-12;

impl<const M: usize> Functional<M> {
    /// - `steering`: grid and array, see [`Steering`]
    /// - `nu`: exponent, typically 10 to 100
    /// - `noise_removal`, `equal_bins`: see [`Functional`]
    pub fn new(steering: Steering<M>, nu: f64, noise_removal: bool, equal_bins: bool) -> Self {
        assert!(nu >= 1.0, "nu must be at least 1");

        let inv_norm2 = (0..steering.n_points())
            .map(|i| 1.0 / steering.gains(i).iter().sum::<f64>())
            .collect();

        Self {
            steering,
            nu,
            noise_removal,
            equal_bins,
            inv_norm2,
        }
    }
}

// The bounds are what `symmetric_eigen` needs; they hold for every concrete M (e.g. 16)
impl<const M: usize> Functional<M>
where
    Const<M>: DimSub<U1>,
    DefaultAllocator: Allocator<DimDiff<Const<M>, U1>>,
{
    /// Map of bin `k` into `out` (overwritten)
    fn bin_map(&self, cov_k: &SMatrix<Complex64, M, M>, k: usize, out: &mut [f64]) {
        // R^(1/nu). Eigenvalues within rounding of 0 are set to 0: the 1/nu power would lift
        // them to a visible floor (1e-16^(1/20) = 0.16)
        let mut eigen = cov_k.symmetric_eigen();
        let mut sorted = eigen.eigenvalues;
        sorted.as_mut_slice().sort_by(f64::total_cmp);
        let noise = if self.noise_removal {
            sorted[(M - 1) / 2]
        } else {
            0.0
        };
        let rounding = EIGEN_ROUNDING * sorted[M - 1].abs();
        eigen.eigenvalues.apply(|lambda| {
            let shifted = *lambda - noise;
            *lambda = if shifted > rounding {
                shifted.powf(1.0 / self.nu)
            } else {
                0.0
            };
        });
        let cov_root = eigen.recompose();

        // Grid points are independent
        let points = self
            .inv_norm2
            .par_iter()
            .zip(out.par_iter_mut())
            .enumerate();
        points.for_each(|(i, (inv_norm2, out_i))| {
            let a = self.steering.vector(i, k);
            // w^H R^(1/nu) w with w = a / |a|
            let q = a.dotc(&(cov_root * a)).re * inv_norm2;
            *out_i = q.max(0.0).powf(self.nu) / M as f64;
        });
    }
}

impl<const M: usize> Beamformer<M> for Functional<M>
where
    Const<M>: DimSub<U1>,
    DefaultAllocator: Allocator<DimDiff<Const<M>, U1>>,
{
    fn n_points(&self) -> usize {
        self.steering.n_points()
    }

    /// For a single source with power S and no noise, P = S (a_0^H a_0) / M at the source,
    /// the same as DAS with true-location normalization (without `equal_bins`).
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

        // One map per bin (bins in parallel), combined below
        let maps: Vec<Vec<f64>> = bins
            .into_par_iter()
            .map(|k| {
                let mut map = vec![0.0; n];
                self.bin_map(&cov[k], k, &mut map);
                map
            })
            .collect();
        let max_of = |map: &[f64]| map.iter().copied().fold(0.0, f64::max);

        out.fill(0.0);
        if self.equal_bins {
            let strongest = maps.iter().map(|map| max_of(map)).fold(0.0, f64::max);
            let gate = strongest * 10f64.powf(-EQUAL_BINS_GATE_DB / 10.0);
            for map in &maps {
                let max = max_of(map);
                if max > 0.0 && max >= gate {
                    out.iter_mut().zip(map).for_each(|(o, p)| *o += p / max);
                }
            }
        } else {
            for map in &maps {
                out.iter_mut().zip(map).for_each(|(o, p)| *o += p);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use nalgebra::SMatrix;

    use crate::{
        Complex64,
        dsp::beamformer::{Beamformer as _, Steering, functional::Functional},
        geometry::mic_array::MicArray,
        test_util::{BIN, F_SINE, FFT_LEN, SAMPLE_RATE, test_scan_grid},
    };

    const NU: f64 = 20.0;

    /// Grid point index of the maximum
    fn argmax(power: &[f64]) -> usize {
        power
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Hand-built R = a_0 a_0^H for a source on grid point p_0, one bin.
    /// Expect the maximum at p_0 with P = |a_0|^2 / M.
    #[test]
    fn hand_built_r_peak_at_source() {
        let scan_grid = test_scan_grid();
        let mic_array = MicArray::uma16();
        let steering = Steering::new(&scan_grid, &mic_array, SAMPLE_RATE, FFT_LEN);
        let functional = Functional::new(steering, NU, false, false);

        let i_0 = scan_grid.index(10, 20);
        let a_0 = mic_array.steering(&scan_grid.points()[i_0], F_SINE);
        let mut cov = vec![SMatrix::zeros(); FFT_LEN / 2 + 1];
        cov[BIN] = a_0 * a_0.adjoint();

        let mut power = vec![0.0; scan_grid.points().len()];
        functional.power_map(&cov, BIN..=BIN, &mut power);

        assert_eq!(argmax(&power), i_0);
        let expected = a_0.norm_squared() / 16.0;
        let rel_error = (power[i_0] - expected).abs() / expected;
        assert!(
            rel_error < 1e-6,
            "P(p_0) = {}, expected {expected}",
            power[i_0]
        );
    }

    /// With white noise sigma^2 I added, the maximum stays at p_0 and the map is sharper than
    /// for nu = 1: fewer grid points within 3 dB of the maximum.
    #[test]
    fn noise_peak_at_source_and_sharper_than_nu_1() {
        let scan_grid = test_scan_grid();
        let mic_array = MicArray::uma16();

        let i_0 = scan_grid.index(10, 20);
        let a_0 = mic_array.steering(&scan_grid.points()[i_0], F_SINE);
        let mut cov = vec![SMatrix::zeros(); FFT_LEN / 2 + 1];
        // Noise 10 dB below the source power per mic
        let sigma2 = 0.1 * a_0.norm_squared() / 16.0;
        cov[BIN] = a_0 * a_0.adjoint()
            + SMatrix::<Complex64, 16, 16>::identity() * Complex64::from(sigma2);

        let within_3db = |nu: f64| {
            let steering = Steering::new(&scan_grid, &mic_array, SAMPLE_RATE, FFT_LEN);
            let mut power = vec![0.0; scan_grid.points().len()];
            Functional::new(steering, nu, false, false).power_map(&cov, BIN..=BIN, &mut power);
            assert_eq!(argmax(&power), i_0, "nu = {nu}");
            let p_max = power[i_0];
            power.iter().filter(|&&p| p >= 0.5 * p_max).count()
        };

        let wide = within_3db(1.0);
        let narrow = within_3db(NU);
        assert!(
            narrow < wide,
            "nu = {NU}: {narrow} points within 3 dB, nu = 1: {wide}"
        );
    }

    /// Map of `cov` over `bins` with the given options
    fn map(
        cov: &[SMatrix<Complex64, 16, 16>],
        bins: std::ops::RangeInclusive<usize>,
        noise_removal: bool,
        equal_bins: bool,
    ) -> Vec<f64> {
        let scan_grid = test_scan_grid();
        let steering = Steering::new(&scan_grid, &MicArray::uma16(), SAMPLE_RATE, FFT_LEN);
        let mut power = vec![0.0; scan_grid.points().len()];
        Functional::new(steering, NU, noise_removal, equal_bins).power_map(cov, bins, &mut power);
        power
    }

    /// Steering vector of grid point `i` at bin `k`
    fn steering_at(i: usize, k: usize) -> nalgebra::SVector<Complex64, 16> {
        let f = k as f64 * SAMPLE_RATE / FFT_LEN as f64;
        MicArray::uma16().steering(&test_scan_grid().points()[i], f)
    }

    /// White noise sigma^2 I as strong as the source per mic: noise removal gives the map of the
    /// noise-free R.
    #[test]
    fn noise_removal_restores_noise_free_map() {
        let i_0 = test_scan_grid().index(10, 20);
        let a_0 = steering_at(i_0, BIN);
        let source = a_0 * a_0.adjoint();
        let sigma2 = a_0.norm_squared() / 16.0;
        let mut clean_cov = vec![SMatrix::zeros(); FFT_LEN / 2 + 1];
        clean_cov[BIN] = source;
        let mut noisy_cov = clean_cov.clone();
        noisy_cov[BIN] += SMatrix::<Complex64, 16, 16>::identity() * Complex64::from(sigma2);

        let expected = map(&clean_cov, BIN..=BIN, false, false);
        let power = map(&noisy_cov, BIN..=BIN, true, false);
        let p_max = expected[i_0];
        let worst = expected
            .iter()
            .zip(&power)
            .map(|(e, p)| (e - p).abs())
            .fold(0.0, f64::max);
        assert!(
            worst < 1e-6 * p_max,
            "largest difference {worst}, peak {p_max}"
        );

        // Without noise removal the noise lifts the map around the peak
        let without = map(&noisy_cov, BIN..=BIN, false, false);
        let width = |power: &[f64]| power.iter().filter(|&&p| p >= 0.5 * power[i_0]).count();
        assert!(width(&power) < width(&without));
    }

    /// Equal bin weighting: a source 15 dB weaker in its own bin gets the same peak as the strong
    /// one; a bin 30 dB below the strongest is left out.
    #[test]
    fn equal_bins_weights_bins_alike() {
        let scan_grid = test_scan_grid();
        let i = [
            scan_grid.index(10, 20),
            scan_grid.index(50, 150),
            scan_grid.index(30, 90),
        ];
        let bins = [BIN, BIN + 10, BIN + 20];
        let levels = [1.0, 10f64.powf(-1.5), 1e-3];
        let mut cov = vec![SMatrix::zeros(); FFT_LEN / 2 + 1];
        for ((i, k), level) in i.iter().zip(bins).zip(levels) {
            let a = steering_at(*i, k);
            cov[k] = (a * a.adjoint()).scale(level);
        }

        let power = map(&cov, BIN..=BIN + 20, false, true);
        let (p_0, p_1, p_2) = (power[i[0]], power[i[1]], power[i[2]]);
        assert!(
            (p_0 - p_1).abs() < 0.05 * p_0,
            "P(p_0) = {p_0}, P(p_1) = {p_1}"
        );
        assert!(p_2 < 0.01 * p_0, "gated bin: P(p_2) = {p_2}");

        // Plain sum: the weak source stays weak
        let power = map(&cov, BIN..=BIN + 20, false, false);
        assert!(power[i[1]] < 0.1 * power[i[0]]);
    }
}
