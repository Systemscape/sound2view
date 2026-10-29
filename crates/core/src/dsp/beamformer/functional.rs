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
pub struct Functional<const M: usize> {
    steering: Steering<M>,
    /// Exponent nu, at least 1
    nu: f64,
    /// 1 / (a^H a) per grid point, normalizes a to w
    inv_norm2: Vec<f64>,
}

impl<const M: usize> Functional<M> {
    /// - `steering`: grid and array, see [`Steering`]
    /// - `nu`: exponent, typically 10 to 100
    pub fn new(steering: Steering<M>, nu: f64) -> Self {
        assert!(nu >= 1.0, "nu must be at least 1");

        let inv_norm2 = (0..steering.n_points())
            .map(|i| 1.0 / steering.gains(i).iter().sum::<f64>())
            .collect();

        Self {
            steering,
            nu,
            inv_norm2,
        }
    }
}

// The bounds are what `symmetric_eigen` needs; they hold for every concrete M (e.g. 16)
impl<const M: usize> Beamformer<M> for Functional<M>
where
    Const<M>: DimSub<U1>,
    DefaultAllocator: Allocator<DimDiff<Const<M>, U1>>,
{
    fn n_points(&self) -> usize {
        self.steering.n_points()
    }

    /// For a single source with power S and no noise, P = S (a_0^H a_0) / M at the source,
    /// the same as DAS with true-location normalization.
    fn power_map(
        &self,
        cov: &[SMatrix<Complex64, M, M>],
        bins: RangeInclusive<usize>,
        out: &mut [f64],
    ) {
        assert_eq!(
            out.len(),
            self.steering.n_points(),
            "length of `out` must match number of grid points"
        );
        assert!(
            *bins.end() < cov.len(),
            "`bins` range must not exceed `cov` length"
        );

        out.fill(0.0);

        for k in bins {
            // R^(1/nu); small negative eigenvalues from rounding are clipped to 0
            let mut eigen = cov[k].symmetric_eigen();
            eigen
                .eigenvalues
                .apply(|lambda| *lambda = lambda.max(0.0).powf(1.0 / self.nu));
            let cov_root = eigen.recompose();

            // Grid points are independent: parallel over points, bins stay sequential
            let points = self.inv_norm2.par_iter().zip(out.par_iter_mut()).enumerate();
            points.for_each(|(i, (inv_norm2, out_i))| {
                let a = self.steering.vector(i, k);
                // w^H R^(1/nu) w with w = a / |a|
                let q = a.dotc(&(cov_root * a)).re * inv_norm2;
                *out_i += q.max(0.0).powf(self.nu) / M as f64;
            });
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
        let functional = Functional::new(steering, NU);

        let i_0 = scan_grid.index(10, 20);
        let a_0 = mic_array.steering(&scan_grid.points()[i_0], F_SINE);
        let mut cov = vec![SMatrix::zeros(); FFT_LEN / 2 + 1];
        cov[BIN] = a_0 * a_0.adjoint();

        let mut power = vec![0.0; scan_grid.points().len()];
        functional.power_map(&cov, BIN..=BIN, &mut power);

        assert_eq!(argmax(&power), i_0);
        let expected = a_0.norm_squared() / 16.0;
        let rel_error = (power[i_0] - expected).abs() / expected;
        assert!(rel_error < 1e-6, "P(p_0) = {}, expected {expected}", power[i_0]);
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
            Functional::new(steering, nu).power_map(&cov, BIN..=BIN, &mut power);
            assert_eq!(argmax(&power), i_0, "nu = {nu}");
            let p_max = power[i_0];
            power.iter().filter(|&&p| p >= 0.5 * p_max).count()
        };

        let wide = within_3db(1.0);
        let narrow = within_3db(NU);
        assert!(narrow < wide, "nu = {NU}: {narrow} points within 3 dB, nu = 1: {wide}");
    }
}
