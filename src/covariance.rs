use nalgebra::{SMatrix, SVector};
use realfft::num_traits::Zero;

use crate::Complex64;

pub struct Covariance<const M: usize> {
    sum: Vec<SMatrix<Complex64, M, M>>,
    count: usize,
}

impl<const M: usize> Covariance<M> {
    /// n_bins is usually fft_len / 2 + 1
    pub fn new(n_bins: usize) -> Self {
        Self {
            sum: vec![SMatrix::zeros(); n_bins],
            count: 0,
        }
    }

    pub fn accumulate(&mut self, snapshots: &[SVector<Complex64, M>]) {
        // zip would stop at the shorter iterator, which would be hard to catch
        assert_eq!(
            snapshots.len(),
            self.sum.len(),
            "snapshot bin count must match"
        );

        for (sum, x) in self.sum.iter_mut().zip(snapshots) {
            *sum += x * x.adjoint();
        }
        self.count += 1;
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// R_est = sum / L
    ///
    /// Panics if L = 0
    ///
    /// Does **not** call [`reset()`]
    pub fn estimate(&self) -> Vec<SMatrix<Complex64, M, M>> {
        assert!(self.count > 0, "Count must be >0");

        self.sum
            .iter()
            .map(|x| x.unscale(self.count as f64))
            .collect()
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.sum.iter_mut().for_each(|x| x.fill(Complex64::zero()));
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::SVector;

    use super::Covariance;
    use crate::Complex64;

    const M: usize = 3;
    const N_BINS: usize = 4;

    /// Deterministic snapshots, different per bin, channel and `seed`.
    fn snapshots(seed: f64) -> Vec<SVector<Complex64, M>> {
        (0..N_BINS)
            .map(|k| {
                SVector::from_fn(|m, _| {
                    Complex64::new(seed + (k + m) as f64, seed * (k as f64 - m as f64) + 0.5)
                })
            })
            .collect()
    }

    #[test]
    fn single_frame_is_outer_product() {
        let x = snapshots(1.0);
        let mut cov = Covariance::<M>::new(N_BINS);
        cov.accumulate(&x);

        let r_est = cov.estimate();
        for k in 0..N_BINS {
            assert_eq!(r_est[k], x[k] * x[k].adjoint(), "bin {k}");
        }
    }

    #[test]
    fn average_over_frames() {
        let x = snapshots(1.0);
        let y = snapshots(-2.0);
        let mut cov = Covariance::<M>::new(N_BINS);
        cov.accumulate(&x);
        cov.accumulate(&y);

        let r_est = cov.estimate();
        for k in 0..N_BINS {
            let expected = (x[k] * x[k].adjoint() + y[k] * y[k].adjoint()) / Complex64::from(2.0);
            assert!((r_est[k] - expected).norm() < 1e-12, "bin {k}");
        }
    }

    #[test]
    fn hermitian_with_real_nonnegative_diagonal() {
        let mut cov = Covariance::<M>::new(N_BINS);
        for seed in [0.3, -1.7, 2.5] {
            cov.accumulate(&snapshots(seed));
        }

        for (k, r) in cov.estimate().iter().enumerate() {
            assert!((r - r.adjoint()).norm() < 1e-12, "bin {k}: not Hermitian");
            for m in 0..M {
                assert!(r[(m, m)].im.abs() < 1e-12, "bin {k}: diagonal {m} not real");
                assert!(r[(m, m)].re >= 0.0, "bin {k}: diagonal {m} negative");
            }
        }
    }

    #[test]
    #[should_panic]
    fn estimate_without_frames_panics() {
        Covariance::<M>::new(N_BINS).estimate();
    }

    #[test]
    #[should_panic]
    fn estimate_after_reset_panics() {
        let mut cov = Covariance::<M>::new(N_BINS);
        cov.accumulate(&snapshots(1.0));
        Covariance::<M>::new(N_BINS).estimate();
        cov.reset();
        cov.estimate();
    }

    #[test]
    #[should_panic]
    fn wrong_bin_count_panics() {
        Covariance::<M>::new(N_BINS + 1).accumulate(&snapshots(1.0));
    }
}
