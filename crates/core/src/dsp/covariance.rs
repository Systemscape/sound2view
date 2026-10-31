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

/// Exponentially weighted covariance for live maps: R_est <- R_est + w (x x^H - R_est) per frame.
///
/// `w = max(1 - alpha, 1/l)` for frame l = 1, 2, ...: the plain mean while l < 1/(1 - alpha)
/// (no start-up bias toward zero), then the exponential average with memory of about
/// 1/(1 - alpha) frames. Never reset; a map can be computed after any frame.
pub struct SlidingCovariance<const M: usize> {
    estimate: Vec<SMatrix<Complex64, M, M>>,
    /// Forgetting factor in [0, 1)
    alpha: f64,
    count: usize,
}

impl<const M: usize> SlidingCovariance<M> {
    /// - `n_bins`: usually fft_len / 2 + 1
    /// - `memory`: effective window in frames, alpha = 1 - 1/memory
    pub fn new(n_bins: usize, memory: usize) -> Self {
        assert!(memory > 0, "memory must be at least one frame");
        Self {
            estimate: vec![SMatrix::zeros(); n_bins],
            alpha: 1.0 - 1.0 / memory as f64,
            count: 0,
        }
    }

    pub fn accumulate(&mut self, snapshots: &[SVector<Complex64, M>]) {
        // zip would stop at the shorter iterator, which would be hard to catch
        assert_eq!(
            snapshots.len(),
            self.estimate.len(),
            "snapshot bin count must match"
        );

        self.count += 1;
        let w = (1.0 - self.alpha).max(1.0 / self.count as f64);
        for (r, x) in self.estimate.iter_mut().zip(snapshots) {
            *r += (x * x.adjoint() - *r).scale(w);
        }
    }

    /// Frames accumulated so far
    pub fn count(&self) -> usize {
        self.count
    }

    /// Current R_est per bin
    ///
    /// Panics if no frame was accumulated yet
    pub fn estimate(&self) -> &[SMatrix<Complex64, M, M>] {
        assert!(self.count > 0, "Count must be >0");
        &self.estimate
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::SVector;

    use super::{Covariance, SlidingCovariance};
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

    /// Within the memory the sliding estimate is the plain mean, like `Covariance`
    #[test]
    fn sliding_equals_mean_during_start() {
        let mut plain = Covariance::<M>::new(N_BINS);
        let mut sliding = SlidingCovariance::<M>::new(N_BINS, 10);
        for seed in [0.3, -1.7, 2.5] {
            plain.accumulate(&snapshots(seed));
            sliding.accumulate(&snapshots(seed));
        }

        for (k, (p, s)) in plain.estimate().iter().zip(sliding.estimate()).enumerate() {
            assert!((p - s).norm() < 1e-12, "bin {k}");
        }
    }

    /// After a change of input, old frames decay: the estimate converges to the new x x^H
    #[test]
    fn sliding_forgets_old_frames() {
        let x = snapshots(1.0);
        let y = snapshots(-2.0);
        let mut cov = SlidingCovariance::<M>::new(N_BINS, 10);
        for _ in 0..10 {
            cov.accumulate(&x);
        }
        // Weight of the x frames after 200 y frames: 0.9^200 ~ 7e-10
        for _ in 0..200 {
            cov.accumulate(&y);
        }

        for k in 0..N_BINS {
            let target = y[k] * y[k].adjoint();
            let error = (cov.estimate()[k] - target).norm() / target.norm();
            assert!(error < 1e-8, "bin {k}: relative error {error}");
        }
    }

    #[test]
    #[should_panic]
    fn sliding_estimate_without_frames_panics() {
        SlidingCovariance::<M>::new(N_BINS, 10).estimate();
    }
}
