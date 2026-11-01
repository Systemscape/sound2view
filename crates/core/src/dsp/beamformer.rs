use std::ops::RangeInclusive;

use nalgebra::{SMatrix, SVector};

use crate::{
    Complex64,
    geometry::{mic_array::MicArray, scan_grid::ScanGrid},
};

pub mod clean_sc;
pub mod delay_and_sum;
pub mod functional;

/// Steering vectors for all scan grid points; the geometry is precomputed, the frequency part is
/// built per call
#[derive(Clone)]
pub struct Steering<const M: usize> {
    /// Precomputed distances per grid point
    distances: Vec<[f64; M]>,
    /// FFT bin spacing in Hz
    df: f64,
    /// [width, height] of the scan grid, row-major points
    grid_size: [usize; 2],
}
impl<const M: usize> Steering<M> {
    /// Create a new Steering struct
    ///
    /// ## Parameters
    /// - `scan_grid`: points the beamformer scans, only used for construction
    /// - `mic_array`: microphone array configuration, only used for construction
    /// - `f_s`: sample rate in Hz
    /// - `fft_len`: FFT length N
    pub fn new(scan_grid: &ScanGrid, mic_array: &MicArray<M>, f_s: f64, fft_len: usize) -> Self {
        // Distances from each mic to each grid point
        let distances = Vec::from_iter(
            scan_grid
                .points()
                .iter()
                .map(|pos| mic_array.get_distance_from(pos)),
        );

        Self {
            distances,
            df: f_s / fft_len as f64,
            grid_size: [scan_grid.width(), scan_grid.height()],
        }
    }

    /// Get the number of points in the scanning grid
    pub fn n_points(&self) -> usize {
        self.distances.len()
    }

    /// [width, height] of the scan grid; points are row-major
    pub fn grid_size(&self) -> [usize; 2] {
        self.grid_size
    }

    /// a(r, f_k) for grid point `point` and FFT bin `bin`, unnormalized (includes 1/r)
    pub fn vector(&self, point: usize, bin: usize) -> SVector<Complex64, M> {
        MicArray::steering_from_distances(&self.distances[point], bin as f64 * self.df)
    }

    /// `g_m = |a_m|^2 = 1/r_m^2` for one grid point
    pub fn gains(&self, point: usize) -> [f64; M] {
        self.distances[point].map(|r_m| r_m.powi(-2))
    }
}

/// Power arriving coherently from grid point `point`, summed over `bins`, and the mean mic
/// auto-power sum_m R_mm / M over `bins` (all the received power).
///
/// The coherent power is the delay-and-sum power with diagonal removal, P = c (a^H R a -
/// sum_m g_m R_mm), with c such that a lone source at the point gives its mean power at the mics
/// (R = S a a^H gives S sum(g)/M). A source elsewhere gives its power times the sidelobe level,
/// noise that is uncorrelated between the mics about 0 (also negative), whatever its level.
/// Independent of the beamformer that made the map, so a peak of any map can be checked
/// against it.
pub fn coherent_and_received_power<const M: usize>(
    cov: &[SMatrix<Complex64, M, M>],
    bins: RangeInclusive<usize>,
    steering: &Steering<M>,
    point: usize,
) -> (f64, f64) {
    debug_assert!(M > 1, "diagonal removal needs at least two mics");
    let gains = steering.gains(point);
    let sum_g: f64 = gains.iter().sum();
    let sum_g2: f64 = gains.iter().map(|g| g * g).sum();
    let c = sum_g / M as f64 / (sum_g * sum_g - sum_g2);

    let (mut coherent, mut received) = (0.0, 0.0);
    for bin in bins {
        let r = &cov[bin];
        let a = steering.vector(point, bin);
        let quad = a.dotc(&(r * a)).re;
        let diag: f64 = (0..M).map(|m| gains[m] * r[(m, m)].re).sum();
        coherent += c * (quad - diag);
        received += (0..M).map(|m| r[(m, m)].re).sum::<f64>() / M as f64;
    }
    (coherent, received)
}

/// Fraction of the power received over `bins` that arrives coherently from `point`, see
/// [`coherent_and_received_power`]: about 1 for a lone source at the point, the sidelobe level
/// for a source elsewhere, about 0 for uncorrelated noise. 0 without received power.
pub fn source_fraction<const M: usize>(
    cov: &[SMatrix<Complex64, M, M>],
    bins: RangeInclusive<usize>,
    steering: &Steering<M>,
    point: usize,
) -> f64 {
    let (coherent, received) = coherent_and_received_power(cov, bins, steering, point);
    if received > 0.0 { coherent / received } else { 0.0 }
}

pub trait Beamformer<const M: usize> {
    /// Get the number of points in the scanning grid
    fn n_points(&self) -> usize;

    /// Compute the power map
    ///
    /// - `cov`: Covariance matrices for the whole spectrum (N/2 + 1 bins)
    /// - `bins`: absolute bin indices
    /// - Sums power over `bins` of the fft
    /// - `out`: row-major like the [`ScanGrid`], overwritten
    fn power_map(
        &self,
        cov: &[SMatrix<Complex64, M, M>],
        bins: RangeInclusive<usize>,
        out: &mut [f64],
    );
}

/// Normalization c in P = c * a^H R a (Sarradj 2012, steering formulations III and IV)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Normalization {
    /// c = 1/(a^H a)^2: P = S at the source, but the peak drifts toward farther points
    TrueLevel,
    /// c = 1/(M a^H a): peak exactly at the source, P = S * (a^H a)/M there
    TrueLocation,
}

#[cfg(test)]
mod test {
    use nalgebra::SMatrix;

    use super::{Steering, coherent_and_received_power, source_fraction};
    use crate::{
        Complex64,
        geometry::mic_array::MicArray,
        test_util::{BIN, FFT_LEN, SAMPLE_RATE, test_scan_grid},
    };

    /// A source at the point with mic noise at the same level gives 1/2 exactly (the diagonal
    /// removal cancels the noise, the mic power counts it); noise alone gives 0; a source
    /// elsewhere gives less than at the point.
    #[test]
    fn fraction_of_source_and_noise() {
        let grid = test_scan_grid();
        let steering = Steering::new(&grid, &MicArray::uma16(), SAMPLE_RATE, FFT_LEN);
        let point = grid.index(grid.height() / 2, grid.width() / 2);
        let a = steering.vector(point, BIN);
        let gains = steering.gains(point);
        let mean_mic_power = gains.iter().sum::<f64>() / 16.0;
        let identity = SMatrix::<Complex64, 16, 16>::identity();

        let mut cov = vec![SMatrix::<Complex64, 16, 16>::zeros(); FFT_LEN / 2 + 1];
        cov[BIN] = a * a.adjoint() + identity * Complex64::from(mean_mic_power);
        let (coherent, received) = coherent_and_received_power(&cov, BIN..=BIN, &steering, point);
        assert!((coherent - mean_mic_power).abs() < 1e-9, "coherent {coherent}");
        assert!((received - 2.0 * mean_mic_power).abs() < 1e-9, "received {received}");
        let at_source = source_fraction(&cov, BIN..=BIN, &steering, point);
        assert!((at_source - 0.5).abs() < 1e-9, "at the source: {at_source}");
        let corner = source_fraction(&cov, BIN..=BIN, &steering, 0);
        assert!(corner < 0.5 * at_source, "corner: {corner}");

        cov[BIN] = identity * Complex64::from(mean_mic_power);
        let noise = source_fraction(&cov, BIN..=BIN, &steering, point);
        assert!(noise.abs() < 1e-12, "noise only: {noise}");
    }
}
