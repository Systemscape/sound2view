use std::ops::RangeInclusive;

use nalgebra::{SMatrix, SVector};

use crate::{
    Complex64,
    geometry::{mic_array::MicArray, scan_grid::ScanGrid},
};

pub mod delay_and_sum;
pub mod functional;

pub struct Steering<const M: usize> {
    /// Precomputed distances per grid point
    distances: Vec<[f64; M]>,
    /// FFT bin spacing in Hz
    df: f64,
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
        }
    }

    /// Get the number of points in the scanning grid
    pub fn n_points(&self) -> usize {
        self.distances.len()
    }

    pub fn vector(&self, point: usize, bin: usize) -> SVector<Complex64, M> {
        MicArray::steering_from_distances(&self.distances[point], bin as f64 * self.df)
    }

    /// `g_m = |a_m|^2 = 1/r_m^2` for one grid point
    pub fn gains(&self, point: usize) -> [f64; M] {
        self.distances[point].map(|r_m| r_m.powi(-2))
    }
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
