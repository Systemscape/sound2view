use nalgebra::SMatrix;

use crate::{
    Complex64, FrameSource as _, Vector3,
    camera::{Camera, Intrinsics, Mount},
    covariance::Covariance,
    mic_array::MicArray,
    scan_grid::ScanGrid,
    simulator::{
        Simulator,
        scene::{MicNoiseParams, Scene, Source},
        signal::{Signal, Sine},
    },
    stft::{Stft, StftStream},
};

pub const SAMPLE_RATE: f64 = 48_000.0;
pub const FFT_LEN: usize = 1000;
pub const BIN: usize = 50;

pub const F_SINE: f64 = BIN as f64 * SAMPLE_RATE / FFT_LEN as f64;

pub fn test_pos() -> Vector3 {
    Vector3::new(0.1, 0.2, 1.0)
}

/// Sine with amplitude 1, frequency `F_SINE` and zero-phase
pub fn test_sine() -> Sine {
    Sine::new(1.0, F_SINE, 0.0)
}

/// One sine at `test_pos()`, mic noise at the given level.
pub fn test_scene(mic_noise_db: f64, duration_s: f64) -> Scene {
    Scene {
        sample_rate: SAMPLE_RATE,
        duration_s,
        mic_noise_params: MicNoiseParams {
            seed: 1,
            level_db: mic_noise_db,
        },
        sources: vec![Source {
            pos: test_pos(),
            signal: Signal::Sine(test_sine()),
        }],
    }
}

/// Read until `Ok(0)` in chunks of `chunk_frames`; returns all samples.
pub fn read_all<const M: usize>(sim: &mut Simulator<M>, chunk_frames: usize) -> Vec<f64> {
    let mut samples = Vec::new();
    let mut buf = vec![0.0; chunk_frames * M];
    loop {
        let frames = sim.read(&mut buf).unwrap();
        if frames == 0 {
            break;
        }
        samples.extend_from_slice(&buf[..frames * M]);
    }
    samples
}

/// Simulator -> StftStream -> Covariance over the whole scene; returns (R_est, L).
pub fn stream_covariance<const M: usize>(
    scene: &Scene,
    mic_array: &MicArray<M>,
) -> (Vec<SMatrix<Complex64, M, M>>, usize) {
    let mut sim = Simulator::new(scene.clone(), mic_array.clone());
    let mut stream = StftStream::<M>::new(Stft::new(FFT_LEN));
    let mut cov = Covariance::<M>::new(FFT_LEN / 2 + 1);

    let samples = read_all(&mut sim, 4096);
    stream.push(&samples, |snapshots, _l| cov.accumulate(snapshots));

    (cov.estimate(), cov.count())
}

/// Camera that matches approx. the UMA16
pub fn test_camera() -> Camera {
    Camera {
        intrinsics: Intrinsics::from_hfov(1920, 1080, 110f64.to_radians()),
        mount: Mount::default(),
    }
}

/// Create a ScanGrid from [`test_camera()`] with width 192 and 1m focus distance
pub fn test_scan_grid() -> ScanGrid {
    ScanGrid::new(&test_camera(), 192, 1.0)
}

/// Bin power of a bin-centered sine after Hann + Stft scaling: (A^2/2)*N/3
pub fn sine_bin_power(amplitude: f64, fft_len: usize) -> f64 {
    amplitude.powi(2) / 2.0 * fft_len as f64 / 3.0
}
