use crate::{
    FrameSource as _, Vector3,
    simulator::{
        Simulator,
        scene::{MicNoiseParams, Scene, Source},
        signal::{Signal, Sine},
    },
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
