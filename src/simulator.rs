use crate::{
    FrameSource,
    mic_array::{self, MicArray},
    simulator::scene::{MicNoise, MicNoiseParams, Scene},
};

pub mod scene;
pub mod signal;

/// Reference sound pressure in air, 20µPa
const P_REF: f64 = 20e-6;

/// Convert a Sound Pressure Level in dB to an RMS pressure in Pa
pub fn level_to_pressure(level_db: f64) -> f64 {
    10f64.powf(level_db / 20.0) * P_REF
}

pub struct Simulator<const M: usize> {
    /// Simulation Scene
    scene: Scene,
    /// Microphone array geometry
    mic_array: MicArray<M>,
    /// White Gaussian noise added to the microphone signal
    mic_noise: MicNoise,
    /// Current frame
    n: u64,
    /// Total number of frames, computed from duration
    n_total: u64,
}

impl<const M: usize> Simulator<M> {
    pub fn new(scene: Scene, mic_array: MicArray<M>, mic_noise_params: MicNoiseParams) -> Self {
        Simulator {
            n: 0,
            n_total: (scene.duration_s * scene.sample_rate).round() as u64,
            scene,
            mic_array,
            mic_noise: MicNoise::from(mic_noise_params),
        }
    }
}

impl<const M: usize> FrameSource<M> for Simulator<M> {
    fn sample_rate(&self) -> f64 {
        self.scene.sample_rate
    }

    fn read(&mut self, buf: &mut [f32]) -> std::io::Result<usize> {
        debug_assert_eq!(
            buf.len() % M,
            0,
            "buf should be aligned to {M} samples per frame"
        );

        // If we already have more samples than we should, tell the caller
        if self.n > self.n_total {
            return Ok(0);
        }

        // Check how many samples we can read for the given buffer length
        let frames_to_read = buf.len() / self.channels();

        for _sample in 0..frames_to_read {
            let t = self.n as f64 / self.sample_rate();
            // Now compute the superposition of signals per mic for the scene

            self.n += 1;
        }

        Ok(frames_to_read)
    }
}
