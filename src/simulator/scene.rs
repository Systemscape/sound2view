use rand::{prelude::*, rngs::ChaCha8Rng};
use rand_distr::Normal;
use serde::Deserialize;

use crate::{
    Vector3,
    simulator::{level_to_pressure, signal::Signal},
};

#[derive(Clone, Debug, Deserialize)]
pub struct Scene {
    /// Sampling rate in Hz
    pub sample_rate: f64,
    /// Duration in seconds
    pub duration_s: f64,
    /// White gaussian noise at the microphones
    pub mic_noise_params: MicNoiseParams,
    /// Signal and Noise sources
    pub sources: Vec<Source>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Source {
    /// Position in meters relative to center of array frame, z into the scene
    pub pos: Vector3,
    /// Signal parameters
    pub signal: Signal,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MicNoiseParams {
    /// RNG Seed
    seed: u64,
    /// Noise level in dB SPL
    level_db: f64,
}

#[derive(Debug, Deserialize)]
#[serde(from = "MicNoiseParams")]
pub struct MicNoise {
    _params: MicNoiseParams,
    rng: ChaCha8Rng,
    distribution: Normal<f64>,
}

impl From<MicNoiseParams> for MicNoise {
    fn from(params: MicNoiseParams) -> Self {
        MicNoise {
            rng: ChaCha8Rng::seed_from_u64(params.seed),
            // p_rms^2 is the power, so p_rms must be the stdev of the normal distribution
            distribution: Normal::new(0.0, level_to_pressure(params.level_db))
                .expect("Construct Normal distribution from given noise level"),
            _params: params,
        }
    }
}

impl MicNoise {
    /// Get noise sample.
    ///
    /// This takes `&mut self` because it advances the RNG state
    pub fn get_sample(&mut self) -> f64 {
        self.rng.sample(self.distribution)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Vector3,
        simulator::{
            level_to_pressure,
            scene::{MicNoise, MicNoiseParams, Scene},
            signal::Signal,
        },
    };

    #[test]
    fn parse_scene() {
        let scene: Scene = toml::from_str(
            r#"
            sample_rate = 16000
            duration_s = 0.5

            [mic_noise_params]
            seed = 7
            level_db = 30.0

            [[sources]]
            pos = [1.0, -2.0, 3.0]
            signal = { type = "sine", level_db = 80.0, freq_hz = 500.0, phase_rad = 0.0 }

            [[sources]]
            pos = [0.0, 0.0, 1.0]
            signal = { type = "noise", seed = 9, n_components = 4, f_lo_hz = 100.0, f_hi_hz = 200.0, level_db = 70.0 }
            "#,
        )
        .unwrap();

        assert_eq!(scene.sample_rate, 16_000.0);
        assert_eq!(scene.duration_s, 0.5);
        assert_eq!(scene.mic_noise_params.seed, 7);
        assert_eq!(scene.mic_noise_params.level_db, 30.0);

        assert_eq!(scene.sources.len(), 2);
        assert_eq!(scene.sources[0].pos, Vector3::new(1.0, -2.0, 3.0));
        assert!(matches!(scene.sources[0].signal, Signal::Sine(_)));
        assert_eq!(scene.sources[1].pos, Vector3::new(0.0, 0.0, 1.0));
        assert!(matches!(scene.sources[1].signal, Signal::Noise(_)));
    }

    #[test]
    fn mic_noise() {
        let noise_level_db = 6.0;
        let mut mic_noise = MicNoise::from(MicNoiseParams {
            seed: 42,
            level_db: noise_level_db,
        });
        let n_samples = 1_000_000;

        let noise_power: f64 = (0..n_samples)
            .map(|_| mic_noise.get_sample().powi(2))
            .sum::<f64>()
            / n_samples as f64;

        // power is p_rms^2
        let noise_power_expected = level_to_pressure(noise_level_db).powi(2);

        // Var(signal^2) is 2*sigma^4
        // So Std(signal^2) is sqrt(2)*sigma^2
        //
        // With n_samples, the stdev should converge to sigma^2*sqrt(2/N)
        // where sigma^2 is the noise power
        // Let's alloow up to 5 times that, to prevent flaky tests
        assert!(
            (noise_power - noise_power_expected).abs()
                < noise_power_expected * (2.0 / n_samples as f64).sqrt() * 5.0,
            "expected noise power {noise_power_expected}, got {noise_power}"
        )
    }
}
