use std::f64::consts::{PI, SQRT_2};

use rand::{RngExt, SeedableRng, rngs::ChaCha8Rng};
use serde::Deserialize;

use crate::simulator::level_to_pressure;

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Signal {
    Sine(Sine),
    Noise(Noise),
}

impl Signal {
    /// Evaluate the Signal at time `t`, valid for all `t`, including `t<0`
    pub fn eval(&self, t: f64) -> f64 {
        match self {
            Self::Sine(sine) => sine.eval(t),
            Self::Noise(noise) => noise.eval(t),
        }
    }

    /// Highest frequency contained in the signal, Hz
    pub fn max_freq(&self) -> f64 {
        match self {
            Self::Sine(sine) => sine.freq_hz,
            Self::Noise(noise) => noise
                .components
                .iter()
                .map(|c| c.freq_hz)
                .fold(0.0, f64::max),
        }
    }

    /// Get the Signal's components
    pub fn components(&self) -> &[Sine] {
        match self {
            Signal::Sine(sine) => std::slice::from_ref(sine),
            Signal::Noise(noise) => &noise.components,
        }
    }
}

/// Sine as specified in the config.
#[derive(Clone, Debug, Deserialize)]
pub struct SineParams {
    /// Level in dB SPL at 1 m
    pub level_db: f64,
    /// Frequency in Hz
    pub freq_hz: f64,
    /// Phase phi, such that `s(0) = A * sin(phi)`
    pub phase_rad: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(from = "SineParams")]
pub struct Sine {
    /// Amplitude in Pa at 1m
    pub amplitude: f64,
    /// Frequency in Hz
    pub freq_hz: f64,
    /// Phase phi, such that `s(0) = A * sin(phi)`
    pub phase_rad: f64,
}

impl Sine {
    pub fn new(amplitude: f64, freq_hz: f64, phase_rad: f64) -> Self {
        Self {
            amplitude,
            freq_hz,
            phase_rad,
        }
    }

    /// Get Sine signal value at time `t`
    pub fn eval(&self, t: f64) -> f64 {
        self.amplitude * (2.0 * PI * self.freq_hz * t + self.phase_rad).sin()
    }
}

impl From<SineParams> for Sine {
    fn from(params: SineParams) -> Self {
        Self {
            amplitude: SQRT_2 * level_to_pressure(params.level_db),
            freq_hz: params.freq_hz,
            phase_rad: params.phase_rad,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct NoiseParams {
    /// RNG Seed for reproducible noise
    pub seed: u64,
    /// Number of noise components `Q`
    pub n_components: usize,
    /// Lower frequency limit (inclusive), Hz
    pub f_lo_hz: f64,
    /// Upper frequency limit (inclusive), Hz
    pub f_hi_hz: f64,
    /// Total level in dB SPL at 1 m
    pub level_db: f64,
}

/// Noise constructed from a sum of `Sine`s
#[derive(Clone, Debug, Deserialize)]
#[serde(from = "NoiseParams")]
pub struct Noise {
    /// Parameters that were used for constructing this instance.
    /// Kept for debugging
    _params: NoiseParams,
    /// `Sine`s that comprise the noise
    components: Vec<Sine>,
}

impl From<NoiseParams> for Noise {
    fn from(params: NoiseParams) -> Self {
        assert!(params.n_components > 0, "n_components must be >0");
        assert!(
            params.f_lo_hz < params.f_hi_hz,
            "f_hi_hz must be higher than f_lo_hz"
        );

        let mut rng = ChaCha8Rng::seed_from_u64(params.seed);
        let n_components = params.n_components;

        // Normalize amplitude by sqrt(2/Q)
        let amplitude = (2.0 / n_components as f64).sqrt() * level_to_pressure(params.level_db);

        let components = (0..n_components)
            .map(|_| {
                Sine::new(
                    amplitude,
                    rng.random_range(params.f_lo_hz..=params.f_hi_hz),
                    rng.random_range(0.0..2.0 * PI),
                )
            })
            .collect();

        Noise {
            _params: params,
            components,
        }
    }
}

impl Noise {
    /// Get noise signal value at time `t`
    pub fn eval(&self, t: f64) -> f64 {
        let mut res = 0.0;
        for component in &self.components {
            res += component.eval(t);
        }
        res
    }
}

#[cfg(test)]
mod tests {

    use crate::simulator::P_REF;

    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn sine_known_values() {
        let s = Signal::Sine(Sine::new(2.0, 1000.0, 0.0));
        let quarter_period = 0.25 / 1000.0;

        assert!(s.eval(0.0).abs() < TOL);
        assert!((s.eval(quarter_period) - 2.0).abs() < TOL);
        assert!((s.eval(-quarter_period) + 2.0).abs() < TOL);
    }

    #[test]
    fn sine_phase() {
        let s = Signal::Sine(Sine::new(2.0, 1000.0, PI / 2.0));

        assert!((s.eval(0.0) - 2.0).abs() < TOL);
    }

    #[test]
    fn sine_from_toml() {
        let s: Signal = toml::from_str(
            r#"
            type = "sine"
            level_db = 0.0
            freq_hz = 1000.0
            phase_rad = 0.5
            "#,
        )
        .unwrap();

        let Signal::Sine(sine) = &s else {
            panic!("expected Signal::Sine, got {s:?}");
        };
        // 0 dB SPL: RMS = 20 µPa, A = sqrt(2) * RMS
        let expected = SQRT_2 * P_REF;
        assert!((sine.amplitude - expected).abs() < expected * TOL);
        assert_eq!(sine.freq_hz, 1000.0);
        assert_eq!(sine.phase_rad, 0.5);
    }

    #[test]
    fn level_to_pressure_conversion() {
        fn pressure_to_level(pressure: f64) -> f64 {
            20.0 * (pressure / P_REF).log10()
        }

        assert_eq!(
            level_to_pressure(0.0),
            P_REF,
            "0.0 dB should match the reference pressure"
        );

        for pressure in [20e-6, 1e-3, 1.0, 10.0] {
            let level = pressure_to_level(pressure);
            let computed_pressure = level_to_pressure(level);

            assert!(
                (computed_pressure - pressure).abs() < 1e-12,
                "expected pressure {pressure}, got {computed_pressure}"
            )
        }
    }

    const NOISE_PARAMS: NoiseParams = NoiseParams {
        seed: 4234,
        level_db: 60.0,
        n_components: 10,
        f_lo_hz: 1000.0,
        f_hi_hz: 4000.0,
    };

    #[test]
    fn noise_frequencies_in_band() {
        let noise = Noise::from(NOISE_PARAMS);

        for component in &noise.components {
            assert!(
                (NOISE_PARAMS.f_lo_hz..=NOISE_PARAMS.f_hi_hz).contains(&component.freq_hz),
                "frequency {} out of band",
                component.freq_hz
            );
        }
    }

    #[test]
    fn noise_same_seed() {
        let noise = Noise::from(NOISE_PARAMS);
        let noise_2 = Noise::from(NOISE_PARAMS);

        assert_eq!(noise.components, noise_2.components);
    }

    #[test]
    fn noise_different_seed() {
        let noise = Noise::from(NOISE_PARAMS);
        let noise_2 = Noise::from(NoiseParams {
            seed: NOISE_PARAMS.seed + 1,
            ..NOISE_PARAMS
        });

        assert_ne!(noise.components, noise_2.components);
    }

    #[test]
    fn noise_power() {
        // Sampling frequency in Hz
        const F_SAMPLE: f64 = 48_000.0;

        let noise_params = NOISE_PARAMS;
        let noise = Noise::from(noise_params);

        let p_signal: f64 = noise
            .components
            .iter()
            .map(|c| c.amplitude.powi(2) / 2.0)
            .sum();

        let expected_p_signal = level_to_pressure(noise_params.level_db).powi(2);

        assert!(
            (p_signal - expected_p_signal).abs() < 1e-12,
            "signal power {p_signal} deviates from the setting {expected_p_signal}"
        );

        let n_samples = 1_000_000;
        let mean_square = (0..n_samples)
            .map(|t| noise.eval(t as f64 / F_SAMPLE))
            .map(|n| n.powi(2))
            .sum::<f64>()
            / n_samples as f64;

        // Check that the deviation is at most 2% of pressure_squared
        //
        // Since noise is generated by the RNG, the actual power depends on
        // the frequency distribution. So the deviation depends on the seed
        // and this tolerance might fail for a different seed.
        assert!(
            (mean_square - expected_p_signal).abs() < expected_p_signal * 0.02,
            "expected expected_p_signal {expected_p_signal}, got {mean_square}"
        );
    }
}
