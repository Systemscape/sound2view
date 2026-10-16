use std::f64::consts::{PI, SQRT_2};

use rand::{RngExt, SeedableRng, rngs::ChaCha8Rng};
use serde::Deserialize;

/// Reference sound pressure in air, 20µPa
const P_REF: f64 = 20e-6;

/// Convert a Sound Pressure Level in dB to a pressure value in Pa
pub fn level_to_pressure(level_db: f64) -> f64 {
    // Level is computed as 20*log10(p_rms / p_ref), where p_ref = 20µPa
    // Inverted: 10^(level/20) * p_ref

    10f64.powf(level_db / 20.0) * P_REF
}

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
    pub seed: u64,
    pub n_components: usize,
    pub f_lo_hz: f64,
    pub f_hi_hz: f64,
    /// Total level in dB SPL at 1 m
    pub level_db: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(from = "NoiseParams")]
pub struct Noise {
    pub params: NoiseParams, // keep it for Serialize (`into = ...`) and debugging
    pub components: Vec<Sine>, // amplitude, freq, phase per component
}

impl From<NoiseParams> for Noise {
    fn from(params: NoiseParams) -> Self {
        assert!(params.n_components > 0);
        assert!(params.f_lo_hz < params.f_hi_hz);

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

        Noise { params, components }
    }
}

impl Noise {
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
        let expected = std::f64::consts::SQRT_2 * 20e-6;
        assert!((sine.amplitude - expected).abs() < expected * TOL);
        assert_eq!(sine.freq_hz, 1000.0);
        assert_eq!(sine.phase_rad, 0.5);
    }

    #[test]
    fn level_to_pressure_conversion() {
        fn pressure_to_level(pressure: f64) -> f64 {
            20.0 * (pressure / P_REF).log10()
        }

        for pressure in [20e-6, 1e-3, 1.0, 10.0] {
            let level = pressure_to_level(pressure);
            let computed_pressure = level_to_pressure(level);

            assert!(
                (computed_pressure - pressure).abs() < 1e-12,
                "expected pressure {pressure}, got {computed_pressure}"
            )
        }
    }

    #[test]
    fn noise_power() {
        let noise_params = NoiseParams {
            seed: 1234,
            level_db: 60.0,
            n_components: 10,
            f_lo_hz: 1000.0,
            f_hi_hz: 4000.0,
        };

        let noise = Noise::from(noise_params);

        // Ensure noise components' frequencies are in the defined range
        assert!(noise.components.iter().all(|component| {
            (noise_params.f_lo_hz..=noise_params.f_hi_hz).contains(&component.freq_hz)
        }));

        let noise_samples: Vec<_> = (0..100000).map(|t| noise.eval(t as f64)).collect();

        let mean_square =
            noise_samples.iter().map(|n| n.powi(2)).sum::<f64>() / noise_samples.len() as f64;

        let pressure_squared = level_to_pressure(noise_params.level_db).powi(2);
        assert!(
            (mean_square - pressure_squared).abs() < 10e-8,
            "expected pressure_squared {pressure_squared}, got {mean_square}"
        );

        // Generate new noise from the same seed
        let noise_2 = Noise::from(noise_params);

        assert!(
            noise.components == noise_2.components,
            "noise_2's components should match noise's components for the same seed"
        )
    }
}
