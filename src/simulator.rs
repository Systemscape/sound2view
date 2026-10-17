use std::cmp::min;

use crate::{
    FrameSource, distance_to_delay,
    mic_array::MicArray,
    simulator::{
        scene::{MicNoise, Scene},
        signal::Signal,
    },
};

pub mod scene;
pub mod signal;

/// Reference sound pressure in air, 20µPa
const P_REF: f64 = 20e-6;

/// Convert a Sound Pressure Level in dB to an RMS pressure in Pa
pub fn level_to_pressure(level_db: f64) -> f64 {
    10f64.powf(level_db / 20.0) * P_REF
}

struct SimSource<const M: usize> {
    signal: Signal,
    /// 1/r_mk
    gain: [f64; M],
    /// tau_mk in s
    delay: [f64; M],
}

pub struct Simulator<const M: usize> {
    /// Simulation Scene
    scene: Scene,
    /// Sources from the scene with precomputed values for the microphones
    sim_sources: Vec<SimSource<M>>,
    /// White Gaussian noise added to the microphone signal
    mic_noise: MicNoise,
    /// Index of the next frame
    n: u64,
    /// Total number of frames, computed from duration
    n_total: u64,
}

impl<const M: usize> Simulator<M> {
    pub fn new(scene: Scene, mic_array: MicArray<M>) -> Self {
        let _: () = scene.validate();
        let mut sim_sources = Vec::with_capacity(scene.sources.len());

        for source in &scene.sources {
            let r_mk = mic_array.get_distance_from(&source.pos);
            let gain = r_mk.map(|r| 1.0 / r);
            let delay = r_mk.map(distance_to_delay);

            sim_sources.push(SimSource {
                signal: source.signal.clone(),
                gain,
                delay,
            });
        }

        Simulator {
            n: 0,
            n_total: (scene.duration_s * scene.sample_rate).round() as u64 as u64,
            sim_sources,
            mic_noise: MicNoise::from(scene.mic_noise_params.clone()),
            scene,
        }
    }
}

impl<const M: usize> FrameSource<M> for Simulator<M> {
    fn sample_rate(&self) -> f64 {
        self.scene.sample_rate
    }

    fn read(&mut self, buf: &mut [f64]) -> std::io::Result<usize> {
        debug_assert_eq!(
            buf.len() % M,
            0,
            "buf should be aligned to {M} samples per frame"
        );

        // Check how many samples we can read for the given buffer length
        let frames_to_read = min((buf.len() / M) as u64, self.n_total - self.n) as usize;

        for frame in 0..frames_to_read {
            let t = self.n as f64 / self.sample_rate();

            for m in 0..M {
                let mut x: f64 = self
                    .sim_sources
                    .iter()
                    .map(|s| s.gain[m] * s.signal.eval(t - s.delay[m]))
                    .sum();
                x += self.mic_noise.get_sample();

                buf[frame * M + m] = x;
            }

            self.n += 1;
        }

        Ok(frames_to_read)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use crate::{
        FrameSource, Vector3, distance_to_delay,
        mic_array::MicArray,
        simulator::{
            Simulator,
            scene::{MicNoiseParams, Scene, Source},
            signal::{Signal, Sine},
        },
    };

    const SAMPLE_RATE: f64 = 48_000.0;
    const DURATION_S: f64 = 1.0;
    const N_FRAMES: usize = 48_000;

    fn test_pos() -> Vector3 {
        Vector3::new(0.1, 0.2, 1.0)
    }

    fn test_sine() -> Sine {
        Sine::new(1.0, 2400.0, 0.0)
    }

    /// One sine at `test_pos()`, mic noise at the given level.
    fn test_scene(mic_noise_db: f64) -> Scene {
        Scene {
            sample_rate: SAMPLE_RATE,
            duration_s: DURATION_S,
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
    fn read_all<const M: usize>(sim: &mut Simulator<M>, chunk_frames: usize) -> Vec<f64> {
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

    #[test]
    fn single_source_propagation() {
        let pos = test_pos();
        let sine = test_sine();
        let mic_array = MicArray::uma16();

        let mut sim = Simulator::new(test_scene(-1000.0), mic_array.clone());
        let buf = read_all(&mut sim, 1024);

        assert_eq!(
            buf.len(),
            N_FRAMES * sim.channels(),
            "total frames until EOF"
        );

        for frame in 0..N_FRAMES {
            let t = frame as f64 / SAMPLE_RATE;
            for m in 0..sim.channels() {
                let r_mk = (pos - mic_array.mic_pos[m]).norm();
                let tau_mk = distance_to_delay(r_mk);

                let expected =
                    sine.amplitude / r_mk * (2.0 * PI * sine.freq_hz * (t - tau_mk)).sin();

                let simulated = buf[frame * sim.channels() + m];
                assert!(
                    (simulated - expected).abs() < 1e-6,
                    "expected {expected}, simulator generated {simulated}. Frame {frame}, m {m}"
                );
            }
        }
    }

    #[test]
    fn chunked_read_matches_single_read() {
        let scene = test_scene(54.0);

        // 7 does not divide N_FRAMES, so the last read is partial.
        let chunked = read_all(&mut Simulator::new(scene.clone(), MicArray::uma16()), 7);
        let single = read_all(&mut Simulator::new(scene, MicArray::uma16()), N_FRAMES);

        assert_eq!(chunked.len(), single.len());
        if let Some(i) = chunked.iter().zip(&single).position(|(a, b)| a != b) {
            panic!(
                "first mismatch at sample {i}: chunked {}, single {}",
                chunked[i], single[i]
            );
        }
    }
}
