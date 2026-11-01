//! Audio from one direction: delay and sum toward a point per STFT frame, back to a time signal
//! by overlap-add, decimated to `TRACK_SAMPLE_RATE` for a speech classifier.

use std::{f64::consts::PI, sync::Arc};

use nalgebra::SVector;
use realfft::{ComplexToReal, RealFftPlanner};

use crate::{Complex64, distance_to_delay, geometry::mic_array::MicArray};

/// Sample rate of the beam signal in Hz (what speech classifiers expect)
pub const TRACK_SAMPLE_RATE: f64 = 16_000.0;

/// Integer factor from `sample_rate` down to [`TRACK_SAMPLE_RATE`], `None` if there is none
pub fn decimation_factor(sample_rate: f64) -> Option<usize> {
    let factor = sample_rate / TRACK_SAMPLE_RATE;
    (factor >= 1.0 && factor.fract() == 0.0).then_some(factor as usize)
}

/// Delay and sum toward one point, as a stream of time samples.
///
/// Per STFT frame y(f_k) = w_k^H x(f_k) with w = a e^{j omega r_ref / c} / (a^H a), so a source at
/// the point comes out as its signal s(t - r_ref / c) referenced to 1 m, with r_ref the mean
/// mic distance. The inverse FFT of y gives the windowed segment; consecutive segments overlap
/// by N/2 and the periodic Hann window sums to 1, so overlap-add reconstructs the signal.
///
/// The weights shift each mic's frame circularly by (r_m - r_ref) / c, at most about 13 samples
/// at 48 kHz, where the window is nearly 0. Removing the whole delay r_m / c instead (about 280
/// samples at 2 m) wraps the window around the frame and breaks the overlap-add: an error at
/// every hop, heard as a buzz.
pub struct BeamSignal<const M: usize> {
    /// w_k per bin for the current point
    weights: Vec<SVector<Complex64, M>>,
    /// FFT bin spacing in Hz
    df: f64,
    plan: Arc<dyn ComplexToReal<f64>>,
    spectrum: Vec<Complex64>,
    segment: Vec<f64>,
    scratch: Vec<Complex64>,
    /// Overlap-add accumulator, N samples; the first N/2 are complete after each frame
    ola: Vec<f64>,
    /// 1 / (N * stft_scale): undoes the gain of the unnormalized inverse FFT and the STFT scale
    gain: f64,
    decimator: Decimator,
    /// Samples at `TRACK_SAMPLE_RATE` since the last `take_output`
    output: Vec<f32>,
}

impl<const M: usize> BeamSignal<M> {
    /// - `fft_len`, `stft_scale`: of the STFT that produces the frames, see
    ///   [`crate::dsp::stft::Stft::scale`]
    /// - `sample_rate`: in Hz, an integer multiple of [`TRACK_SAMPLE_RATE`]
    /// - `distances`: r_m from each mic to the point, see [`MicArray::get_distance_from`]
    ///
    /// Panics if `sample_rate` has no integer decimation factor.
    pub fn new(fft_len: usize, stft_scale: f64, sample_rate: f64, distances: &[f64; M]) -> Self {
        let factor = decimation_factor(sample_rate)
            .expect("sample rate must be an integer multiple of TRACK_SAMPLE_RATE");
        let plan = RealFftPlanner::new().plan_fft_inverse(fft_len);

        let mut beam = Self {
            weights: vec![SVector::zeros(); fft_len / 2 + 1],
            df: sample_rate / fft_len as f64,
            spectrum: plan.make_input_vec(),
            segment: plan.make_output_vec(),
            scratch: plan.make_scratch_vec(),
            plan,
            ola: vec![0.0; fft_len],
            gain: 1.0 / (fft_len as f64 * stft_scale),
            decimator: Decimator::new(factor),
            output: Vec::new(),
        };
        beam.steer(distances);
        beam
    }

    /// Point the beam at the point with mic distances `distances`
    pub fn steer(&mut self, distances: &[f64; M]) {
        let delay_ref = distance_to_delay(distances.iter().sum::<f64>() / M as f64);
        for (k, w) in self.weights.iter_mut().enumerate() {
            let freq = k as f64 * self.df;
            let a = MicArray::steering_from_distances(distances, freq);
            // Keep the common delay: w^H gets e^{-j omega tau_ref}
            let keep_delay = Complex64::from_polar(1.0, 2.0 * PI * freq * delay_ref);
            *w = a.unscale(a.norm_squared()) * keep_delay;
        }
    }

    /// Add one STFT frame (all N/2 + 1 bins); appends N/2 / factor samples to the output.
    pub fn push_frame(&mut self, snapshots: &[SVector<Complex64, M>]) {
        assert_eq!(snapshots.len(), self.weights.len(), "one snapshot per bin");

        for ((y, w), x) in self.spectrum.iter_mut().zip(&self.weights).zip(snapshots) {
            *y = w.dotc(x);
        }
        // A real signal has real DC and Nyquist bins; the inverse FFT requires it
        let last = self.spectrum.len() - 1;
        self.spectrum[0].im = 0.0;
        self.spectrum[last].im = 0.0;

        self.plan
            .process_with_scratch(&mut self.spectrum, &mut self.segment, &mut self.scratch)
            .expect("buffers have the planned lengths");

        for (acc, s) in self.ola.iter_mut().zip(&self.segment) {
            *acc += s * self.gain;
        }
        let hop = self.ola.len() / 2;
        self.decimator.push(&self.ola[..hop], &mut self.output);
        self.ola.copy_within(hop.., 0);
        self.ola[hop..].fill(0.0);
    }

    /// Samples at [`TRACK_SAMPLE_RATE`] since the previous call
    pub fn take_output(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.output)
    }
}

/// Low-pass FIR and every `factor`-th sample, streaming
struct Decimator {
    taps: Vec<f64>,
    factor: usize,
    /// The last `taps.len() - 1` input samples, then new input
    buf: Vec<f64>,
    /// Index in `buf` of the next output sample
    next: usize,
}

/// FIR length per decimation factor; 95 taps at factor 3 give a transition of about 2.8 kHz
const TAPS_PER_FACTOR: usize = 32;
/// Cutoff relative to the output rate: 0.4 * 16 kHz = 6.4 kHz, stopband from about 7.8 kHz
const CUTOFF: f64 = 0.4;

impl Decimator {
    fn new(factor: usize) -> Self {
        let taps = if factor == 1 {
            vec![1.0]
        } else {
            lowpass_taps(TAPS_PER_FACTOR * factor - 1, CUTOFF / factor as f64)
        };
        let history = taps.len() - 1;
        Self {
            taps,
            factor,
            buf: vec![0.0; history],
            next: history,
        }
    }

    fn push(&mut self, input: &[f64], out: &mut Vec<f32>) {
        self.buf.extend_from_slice(input);
        while self.next < self.buf.len() {
            let y: f64 = self
                .taps
                .iter()
                .enumerate()
                .map(|(j, h)| h * self.buf[self.next - j])
                .sum();
            out.push(y as f32);
            self.next += self.factor;
        }
        // Keep the history the filter needs for the next output
        let remove = self.buf.len() - (self.taps.len() - 1);
        self.buf.drain(..remove);
        self.next -= remove;
    }
}

/// Windowed-sinc low-pass (Blackman), DC gain 1; `cutoff` relative to the sample rate
fn lowpass_taps(len: usize, cutoff: f64) -> Vec<f64> {
    let center = (len - 1) as f64 / 2.0;
    let taps: Vec<f64> = (0..len)
        .map(|n| {
            let t = n as f64 - center;
            let sinc = if t == 0.0 {
                2.0 * cutoff
            } else {
                (2.0 * PI * cutoff * t).sin() / (PI * t)
            };
            let x = 2.0 * PI * n as f64 / (len - 1) as f64;
            let blackman = 0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos();
            sinc * blackman
        })
        .collect();
    let sum: f64 = taps.iter().sum();
    taps.iter().map(|h| h / sum).collect()
}

#[cfg(test)]
mod test {
    use std::f64::consts::PI;

    use super::{BeamSignal, Decimator, TRACK_SAMPLE_RATE, decimation_factor};
    use crate::{
        Point3, distance_to_delay,
        dsp::stft::{Stft, StftStream},
        geometry::mic_array::MicArray,
        source::simulator::{
            Simulator,
            scene::{MicNoiseParams, Scene, Source},
            signal::{Noise, NoiseParams, Signal, Sine},
        },
        test_util::{FFT_LEN, SAMPLE_RATE, read_all},
    };

    /// Scene of bin-centered sines (bin, position), no mic noise
    fn sine_scene(sines: &[(usize, Point3)], duration_s: f64) -> Scene {
        Scene {
            sample_rate: SAMPLE_RATE,
            duration_s,
            mic_noise_params: MicNoiseParams {
                seed: 1,
                level_db: -1000.0,
            },
            sources: sines
                .iter()
                .map(|&(bin, pos)| Source {
                    pos,
                    signal: Signal::Sine(Sine::new(1.0, bin_freq(bin), 0.0)),
                })
                .collect(),
        }
    }

    fn bin_freq(bin: usize) -> f64 {
        bin as f64 * SAMPLE_RATE / FFT_LEN as f64
    }

    /// Beam signal toward `target` over the whole scene, with decimation `factor`
    fn beam_output(scene: &Scene, target: &Point3, factor: usize) -> Vec<f32> {
        let mic_array = MicArray::uma16();
        let samples = read_all(&mut Simulator::new(scene.clone(), mic_array.clone()).unwrap(), 4096);
        let stft = Stft::new(FFT_LEN);
        let sample_rate = factor as f64 * TRACK_SAMPLE_RATE;
        let mut beam =
            BeamSignal::new(FFT_LEN, stft.scale(), sample_rate, &mic_array.get_distance_from(target));
        let mut stream = StftStream::<16>::new(stft);
        stream.push(&samples, |snapshots, _l| beam.push_frame(snapshots));
        beam.take_output()
    }

    /// Amplitude of the component at `freq` (least squares fit of sine and cosine)
    fn amplitude_at(signal: &[f32], sample_rate: f64, freq: f64) -> f64 {
        let (mut c, mut s) = (0.0, 0.0);
        for (n, &y) in signal.iter().enumerate() {
            let phase = 2.0 * PI * freq * n as f64 / sample_rate;
            c += y as f64 * phase.cos();
            s += y as f64 * phase.sin();
        }
        2.0 * (c * c + s * s).sqrt() / signal.len() as f64
    }

    #[test]
    fn decimation_factors() {
        assert_eq!(decimation_factor(48_000.0), Some(3));
        assert_eq!(decimation_factor(16_000.0), Some(1));
        assert_eq!(decimation_factor(44_100.0), None);
        assert_eq!(decimation_factor(8_000.0), None);
    }

    /// Without decimation, a source at the beam point comes out as s(t - r_ref / c) (referenced
    /// to 1 m, r_ref the mean mic distance) once the first frame has passed. Band noise at
    /// off-bin frequencies is not periodic in the frame, so a circular shift that wraps the
    /// window would show up here (a bin-centered sine hides it).
    #[test]
    fn reconstructs_source_signal() {
        let pos = Point3::new(0.4, 0.2, 2.0);
        let signal = Signal::Noise(Noise::from(NoiseParams {
            seed: 3,
            n_components: 200,
            f_lo_hz: 200.0,
            f_hi_hz: 6000.0,
            level_db: 94.0,
        }));
        // Factor 1 needs a 16 kHz scene
        let mut scene = sine_scene(&[(50, pos)], 1.0);
        scene.sample_rate = TRACK_SAMPLE_RATE;
        scene.sources[0].signal = signal.clone();

        let output = beam_output(&scene, &pos, 1);
        let distances = MicArray::uma16().get_distance_from(&pos);
        let delay_ref = distance_to_delay(distances.iter().sum::<f64>() / 16.0);
        let (mut error, mut power) = (0.0, 0.0);
        for (n, &y) in output.iter().enumerate().skip(FFT_LEN) {
            let expected = signal.eval(n as f64 / TRACK_SAMPLE_RATE - delay_ref);
            error += (y as f64 - expected).powi(2);
            power += expected.powi(2);
        }
        let relative = (error / power).sqrt();
        // About 1e-7 (wrap at the window tails); 2e-2 with the whole delay removed
        assert!(relative < 1e-5, "relative error {relative}");
    }

    /// At 48 kHz the decimated output keeps the source amplitude.
    #[test]
    fn decimated_amplitude() {
        let pos = Point3::new(0.1, 0.2, 1.0);
        let scene = sine_scene(&[(50, pos)], 0.5);
        let output = beam_output(&scene, &pos, 3);
        // N/2 samples at 48 kHz per frame, a third of them at 16 kHz
        let skip = FFT_LEN / 2;
        let amplitude = amplitude_at(&output[skip..], TRACK_SAMPLE_RATE, bin_freq(50));
        assert!((amplitude - 1.0).abs() < 1e-3, "amplitude {amplitude}");
    }

    /// Two sources: the beam toward one attenuates the other.
    #[test]
    fn attenuates_other_direction() {
        let (pos_a, pos_b) = (Point3::new(-0.6, 0.0, 1.0), Point3::new(0.6, 0.0, 1.0));
        let (bin_a, bin_b) = (70, 75);
        let scene = sine_scene(&[(bin_a, pos_a), (bin_b, pos_b)], 0.5);

        let output = beam_output(&scene, &pos_a, 3);
        let skip = FFT_LEN / 2;
        let a = amplitude_at(&output[skip..], TRACK_SAMPLE_RATE, bin_freq(bin_a));
        let b = amplitude_at(&output[skip..], TRACK_SAMPLE_RATE, bin_freq(bin_b));
        // Both sources have amplitude 1 at 1 m; B is about 1.17 m from the array center
        let suppression_db = 20.0 * (a / b).log10();
        // About 17 dB at 3.4 kHz with sources 62 degrees apart
        assert!(suppression_db > 15.0, "A {a}, B {b}: {suppression_db:.1} dB");
    }

    /// The decimation filter passes speech frequencies and blocks what would alias.
    #[test]
    fn decimator_passband_and_stopband() {
        let gain = |freq: f64| {
            let input: Vec<f64> = (0..48_000)
                .map(|n| (2.0 * PI * freq * n as f64 / SAMPLE_RATE).sin())
                .collect();
            let mut decimator = Decimator::new(3);
            let mut out = Vec::new();
            // Uneven chunks: the stream must not depend on the chunking
            for chunk in input.chunks(500) {
                decimator.push(chunk, &mut out);
            }
            assert_eq!(out.len(), 16_000);
            amplitude_at(&out[200..], TRACK_SAMPLE_RATE, freq)
        };

        assert!((gain(1000.0) - 1.0).abs() < 1e-3, "1 kHz: {}", gain(1000.0));
        assert!((gain(5000.0) - 1.0).abs() < 1e-2, "5 kHz: {}", gain(5000.0));
        // 12 kHz would alias to 4 kHz; amplitude_at(.., 12 kHz) at 16 kHz measures that alias
        let stop = gain(12_000.0);
        assert!(stop < 1e-4, "12 kHz: {stop}");
    }
}
