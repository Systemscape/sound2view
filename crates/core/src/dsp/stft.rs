use std::{f64::consts::PI, sync::Arc};

use nalgebra::SVector;
use realfft::{RealFftPlanner, RealToComplex};

use crate::Complex64;

pub struct Stft {
    plan: Arc<dyn RealToComplex<f64>>,
    /// Scratch buffer for FFT
    scratch_buf: Vec<Complex64>,
    /// FFT input buffer
    input_buf: Vec<f64>,
    output_buf: Vec<Complex64>,
    window: Vec<f64>,
    /// Kept for debugging and testing
    _scale: f64,
    weights: Vec<f64>,
}

impl Stft {
    pub fn new(fft_len: usize) -> Self {
        assert_eq!(fft_len % 2, 0, "FFT length must be even (50% hops)");
        let plan = RealFftPlanner::new().plan_fft_forward(fft_len);
        let hann_window: Vec<f64> = (0..fft_len)
            .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f64 / fft_len as f64).cos())
            .collect();

        // normalize with 1/sqrt(sum(weights^2))
        let scale = 1.0 / hann_window.iter().map(|w| w * w).sum::<f64>().sqrt();

        // pre-compute weights as scaled hann window
        let weights = hann_window.iter().map(|w| w * scale).collect();

        Self {
            scratch_buf: plan.make_scratch_vec(),
            input_buf: plan.make_input_vec(),
            output_buf: plan.make_output_vec(),
            plan,
            window: hann_window,
            _scale: scale,
            weights,
        }
    }

    pub fn fft_len(&self) -> usize {
        self.window.len()
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn transform(&mut self, frame: &[f64]) -> &[Complex64] {
        assert_eq!(
            frame.len(),
            self.input_buf.len(),
            "frame length must equal FFT length"
        );

        // apply window function and scaling
        self.input_buf
            .iter_mut()
            .enumerate()
            .for_each(|(idx, input)| *input = self.weights[idx] * frame[idx]);

        self.plan
            .process_with_scratch(
                &mut self.input_buf,
                &mut self.output_buf,
                &mut self.scratch_buf,
            )
            .expect("all slices should have the correct length");

        &self.output_buf
    }
}

pub struct StftStream<const M: usize> {
    stft: Stft,
    /// fft_len samples per channel for M channels
    buf: [Vec<f64>; M],
    /// fft_len/2+1 entries, allocated once and overwritten per frame
    snapshots: Vec<SVector<Complex64, M>>,
    /// Number of elements currently in the buffer
    fill: usize,
    /// Current frame
    frame_count: u64,
}

impl<const M: usize> StftStream<M> {
    pub fn new(stft: Stft) -> Self {
        let fft_len = stft.fft_len();
        Self {
            stft,
            buf: std::array::from_fn(|_| vec![0.0; fft_len]),
            snapshots: vec![SVector::zeros(); fft_len / 2 + 1],
            fill: 0,
            frame_count: 0,
        }
    }

    pub fn push(&mut self, chunk: &[f64], mut on_frame: impl FnMut(&[SVector<Complex64, M>], u64)) {
        assert_eq!(chunk.len() % M, 0, "chunk len must be a multiple of M");
        let fft_len = self.stft.fft_len();

        for frame in chunk.as_chunks::<M>().0 {
            for m in 0..M {
                self.buf[m][self.fill] = frame[m];
            }
            self.fill += 1;

            if self.fill == self.stft.fft_len() {
                for m in 0..M {
                    for (bin, value) in self.stft.transform(&self.buf[m]).iter().enumerate() {
                        self.snapshots[bin][m] = *value;
                    }
                    // Shift left by N/2
                    self.buf[m].copy_within(fft_len / 2.., 0);
                }

                self.fill = fft_len / 2;
                on_frame(&self.snapshots, self.frame_count);

                // Increment per FFT frame
                self.frame_count += 1;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::f64::consts::PI;

    use nalgebra::SVector;

    use crate::{
        Complex64, distance_to_delay,
        dsp::stft::{Stft, StftStream},
        geometry::mic_array::MicArray,
        source::simulator::Simulator,
        test_util::*,
    };

    const STREAM_FFT_LEN: usize = 64;
    const STREAM_CHANNELS: usize = 3;
    const STREAM_FRAMES: usize = 5000;

    /// Interleaved test input, a different sine per channel.
    fn stream_input() -> Vec<f64> {
        (0..STREAM_FRAMES)
            .flat_map(|n| {
                (0..STREAM_CHANNELS).map(move |m| (0.01 * (m + 1) as f64 * n as f64).sin())
            })
            .collect()
    }

    /// Push `input` in chunks of `chunk_frames`; returns all (l, snapshots).
    fn stream_frames(
        input: &[f64],
        chunk_frames: usize,
    ) -> Vec<(u64, Vec<SVector<Complex64, STREAM_CHANNELS>>)> {
        let mut stream = StftStream::<STREAM_CHANNELS>::new(Stft::new(STREAM_FFT_LEN));
        let mut frames = Vec::new();
        for chunk in input.chunks(chunk_frames * STREAM_CHANNELS) {
            stream.push(chunk, |snapshots, l| frames.push((l, snapshots.to_vec())));
        }
        frames
    }

    #[test]
    fn stream_frame_count() {
        let frames = stream_frames(&stream_input(), 100);

        let hop = STREAM_FFT_LEN / 2;
        let expected = (STREAM_FRAMES - STREAM_FFT_LEN) / hop + 1;
        assert_eq!(frames.len(), expected);

        // l = 0, 1, 2, ... without gaps
        for (i, (l, snapshots)) in frames.iter().enumerate() {
            assert_eq!(*l, i as u64, "frame index");
            assert_eq!(snapshots.len(), STREAM_FFT_LEN / 2 + 1, "bins per frame");
        }
    }

    #[test]
    fn stream_chunk_independence() {
        let input = stream_input();

        // 7 does not divide the hop, so frames complete in the middle of chunks.
        let chunked = stream_frames(&input, 7);
        let single = stream_frames(&input, STREAM_FRAMES);

        assert_eq!(chunked.len(), single.len());
        for (a, b) in chunked.iter().zip(&single) {
            assert_eq!(a, b, "frame {} differs", b.0);
        }
    }

    #[test]
    fn scale() {
        let fft_len = 256;
        let stft = Stft::new(fft_len);

        // Scaling factor should be approx sqrt(8 / 3N)
        assert!(
            (stft._scale - (8.0 / 3.0 / fft_len as f64).sqrt()).abs() < 1e-12,
            "Scaling factor should be sqrt(8 / 3N)"
        )
    }

    #[test]
    fn constant_overlap_add() {
        let fft_len = 256;
        let stft = Stft::new(fft_len);

        // assert the constant overlap-add property at 50% overlap
        for n in 0..fft_len / 2 {
            assert!(
                (stft.window[n] + stft.window[n + fft_len / 2] - 1.0).abs() < 1e-12,
                "w[n] + w[n + N/2] = 1"
            );
        }
    }

    #[test]
    fn sine_peak() {
        let fft_len = 1024;
        let sample_rate = 48_000.0;
        let bin = 50;
        let amplitude = 2.0;

        // Bin center: an integer number of periods fits into the frame
        let freq_hz = bin as f64 * sample_rate / fft_len as f64;
        let frame: Vec<f64> = (0..fft_len)
            .map(|n| amplitude * (2.0 * PI * freq_hz * n as f64 / sample_rate + 0.3).sin())
            .collect();

        let mut stft = Stft::new(fft_len);
        let power: Vec<f64> = stft
            .transform(&frame)
            .iter()
            .map(|x| x.norm_sqr())
            .collect();

        // Hann on a bin center: (A^2/2)*N/3 in the peak bin, a quarter of that in
        // each neighbor, zero elsewhere.
        let p_peak = amplitude.powi(2) / 2.0 * fft_len as f64 / 3.0;
        for (k, p) in power.iter().enumerate() {
            let expected = match k.abs_diff(bin) {
                0 => p_peak,
                1 => p_peak / 4.0,
                _ => 0.0,
            };
            assert!(
                (p - expected).abs() < 1e-9 * p_peak,
                "bin {k}: expected {expected}, got {p}"
            );
        }
    }

    /// Simulator -> STFT end to end: relative to mic 0, each mic must show
    /// x_m / x_0 = (r_0/r_m)*e^{-j*2*PI*f*(tau_m - tau_0)}. Pins the sign convention
    /// e^{-j * omega * tau}, the channel order and the 1/r gain.
    #[test]
    fn phase() {
        // Asymmetric position, so all 16 distances differ. F_SINE is a bin center:
        // a delayed tone stays a pure bin-center tone, so the ratio is exact.
        let mic_array = MicArray::uma16();
        let mut sim = Simulator::new(test_scene(-1000.0, 0.1), mic_array.clone());
        let samples = read_all(&mut sim, 4096);

        // Bin BIN of the last STFT frame; the sine is in steady state from t = 0
        let mut x: Option<SVector<Complex64, 16>> = None;

        StftStream::new(Stft::new(FFT_LEN))
            .push(&samples, |snapshots, _l| x = Some(snapshots[BIN]));

        let x = x.expect("at least one STFT frame");

        // r_m and tau_m for every mic
        let r = mic_array.get_distance_from(&test_pos());
        let tau = r.map(distance_to_delay);

        for m in 0..16 {
            // x_m / x_0, written so the unknown source spectrum cancels
            let z = x[m] * x[0].conj() / x[0].norm_sqr();
            let expected =
                Complex64::from_polar(r[0] / r[m], -2.0 * PI * F_SINE * (tau[m] - tau[0]));

            // Distance in the complex plane: checks magnitude and phase, immune to wrapping
            assert!(
                (z - expected).norm() < 1e-9,
                "mic {m}: z {z}, expected {expected}"
            );
        }
    }
}
