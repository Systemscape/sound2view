//! Processing chain after the `FrameSource`: STFT, covariance, beamformer and map output.

use std::{
    error::Error,
    ops::{ControlFlow, RangeInclusive},
    time::Instant,
};

use serde::Deserialize;

use crate::{
    FrameSource,
    dsp::{
        beamformer::{Beamformer, Steering},
        covariance::SlidingCovariance,
        stft::{Stft, StftStream},
    },
    geometry::{camera::Camera, mic_array::MicArray, scan_grid::ScanGrid},
};

/// Processing parameters, e.g. from a config file; the defaults are tested with the UMA-16 at 48 kHz
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PipelineConfig {
    /// STFT length N, hop is N/2
    pub fft_len: usize,
    /// Memory of the sliding covariance in STFT frames (40: about 0.43 s at 48 kHz)
    pub memory_frames: usize,
    /// One map every this many STFT frames (10: about 0.11 s at 48 kHz); `power_map` must take less
    pub map_every: usize,
    /// Lower end of the band summed in the power map, in Hz
    pub f_lo_hz: f64,
    /// Upper end of the band, in Hz
    pub f_hi_hz: f64,
    /// Scan grid width in grid pixels; the height follows from the camera aspect ratio
    pub grid_width: u32,
    /// Focus distance of the scan grid in meters
    pub grid_z: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            fft_len: 1024,
            memory_frames: 40,
            map_every: 10,
            f_lo_hz: 900.0,
            f_hi_hz: 4100.0,
            grid_width: 64,
            grid_z: 2.0,
        }
    }
}

/// Everything after the `FrameSource`: STFT, covariance, beamformer and output
pub struct Pipeline<const M: usize, B: Beamformer<M>> {
    stream: StftStream<M>,
    cov: SlidingCovariance<M>,
    beamformer: B,
    /// Bins summed in the power map
    bins: RangeInclusive<usize>,
    /// One map every this many STFT frames
    map_every: usize,
    /// Layout of the power map
    layout: MapLayout,
}

#[derive(Clone)]
pub struct MapLayout {
    pub camera: Camera,
    pub grid_width: usize,
    pub grid_height: usize,
}

impl<B: Beamformer<16>> Pipeline<16, B> {
    /// UMA-16 pipeline for `sample_rate` in Hz; the scan grid covers the field of view of `camera`.
    ///
    /// `make_beamformer` builds the beamformer from the steering of this pipeline's scan grid.
    pub fn new(
        sample_rate: f64,
        config: &PipelineConfig,
        camera: Camera,
        make_beamformer: impl FnOnce(Steering<16>) -> B,
    ) -> Self {
        // TODO(Julian): config errors as `Result` with the thiserror plan
        assert!(
            config.f_lo_hz < config.f_hi_hz,
            "band: f_lo_hz must be below f_hi_hz"
        );
        assert!(config.map_every > 0, "map_every must be at least 1");

        let scan_grid = ScanGrid::new(&camera, config.grid_width, config.grid_z);

        let n_bins = config.fft_len / 2 + 1;
        let df = sample_rate / config.fft_len as f64;
        // The upper end is clipped to the last bin (f_s/2)
        let bin_lo = (config.f_lo_hz / df).round() as usize;
        let bin_hi = ((config.f_hi_hz / df).round() as usize).min(n_bins - 1);
        let bins = bin_lo..=bin_hi;

        println!(
            "Grid {}x{}, bins {bins:?}, memory {} frames, map every {} frames",
            scan_grid.width(),
            scan_grid.height(),
            config.memory_frames,
            config.map_every
        );

        Self {
            stream: StftStream::new(Stft::new(config.fft_len)),
            cov: SlidingCovariance::new(n_bins, config.memory_frames),
            beamformer: make_beamformer(Steering::new(
                &scan_grid,
                &MicArray::uma16(),
                sample_rate,
                config.fft_len,
            )),
            bins,
            map_every: config.map_every,
            layout: MapLayout {
                camera,
                grid_width: scan_grid.width(),
                grid_height: scan_grid.height(),
            },
        }
    }
}

/// Read `source` through the pipeline; one power map every `map_every` STFT frames.
///
/// The covariance is a sliding average over about `memory_frames` frames, so consecutive maps
/// overlap.
/// Each map is passed to `on_map` together with its [`MapLayout`]. Stops when the source ends
/// (`Ok(0)`), when `on_map` returns `Break`, or on the first error from `source` or `on_map`.
pub fn run<S: FrameSource<M>, B: Beamformer<M>, const M: usize>(
    source: &mut S,
    mut pipeline: Pipeline<M, B>,
    mut on_map: impl FnMut(&[f64], &MapLayout) -> Result<ControlFlow<()>, Box<dyn Error + Send + Sync>>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut buf = vec![0.0; 1024 * M];
    let mut total_frames = 0;
    let mut out = vec![0.0; pipeline.beamformer.n_points()];
    let mut map_index = 0;
    // STFT frames since the last map
    let mut new_frames = 0;

    loop {
        let frames = source.read(&mut buf)?;
        if frames == 0 {
            break;
        }
        total_frames += frames;

        // Closures capture disjoint fields, so `stream` and `cov` can be borrowed together
        pipeline.stream.push(&buf[..frames * M], |snapshots, _l| {
            pipeline.cov.accumulate(snapshots);
            new_frames += 1;
        });

        if new_frames >= pipeline.map_every {
            let start = Instant::now();
            pipeline
                .beamformer
                .power_map(pipeline.cov.estimate(), pipeline.bins.clone(), &mut out);
            println!(
                "map {map_index}: power_map {:.1} ms",
                start.elapsed().as_secs_f64() * 1e3
            );

            new_frames = 0;
            map_index += 1;

            if on_map(&out, &pipeline.layout)?.is_break() {
                break;
            }
        }
    }

    println!(
        "read {total_frames} frames ({:.2} s, {M} channels), {map_index} maps",
        total_frames as f64 / source.sample_rate()
    );
    Ok(())
}
