//! Processing chain after the `FrameSource`: STFT, covariance, beamformer and map output.

use std::{
    error::Error,
    ops::{ControlFlow, RangeInclusive},
    time::Instant,
};

use crate::{
    FrameSource,
    dsp::{
        beamformer::{Beamformer, Steering},
        covariance::SlidingCovariance,
        stft::{Stft, StftStream},
    },
    geometry::{
        camera::{Camera, Intrinsics, Mount},
        mic_array::MicArray,
        scan_grid::ScanGrid,
    },
};

/// STFT length N, hop is N/2
const FFT_LEN: usize = 1024;
/// Memory of the sliding covariance in STFT frames (about 0.43 s at 48 kHz)
const L_FRAMES: usize = 40;
/// One map every this many STFT frames (about 0.11 s at 48 kHz); `power_map` must take less
const MAP_EVERY: usize = 10;
/// Band summed in the power map, in Hz
const F_LO_HZ: f64 = 900.0;
const F_HI_HZ: f64 = 4100.0;
/// Scan grid width in grid pixels
const GRID_WIDTH: u32 = 64;
/// Focus distance of the scan grid in meters
const GRID_Z: f64 = 2.0;
/// Camera horizontal field of view in degrees
const HFOV_DEG: f64 = 83.0;

/// Everything after the `FrameSource`: STFT, covariance, beamformer and output
pub struct Pipeline<const M: usize, B: Beamformer<M>> {
    stream: StftStream<M>,
    cov: SlidingCovariance<M>,
    beamformer: B,
    /// Bins summed in the power map
    bins: RangeInclusive<usize>,
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
    /// UMA-16 pipeline for `sample_rate` in Hz; `src_pos` are the true sources for the markers.
    ///
    /// `make_beamformer` builds the beamformer from the steering of this pipeline's scan grid.
    pub fn new(sample_rate: f64, make_beamformer: impl FnOnce(Steering<16>) -> B) -> Self {
        let camera = Camera {
            intrinsics: Intrinsics::from_hfov(1920, 1080, HFOV_DEG.to_radians()),
            mount: Mount::default(),
        };
        let scan_grid = ScanGrid::new(&camera, GRID_WIDTH, GRID_Z);

        let df = sample_rate / FFT_LEN as f64;
        let bins = (F_LO_HZ / df).round() as usize..=(F_HI_HZ / df).round() as usize;

        println!(
            "Grid {}x{}, bins {bins:?}, L = {L_FRAMES}, map every {MAP_EVERY} frames",
            scan_grid.width(),
            scan_grid.height()
        );

        Self {
            stream: StftStream::new(Stft::new(FFT_LEN)),
            cov: SlidingCovariance::new(FFT_LEN / 2 + 1, L_FRAMES),
            beamformer: make_beamformer(Steering::new(
                &scan_grid,
                &MicArray::uma16(),
                sample_rate,
                FFT_LEN,
            )),
            bins,
            layout: MapLayout {
                camera,
                grid_width: scan_grid.width(),
                grid_height: scan_grid.height(),
            },
        }
    }
}

/// Read `source` through the pipeline; one power map every `MAP_EVERY` STFT frames.
///
/// The covariance is a sliding average over about `L_FRAMES` frames, so consecutive maps overlap.
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

        if new_frames >= MAP_EVERY {
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
