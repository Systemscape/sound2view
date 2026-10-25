//! Processing chain after the `FrameSource`: STFT, covariance, beamformer and map output.

use std::{error::Error, ops::RangeInclusive, time::Instant};

use crate::{
    FrameSource, Point3,
    dsp::{
        beamformer::{Beamformer, Steering},
        covariance::Covariance,
        stft::{Stft, StftStream},
    },
    geometry::{
        camera::{Camera, Intrinsics, Mount},
        mic_array::MicArray,
        scan_grid::ScanGrid,
    },
    render,
};

/// STFT length N, hop is N/2
const FFT_LEN: usize = 1024;
/// Frames per covariance estimate, one map per L frames
const L_FRAMES: usize = 40;
/// Band summed in the power map, in Hz
const F_LO_HZ: f64 = 2500.0;
const F_HI_HZ: f64 = 4100.0;
/// Scan grid width in grid pixels
const GRID_WIDTH: u32 = 64;
/// Focus distance of the scan grid in meters
const GRID_Z: f64 = 2.0;
/// Camera horizontal field of view in degrees
const HFOV_DEG: f64 = 110.0;

/// Everything after the `FrameSource`: STFT, covariance, beamformer and output
pub struct Pipeline<const M: usize, B: Beamformer<M>> {
    stream: StftStream<M>,
    cov: Covariance<M>,
    beamformer: B,
    /// Bins summed in the power map
    bins: RangeInclusive<usize>,
    /// Scan grid size in grid pixels
    grid_width: usize,
    grid_height: usize,
    camera: Camera,
    /// Ground truth source positions for the markers, array frame
    src_pos: Vec<Point3>,
}

impl<B: Beamformer<16>> Pipeline<16, B> {
    /// UMA-16 pipeline for `sample_rate` in Hz; `src_pos` are the true sources for the markers.
    ///
    /// `make_beamformer` builds the beamformer from the steering of this pipeline's scan grid.
    pub fn new(
        sample_rate: f64,
        src_pos: Vec<Point3>,
        make_beamformer: impl FnOnce(Steering<16>) -> B,
    ) -> Self {
        let camera = Camera {
            intrinsics: Intrinsics::from_hfov(1920, 1080, HFOV_DEG.to_radians()),
            mount: Mount::default(),
        };
        let scan_grid = ScanGrid::new(&camera, GRID_WIDTH, GRID_Z);

        let df = sample_rate / FFT_LEN as f64;
        let bins = (F_LO_HZ / df).round() as usize..=(F_HI_HZ / df).round() as usize;

        println!(
            "Grid {}x{}, bins {bins:?}, L = {L_FRAMES}",
            scan_grid.width(),
            scan_grid.height()
        );

        Self {
            stream: StftStream::new(Stft::new(FFT_LEN)),
            cov: Covariance::new(FFT_LEN / 2 + 1),
            beamformer: make_beamformer(Steering::new(
                &scan_grid,
                &MicArray::uma16(),
                sample_rate,
                FFT_LEN,
            )),
            bins,
            grid_width: scan_grid.width(),
            grid_height: scan_grid.height(),
            camera,
            src_pos,
        }
    }
}

/// Read the stream until the end or until `max_maps` maps are written; one power map per `L_FRAMES` STFT frames.
pub fn run<S: FrameSource<M>, B: Beamformer<M>, const M: usize>(
    source: &mut S,
    mut pipeline: Pipeline<M, B>,
    max_maps: Option<usize>,
) -> Result<(), Box<dyn Error>> {
    let mut buf = vec![0.0; 1024 * M];
    let mut total_frames = 0;
    let mut out = vec![0.0; pipeline.beamformer.n_points()];
    let mut map_index = 0;

    loop {
        let frames = source.read(&mut buf)?;
        if frames == 0 {
            break;
        }
        total_frames += frames;

        // Closures capture disjoint fields, so `stream` and `cov` can be borrowed together
        pipeline.stream.push(&buf[..frames * M], |snapshots, _l| {
            pipeline.cov.accumulate(snapshots)
        });

        if pipeline.cov.count() >= L_FRAMES {
            let cov_est = pipeline.cov.estimate();
            let start = Instant::now();
            pipeline
                .beamformer
                .power_map(&cov_est, pipeline.bins.clone(), &mut out);
            println!(
                "map {map_index}: power_map {:.1} ms",
                start.elapsed().as_secs_f64() * 1e3
            );

            render::map_image(
                &out,
                pipeline.grid_width,
                pipeline.grid_height,
                &pipeline.camera,
                &pipeline.src_pos,
            )
            .save(format!("map_{map_index:03}.png"))?;
            map_index += 1;

            pipeline.cov.reset();

            if max_maps.is_some_and(|max| map_index >= max) {
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
