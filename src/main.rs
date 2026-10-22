use std::{env, error::Error, fs, ops::RangeInclusive, time::Instant};

use image::{
    Rgb, RgbImage,
    imageops::{self, FilterType},
};
use sound2view::{
    FrameSource, Point3,
    beamformer::{Das, Normalization},
    camera::{Camera, Intrinsics, Mount},
    capture::{self, CpalSource},
    covariance::Covariance,
    mic_array::MicArray,
    scan_grid::ScanGrid,
    simulator::{Simulator, scene::Scene},
    stft::{Stft, StftStream},
};

/// STFT length N, hop is N/2
const FFT_LEN: usize = 1024;
/// Frames per covariance estimate, one map per L frames
const L_FRAMES: usize = 40;
/// Band summed in the power map, in Hz
const F_LO_HZ: f64 = 900.0;
const F_HI_HZ: f64 = 4100.0;
/// Scan grid width in grid pixels
const GRID_WIDTH: u32 = 64;
/// Focus distance of the scan grid in meters
const GRID_Z: f64 = 2.0;
/// Camera horizontal field of view in degrees
const HFOV_DEG: f64 = 110.0;

/// Everything after the `FrameSource`: STFT, covariance, beamformer and output
struct Pipeline<const M: usize> {
    stream: StftStream<M>,
    cov: Covariance<M>,
    das: Das<M>,
    /// Bins summed in the power map
    bins: RangeInclusive<usize>,
    /// Scan grid size in grid pixels
    grid_width: usize,
    grid_height: usize,
    camera: Camera,
    /// Ground truth source positions for the markers, array frame
    src_pos: Vec<Point3>,
}

impl Pipeline<16> {
    /// UMA-16 pipeline for `sample_rate` in Hz; `src_pos` are the true sources for the markers.
    fn new(sample_rate: f64, src_pos: Vec<Point3>) -> Self {
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
            das: Das::new(
                &scan_grid,
                &MicArray::uma16(),
                sample_rate,
                FFT_LEN,
                true,
                Normalization::TrueLocation,
            ),
            bins,
            grid_width: scan_grid.width(),
            grid_height: scan_grid.height(),
            camera,
            src_pos,
        }
    }
}

/// Input device id pattern of the UMA-16, see `sound2view probe`
const MIC_DEVICE: &str = "hw:CARD=UMA16v2";
const MIC_SAMPLE_RATE: u32 = 48_000;
/// Number of maps written in mic mode
const MIC_MAPS: usize = 20;

/// Usage:
/// - `sound2view [scene.toml]`: simulate a scene, defaults to `config.toml`
/// - `sound2view mic`: capture from the UMA-16
/// - `sound2view probe`: list input devices
fn main() -> Result<(), Box<dyn Error>> {
    let arg = env::args()
        .nth(1)
        .unwrap_or_else(|| "config.toml".to_string());

    match arg.as_str() {
        "probe" => capture::probe(),
        "mic" => run_mic(),
        path => run_sim(path),
    }
}

fn run_sim(path: &str) -> Result<(), Box<dyn Error>> {
    println!("Reading scene from {path}");
    let scene = Scene::from_toml(&fs::read_to_string(path)?)?;
    scene.validate();

    let pipeline = Pipeline::new(
        scene.sample_rate,
        scene.sources.iter().map(|s| s.pos).collect(),
    );
    let mut sim = Simulator::new(scene, MicArray::uma16());
    run(&mut sim, pipeline, None)
}

fn run_mic() -> Result<(), Box<dyn Error>> {
    println!("Capturing from {MIC_DEVICE} at {MIC_SAMPLE_RATE} Hz");
    let mut mic = CpalSource::<16>::new(MIC_DEVICE, MIC_SAMPLE_RATE)?;

    let pipeline = Pipeline::new(MIC_SAMPLE_RATE as f64, Vec::new());
    run(&mut mic, pipeline, Some(MIC_MAPS))?;

    println!("dropped {} chunks", mic.dropped());
    Ok(())
}

/// Read the stream until the end or until `max_maps` maps are written; one power map per `L_FRAMES` STFT frames.
fn run<S: FrameSource<M>, const M: usize>(
    source: &mut S,
    mut pipeline: Pipeline<M>,
    max_maps: Option<usize>,
) -> Result<(), Box<dyn Error>> {
    let mut buf = vec![0.0; 1024 * M];
    let mut total_frames = 0;
    let mut out = vec![0.0; pipeline.das.n_points()];
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
                .das
                .power_map(&cov_est, pipeline.bins.clone(), &mut out);
            println!("map {map_index}: power_map {:.1} ms", start.elapsed().as_secs_f64() * 1e3);

            write_map(&out, &pipeline, &format!("map_{map_index:03}.png"))?;
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

/// Marker half length in camera pixels
const MARKER_HALF_LEN: i64 = 10;

/// Write a power map as PNG at camera resolution, with a cross per true source.
///
/// - `power`: row-major like the scan grid
fn write_map<const M: usize>(
    power: &[f64],
    pipeline: &Pipeline<M>,
    path: &str,
) -> image::ImageResult<()> {
    // One grid pixel per map value; x = col, y = row, same as `power[row * w + col]`
    let w = pipeline.grid_width;
    let h = pipeline.grid_height;
    let p_max = power.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let img = RgbImage::from_fn(w as u32, h as u32, |x, y| {
        power_to_rgb(power[y as usize * w + x as usize], p_max)
    });

    // Upscale to camera resolution; Nearest keeps the grid pixels visible
    let intrinsics = &pipeline.camera.intrinsics;
    let mut big = imageops::resize(&img, intrinsics.width, intrinsics.height, FilterType::Nearest);

    for pos in &pipeline.src_pos {
        // `project` panics behind the camera
        if pos.z <= 0.0 {
            continue;
        }
        let (u, v) = pipeline.camera.project(pos);
        let (u, v) = (u.round() as i64, v.round() as i64);

        for d in -MARKER_HALF_LEN..=MARKER_HALF_LEN {
            for (x, y) in [(u + d, v), (u, v + d)] {
                // Skips pixels outside the image, including sources outside the FoV
                if (0..big.width() as i64).contains(&x) && (0..big.height() as i64).contains(&y) {
                    big.put_pixel(x as u32, y as u32, Rgb([255, 0, 0]));
                }
            }
        }
    }

    big.save(path)
}

/// Map power to gray: 0 dB (the maximum) is white, -15 dB and below is black.
fn power_to_rgb(p: f64, p_max: f64) -> Rgb<u8> {
    let db = 10.0 * (p / p_max).max(1e-10).log10();
    // -15 dB -> 0, 0 dB -> 1
    let t = ((db + 15.0) / 15.0).clamp(0.0, 1.0);
    let g = (t * 255.0) as u8;
    Rgb([g, g, g])
}
