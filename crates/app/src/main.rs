use std::{
    collections::{HashMap, hash_map::Entry},
    env,
    error::Error,
    fs::{self, File},
    io::BufWriter,
    ops::ControlFlow,
    thread,
    time::{Duration, Instant},
};

use image::RgbaImage;
use sound2view_core::{
    BoxError, FrameSource, ImageSource, Point3,
    dsp::beamformer::{
        Normalization, clean_sc::CleanSc, delay_and_sum::Das, functional::Functional,
    },
    geometry::mic_array::MicArray,
    dsp::beam_signal::TRACK_SAMPLE_RATE,
    pipeline::{MapLayout, Pipeline, PipelineError, run},
    render,
    source::{
        capture::{self, CpalSource},
        simulator::{Simulator, scene::Scene},
        video::{self, CameraSource},
    },
    tracking::Track,
};

use crate::config::{CONFIG_PATH, Config, Method};

mod config;
mod gui;
mod theme;

/// Number of maps written in mic mode, unless given as argument
const MIC_MAPS: usize = 20;
/// Frames timed in camshot mode
const CAMSHOT_FRAMES: usize = 60;
/// CLI modes wait this long for the user to grant camera access (macOS asks on the first run)
const CAMERA_AUTH_TIMEOUT: Duration = Duration::from_secs(10);

/// Usage (settings from `sound2view.toml`, see `config.rs`):
/// - `sound2view [scene.toml]`: simulate a scene, defaults to `config.toml`
/// - `sound2view mic [maps]`: capture from the UMA-16, `MIC_MAPS` maps unless given (about 9 per
///   second)
/// - `sound2view probe`: list audio input devices
/// - `sound2view camprobe`: list cameras and their formats
/// - `sound2view camshot`: time camera frames and save one as `camshot.png`
/// - `sound2view gui`: live window with the camera image
fn main() -> Result<(), Box<dyn Error>> {
    let arg = env::args()
        .nth(1)
        .unwrap_or_else(|| "config.toml".to_string());

    match arg.as_str() {
        "probe" => capture::probe(),
        "camprobe" => {
            init_camera();
            video::probe()
        }
        "camshot" => {
            init_camera();
            camshot(&Config::load(CONFIG_PATH)?)
        }
        "gui" => {
            // The camera thread retries until access is granted
            video::initialize();
            run_gui(Config::load(CONFIG_PATH)?)
        }
        "mic" => {
            let maps = match env::args().nth(2) {
                Some(n) => n.parse().map_err(|_| format!("mic: not a map count: {n}"))?,
                None => MIC_MAPS,
            };
            run_mic(&Config::load(CONFIG_PATH)?, maps)
        }
        path => run_sim(path, &Config::load(CONFIG_PATH)?),
    }
}

/// Request camera access and wait until it is granted or `CAMERA_AUTH_TIMEOUT` passed; instant on
/// Linux
fn init_camera() {
    video::initialize();
    let start = Instant::now();
    while !video::is_authorized() && start.elapsed() < CAMERA_AUTH_TIMEOUT {
        thread::sleep(Duration::from_millis(100));
    }
}

fn camshot(config: &Config) -> Result<(), Box<dyn Error>> {
    let c = &config.camera;
    let mut camera = CameraSource::open_by_name(&c.name, c.width, c.height, c.fps)?;

    // The first frame includes stream startup, so it is not timed
    let mut frame = camera.next_frame()?;
    let start = Instant::now();
    for _ in 0..CAMSHOT_FRAMES {
        frame = camera.next_frame()?;
    }
    let dt = start.elapsed().as_secs_f64() / CAMSHOT_FRAMES as f64;
    println!(
        "{}x{}: {:.1} ms per frame ({:.1} fps), capture and MJPEG decode",
        frame.width,
        frame.height,
        dt * 1e3,
        1.0 / dt
    );

    RgbaImage::from_raw(frame.width, frame.height, frame.pixels)
        .ok_or("pixel buffer does not match the frame size")?
        .save("camshot.png")?;
    println!("saved camshot.png");
    Ok(())
}

fn run_sim(path: &str, config: &Config) -> Result<(), Box<dyn Error>> {
    println!("Reading scene from {path}");
    let scene = Scene::from_toml(&fs::read_to_string(path)?)?;

    let src_pos = scene.sources.iter().map(|s| s.pos).collect();
    let sample_rate = scene.sample_rate;
    let mut sim = Simulator::new(scene, MicArray::uma16())?;

    let on_map = save_maps(src_pos, None, config.peaks.enabled);
    run_configured(&mut sim, sample_rate, config, on_map)?;
    Ok(())
}

fn run_mic(config: &Config, maps: usize) -> Result<(), Box<dyn Error>> {
    let audio = &config.audio;
    println!(
        "Capturing from {} at {} Hz",
        audio.device, audio.sample_rate
    );
    let mut mic = CpalSource::<16>::new(&audio.device, audio.sample_rate)?;

    let sample_rate = audio.sample_rate as f64;
    run_configured(
        &mut mic,
        sample_rate,
        config,
        save_maps(Vec::new(), Some(maps), config.peaks.enabled),
    )?;

    println!("dropped {} chunks", mic.dropped());
    Ok(())
}

/// Run `source` through a pipeline with the configured beamformer and camera.
///
/// One `run` instantiation per method (static dispatch); the `match` picks it at runtime.
fn run_configured<S: FrameSource<16>>(
    source: &mut S,
    sample_rate: f64,
    config: &Config,
    on_map: impl FnMut(&[f64], &MapLayout, &[Track]) -> Result<ControlFlow<()>, BoxError>,
) -> Result<(), PipelineError> {
    let camera = config.camera.camera();
    let pipeline = &config.pipeline;
    let b = &config.beamformer;
    let (peaks, tracking) = (&config.peaks, &config.tracking);

    match b.method {
        Method::Das => {
            let pipeline = Pipeline::new(sample_rate, pipeline, peaks, tracking, camera, |steering| {
                Das::new(steering, true, Normalization::TrueLocation)
            })?;
            run(source, pipeline, on_map)
        }
        Method::Functional => {
            let pipeline = Pipeline::new(sample_rate, pipeline, peaks, tracking, camera, |steering| {
                Functional::new(
                    steering,
                    b.nu,
                    b.functional_noise_removal,
                    b.functional_equal_bins,
                )
            })?;
            run(source, pipeline, on_map)
        }
        Method::CleanSc => {
            let pipeline = Pipeline::new(sample_rate, pipeline, peaks, tracking, camera, |steering| {
                CleanSc::new(
                    steering,
                    b.clean_gain,
                    b.clean_max_iter,
                    b.clean_stop_db,
                    b.clean_beam_cells,
                    true,
                )
            })?;
            run(source, pipeline, on_map)
        }
    }
}

/// Map handler for `run`: writes each map as `map_NNN.png` with a red cross per `src_pos` and a
/// blue dot per confirmed track (if `show_tracks`), and the audio of each confirmed track to
/// `track_NNN.wav` (see [`TrackRecorder`]).
///
/// Stops after `max_maps` maps; `None` runs until the stream ends.
fn save_maps(
    src_pos: Vec<Point3>,
    max_maps: Option<usize>,
    show_tracks: bool,
) -> impl FnMut(&[f64], &MapLayout, &[Track]) -> Result<ControlFlow<()>, BoxError> {
    let mut i = 0;
    let mut recorder = TrackRecorder::default();
    move |power, layout, tracks| {
        let shown = if show_tracks { tracks } else { &[] };
        render::map_image(power, layout, &src_pos, shown).save(format!("map_{i:03}.png"))?;
        recorder.record(tracks)?;
        i += 1;
        if max_maps.is_some_and(|max| i >= max) {
            Ok(ControlFlow::Break(()))
        } else {
            Ok(ControlFlow::Continue(()))
        }
    }
}

/// Writes the audio of each confirmed track to `track_NNN.wav` (NNN = track id), mono 32-bit
/// float at `TRACK_SAMPLE_RATE`, to check by ear whether a beam isolates its source. Audio from
/// before the confirmation is not written. The files are finalized when the recorder is dropped.
#[derive(Default)]
struct TrackRecorder {
    writers: HashMap<u64, hound::WavWriter<BufWriter<File>>>,
}

impl TrackRecorder {
    fn record(&mut self, tracks: &[Track]) -> Result<(), BoxError> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: TRACK_SAMPLE_RATE as u32,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        for track in tracks.iter().filter(|track| track.confirmed) {
            let writer = match self.writers.entry(track.id) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let path = format!("track_{:03}.wav", track.id);
                    entry.insert(hound::WavWriter::create(path, spec)?)
                }
            };
            for &sample in &track.samples {
                writer.write_sample(sample)?;
            }
        }
        Ok(())
    }
}

fn run_gui(config: Config) -> Result<(), Box<dyn Error>> {
    eframe::run_native(
        "sound2view",
        eframe::NativeOptions {
            viewport: eframe::egui::ViewportBuilder::default().with_maximized(true),
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(gui::App::new(cc, config)))),
    )?;
    Ok(())
}
