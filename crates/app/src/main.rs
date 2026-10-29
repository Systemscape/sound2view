use std::{env, error::Error, fs, ops::ControlFlow, time::Instant};

use image::RgbaImage;
use sound2view_core::{
    ImageSource, Point3,
    dsp::beamformer::{Normalization, Steering, delay_and_sum::Das, functional::Functional},
    geometry::mic_array::MicArray,
    pipeline::{MapLayout, Pipeline, run},
    render,
    source::{
        capture::{self, CpalSource},
        simulator::{Simulator, scene::Scene},
        video::{self, CameraSource},
    },
};

mod gui;

/// Input device id pattern of the UMA-16, see `sound2view probe`
const MIC_DEVICE: &str = "hw:CARD=UMA16v2";
const MIC_SAMPLE_RATE: u32 = 48_000;
/// Number of maps written in mic mode
const MIC_MAPS: usize = 20;

/// V4L2 index of the camera (`/dev/videoN`), see `sound2view camprobe`
const CAMERA_INDEX: u32 = 6;
/// Frames timed in camshot mode
const CAMSHOT_FRAMES: usize = 60;

/// Usage:
/// - `sound2view [scene.toml]`: simulate a scene, defaults to `config.toml`
/// - `sound2view mic`: capture from the UMA-16
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
        "camprobe" => video::probe(),
        "camshot" => camshot(),
        "gui" => run_gui(),
        "mic" => run_mic(),
        path => run_sim(path),
    }
}

fn camshot() -> Result<(), Box<dyn Error>> {
    let mut camera = CameraSource::new(CAMERA_INDEX)?;

    // The first frame includes stream startup, so it is not timed
    let mut frame = camera.next_frame().ok_or("no frame from camera")?;
    let start = Instant::now();
    for _ in 0..CAMSHOT_FRAMES {
        frame = camera.next_frame().ok_or("no frame from camera")?;
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

fn run_sim(path: &str) -> Result<(), Box<dyn Error>> {
    println!("Reading scene from {path}");
    let scene = Scene::from_toml(&fs::read_to_string(path)?)?;
    scene.validate();

    let src_pos = scene.sources.iter().map(|s| s.pos).collect();
    let pipeline = Pipeline::new(scene.sample_rate, make_das);
    let mut sim = Simulator::new(scene, MicArray::uma16());

    // `?` does not convert `+ Send + Sync` away, the cast does
    run(&mut sim, pipeline, save_maps(src_pos, None)).map_err(|e| e as Box<dyn Error>)?;
    Ok(())
}

fn run_mic() -> Result<(), Box<dyn Error>> {
    println!("Capturing from {MIC_DEVICE} at {MIC_SAMPLE_RATE} Hz");
    let mut mic = CpalSource::<16>::new(MIC_DEVICE, MIC_SAMPLE_RATE)?;

    let pipeline = Pipeline::new(MIC_SAMPLE_RATE as f64, make_das);
    run(&mut mic, pipeline, save_maps(Vec::new(), Some(MIC_MAPS)))
        .map_err(|e| e as Box<dyn Error>)?;

    println!("dropped {} chunks", mic.dropped());
    Ok(())
}

/// DAS with diagonal removal and true-location normalization
fn make_das(steering: Steering<16>) -> Das<16> {
    Das::new(steering, true, Normalization::TrueLocation)
}

/// Exponent nu of functional beamforming; larger is sharper but more sensitive to model errors
const FUNCTIONAL_NU: f64 = 20.0;

fn make_functional(steering: Steering<16>) -> Functional<16> {
    Functional::new(steering, FUNCTIONAL_NU)
}

/// Map handler for `run`: writes each map as `map_NNN.png` with a marker per `src_pos`.
///
/// Stops after `max_maps` maps; `None` runs until the stream ends.
fn save_maps(
    src_pos: Vec<Point3>,
    max_maps: Option<usize>,
) -> impl FnMut(&[f64], &MapLayout) -> Result<ControlFlow<()>, Box<dyn Error + Send + Sync>> {
    let mut i = 0;
    move |power, layout| {
        render::map_image(power, layout, &src_pos).save(format!("map_{i:03}.png"))?;
        i += 1;
        if max_maps.is_some_and(|max| i >= max) {
            Ok(ControlFlow::Break(()))
        } else {
            Ok(ControlFlow::Continue(()))
        }
    }
}

fn run_gui() -> Result<(), Box<dyn Error>> {
    eframe::run_native(
        "sound2view",
        eframe::NativeOptions::default(),
        Box::new(|cc| Ok(Box::new(gui::App::new(cc, CAMERA_INDEX)))),
    )?;
    Ok(())
}
