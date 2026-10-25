use std::{env, error::Error, fs, ops::ControlFlow};

use sound2view_core::{
    Point3,
    dsp::beamformer::{Normalization, Steering, delay_and_sum::Das},
    geometry::mic_array::MicArray,
    pipeline::{MapLayout, Pipeline, run},
    render,
    source::{
        capture::{self, CpalSource},
        simulator::{Simulator, scene::Scene},
    },
};

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
