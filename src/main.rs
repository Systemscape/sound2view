use std::{env, error::Error, fs};

use sound2view::{
    dsp::beamformer::{Normalization, Steering, delay_and_sum::Das},
    geometry::mic_array::MicArray,
    pipeline::{Pipeline, run},
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

    let pipeline = Pipeline::new(
        scene.sample_rate,
        scene.sources.iter().map(|s| s.pos).collect(),
        make_das,
    );
    let mut sim = Simulator::new(scene, MicArray::uma16());
    run(&mut sim, pipeline, None)
}

fn run_mic() -> Result<(), Box<dyn Error>> {
    println!("Capturing from {MIC_DEVICE} at {MIC_SAMPLE_RATE} Hz");
    let mut mic = CpalSource::<16>::new(MIC_DEVICE, MIC_SAMPLE_RATE)?;

    let pipeline = Pipeline::new(MIC_SAMPLE_RATE as f64, Vec::new(), make_das);
    run(&mut mic, pipeline, Some(MIC_MAPS))?;

    println!("dropped {} chunks", mic.dropped());
    Ok(())
}

/// DAS with diagonal removal and true-location normalization
fn make_das(steering: Steering<16>) -> Das<16> {
    Das::new(steering, true, Normalization::TrueLocation)
}
