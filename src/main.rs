use std::{env, error::Error, fs};

use audio2video_rs::{
    FrameSource,
    mic_array::MicArray,
    simulator::{Simulator, scene::Scene},
};

/// Usage: `audio2video-rs [scene.toml]`, defaults to `config.toml`.
fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args().nth(1).unwrap_or_else(|| "config.toml".to_string());
    let scene: Scene = toml::from_str(&fs::read_to_string(&path)?)?;

    let sim = Simulator::new(scene, MicArray::uma16());
    run(sim)
}

/// Read the stream until the end. Placeholder for the processing pipeline.
fn run<S: FrameSource<M>, const M: usize>(mut source: S) -> Result<(), Box<dyn Error>> {
    let mut buf = vec![0.0; 1024 * M];
    let mut total_frames = 0;

    loop {
        let frames = source.read(&mut buf)?;
        if frames == 0 {
            break;
        }
        total_frames += frames;
    }

    println!(
        "read {total_frames} frames ({:.2} s, {M} channels)",
        total_frames as f64 / source.sample_rate()
    );
    Ok(())
}
