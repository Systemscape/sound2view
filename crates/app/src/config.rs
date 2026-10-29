//! Run configuration from `sound2view.toml`. Every field has a default, so the file, each
//! section and each entry are optional; unknown entries are an error (catches typos).

use std::{error::Error, fs, io};

use serde::Deserialize;
use sound2view_core::{
    geometry::camera::{Camera, Intrinsics, Mount},
    pipeline::PipelineConfig,
};

/// Config file, looked up in the working directory
pub const CONFIG_PATH: &str = "sound2view.toml";

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub audio: AudioConfig,
    pub camera: CameraConfig,
    pub pipeline: PipelineConfig,
    pub beamformer: BeamformerConfig,
    pub display: DisplayConfig,
}

impl Config {
    /// Read `path`; a missing file gives the defaults.
    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        match fs::read_to_string(path) {
            Ok(text) => Ok(toml::from_str(&text)?),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                println!("{path} not found, using defaults");
                Ok(Self::default())
            }
            Err(e) => Err(format!("{path}: {e}").into()),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AudioConfig {
    /// Input device id pattern, see `sound2view probe`
    pub device: String,
    /// In Hz, must be supported by the device
    pub sample_rate: u32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device: "hw:CARD=UMA16v2".to_string(),
            sample_rate: 48_000,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CameraConfig {
    /// Part of the camera name (case-insensitive), see `sound2view camprobe`
    pub name: String,
    /// Capture format (MJPEG), must be listed by `sound2view camprobe`
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// Horizontal field of view in degrees at this resolution
    pub hfov_deg: f64,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            name: "icspring".to_string(),
            width: 1920,
            height: 1080,
            fps: 60,
            // Measured on a wall, see CLAUDE.md
            hfov_deg: 83.0,
        }
    }
}

impl CameraConfig {
    /// Pinhole model: centered principal point, square pixels, mounted in the array center
    pub fn camera(&self) -> Camera {
        Camera {
            intrinsics: Intrinsics::from_hfov(self.width, self.height, self.hfov_deg.to_radians()),
            mount: Mount::default(),
        }
    }
}

/// Beamforming method
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    /// Delay and sum with diagonal removal and true-location normalization
    Das,
    Functional,
}

impl Method {
    pub const ALL: [Method; 2] = [Method::Das, Method::Functional];

    pub fn label(self) -> &'static str {
        match self {
            Method::Das => "Delay and sum",
            Method::Functional => "Functional",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BeamformerConfig {
    pub method: Method,
    /// Exponent nu of functional beamforming; larger is sharper but more sensitive to model
    /// errors
    pub nu: f64,
}

impl Default for BeamformerConfig {
    fn default() -> Self {
        Self {
            method: Method::Functional,
            nu: 20.0,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DisplayConfig {
    /// Overlay range in dB below the map maximum; lower values are transparent
    pub db_range: f64,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self { db_range: 3.0 }
    }
}
