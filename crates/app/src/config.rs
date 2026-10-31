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
            Ok(text) => {
                let config: Self = toml::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
                config.validate().map_err(|e| format!("{path}: {e}"))?;
                Ok(config)
            }
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
    /// Part of the input device id or name, see `sound2view probe`; default per platform
    /// (`DEFAULT_AUDIO_DEVICE`)
    pub device: String,
    /// In Hz, must be supported by the device
    pub sample_rate: u32,
}

/// ALSA lists the UMA-16 several times (hw, plughw, dsnoop, ...); only `hw:` is the raw device
#[cfg(not(target_os = "macos"))]
const DEFAULT_AUDIO_DEVICE: &str = "hw:CARD=UMA16v2";
/// CoreAudio lists it once; id (device UID) and name contain the product name
#[cfg(target_os = "macos")]
const DEFAULT_AUDIO_DEVICE: &str = "UMA16";

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device: DEFAULT_AUDIO_DEVICE.to_string(),
            sample_rate: 48_000,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CameraConfig {
    /// Part of the camera name (case-insensitive), see `sound2view camprobe`
    pub name: String,
    /// Capture format, see `sound2view camprobe`; MJPEG if offered, else the closest format
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// Horizontal field of view in degrees at this resolution; ignored with `calibration`
    pub hfov_deg: f64,
    /// Checkerboard calibration at `width` x `height`; `None`: pinhole from `hfov_deg`
    pub calibration: Option<Calibration>,
}

/// Intrinsics from a checkerboard calibration (OpenCV), only valid at the resolution it was
/// taken at. Pixel-center convention as OpenCV: pixel centers at integer coordinates.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Calibration {
    /// Focal lengths in pixels
    pub fx: f64,
    pub fy: f64,
    /// Principal point in pixels
    pub cx: f64,
    pub cy: f64,
    /// Distortion coefficients (k1, k2, p1, p2, k3); default zero
    #[serde(default)]
    pub dist: [f64; 5],
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
            calibration: None,
        }
    }
}

impl CameraConfig {
    /// Calibrated intrinsics if given, else a pinhole from `hfov_deg` (centered principal
    /// point, square pixels); mounted in the array center
    pub fn camera(&self) -> Camera {
        let intrinsics = match &self.calibration {
            Some(c) => Intrinsics {
                width: self.width,
                height: self.height,
                fx: c.fx,
                fy: c.fy,
                cx: c.cx,
                cy: c.cy,
                dist: c.dist,
            },
            None => Intrinsics::from_hfov(self.width, self.height, self.hfov_deg.to_radians()),
        };
        Camera {
            intrinsics,
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
    CleanSc,
}

impl Method {
    pub const ALL: [Method; 3] = [Method::Das, Method::Functional, Method::CleanSc];

    pub fn label(self) -> &'static str {
        match self {
            Method::Das => "Delay and sum",
            Method::Functional => "Functional",
            Method::CleanSc => "CLEAN-SC",
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
    /// CLEAN-SC loop gain in (0, 1]
    pub clean_gain: f64,
    /// CLEAN-SC: maximum number of removed sources per frequency bin
    pub clean_max_iter: usize,
    /// CLEAN-SC: stop when the strongest remaining source is this many dB below the first
    pub clean_stop_db: f64,
    /// CLEAN-SC: standard deviation of the Gaussian clean beam in grid cells (0: single points)
    pub clean_beam_cells: f64,
}

impl Default for BeamformerConfig {
    fn default() -> Self {
        Self {
            method: Method::Functional,
            nu: 20.0,
            clean_gain: 0.9,
            clean_max_iter: 10,
            clean_stop_db: 20.0,
            clean_beam_cells: 1.5,
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

impl Config {
    /// Error for values the pipeline or GUI cannot use. Pipeline values that depend on the
    /// sample rate are checked again by `Pipeline::new`.
    pub fn validate(&self) -> Result<(), String> {
        let b = &self.beamformer;
        if b.nu < 1.0 {
            return Err("beamformer.nu must be at least 1".to_string());
        }
        if !(b.clean_gain > 0.0 && b.clean_gain <= 1.0) {
            return Err("beamformer.clean_gain must be in (0, 1]".to_string());
        }
        if b.clean_max_iter == 0 {
            return Err("beamformer.clean_max_iter must be at least 1".to_string());
        }
        if b.clean_stop_db < 0.0 || b.clean_beam_cells < 0.0 {
            return Err("beamformer.clean_stop_db and clean_beam_cells must be >= 0".to_string());
        }
        if let Some(c) = &self.camera.calibration {
            if !(c.fx > 0.0 && c.fy > 0.0) {
                return Err("camera.calibration: fx and fy must be > 0".to_string());
            }
            let (w, h) = (self.camera.width as f64, self.camera.height as f64);
            if !((0.0..w).contains(&c.cx) && (0.0..h).contains(&c.cy)) {
                return Err(format!(
                    "camera.calibration: principal point ({}, {}) outside the {w}x{h} image",
                    c.cx, c.cy
                ));
            }
            if !c.dist.iter().all(|k| k.is_finite()) {
                return Err("camera.calibration: dist must be finite".to_string());
            }
        }
        if self.display.db_range <= 0.0 {
            return Err("display.db_range must be > 0".to_string());
        }
        self.pipeline
            .validate(self.audio.sample_rate as f64)
            .map_err(|e| e.to_string())
    }
}
