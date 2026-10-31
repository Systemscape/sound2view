//! Camera capture: V4L2 via nokhwa on Linux, AVFoundation via objc2 on macOS.
//!
//! Both backends provide `CameraSource` (with `new` and `ImageSource`), `list_cameras`,
//! `initialize`, `is_authorized` and `probe`; the shared parts are here.

use std::time::Duration;

#[cfg(target_os = "macos")]
mod avfoundation;
#[cfg(target_os = "linux")]
mod v4l;

#[cfg(target_os = "macos")]
pub use avfoundation::{CameraSource, initialize, is_authorized, list_cameras, probe};
#[cfg(target_os = "linux")]
pub use v4l::{CameraSource, initialize, is_authorized, list_cameras, probe};

#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    #[error("no camera with {0:?} in its name")]
    NotFound(String),
    #[error("no camera with index {0}")]
    NoSuchIndex(u32),
    #[error("camera access not granted (macOS: System Settings, Privacy & Security, Camera)")]
    NotAuthorized,
    #[cfg(target_os = "linux")]
    #[error("camera: {0}")]
    Camera(#[from] nokhwa::NokhwaError),
    /// Error reported by AVFoundation, e.g. from `NSError`
    #[cfg(target_os = "macos")]
    #[error("camera: {0}")]
    Camera(String),
    #[error("no camera frame for {} s", .0.as_secs_f64())]
    NoFrame(Duration),
    #[error("camera stream closed")]
    Closed,
}

/// A camera as listed by the OS
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CameraEntry {
    /// V4L2: N of `/dev/videoN`; AVFoundation: position in the device list
    pub index: u32,
    pub name: String,
}

impl CameraSource {
    /// Open the first camera whose name contains `name` (case-insensitive) and that streams the
    /// requested format, see `CameraSource::new`.
    ///
    /// One camera can show up as several devices (e.g. a V4L2 metadata node that cannot
    /// stream), so all matches are tried in order. Unlike the index, the name survives replugging.
    /// Returns the error of the last match that failed, or `NotFound` if none matched.
    pub fn open_by_name(name: &str, width: u32, height: u32, fps: u32) -> Result<Self, VideoError> {
        let lower = name.to_lowercase();
        let mut last_error = None;

        for entry in list_cameras()? {
            if !entry.name.to_lowercase().contains(&lower) {
                continue;
            }
            match Self::new(entry.index, width, height, fps) {
                Ok(camera) => return Ok(camera),
                Err(e) => last_error = Some(e),
            }
        }

        Err(last_error.unwrap_or_else(|| VideoError::NotFound(name.to_string())))
    }
}
