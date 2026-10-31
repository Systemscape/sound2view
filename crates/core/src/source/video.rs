//! Camera capture via nokhwa (V4L2 on Linux, AVFoundation on macOS).

use std::error::Error;

use nokhwa::{
    NokhwaError,
    Camera,
    pixel_format::RgbAFormat,
    query,
    utils::{
        ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
        Resolution,
    },
};

use crate::{ImageSource, VideoFrame};

/// Request camera access; required once on macOS before any other camera call (the OS asks the
/// user on the first run). Returns immediately, see [`is_authorized`]. No-op on Linux.
pub fn initialize() {
    nokhwa::nokhwa_initialize(|granted| {
        if !granted {
            eprintln!("{}", VideoError::NotAuthorized);
        }
    });
}

/// True once camera access is granted; always true on Linux
pub fn is_authorized() -> bool {
    nokhwa::nokhwa_check()
}

/// Print all cameras with their supported formats, one line per format.
///
/// On Linux the user must be in the `video` group, on macOS the terminal needs camera access,
/// see the README.
pub fn probe() -> Result<(), Box<dyn Error>> {
    if !is_authorized() {
        println!("{}", VideoError::NotAuthorized);
    }
    let cameras = query(ApiBackend::Auto)?;
    if cameras.is_empty() {
        println!("no cameras found");
    }

    for info in cameras {
        println!(
            "{} {} ({})",
            info.index(),
            info.human_name(),
            info.description()
        );

        // Formats can only be queried from an opened camera
        let format = RequestedFormat::new::<RgbAFormat>(RequestedFormatType::None);
        match Camera::new(info.index().clone(), format)
            .and_then(|mut camera| camera.compatible_camera_formats())
        {
            Ok(formats) => {
                for f in formats {
                    println!("    {f}");
                }
            }
            Err(e) => println!("    error getting formats: {e}"),
        }
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    #[error("no camera with {0:?} in its name")]
    NotFound(String),
    #[error("camera access not granted (macOS: System Settings, Privacy & Security, Camera)")]
    NotAuthorized,
    #[error("camera: {0}")]
    Camera(#[from] NokhwaError),
}

/// Live camera capture, frames converted to RGBA
pub struct CameraSource {
    camera: Camera,
}

impl CameraSource {
    /// Open camera `index` (see [`probe`]) at `width` x `height` and `fps`, and start streaming.
    ///
    /// Tries MJPEG first; if the camera does not offer it exactly (AVFoundation often lists only
    /// uncompressed formats such as NV12 or YUYV), takes the closest format. The RGBA decoder
    /// handles both. A different resolution is printed, because the field of view in the config
    /// belongs to the requested one.
    pub fn new(index: u32, width: u32, height: u32, fps: u32) -> Result<Self, VideoError> {
        if !is_authorized() {
            return Err(VideoError::NotAuthorized);
        }

        let requested = CameraFormat::new(Resolution::new(width, height), FrameFormat::MJPEG, fps);
        let open = |format: RequestedFormatType| -> Result<Camera, NokhwaError> {
            let mut camera = Camera::new(
                CameraIndex::Index(index),
                RequestedFormat::new::<RgbAFormat>(format),
            )?;
            camera.open_stream()?;
            Ok(camera)
        };

        let camera = open(RequestedFormatType::Exact(requested))
            .or_else(|_| open(RequestedFormatType::Closest(requested)))?;

        let actual = camera.camera_format();
        if actual.resolution() != requested.resolution() {
            eprintln!("camera {index}: requested {requested}, got {actual}");
        }

        Ok(Self { camera })
    }

    /// Open the first camera whose name contains `name` (case-insensitive) and that streams the
    /// requested format, see [`Self::new`].
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

/// A camera as listed by the OS
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CameraEntry {
    /// V4L2: N of `/dev/videoN`; AVFoundation: position in the device list
    pub index: u32,
    pub name: String,
}

/// All cameras, in OS order. One camera can show up several times (e.g. a V4L2 metadata node
/// that cannot stream).
pub fn list_cameras() -> Result<Vec<CameraEntry>, VideoError> {
    if !is_authorized() {
        return Err(VideoError::NotAuthorized);
    }
    query(ApiBackend::Auto)?
        .into_iter()
        .map(|info| {
            Ok(CameraEntry {
                index: info.index().as_index()?,
                name: info.human_name(),
            })
        })
        .collect()
}

impl ImageSource for CameraSource {
    type Error = VideoError;

    /// Blocks until the next frame; capture and MJPEG decode errors are returned.
    fn next_frame(&mut self) -> Result<VideoFrame, VideoError> {
        let image = self.camera.frame()?.decode_image::<RgbAFormat>()?;

        Ok(VideoFrame {
            width: image.width(),
            height: image.height(),
            pixels: image.into_raw(),
        })
    }
}
