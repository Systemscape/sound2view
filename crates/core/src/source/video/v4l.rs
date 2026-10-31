//! Linux backend: V4L2 via nokhwa.

use std::error::Error;

use nokhwa::{
    Camera, NokhwaError,
    pixel_format::RgbAFormat,
    query,
    utils::{
        ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
        Resolution,
    },
};

use super::{CameraEntry, VideoError};
use crate::{ImageSource, VideoFrame};

/// No-op on Linux; access is controlled by the `video` group, see the README.
pub fn initialize() {}

/// Always true on Linux; missing `video` group membership shows up as an open error
pub fn is_authorized() -> bool {
    true
}

/// Print all cameras with their supported formats, one line per format.
///
/// The user must be in the `video` group, see the README.
pub fn probe() -> Result<(), Box<dyn Error>> {
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

/// All cameras, in OS order. One camera can show up several times (e.g. a V4L2 metadata node
/// that cannot stream).
pub fn list_cameras() -> Result<Vec<CameraEntry>, VideoError> {
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

/// Live camera capture, frames converted to RGBA
pub struct CameraSource {
    camera: Camera,
}

impl CameraSource {
    /// Open camera `index` (`/dev/videoN`, see [`probe`]) at `width` x `height` and `fps`, and
    /// start streaming.
    ///
    /// Tries MJPEG first; if the camera does not offer it exactly, takes the closest format. The
    /// RGBA decoder handles MJPEG and uncompressed formats. A different resolution is printed,
    /// because the field of view in the config belongs to the requested one.
    pub fn new(index: u32, width: u32, height: u32, fps: u32) -> Result<Self, VideoError> {
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
