//! Camera capture via nokhwa (V4L2 on Linux, AVFoundation on macOS).

use std::error::Error;

use nokhwa::{
    Camera,
    pixel_format::RgbAFormat,
    query,
    utils::{
        ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
        Resolution,
    },
};

use crate::{ImageSource, VideoFrame};

/// Print all cameras with their supported formats, one line per format.
///
/// On Linux the user must be in the `video` group, see the README.
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

/// Live camera capture, frames converted to RGBA
pub struct CameraSource {
    camera: Camera,
}

impl CameraSource {
    /// Open camera `index` (see [`probe`]) with MJPEG at `width` x `height` and `fps`, and start
    /// streaming.
    pub fn new(index: u32, width: u32, height: u32, fps: u32) -> Result<Self, Box<dyn Error>> {
        let format = RequestedFormatType::Exact(CameraFormat::new(
            Resolution::new(width, height),
            FrameFormat::MJPEG,
            fps,
        ));

        let mut camera = Camera::new(
            CameraIndex::Index(index),
            RequestedFormat::new::<RgbAFormat>(format),
        )?;
        camera.open_stream()?;

        Ok(Self { camera })
    }

    /// Open the first camera whose name contains `name` (case-insensitive) and that streams the
    /// requested format, see [`Self::new`].
    ///
    /// One camera can show up as several devices (e.g. a V4L2 metadata node that cannot
    /// stream), so all matches are tried in order. Unlike the index, the name survives replugging.
    pub fn open_by_name(
        name: &str,
        width: u32,
        height: u32,
        fps: u32,
    ) -> Result<Self, Box<dyn Error>> {
        let name = name.to_lowercase();
        let mut last_error = None;

        for info in query(ApiBackend::Auto)? {
            if !info.human_name().to_lowercase().contains(&name) {
                continue;
            }
            match Self::new(info.index().as_index()?, width, height, fps) {
                Ok(camera) => return Ok(camera),
                Err(e) => last_error = Some(e),
            }
        }

        Err(last_error.unwrap_or_else(|| format!("no camera with {name:?} in its name").into()))
    }
}

impl ImageSource for CameraSource {
    fn next_frame(&mut self) -> Option<VideoFrame> {
        let frame = match self.camera.frame() {
            Ok(frame) => frame,
            Err(e) => {
                eprintln!("camera frame: {e}");
                return None;
            }
        };

        let image = frame.decode_image::<RgbAFormat>().ok()?;

        Some(VideoFrame {
            width: image.width(),
            height: image.height(),
            pixels: image.into_raw(),
        })
    }
}
