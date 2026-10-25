//! Camera capture via nokhwa (V4L2 on Linux, AVFoundation on macOS).

use std::error::Error;

use nokhwa::{
    Camera,
    pixel_format::RgbAFormat,
    query,
    utils::{ApiBackend, CameraFormat, RequestedFormat, RequestedFormatType, Resolution},
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
    /// Open camera `index` (see [`probe`]) and start streaming.
    pub fn new(index: u32) -> Result<Self, Box<dyn Error>> {
        let format = RequestedFormatType::Exact(CameraFormat::new(
            Resolution::new(1920, 1080),
            nokhwa::utils::FrameFormat::MJPEG,
            60,
        ));

        let mut camera = Camera::new(
            nokhwa::utils::CameraIndex::Index(index),
            RequestedFormat::new::<RgbAFormat>(format),
        )?;
        camera.open_stream()?;

        Ok(Self { camera })
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
