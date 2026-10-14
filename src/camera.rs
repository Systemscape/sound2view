use nalgebra::Matrix4;
use serde::{Deserialize, Serialize};

use crate::Vector3;

#[derive(Debug, Deserialize, Serialize)]
pub struct Camera {
    pub intrinsics: Intrinsics,
    pub mount: Mount,
}

/// Camera intrinsic properties
#[derive(Debug, Deserialize, Serialize)]
pub struct Intrinsics {
    /// image width in pixels
    pub width: u32,
    /// image height in pixels
    pub height: u32,
    /// focal length, x-coordinate
    pub fx: f64,
    /// focal length, y-coordinate
    pub fy: f64,
    /// principal point, x-coordinate of pixel center
    pub cx: f64,
    /// principal point, y-coordinate of pixel center
    pub cy: f64,
}

impl Intrinsics {
    /// Construct an "ideal" camera from width, height and the
    /// horizontal fov.
    ///
    /// Parameters:
    /// - `width` and `height` in pixels
    /// - `hfov` in radians
    pub fn from_hfov(width: u32, height: u32, hfov: f64) -> Self {
        Self {
            width,
            height,
            fx: (width as f64 / 2.0) / (hfov / 2.0).tan(),
            fy: (width as f64 / 2.0) / (hfov / 2.0).tan(),
            cx: (width as f64 - 1.0) / 2.0,
            cy: (height as f64 - 1.0) / 2.0,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Mount {
    /// Offset
    offset: Vector3,
}

impl Mount {
    /// Transformation matrix to compensate the mounting position
    pub fn get_transformation_matrix(&self) -> Matrix4<f64> {
        unimplemented!()
    }
}
