use std::{cmp, f64::consts::PI};

use nalgebra::{Isometry3, Translation3, UnitQuaternion};
use serde::{Deserialize, Serialize};

use crate::{Point3, Vector3};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Camera {
    pub intrinsics: Intrinsics,
    pub mount: Mount,
}

impl Camera {
    /// Project a pixel center onto the array frame
    ///
    /// Applies the mount pose (rotation and offset)
    ///
    /// - `z`: focus distance from the camera along the optical axis, must be >0
    ///
    /// Panics if `z <= 0`
    pub fn unproject(&self, u: f64, v: f64, z: f64) -> Point3 {
        self.mount
            .pose()
            .transform_point(&self.intrinsics.unproject(u, v, z))
    }

    /// Project position `p` onto the image plane
    ///
    /// Panics if `p` is not in front of the camera
    pub fn project(&self, p: &Point3) -> (f64, f64) {
        self.intrinsics
            .project(&self.mount.pose().inverse_transform_point(p))
    }
}

/// Camera intrinsic properties
#[derive(Clone, Debug, Deserialize, Serialize)]
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

    /// Compute a scaled version of [`Self`] for a given target width
    pub fn scaled(&self, w_target: u32) -> Self {
        let h_target = (w_target as f64 * (self.height as f64 / self.width as f64)).round() as u32;
        // Guard against height becoming zero
        let h_target = cmp::max(1, h_target);

        let s_x = w_target as f64 / self.width as f64;
        let s_y = h_target as f64 / self.height as f64;

        Self {
            width: w_target,
            height: h_target,
            fx: self.fx * s_x,
            fy: self.fy * s_y,
            cx: (self.cx + 0.5) * s_x - 0.5,
            cy: (self.cy + 0.5) * s_y - 0.5,
        }
    }

    /// Project a pixel center onto the camera frame
    ///
    /// - `z`: focus distance along the optical axis, must be >0
    ///
    /// Panics if `z <= 0`
    pub fn unproject(&self, u: f64, v: f64, z: f64) -> Point3 {
        assert!(z > 0.0, "z must be > 0");
        Point3::new((u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z)
    }

    /// Project position `p` onto the image plane, returns (u,v)
    ///
    /// Panics if `p`'s z-coordinate is <=0
    pub fn project(&self, p: &Point3) -> (f64, f64) {
        assert!(p.z > 0.0, "p.z must be > 0");
        let u = self.cx + self.fx * p.x / p.z;
        let v = self.cy + self.fy * p.y / p.z;

        (u, v)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Mount {
    /// Camera position in the array frame, in meters
    pub offset: Vector3,
    /// Rotation, normally PI about z
    pub rotation: UnitQuaternion<f64>,
}

impl Default for Mount {
    fn default() -> Self {
        Self::from_offset(Vector3::zeros())
    }
}

impl Mount {
    pub fn from_offset(offset: Vector3) -> Self {
        Self {
            offset,
            rotation: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), PI),
        }
    }

    /// Camera pose in the array frame: p_array = R * p_cam + t
    pub fn pose(&self) -> Isometry3<f64> {
        Isometry3::from_parts(Translation3::from(self.offset), self.rotation)
    }
}

#[cfg(test)]
mod test {

    use crate::{
        Point3, Vector3,
        geometry::camera::{Camera, Mount},
        test_util::test_camera,
    };

    #[test]
    fn roundtrip_project_unproject() {
        let camera = test_camera();

        let pos = Point3::new(0.1, 0.2, 3.0);
        let (u, v) = camera.project(&pos);
        let unprojected = camera.unproject(u, v, pos.z);

        assert!(
            (unprojected - pos).norm() < 1e-12,
            "expected {pos}, got {unprojected}"
        );
    }

    #[test]
    fn image_center_on_axis() {
        let camera = test_camera();
        let z = 3.0;

        let image_center = camera.unproject(camera.intrinsics.cx, camera.intrinsics.cy, z);
        assert!(
            image_center.x.abs() < 1e-12,
            "center should project to 0, got x={}",
            image_center.x
        );
        assert!(
            image_center.y.abs() < 1e-12,
            "center should project to 0, got y={}",
            image_center.y
        );
        assert_eq!(image_center.z, z);
    }

    /// Pins the 180 deg rotation: image top-left is +x, +y in the array frame.
    #[test]
    fn top_left_is_positive_x_y() {
        let camera = test_camera();
        let z = 3.0;

        let top_left_pixel = camera.unproject(0.0, 0.0, z);
        assert!(
            top_left_pixel.x > 0.0,
            "top left pixel should be x>0, got x={}",
            top_left_pixel.x
        );
        assert!(
            top_left_pixel.y > 0.0,
            "top left pixel should be y>0, got y={}",
            top_left_pixel.y
        );
    }

    /// Camera offset from the array center: z is the depth from the camera,
    /// and the optical axis passes through `offset`.
    #[test]
    fn offset_mount() {
        let offset = Vector3::new(0.02, -0.01, 0.005);
        let camera = Camera {
            mount: Mount::from_offset(offset),
            ..test_camera()
        };
        let z = 3.0;

        // Round trip; the rotation about z keeps depth, so camera depth = pos.z - offset.z
        let pos = Point3::new(0.1, 0.2, z);
        let (u, v) = camera.project(&pos);
        let unprojected = camera.unproject(u, v, pos.z - offset.z);
        assert!(
            (unprojected - pos).norm() < 1e-12,
            "expected {pos}, got {unprojected}"
        );

        let image_center = camera.unproject(camera.intrinsics.cx, camera.intrinsics.cy, z);
        let expected = Point3::from(offset + Vector3::new(0.0, 0.0, z));
        assert!(
            (image_center - expected).norm() < 1e-12,
            "image center: expected {expected}, got {image_center}"
        );
    }
}
