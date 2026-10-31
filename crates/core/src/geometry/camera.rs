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
    /// Distortion coefficients in OpenCV order (k1, k2, p1, p2, k3), acting on normalized
    /// coordinates; zero = plain pinhole
    #[serde(default)]
    pub dist: [f64; 5],
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
            dist: [0.0; 5],
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
            // Acts on normalized coordinates, independent of the resolution
            dist: self.dist,
        }
    }

    /// Project a pixel center onto the camera frame, removing lens distortion
    ///
    /// - `z`: focus distance along the optical axis, must be >0
    ///
    /// Panics if `z <= 0`
    pub fn unproject(&self, u: f64, v: f64, z: f64) -> Point3 {
        assert!(z > 0.0, "z must be > 0");
        let (x, y) = self.undistort((u - self.cx) / self.fx, (v - self.cy) / self.fy);
        Point3::new(x * z, y * z, z)
    }

    /// Project position `p` onto the image plane, including lens distortion, returns (u,v)
    ///
    /// Panics if `p`'s z-coordinate is <=0
    pub fn project(&self, p: &Point3) -> (f64, f64) {
        assert!(p.z > 0.0, "p.z must be > 0");
        let (x, y) = self.distort(p.x / p.z, p.y / p.z);
        let u = self.cx + self.fx * x;
        let v = self.cy + self.fy * y;

        (u, v)
    }

    /// Brown-Conrady distortion (OpenCV's 5-coefficient model) of normalized coordinates
    /// x = X/Z, y = Y/Z
    fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let radial = self.radial(x, y);
        let (dx, dy) = self.tangential(x, y);
        (x * radial + dx, y * radial + dy)
    }

    /// Inverse of [`Self::distort`] by fixed-point iteration, as OpenCV's `undistortPoints`.
    ///
    /// Converges linearly; for the calibrated camera the error shrinks by ~0.4 per step in the
    /// worst image corner, so `UNDISTORT_ITER` steps reach machine precision. Runs once per scan
    /// grid point, so the cost does not matter.
    fn undistort(&self, x_d: f64, y_d: f64) -> (f64, f64) {
        let (mut x, mut y) = (x_d, y_d);
        for _ in 0..UNDISTORT_ITER {
            let radial = self.radial(x, y);
            let (dx, dy) = self.tangential(x, y);
            x = (x_d - dx) / radial;
            y = (y_d - dy) / radial;
        }
        (x, y)
    }

    /// Radial factor 1 + k1 r^2 + k2 r^4 + k3 r^6
    fn radial(&self, x: f64, y: f64) -> f64 {
        let [k1, k2, _, _, k3] = self.dist;
        let r2 = x * x + y * y;
        1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    }

    /// Tangential offset (decentered lens)
    fn tangential(&self, x: f64, y: f64) -> (f64, f64) {
        let [_, _, p1, p2, _] = self.dist;
        let r2 = x * x + y * y;
        (
            2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
            p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
        )
    }
}

/// Fixed-point iterations in [`Intrinsics::undistort`]
const UNDISTORT_ITER: usize = 40;

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
        geometry::camera::{Camera, Intrinsics, Mount},
        test_util::test_camera,
    };

    /// Calibrated icspring camera (OpenCV, 1920x1080)
    fn calibrated_intrinsics() -> Intrinsics {
        Intrinsics {
            width: 1920,
            height: 1080,
            fx: 1046.19,
            fy: 1046.79,
            cx: 986.79,
            cy: 608.80,
            dist: [-0.0382, 0.0720, -0.0000794, 0.000768, -0.0516],
        }
    }

    /// Reference values from OpenCV `projectPoints` with the same intrinsics
    #[test]
    fn project_matches_opencv() {
        let intrinsics = calibrated_intrinsics();
        let cases = [
            (
                Point3::new(0.5, 0.3, 1.0),
                (1507.033982435592, 920.933146584083),
            ),
            (
                Point3::new(-0.9, -0.5, 1.0),
                (67.116889765618, 97.015941084352),
            ),
            (
                Point3::new(0.2, -0.4, 2.0),
                (1091.28690021129, 399.76257734392),
            ),
        ];
        for (p, (u_ref, v_ref)) in cases {
            let (u, v) = intrinsics.project(&p);
            assert!(
                (u - u_ref).abs() < 1e-6 && (v - v_ref).abs() < 1e-6,
                "{p}: expected ({u_ref}, {v_ref}), got ({u}, {v})"
            );
        }
    }

    /// Undistortion inverts distortion over the whole image, corners included
    #[test]
    fn distorted_roundtrip_unproject_project() {
        let intrinsics = calibrated_intrinsics();
        let z = 2.0;
        let (w, h) = (
            intrinsics.width as f64 - 1.0,
            intrinsics.height as f64 - 1.0,
        );

        let pixels = [
            (0.0, 0.0),
            (w, 0.0),
            (0.0, h),
            (w, h),
            (w / 2.0, h / 2.0),
            (100.0, 700.0),
        ];
        for (u, v) in pixels {
            let p = intrinsics.unproject(u, v, z);
            assert_eq!(p.z, z);
            let (u_back, v_back) = intrinsics.project(&p);
            assert!(
                (u_back - u).abs() < 1e-9 && (v_back - v).abs() < 1e-9,
                "({u}, {v}) came back as ({u_back}, {v_back})"
            );
        }
    }

    /// Distortion keeps the principal point on the optical axis
    #[test]
    fn distorted_principal_point_on_axis() {
        let intrinsics = calibrated_intrinsics();
        let p = intrinsics.unproject(intrinsics.cx, intrinsics.cy, 3.0);
        assert!(p.x.abs() < 1e-12 && p.y.abs() < 1e-12, "got {p}");
    }

    /// Scaling keeps the distortion: the same ray hits the same relative image position
    #[test]
    fn scaled_keeps_distortion() {
        let full = calibrated_intrinsics();
        let small = full.scaled(64);
        let p = Point3::new(-0.7, 0.4, 1.0);

        let (u, v) = full.project(&p);
        let (u_s, v_s) = small.project(&p);
        // Pixel-center convention: (u + 0.5) scales with the resolution
        let s_x = small.width as f64 / full.width as f64;
        let s_y = small.height as f64 / full.height as f64;
        assert!(((u + 0.5) * s_x - 0.5 - u_s).abs() < 1e-9);
        assert!(((v + 0.5) * s_y - 0.5 - v_s).abs() < 1e-9);
    }

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
