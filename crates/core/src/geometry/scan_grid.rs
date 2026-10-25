use crate::{Point3, geometry::camera::Camera};

pub struct ScanGrid {
    /// Scan grid width in "pixels"
    width: usize,
    /// Scan grid height in "pixels"
    height: usize,
    /// Points of the grid stored in row-major form
    points: Vec<Point3>,
}

impl ScanGrid {
    /// Array scan grid that covers exactly the `camera`s field of view
    ///
    /// - `width` is the grid's resolution along the x-axis
    /// - `z` is the depth from the camera in meters
    pub fn new(camera: &Camera, width: u32, z: f64) -> Self {
        let scaled_camera = Camera {
            intrinsics: camera.intrinsics.scaled(width),
            mount: camera.mount.clone(),
        };
        let height = scaled_camera.intrinsics.height;

        // workaround so `move` copies the reference
        let scaled_camera_ref = &scaled_camera;
        let points = Vec::from_iter((0..height).flat_map(|row| {
            (0..width).map(move |col| scaled_camera_ref.unproject(col as f64, row as f64, z))
        }));

        Self {
            width: width as usize,
            height: height as usize,
            points,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn points(&self) -> &[Point3] {
        &self.points
    }

    pub fn index(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.height && col < self.width);
        row * self.width + col
    }
}

#[cfg(test)]
mod test {
    use crate::{geometry::scan_grid::ScanGrid, test_util::test_camera};

    #[test]
    fn points_len_w_times_h() {
        let grid = ScanGrid::new(&test_camera(), 100, 100.0);
        assert_eq!(
            grid.width() * grid.height(),
            grid.points().len(),
            "number of grid points should be width*height"
        )
    }

    #[test]
    fn all_points_same_z() {
        let z = 123.0;
        let grid = ScanGrid::new(&test_camera(), 100, z);
        assert!(
            grid.points().iter().all(|p| p.z == z),
            "all grid points should have the same z"
        )
    }

    #[test]
    /// Corner grid points, projected with the full-resolution camera, land on the
    /// full-resolution position of the grid pixel centers: (i + 0.5) / s - 0.5.
    /// Checks `scaled`, u/v order and row direction together.
    fn corners() {
        let camera = test_camera();
        let grid = ScanGrid::new(&camera, 100, 100.0);
        let (w, h) = (grid.width(), grid.height());

        // 1920 -> 100, 1080 -> round(56.25) = 56
        assert_eq!(h, 56);
        let s_x = 100.0 / 1920.0;
        let s_y = 56.0 / 1080.0;

        for (row, col) in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)] {
            let (u, v) = camera.project(&grid.points()[grid.index(row, col)]);
            let u_expected = (col as f64 + 0.5) / s_x - 0.5;
            let v_expected = (row as f64 + 0.5) / s_y - 0.5;

            assert!(
                (u - u_expected).abs() < 1e-9 && (v - v_expected).abs() < 1e-9,
                "corner ({row}, {col}): expected ({u_expected}, {v_expected}), got ({u}, {v})"
            );
        }
    }

    #[test]
    fn height() {
        // 64 * 1080 / 1920 = 36
        let grid = ScanGrid::new(&test_camera(), 64, 100.0);
        assert_eq!(grid.height(), 36);
    }

    #[test]
    fn symmetry() {
        let grid = ScanGrid::new(&test_camera(), 100, 100.0);

        let points = grid.points();

        for row in 0..grid.height() {
            for col in 0..grid.width() / 2 {
                let a = points[grid.index(row, col)].x;
                let b = points[grid.index(row, grid.width() - 1 - col)].x;
                assert!((a + b).abs() < 1e-9, "row {row}, col {col}: {a} vs {b}");
            }
        }
    }
}
