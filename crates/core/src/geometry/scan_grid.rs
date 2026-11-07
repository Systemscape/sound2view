use crate::{Point3, geometry::camera::Camera};

/// Scan grid cells farther off the camera axis than this are never added as margin: the pinhole
/// plane grows as tan(angle), and the array has no useful resolution at grazing angles
pub const MAX_OFF_AXIS_DEG: f64 = 55.0;

/// A rectangle of scan grid cells: the part of the grid that the camera sees
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Viewport {
    /// First visible column and row in the full grid
    pub col: usize,
    pub row: usize,
    /// Visible size in cells
    pub width: usize,
    pub height: usize,
}

impl Viewport {
    /// The whole grid
    pub fn full(width: usize, height: usize) -> Self {
        Self {
            col: 0,
            row: 0,
            width,
            height,
        }
    }

    /// Whether (col, row) in full grid cells (sub-cell positions allowed) lies within the visible
    /// cell centers
    pub fn contains(&self, col: f64, row: f64) -> bool {
        let (c, r) = (col - self.col as f64, row - self.row as f64);
        0.0 <= c && c <= (self.width - 1) as f64 && 0.0 <= r && r <= (self.height - 1) as f64
    }

    /// Copy the visible part of `full` (row-major, `full_width` wide) into `out`
    /// (`width * height`)
    pub fn crop(&self, full: &[f64], full_width: usize, out: &mut [f64]) {
        assert_eq!(out.len(), self.width * self.height);
        for (row, out) in out.chunks_exact_mut(self.width).enumerate() {
            let start = (self.row + row) * full_width + self.col;
            out.copy_from_slice(&full[start..start + self.width]);
        }
    }
}

pub struct ScanGrid {
    /// Scan grid width in "pixels", including the margin
    width: usize,
    /// Scan grid height in "pixels", including the margin
    height: usize,
    /// Points of the grid stored in row-major form
    points: Vec<Point3>,
    /// The cells inside the camera image
    viewport: Viewport,
}

impl ScanGrid {
    /// Array scan grid that covers exactly the `camera`s field of view
    ///
    /// - `width` is the grid's resolution along the x-axis
    /// - `z` is the depth from the camera in meters
    pub fn new(camera: &Camera, width: u32, z: f64) -> Self {
        Self::with_margin(camera, width, z, 0.0)
    }

    /// Like [`new`](Self::new), extended on every side by `margin` radians beyond the field of
    /// view (as far as [`MAX_OFF_AXIS_DEG`] allows), at the same cell size.
    ///
    /// A source outside the image then has its maximum at its own direction instead of on the
    /// image border, and a deconvolving beamformer (CLEAN-SC) takes its contribution out of
    /// the visible part. The visible cells are [`viewport`](Self::viewport).
    pub fn with_margin(camera: &Camera, width: u32, z: f64, margin: f64) -> Self {
        let scaled_camera = Camera {
            intrinsics: camera.intrinsics.scaled(width),
            mount: camera.mount.clone(),
        };
        let intrinsics = &scaled_camera.intrinsics;
        let height = intrinsics.height;
        let margin_cols = margin_cells(intrinsics.fx, intrinsics.cx, width, margin);
        let margin_rows = margin_cells(intrinsics.fy, intrinsics.cy, height, margin);

        let cols = -(margin_cols as i64)..(width as i64 + margin_cols as i64);
        let rows = -(margin_rows as i64)..(height as i64 + margin_rows as i64);
        // workaround so `move` copies the reference
        let scaled_camera_ref = &scaled_camera;
        let points = Vec::from_iter(rows.clone().flat_map(|row| {
            cols.clone()
                .map(move |col| scaled_camera_ref.unproject(col as f64, row as f64, z))
        }));

        Self {
            width: width as usize + 2 * margin_cols,
            height: height as usize + 2 * margin_rows,
            points,
            viewport: Viewport {
                col: margin_cols,
                row: margin_rows,
                width: width as usize,
                height: height as usize,
            },
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

    /// The cells inside the camera image (all of them without margin)
    pub fn viewport(&self) -> Viewport {
        self.viewport
    }

    pub fn index(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.height && col < self.width);
        row * self.width + col
    }
}

/// Cells to add beyond the outermost cell center of an axis with focal length `f` and principal
/// point `c` (grid pixels, `n` cells) so that the grid reaches `margin` radians further, but
/// not beyond `MAX_OFF_AXIS_DEG`
fn margin_cells(f: f64, c: f64, n: u32, margin: f64) -> usize {
    let edge = ((n as f64 - 1.0 - c) / f).atan();
    let outer = (edge + margin).min(MAX_OFF_AXIS_DEG.to_radians()).max(edge);
    (f * (outer.tan() - edge.tan())).round() as usize
}

#[cfg(test)]
mod test {
    use crate::{
        geometry::scan_grid::{MAX_OFF_AXIS_DEG, ScanGrid, Viewport},
        test_util::test_camera,
    };

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

    /// With a margin the visible cells are the plain grid, shifted by the margin; the margin
    /// cells continue the pixel raster beyond the image.
    #[test]
    fn margin_keeps_visible_cells() {
        let camera = test_camera();
        let plain = ScanGrid::new(&camera, 64, 2.0);
        let grid = ScanGrid::with_margin(&camera, 64, 2.0, 15f64.to_radians());
        let vp = grid.viewport();

        assert_eq!((vp.width, vp.height), (plain.width(), plain.height()));
        assert!(vp.row > 0, "a 15 degree margin adds rows: {vp:?}");
        assert_eq!(grid.width(), vp.width + 2 * vp.col);
        assert_eq!(grid.height(), vp.height + 2 * vp.row);
        assert_eq!(grid.points().len(), grid.width() * grid.height());
        for row in 0..vp.height {
            for col in 0..vp.width {
                let a = plain.points()[plain.index(row, col)];
                let b = grid.points()[grid.index(row + vp.row, col + vp.col)];
                assert!((a - b).norm() < 1e-9, "({row}, {col}): {a} vs {b}");
            }
        }
        // The cell above the image projects one grid pixel above it
        let (_, v) = camera.project(&grid.points()[grid.index(vp.row - 1, vp.col)]);
        let s_y = plain.height() as f64 / camera.intrinsics.height as f64;
        assert!((v - (-0.5 / s_y - 0.5)).abs() < 1e-9, "v = {v}");
    }

    /// The margin ends at `MAX_OFF_AXIS_DEG`: the 110 degree test camera already sees almost
    /// that far horizontally (its outermost cell center is at 54.6 degrees), so at most one
    /// column is added, and a huge margin adds the same rows as the cap.
    #[test]
    fn margin_capped() {
        let camera = test_camera();
        let grid = ScanGrid::with_margin(&camera, 64, 2.0, 15f64.to_radians());
        assert!(grid.viewport().col <= 1, "{:?}", grid.viewport());

        let capped = ScanGrid::with_margin(&camera, 64, 2.0, 80f64.to_radians());
        let vp = capped.viewport();
        let top = capped.points()[capped.index(0, vp.col + vp.width / 2)];
        let angle = (top.y / top.z).abs().atan().to_degrees();
        assert!(
            (angle - MAX_OFF_AXIS_DEG).abs() < 1.5,
            "top row at {angle} degrees"
        );
        assert_eq!(
            ScanGrid::new(&camera, 64, 2.0).viewport(),
            Viewport::full(64, 36)
        );
    }

    #[test]
    fn viewport_crop_and_contains() {
        let vp = Viewport {
            col: 1,
            row: 2,
            width: 2,
            height: 2,
        };
        let full: Vec<f64> = (0..4 * 5).map(|i| i as f64).collect();
        let mut out = vec![0.0; 4];
        vp.crop(&full, 4, &mut out);
        assert_eq!(out, [9.0, 10.0, 13.0, 14.0]);
        assert!(vp.contains(1.0, 2.0) && vp.contains(2.0, 3.0));
        assert!(!vp.contains(0.9, 2.0) && !vp.contains(2.0, 3.1));
    }
}
