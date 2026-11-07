//! Power map to image: dB scaling, colormap and markers.

use image::{
    Rgb, RgbImage,
    imageops::{self, FilterType},
};

use crate::{Point3, pipeline::MapLayout, tracking::Track};

/// Marker half length in camera pixels
const MARKER_HALF_LEN: i64 = 10;
/// Radius of the dot per confirmed track in camera pixels
const PEAK_RADIUS: i64 = 8;
/// Dot colors: track, and track classified as speech
const TRACK_RGB: Rgb<u8> = Rgb([40, 120, 255]);
const SPEECH_RGB: Rgb<u8> = Rgb([63, 185, 80]);

/// Power map as image at camera resolution, with a red cross per true source and a dot per
/// confirmed track: blue, or green if its id is in `speech`.
///
/// - `power`: row-major like the scan grid, `grid_width * grid_height` values
/// - `src_pos`: true source positions, array frame
/// - `tracks`: estimated sources in grid cells, see [`crate::tracking::Tracker`]
/// - `speech`: ids of the tracks classified as speech
pub fn map_image(
    power: &[f64],
    layout: &MapLayout,
    src_pos: &[Point3],
    tracks: &[Track],
    speech: &[u64],
) -> RgbImage {
    // One grid pixel per map value; x = col, y = row, same as `power[row * w + col]`
    let w = layout.grid_width;
    let p_max = power.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let img = RgbImage::from_fn(w as u32, layout.grid_height as u32, |x, y| {
        power_to_rgb(power[y as usize * w + x as usize], p_max)
    });

    // Upscale to camera resolution; Nearest keeps the grid pixels visible
    let intrinsics = &layout.camera.intrinsics;
    let mut big = imageops::resize(
        &img,
        intrinsics.width,
        intrinsics.height,
        FilterType::Nearest,
    );

    // Grid cell centers to camera pixel centers, as `Intrinsics::scaled`
    let s_x = intrinsics.width as f64 / layout.grid_width as f64;
    let s_y = intrinsics.height as f64 / layout.grid_height as f64;
    for track in tracks.iter().filter(|track| track.confirmed) {
        let u = ((track.col + 0.5) * s_x - 0.5).round() as i64;
        let v = ((track.row + 0.5) * s_y - 0.5).round() as i64;
        let color = if speech.contains(&track.id) {
            SPEECH_RGB
        } else {
            TRACK_RGB
        };
        for dy in -PEAK_RADIUS..=PEAK_RADIUS {
            for dx in -PEAK_RADIUS..=PEAK_RADIUS {
                let (x, y) = (u + dx, v + dy);
                let inside =
                    (0..big.width() as i64).contains(&x) && (0..big.height() as i64).contains(&y);
                if dx * dx + dy * dy <= PEAK_RADIUS * PEAK_RADIUS && inside {
                    big.put_pixel(x as u32, y as u32, color);
                }
            }
        }
    }

    for pos in src_pos {
        // `project` panics behind the camera
        if pos.z <= 0.0 {
            continue;
        }
        let (u, v) = layout.camera.project(pos);
        let (u, v) = (u.round() as i64, v.round() as i64);

        for d in -MARKER_HALF_LEN..=MARKER_HALF_LEN {
            for (x, y) in [(u + d, v), (u, v + d)] {
                // Skips pixels outside the image, including sources outside the FoV
                if (0..big.width() as i64).contains(&x) && (0..big.height() as i64).contains(&y) {
                    big.put_pixel(x as u32, y as u32, Rgb([255, 0, 0]));
                }
            }
        }
    }

    big
}

/// Largest overlay alpha, at 0 dB; the camera image stays visible underneath
const OVERLAY_ALPHA_MAX: f64 = 200.0;

/// Power map as RGBA overlay at grid resolution, row-major like the scan grid, unmultiplied alpha.
///
/// Normalized to the map maximum. Fully transparent at `db_range` dB below the maximum and lower;
/// from there yellow to red and increasingly opaque up to 0 dB.
pub fn map_rgba(power: &[f64], db_range: f64) -> Vec<u8> {
    let p_max = power.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    power
        .iter()
        .flat_map(|&p| {
            let db = 10.0 * (p / p_max).max(1e-10).log10();
            // -db_range dB -> 0, 0 dB -> 1
            let t = ((db + db_range) / db_range).clamp(0.0, 1.0);
            let green = (255.0 * (1.0 - t)) as u8;
            let alpha = (OVERLAY_ALPHA_MAX * t) as u8;
            [255, green, 0, alpha]
        })
        .collect()
}

/// Map power to gray: 0 dB (the maximum) is white, -15 dB and below is black.
fn power_to_rgb(p: f64, p_max: f64) -> Rgb<u8> {
    let db = 10.0 * (p / p_max).max(1e-10).log10();
    // -15 dB -> 0, 0 dB -> 1
    let t = ((db + 15.0) / 15.0).clamp(0.0, 1.0);
    let g = (t * 255.0) as u8;
    Rgb([g, g, g])
}
