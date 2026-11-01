//! Source positions estimated from a power map: local maxima with sub-cell refinement.

use serde::Deserialize;

/// Peak search parameters, e.g. from a config file
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PeakConfig {
    /// Show peaks at all
    pub enabled: bool,
    /// Only local maxima at most this many dB below the map maximum
    pub range_db: f64,
    /// Minimum distance between two peaks in grid cells; the weaker one is dropped
    pub min_dist_cells: f64,
    /// At most this many peaks, strongest first
    pub max_count: usize,
    /// A track is shown (and recorded) only if the power arriving coherently from its direction
    /// (see [`crate::dsp::beamformer::coherent_and_received_power`]) is at least this many dB
    /// above the noise floor: the most coherent power from any direction while nothing is
    /// there, learned from the quietest map of the last minute (see
    /// [`crate::pipeline::FLOOR_WINDOW_S`]). A steady background source (fan) is the floor and
    /// gets no dot. The pipeline measures it (`Track::source_db`); the app applies it when
    /// showing tracks, so that it can change without a restart.
    pub min_above_floor_db: f64,
}

impl Default for PeakConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            range_db: 6.0,
            min_dist_cells: 4.0,
            max_count: 5,
            min_above_floor_db: 8.0,
        }
    }
}

/// Local maximum of a power map
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peak {
    /// Column and row in grid cells, refined below cell size; cell centers at integer values
    pub col: f64,
    pub row: f64,
    /// Level of the peak cell relative to the map maximum, in dB (0 for the strongest)
    pub level_db: f64,
}

/// Local maxima of `power` (row-major, `width * height`), strongest first.
///
/// A cell is a local maximum if no 8-neighbor is larger (on a plateau only the first cell in
/// row-major order counts). Maxima more than `range_db` below the map maximum are dropped, then
/// maxima closer than `min_dist_cells` to a stronger kept one, then all beyond `max_count`.
/// Positions are refined with a parabola through the log power of the cell and its two
/// neighbors, separately per axis (exact for a Gaussian peak). Empty if the maximum is <= 0.
pub fn find_peaks(power: &[f64], [width, height]: [usize; 2], config: &PeakConfig) -> Vec<Peak> {
    assert_eq!(
        power.len(),
        width * height,
        "power must have width * height values"
    );

    let p_max = power.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if p_max <= 0.0 || config.max_count == 0 {
        return Vec::new();
    }
    let threshold = p_max * 10f64.powf(-config.range_db / 10.0);
    let at = |row: usize, col: usize| power[row * width + col];

    let mut candidates: Vec<(usize, usize)> = (0..height)
        .flat_map(|row| (0..width).map(move |col| (row, col)))
        .filter(|&(row, col)| {
            let p = at(row, col);
            p >= threshold && is_local_max(row, col, p, [width, height], &at)
        })
        .collect();
    candidates.sort_by(|&(r1, c1), &(r2, c2)| at(r2, c2).total_cmp(&at(r1, c1)));

    let mut peaks: Vec<Peak> = Vec::new();
    for (row, col) in candidates {
        if peaks.len() == config.max_count {
            break;
        }
        let too_close = peaks.iter().any(|peak| {
            let (dr, dc) = (peak.row - row as f64, peak.col - col as f64);
            (dr * dr + dc * dc).sqrt() < config.min_dist_cells
        });
        if too_close {
            continue;
        }

        let p = at(row, col);
        // No refinement on the border
        let col_offset = if col > 0 && col + 1 < width {
            parabola_offset(at(row, col - 1), p, at(row, col + 1))
        } else {
            0.0
        };
        let row_offset = if row > 0 && row + 1 < height {
            parabola_offset(at(row - 1, col), p, at(row + 1, col))
        } else {
            0.0
        };
        peaks.push(Peak {
            col: col as f64 + col_offset,
            row: row as f64 + row_offset,
            level_db: 10.0 * (p / p_max).log10(),
        });
    }
    peaks
}

/// No 8-neighbor is larger; on ties only the first cell in row-major order wins
fn is_local_max(
    row: usize,
    col: usize,
    p: f64,
    [width, height]: [usize; 2],
    at: &impl Fn(usize, usize) -> f64,
) -> bool {
    for dr in -1..=1isize {
        for dc in -1..=1isize {
            if dr == 0 && dc == 0 {
                continue;
            }
            let (r, c) = (row as isize + dr, col as isize + dc);
            if !(0..height as isize).contains(&r) || !(0..width as isize).contains(&c) {
                continue;
            }
            let q = at(r as usize, c as usize);
            // Neighbors before this cell in row-major order must be strictly smaller
            let before = (dr, dc) < (0, 0);
            if q > p || (before && q == p) {
                return false;
            }
        }
    }
    true
}

/// Vertex of the parabola through log(left), log(center), log(right) at -1, 0, 1, clamped to
/// half a cell; 0 if a value is not positive or the parabola is not curved downward
fn parabola_offset(left: f64, center: f64, right: f64) -> f64 {
    if left <= 0.0 || center <= 0.0 || right <= 0.0 {
        return 0.0;
    }
    let (l, c, r) = (left.ln(), center.ln(), right.ln());
    let curvature = l - 2.0 * c + r;
    if curvature >= 0.0 {
        return 0.0;
    }
    (0.5 * (l - r) / curvature).clamp(-0.5, 0.5)
}

#[cfg(test)]
mod test {
    use super::{PeakConfig, find_peaks};

    const SIZE: [usize; 2] = [40, 30];

    /// Sum of Gaussians (col, row, peak value, sigma in cells)
    fn gaussians(blobs: &[(f64, f64, f64, f64)]) -> Vec<f64> {
        let [width, height] = SIZE;
        (0..height)
            .flat_map(|row| (0..width).map(move |col| (row as f64, col as f64)))
            .map(|(row, col)| {
                blobs
                    .iter()
                    .map(|&(c0, r0, p, sigma)| {
                        let d2 = (col - c0).powi(2) + (row - r0).powi(2);
                        p * (-d2 / (2.0 * sigma * sigma)).exp()
                    })
                    .sum()
            })
            .collect()
    }

    /// A single Gaussian off the cell centers: the log parabola hits its center exactly.
    #[test]
    fn subcell_position_of_gaussian() {
        let power = gaussians(&[(10.3, 7.6, 1.0, 2.0)]);
        let peaks = find_peaks(&power, SIZE, &PeakConfig::default());

        assert_eq!(peaks.len(), 1);
        assert!((peaks[0].col - 10.3).abs() < 1e-9, "col {}", peaks[0].col);
        assert!((peaks[0].row - 7.6).abs() < 1e-9, "row {}", peaks[0].row);
        assert_eq!(peaks[0].level_db, 0.0);
    }

    /// Two separate sources, strongest first, with their level relative to the maximum.
    #[test]
    fn two_peaks_sorted_with_levels() {
        let power = gaussians(&[(8.0, 8.0, 0.5, 1.5), (30.0, 20.0, 1.0, 1.5)]);
        let peaks = find_peaks(&power, SIZE, &PeakConfig::default());

        assert_eq!(peaks.len(), 2);
        assert!((peaks[0].col - 30.0).abs() < 1e-6 && (peaks[0].row - 20.0).abs() < 1e-6);
        assert!((peaks[1].col - 8.0).abs() < 1e-6 && (peaks[1].row - 8.0).abs() < 1e-6);
        assert!((peaks[1].level_db - 10.0 * 0.5f64.log10()).abs() < 1e-6);
    }

    /// Range, minimum distance and count each drop the weaker peaks.
    #[test]
    fn range_distance_and_count_limits() {
        // Strong peak, a weaker one 3 cells away, one at -10 dB, one at -3 dB far away
        let blobs = [
            (10.0, 10.0, 1.0, 1.0),
            (13.0, 10.0, 0.8, 1.0),
            (30.0, 5.0, 0.1, 1.0),
            (30.0, 25.0, 0.5, 1.0),
        ];
        let power = gaussians(&blobs);
        let config = |range_db, min_dist_cells, max_count| PeakConfig {
            enabled: true,
            range_db,
            min_dist_cells,
            max_count,
            min_above_floor_db: 8.0,
        };
        let cols_rows = |config: PeakConfig| -> Vec<(i64, i64)> {
            find_peaks(&power, SIZE, &config)
                .iter()
                .map(|p| (p.col.round() as i64, p.row.round() as i64))
                .collect()
        };

        assert_eq!(
            cols_rows(config(20.0, 0.0, 10)),
            [(10, 10), (13, 10), (30, 25), (30, 5)]
        );
        assert_eq!(
            cols_rows(config(6.0, 0.0, 10)),
            [(10, 10), (13, 10), (30, 25)]
        );
        assert_eq!(cols_rows(config(6.0, 4.0, 10)), [(10, 10), (30, 25)]);
        assert_eq!(cols_rows(config(6.0, 4.0, 1)), [(10, 10)]);
    }

    /// A flat plateau gives one peak, a map without positive values none.
    #[test]
    fn plateau_and_empty_map() {
        let flat = vec![1.0; SIZE[0] * SIZE[1]];
        assert_eq!(find_peaks(&flat, SIZE, &PeakConfig::default()).len(), 1);

        let negative = vec![-1.0; SIZE[0] * SIZE[1]];
        assert!(find_peaks(&negative, SIZE, &PeakConfig::default()).is_empty());
    }
}
