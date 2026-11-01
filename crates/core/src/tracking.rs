//! Peaks followed from map to map: stable ids, smoothed positions and the audio of each source.

use serde::Deserialize;

use crate::peaks::Peak;

/// Tracker parameters, e.g. from a config file
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrackerConfig {
    /// A peak continues a track if it is at most this far away, in grid cells
    pub max_dist_cells: f64,
    /// Maps with a matching peak until a track counts as confirmed (not in a row: a source at
    /// the detection limit comes and goes, the misses are limited by `max_missed`)
    pub min_hits: usize,
    /// Maps in a row without a matching peak until a track is dropped
    pub max_missed: usize,
    /// Weight of the new peak in the smoothed position, in (0, 1]; 1 follows the peak exactly
    /// (less puts the dot between two spots when the map's maximum jumps, e.g. to a reflection)
    pub smoothing: f64,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            max_dist_cells: 6.0,
            min_hits: 3,
            max_missed: 5,
            smoothing: 1.0,
        }
    }
}

impl TrackerConfig {
    /// Error message if a value cannot work
    pub fn validate(&self) -> Result<(), String> {
        if self.max_dist_cells <= 0.0 || self.min_hits == 0 {
            return Err("tracking: max_dist_cells must be > 0 and min_hits >= 1".to_string());
        }
        if !(self.smoothing > 0.0 && self.smoothing <= 1.0) {
            return Err("tracking: smoothing must be in (0, 1]".to_string());
        }
        Ok(())
    }
}

/// A source followed over several maps
#[derive(Clone, Debug)]
pub struct Track {
    /// Unique within one pipeline run, never reused
    pub id: u64,
    /// Smoothed position in grid cells, like [`Peak`]
    pub col: f64,
    pub row: f64,
    /// Level of the last matching peak relative to the map maximum, in dB
    pub level_db: f64,
    /// Filled by the pipeline: the power arriving coherently from the track over the map band,
    /// in dB above the noise floor (see [`crate::peaks::PeakConfig::min_above_floor_db`]), and
    /// its share of the power received over the band's upper part (see
    /// [`crate::pipeline::UPPER_BAND_HZ`]) that the other tracks do not explain, in dB
    pub source_db: f64,
    pub source_upper_db: f64,
    /// Also a peak of the map without transient frames (see
    /// [`crate::pipeline::PipelineConfig::skip_transients`]): a continuous sound, not snapping
    /// or clapping; filled by the pipeline. Only steady tracks can be speech.
    pub steady: bool,
    /// Matching peaks since the track started
    pub hits: usize,
    /// Maps in a row without a matching peak (0: seen in the current map)
    pub missed: usize,
    /// Seen in `min_hits` maps; stays confirmed until dropped
    pub confirmed: bool,
    /// Delay-and-sum audio toward the track since the previous map, mono at
    /// [`crate::dsp::beam_signal::TRACK_SAMPLE_RATE`]; filled by the pipeline
    pub samples: Vec<f32>,
}

/// Matches the peaks of each map to the tracks of the previous one
pub struct Tracker {
    config: TrackerConfig,
    tracks: Vec<Track>,
    next_id: u64,
}

impl Tracker {
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            next_id: 0,
        }
    }

    /// Current tracks, oldest first
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    pub fn tracks_mut(&mut self) -> &mut [Track] {
        &mut self.tracks
    }

    /// Continue the tracks with the peaks of a new map.
    ///
    /// Greedy nearest neighbor: the closest (track, peak) pair within `max_dist_cells` is matched
    /// first, then the next closest among the rest. Unmatched peaks start new tracks; unmatched
    /// tracks count a miss and are dropped after more than `max_missed`.
    pub fn update(&mut self, peaks: &[Peak]) {
        let max_dist = self.config.max_dist_cells;
        let mut pairs: Vec<(f64, usize, usize)> = self
            .tracks
            .iter()
            .enumerate()
            .flat_map(|(t, track)| {
                peaks.iter().enumerate().filter_map(move |(p, peak)| {
                    let dist = (track.col - peak.col).hypot(track.row - peak.row);
                    (dist <= max_dist).then_some((dist, t, p))
                })
            })
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut track_matched = vec![false; self.tracks.len()];
        let mut peak_matched = vec![false; peaks.len()];
        for (_, t, p) in pairs {
            if track_matched[t] || peak_matched[p] {
                continue;
            }
            track_matched[t] = true;
            peak_matched[p] = true;

            let (track, peak) = (&mut self.tracks[t], &peaks[p]);
            let alpha = self.config.smoothing;
            track.col += alpha * (peak.col - track.col);
            track.row += alpha * (peak.row - track.row);
            track.level_db = peak.level_db;
            track.hits += 1;
            track.missed = 0;
            track.confirmed |= track.hits >= self.config.min_hits;
        }

        for (track, matched) in self.tracks.iter_mut().zip(&track_matched) {
            if !matched {
                track.missed += 1;
            }
        }
        let max_missed = self.config.max_missed;
        self.tracks.retain(|track| track.missed <= max_missed);

        for (peak, _) in peaks.iter().zip(&peak_matched).filter(|(_, matched)| !**matched) {
            self.tracks.push(Track {
                id: self.next_id,
                col: peak.col,
                row: peak.row,
                level_db: peak.level_db,
                source_db: 0.0,
                source_upper_db: 0.0,
                steady: false,
                hits: 1,
                missed: 0,
                confirmed: self.config.min_hits <= 1,
                samples: Vec::new(),
            });
            self.next_id += 1;
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Tracker, TrackerConfig};
    use crate::peaks::Peak;

    fn peak(col: f64, row: f64) -> Peak {
        Peak {
            col,
            row,
            level_db: 0.0,
        }
    }

    fn ids(tracker: &Tracker) -> Vec<u64> {
        tracker.tracks().iter().map(|t| t.id).collect()
    }

    /// Tracker with position smoothing 0.5
    fn smoothed() -> Tracker {
        Tracker::new(TrackerConfig {
            smoothing: 0.5,
            ..TrackerConfig::default()
        })
    }

    /// A moving peak keeps its id; its position follows with the smoothing weight.
    #[test]
    fn moving_peak_keeps_id() {
        let mut tracker = smoothed();
        tracker.update(&[peak(10.0, 10.0)]);
        tracker.update(&[peak(12.0, 10.0)]);
        tracker.update(&[peak(14.0, 11.0)]);

        assert_eq!(ids(&tracker), [0]);
        let track = &tracker.tracks()[0];
        // 10 -> 11 -> 12.5, 10 -> 10 -> 10.5 with smoothing 0.5
        assert_eq!((track.col, track.row), (12.5, 10.5));
        assert_eq!(track.hits, 3);
    }

    /// A peak farther than `max_dist_cells` starts a new track; the old one counts a miss.
    #[test]
    fn jump_starts_new_track() {
        let mut tracker = Tracker::new(TrackerConfig::default());
        tracker.update(&[peak(10.0, 10.0)]);
        tracker.update(&[peak(30.0, 10.0)]);

        assert_eq!(ids(&tracker), [0, 1]);
        assert_eq!(tracker.tracks()[0].missed, 1);
        assert_eq!(tracker.tracks()[1].missed, 0);
    }

    /// Confirmed after `min_hits` matching maps, also with misses in between, stays confirmed
    /// through misses, dropped after more than `max_missed`.
    #[test]
    fn confirmation_and_drop() {
        let config = TrackerConfig::default();
        let mut tracker = Tracker::new(config.clone());
        for _ in 0..config.min_hits - 1 {
            tracker.update(&[peak(5.0, 5.0)]);
            tracker.update(&[]);
        }
        assert!(!tracker.tracks()[0].confirmed);
        tracker.update(&[peak(5.0, 5.0)]);
        assert!(tracker.tracks()[0].confirmed);

        for _ in 0..config.max_missed {
            tracker.update(&[]);
        }
        assert!(tracker.tracks()[0].confirmed, "kept while missed <= max_missed");
        tracker.update(&[]);
        assert!(tracker.tracks().is_empty());
    }

    /// Two close tracks and two peaks: each peak goes to its nearest track, not both to one.
    #[test]
    fn nearest_pairs_first() {
        let mut tracker = smoothed();
        tracker.update(&[peak(10.0, 10.0), peak(14.0, 10.0)]);
        // Both new peaks are within the gate of both tracks
        tracker.update(&[peak(13.0, 10.0), peak(11.0, 10.0)]);

        assert_eq!(ids(&tracker), [0, 1]);
        let cols: Vec<f64> = tracker.tracks().iter().map(|t| t.col).collect();
        assert_eq!(cols, [10.5, 13.5]);
    }
}
