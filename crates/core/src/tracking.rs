//! Peaks followed from map to map: stable ids, smoothed positions and the audio of each source.

use serde::Deserialize;

use crate::peaks::{Peak, PeakConfig};

/// Tracker parameters, e.g. from a config file
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrackerConfig {
    /// A peak continues a track if it is at most this far away, in grid cells. Small enough that
    /// the maxima of noise, which jump around, do not keep continuing one track (4 cells is
    /// about 5 degrees, a person walking at 2 m moves about 2.5 cells per map)
    pub max_dist_cells: f64,
    /// A track counts as confirmed after a matching peak in `min_hits` of the last
    /// `confirm_window` maps (not necessarily in a row: a source at the detection limit comes and
    /// goes; noise maxima do not stay in one place long enough). A snap stays in the covariance
    /// for about `memory_frames`, i.e. 4 maps, and is confirmed.
    pub min_hits: usize,
    pub confirm_window: usize,
    /// Maps in a row without a matching peak until a track is dropped
    pub max_missed: usize,
    /// Weight of the new peak in the smoothed position, in (0, 1]; 1 follows the peak exactly
    /// (less puts the dot between two spots when the map's maximum jumps, e.g. to a reflection)
    pub smoothing: f64,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            max_dist_cells: 4.0,
            min_hits: 3,
            confirm_window: 5,
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
        if !(self.min_hits..=MAX_CONFIRM_WINDOW).contains(&self.confirm_window) {
            return Err(format!(
                "tracking: confirm_window must be in min_hits..={MAX_CONFIRM_WINDOW}"
            ));
        }
        if !(self.smoothing > 0.0 && self.smoothing <= 1.0) {
            return Err("tracking: smoothing must be in (0, 1]".to_string());
        }
        Ok(())
    }
}

/// `confirm_window` is kept as bits of a `u64`
const MAX_CONFIRM_WINDOW: usize = 64;

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
    /// Filled by the pipeline: how much more `source_db` a track needs at its position to be
    /// shown, see [`crate::peaks::center_penalty_db`]
    pub edge_penalty_db: f64,
    /// Also a peak of the map without transient frames (see
    /// [`crate::pipeline::PipelineConfig::skip_transients`]): a continuous sound, not snapping
    /// or clapping; filled by the pipeline. Only steady tracks can be speech.
    pub steady: bool,
    /// Matching peaks since the track started
    pub hits: usize,
    /// One bit per map, newest in bit 0: 1 if a peak matched
    pub recent: u64,
    /// Maps in a row without a matching peak (0: seen in the current map)
    pub missed: usize,
    /// Seen in `min_hits` of `confirm_window` maps; stays confirmed until dropped
    pub confirmed: bool,
    /// Delay-and-sum audio toward the track since the previous map, mono at
    /// [`crate::dsp::beam_signal::TRACK_SAMPLE_RATE`]; filled by the pipeline
    pub samples: Vec<f32>,
}

impl Track {
    /// Whether to show (and record) the track: confirmed, `min_above_floor_db` (plus its
    /// `edge_penalty_db`) above the noise floor and at least `min_coherence_db` coherent above
    /// 2 kHz, see [`PeakConfig`]
    pub fn shown(&self, peaks: &PeakConfig) -> bool {
        self.confirmed
            && self.source_db - self.edge_penalty_db >= peaks.min_above_floor_db
            && self.source_upper_db >= peaks.min_coherence_db
    }
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
        for track in &mut self.tracks {
            track.recent <<= 1;
        }
        let window_mask = u64::MAX >> (64 - self.config.confirm_window);

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
            track.recent |= 1;
            track.missed = 0;
            let recent_hits = (track.recent & window_mask).count_ones() as usize;
            track.confirmed |= recent_hits >= self.config.min_hits;
        }

        for (track, matched) in self.tracks.iter_mut().zip(&track_matched) {
            if !matched {
                track.missed += 1;
            }
        }
        let max_missed = self.config.max_missed;
        self.tracks.retain(|track| track.missed <= max_missed);

        for (peak, _) in peaks
            .iter()
            .zip(&peak_matched)
            .filter(|(_, matched)| !**matched)
        {
            self.tracks.push(Track {
                id: self.next_id,
                col: peak.col,
                row: peak.row,
                level_db: peak.level_db,
                source_db: 0.0,
                source_upper_db: 0.0,
                edge_penalty_db: 0.0,
                steady: false,
                hits: 1,
                recent: 1,
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
        assert!(
            tracker.tracks()[0].confirmed,
            "kept while missed <= max_missed"
        );
        tracker.update(&[]);
        assert!(tracker.tracks().is_empty());
    }

    /// A peak in only 2 of every 5 maps (noise at one spot now and then) is never confirmed; the
    /// same spot in 3 of 5 is.
    #[test]
    fn confirmation_needs_hits_within_window() {
        let mut tracker = Tracker::new(TrackerConfig::default());
        for map in 0..20 {
            let hit = map % 5 == 0 || map % 5 == 2;
            let peaks = if hit {
                vec![peak(5.0, 5.0)]
            } else {
                Vec::new()
            };
            tracker.update(&peaks);
        }
        assert!(tracker.tracks().iter().all(|track| !track.confirmed));

        let mut tracker = Tracker::new(TrackerConfig::default());
        for map in 0..5 {
            let hit = map % 2 == 0;
            let peaks = if hit {
                vec![peak(5.0, 5.0)]
            } else {
                Vec::new()
            };
            tracker.update(&peaks);
        }
        assert!(tracker.tracks()[0].confirmed);
    }

    /// Maxima that jump by more than the gate every map (noise) start new tracks and none is
    /// ever confirmed.
    #[test]
    fn jumping_peaks_never_confirmed() {
        let mut tracker = Tracker::new(TrackerConfig::default());
        for map in 0..30 {
            let col = 5.0 + (map % 6) as f64 * 5.0;
            tracker.update(&[peak(col, 10.0)]);
        }
        assert!(tracker.tracks().iter().all(|track| !track.confirmed));
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
