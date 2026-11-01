//! Speech detection per tracked source: the audio of each track goes through the
//! voice-activity-detection classifier, smoothed with hysteresis per track.

use std::collections::{HashMap, VecDeque};

use serde::Deserialize;
use sound2view_core::{BoxError, tracking::Track};
use voice_activity_detection::{
    model::Model,
    preprocessing::{HOP, Preprocessor, SAMPLE_RATE},
};

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpeechConfig {
    /// Classify the tracks at all
    pub enabled: bool,
    /// Model exported by voice-activity-detection, relative to the working directory
    pub model: String,
    /// Smoothed probability at which a source becomes speech, in (0, 1]
    pub on_threshold: f32,
    /// Speech is held this long after the last frame at `on_threshold` or above, in ms: pauses
    /// between words stay speech, other sounds after the speech (clapping, snapping) end it
    pub hold_ms: f32,
    /// Time constant of the probability smoothing, in ms (0: no smoothing)
    pub smoothing_ms: f32,
    /// The smoothed probability must stay at `on_threshold` this long before speech starts, in
    /// ms: transients (snapping, clapping) are shorter than a syllable
    pub min_onset_ms: f32,
    /// A speech track whose audio correlates at least this much (within +-20 ms) with a speech
    /// track that has `echo_margin_db` more speech energy repeats that track's speech (reflection,
    /// beamformer ghost, leakage into another sound) and is not speech, in [0, 1]; 1 or more
    /// turns the check off
    pub echo_correlation: f64,
    /// See `echo_correlation`, in dB
    pub echo_margin_db: f32,
    /// Speech ends at once when the track moves farther than this from where it last was at
    /// `on_threshold`, in grid cells: another source nearby (snapping next to the speaker) that
    /// the tracker continues as the same track does not inherit the speech flag
    pub max_shift_cells: f64,
    /// Gain applied to the beam signal before the classifier, in dB. The classifier is trained on
    /// microphone levels (speech around -25 dBFS); the UMA-16 gives speech at 1 to 2 m around
    /// -60 dBFS.
    pub gain_db: f32,
    /// Classifier frames quieter than this (after `gain_db`) count as non-speech, in dBFS
    pub min_level_db: f32,
}

impl Default for SpeechConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model: "model/speech_detector.json".to_string(),
            on_threshold: 0.95,
            hold_ms: 500.0,
            smoothing_ms: 50.0,
            min_onset_ms: 80.0,
            echo_correlation: 0.2,
            echo_margin_db: 6.0,
            max_shift_cells: 3.0,
            gain_db: 36.0,
            min_level_db: -55.0,
        }
    }
}

impl SpeechConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(self.on_threshold > 0.0 && self.on_threshold <= 1.0) {
            return Err("speech.on_threshold must be in (0, 1]".to_string());
        }
        if !self.gain_db.is_finite() {
            return Err("speech.gain_db must be finite".to_string());
        }
        let non_negative = [self.hold_ms, self.smoothing_ms, self.min_onset_ms];
        if !(non_negative.iter().all(|&v| v >= 0.0) && self.max_shift_cells >= 0.0) {
            return Err(
                "speech: hold_ms, smoothing_ms, min_onset_ms and max_shift_cells must be >= 0"
                    .to_string(),
            );
        }
        if self.echo_correlation.is_nan() || self.echo_correlation < 0.0 {
            return Err("speech.echo_correlation must be >= 0".to_string());
        }
        if !self.echo_margin_db.is_finite() {
            return Err("speech.echo_margin_db must be finite".to_string());
        }
        Ok(())
    }

    /// Hold in classifier frames
    fn hold_frames(&self) -> usize {
        (self.hold_ms / FRAME_MS).round() as usize
    }

    /// Minimum onset in classifier frames
    fn onset_frames(&self) -> usize {
        (self.min_onset_ms / FRAME_MS).round() as usize
    }

    /// Weight of a new frame in the smoothed probability, 1 - exp(-frame / tau)
    fn smoothing_weight(&self) -> f32 {
        if self.smoothing_ms > 0.0 {
            1.0 - (-FRAME_MS / self.smoothing_ms).exp()
        } else {
            1.0
        }
    }
}

/// Length of one classifier frame (`HOP` samples) in ms
const FRAME_MS: f32 = 1000.0 * HOP as f32 / SAMPLE_RATE as f32;

/// Audio per track kept for the echo check, in samples (0.3 s)
const ECHO_WINDOW: usize = SAMPLE_RATE * 3 / 10;
/// Largest delay between a track and its echo, in samples (20 ms, about 7 m of extra path)
const ECHO_MAX_LAG: usize = SAMPLE_RATE / 50;

/// Speech state of one track, for display
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpeechStatus {
    pub speech: bool,
    /// Smoothed speech probability in [0, 1]
    pub probability: f32,
    /// Speech-like, but a reflection or ghost of this louder track, see
    /// [`SpeechConfig::echo_correlation`]
    pub echo_of: Option<u64>,
}

/// Smoothing and hysteresis of the classifier output: speech starts when the smoothed
/// probability has been at `on_threshold` for `onset_frames` and ends `hold_frames` after the
/// last frame that reached it. Only
/// such frames re-arm the hold, so a sound the classifier finds half speech-like (clapping after
/// speech, same track) cannot keep it on. Speech also ends when the track moves more than
/// `max_shift_cells` from where it last reached `on_threshold` (another source took it over).
#[derive(Default)]
struct Hysteresis {
    status: SpeechStatus,
    /// Frames left until speech ends
    hold: usize,
    /// Track position (col, row) at the last frame at `on_threshold`
    anchor: [f64; 2],
    /// Consecutive frames with the smoothed probability at `on_threshold`
    above: usize,
}

/// Per-frame constants derived from [`SpeechConfig`]
struct HysteresisParams {
    on_threshold: f32,
    min_level_db: f32,
    hold_frames: usize,
    onset_frames: usize,
    weight: f32,
    max_shift_cells: f64,
}

impl HysteresisParams {
    fn new(config: &SpeechConfig) -> Self {
        Self {
            on_threshold: config.on_threshold,
            min_level_db: config.min_level_db,
            hold_frames: config.hold_frames(),
            onset_frames: config.onset_frames(),
            weight: config.smoothing_weight(),
            max_shift_cells: config.max_shift_cells,
        }
    }
}

impl Hysteresis {
    /// One classifier frame; `pos` is the track position (col, row) in grid cells
    fn update(&mut self, probability: f32, rms: f32, pos: [f64; 2], params: &HysteresisParams) {
        let level_db = 20.0 * rms.max(1e-5).log10();
        let probability = if level_db < params.min_level_db {
            0.0
        } else {
            probability.clamp(0.0, 1.0)
        };
        let status = &mut self.status;
        status.probability += params.weight * (probability - status.probability);
        if status.probability >= params.on_threshold {
            self.above += 1;
        } else {
            self.above = 0;
        }

        // Starts on the smoothed probability after the onset; the frame's own probability must
        // reach the threshold too, both to start and to re-arm the hold and the anchor, so the
        // smoothing tail of the speech can neither anchor nor restart it on a source that took
        // over the track
        if probability >= params.on_threshold
            && (status.speech || self.above >= params.onset_frames.max(1))
        {
            status.speech = true;
            self.hold = params.hold_frames;
            self.anchor = pos;
        } else if status.speech {
            let shift = (pos[0] - self.anchor[0]).hypot(pos[1] - self.anchor[1]);
            self.hold = if shift > params.max_shift_cells {
                0
            } else {
                self.hold.saturating_sub(1)
            };
            status.speech = self.hold > 0;
        }
    }
}

struct TrackSpeech {
    preprocessor: Preprocessor,
    hysteresis: Hysteresis,
    /// The last `ECHO_WINDOW` samples, without gain
    recent: Vec<f32>,
    /// (probability, rms^2) of the classifier frames in the last `ECHO_WINDOW` samples, newest
    /// last
    frames: VecDeque<(f32, f32)>,
}

impl TrackSpeech {
    /// Energy of the speech in the echo window: frame power weighted by its speech probability.
    /// Snaps and other sounds count little (low probability), leaked or reflected speech is
    /// weaker than the speaker's own.
    fn speech_energy(&self) -> f32 {
        self.frames.iter().map(|&(p, power)| p * power).sum()
    }
}

/// Classifies the audio of every track; state per track id
pub struct SpeechDetector {
    model: Model,
    config: SpeechConfig,
    tracks: HashMap<u64, TrackSpeech>,
}

impl SpeechDetector {
    /// Load the model from `config.model`
    pub fn load(config: &SpeechConfig) -> Result<Self, BoxError> {
        let model = Model::load(&config.model).map_err(|e| format!("{}: {e}", config.model))?;
        Ok(Self {
            model,
            config: config.clone(),
            tracks: HashMap::new(),
        })
    }

    /// Forget all tracks; call when a new pipeline run starts (track ids start over)
    pub fn reset(&mut self) {
        self.tracks.clear();
    }

    /// Change `on_threshold` while running; applies from the next classifier frame. In (0, 1]
    /// (see [`SpeechConfig::validate`]).
    pub fn set_on_threshold(&mut self, threshold: f32) {
        debug_assert!(threshold > 0.0 && threshold <= 1.0);
        self.config.on_threshold = threshold;
    }

    /// Classify the new audio of each track (`Track::samples`, 16 kHz); tracks that are gone are
    /// forgotten.
    pub fn update(&mut self, tracks: &[Track]) -> Result<(), BoxError> {
        self.tracks
            .retain(|id, _| tracks.iter().any(|track| track.id == *id));
        let params = HysteresisParams::new(&self.config);
        let gain = 10f32.powf(self.config.gain_db / 20.0);
        let mut samples = Vec::new();

        for track in tracks {
            let state = self.tracks.entry(track.id).or_insert_with(|| TrackSpeech {
                preprocessor: Preprocessor::new(),
                hysteresis: Hysteresis::default(),
                recent: Vec::new(),
                frames: VecDeque::new(),
            });
            state.recent.extend_from_slice(&track.samples);
            let excess = state.recent.len().saturating_sub(ECHO_WINDOW);
            state.recent.drain(..excess);

            samples.clear();
            samples.extend(track.samples.iter().map(|s| s * gain));
            let frames = state.preprocessor.push(&samples);
            if frames.is_empty() {
                continue;
            }
            let rms: Vec<f32> = frames.iter().map(|frame| frame.rms).collect();
            let features = frames.into_iter().map(|frame| frame.features).collect();
            let probabilities = self.model.predict(features)?;
            for (probability, rms) in probabilities.into_iter().zip(rms) {
                state
                    .hysteresis
                    .update(probability, rms, [track.col, track.row], &params);
                state.frames.push_back((probability.clamp(0.0, 1.0), rms * rms));
            }
            let excess = state.frames.len().saturating_sub(ECHO_WINDOW / HOP);
            state.frames.drain(..excess);
        }

        // Speech tracks that only echo a louder speech track are not speech. Decided on the
        // speech flags before any is cleared, so the order of the tracks does not matter.
        let speech: Vec<bool> = tracks
            .iter()
            .map(|track| self.tracks[&track.id].hysteresis.status.speech)
            .collect();
        let echoes: Vec<Option<u64>> = (0..tracks.len())
            .map(|i| self.echo_source(i, tracks, &speech))
            .collect();
        for (track, echo_of) in tracks.iter().zip(echoes) {
            let hysteresis = &mut self.tracks.get_mut(&track.id).unwrap().hysteresis;
            hysteresis.status.echo_of = echo_of;
            if echo_of.is_some() {
                hysteresis.status.speech = false;
                hysteresis.hold = 0;
            }
        }
        Ok(())
    }

    /// The track whose speech `tracks[i]` repeats (reflection, beamformer ghost, or another
    /// sound with the speech leaking in): another speech track with at least `echo_margin_db`
    /// more speech energy (see [`TrackSpeech::speech_energy`]) whose audio correlates at least
    /// `echo_correlation`. `None` if `tracks[i]` is not speech.
    ///
    /// Below about 1.5 kHz, where most speech energy is, the 12.6 cm array has almost no
    /// directivity, so every beam carries a good part of every talker: a snapping track next to
    /// the speaker is speech-like through the leaked speech and correlates only about 0.35 with
    /// the speaker (the snaps dilute it). Its speech energy is 9 to 18 dB below the speaker's.
    /// Not the louder track: the snapping is louder in the map and in the beam signal. Two
    /// talkers within `echo_margin_db` of each other both stay speech.
    fn echo_source(&self, i: usize, tracks: &[Track], speech: &[bool]) -> Option<u64> {
        let track = &tracks[i];
        let state = &self.tracks[&track.id];
        if !speech[i] || self.config.echo_correlation >= 1.0 || state.recent.len() < ECHO_WINDOW / 2
        {
            return None;
        }
        let margin = 10f32.powf(self.config.echo_margin_db / 10.0);
        let energy = state.speech_energy();
        tracks
            .iter()
            .zip(speech)
            .filter(|&(other, &other_speech)| {
                let other_state = &self.tracks[&other.id];
                other.id != track.id
                    && other_speech
                    && other_state.recent.len() >= ECHO_WINDOW / 2
                    && other_state.speech_energy() >= margin * energy
            })
            .map(|(other, _)| {
                let other_recent = &self.tracks[&other.id].recent;
                (other.id, max_correlation(&state.recent, other_recent, ECHO_MAX_LAG))
            })
            .filter(|&(_, correlation)| correlation >= self.config.echo_correlation)
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(id, _)| id)
    }

    /// Current state of track `id`, `None` before its first classifier frame
    pub fn status(&self, id: u64) -> Option<SpeechStatus> {
        self.tracks.get(&id).map(|state| state.hysteresis.status)
    }

    /// State of every current track
    pub fn statuses(&self) -> HashMap<u64, SpeechStatus> {
        self.tracks
            .iter()
            .map(|(&id, state)| (id, state.hysteresis.status))
            .collect()
    }
}

fn mean_square(x: &[f32]) -> f64 {
    x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / x.len().max(1) as f64
}

/// Largest normalized cross-correlation |sum a[n] b[n + lag]| / sqrt(sum a^2 sum b^2) over
/// |lag| <= `max_lag`; `a` and `b` end at the same time, the longer one is cut at the start
fn max_correlation(a: &[f32], b: &[f32], max_lag: usize) -> f64 {
    let len = a.len().min(b.len());
    let (a, b) = (&a[a.len() - len..], &b[b.len() - len..]);
    let norm = (mean_square(a) * mean_square(b)).sqrt() * len as f64;
    if norm == 0.0 {
        return 0.0;
    }
    let max_lag = max_lag.min(len.saturating_sub(1));
    let dot = |x: &[f32], y: &[f32]| -> f64 {
        x.iter().zip(y).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum()
    };
    (0..=max_lag)
        .flat_map(|lag| [dot(&a[..len - lag], &b[lag..]), dot(&a[lag..], &b[..len - lag])])
        .map(f64::abs)
        .fold(0.0, f64::max)
        / norm
}

#[cfg(test)]
mod test {
    use sound2view_core::tracking::Track;

    use super::{
        FRAME_MS, Hysteresis, HysteresisParams, SpeechConfig, SpeechDetector, max_correlation,
    };

    fn config() -> SpeechConfig {
        SpeechConfig {
            model: concat!(env!("CARGO_MANIFEST_DIR"), "/../../model/speech_detector.json")
                .to_string(),
            // The test signals are at microphone level already
            gain_db: 0.0,
            ..SpeechConfig::default()
        }
    }

    /// Track 0 with `samples`
    fn track(samples: &[f32]) -> Track {
        track_with_id(0, 0.0, samples)
    }

    /// Track `id` with map level `level_db`
    fn track_with_id(id: u64, level_db: f64, samples: &[f32]) -> Track {
        Track {
            id,
            col: 0.0,
            row: 0.0,
            level_db,
            hits: 1,
            missed: 0,
            confirmed: true,
            samples: samples.to_vec(),
        }
    }

    /// Feed `audio` in pieces of one map (about 0.11 s at 16 kHz), return the speech flag after
    /// each piece
    fn speech_per_map(audio: &[f32]) -> Vec<bool> {
        let mut detector = SpeechDetector::load(&config()).unwrap();
        audio
            .chunks(1707)
            .map(|chunk| {
                detector.update(&[track(chunk)]).unwrap();
                detector.status(0).is_some_and(|status| status.speech)
            })
            .collect()
    }

    fn speech_fixture() -> Vec<f32> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/speech_16k.wav");
        let mut reader = hound::WavReader::open(path).unwrap();
        assert_eq!(reader.spec().sample_rate, 16_000);
        reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect()
    }

    /// Defaults with `on_threshold` 0.7: the hysteresis tests use made-up probabilities around it
    fn hysteresis_config() -> SpeechConfig {
        SpeechConfig {
            on_threshold: 0.7,
            ..SpeechConfig::default()
        }
    }

    /// White noise, uniform in [-amplitude, amplitude] (xorshift)
    fn noise(len: usize, seed: u32, amplitude: f32) -> Vec<f32> {
        let mut x = seed;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x as f32 / u32::MAX as f32 - 0.5) * 2.0 * amplitude
            })
            .collect()
    }

    /// Probability `p` at speech level for `ms` milliseconds; returns the speech flag at the end
    fn feed(state: &mut Hysteresis, params: &HysteresisParams, p: f32, ms: f32) -> bool {
        feed_at(state, params, p, ms, [10.0, 10.0])
    }

    /// As `feed`, with the track at `pos`
    fn feed_at(
        state: &mut Hysteresis,
        params: &HysteresisParams,
        p: f32,
        ms: f32,
        pos: [f64; 2],
    ) -> bool {
        for _ in 0..(ms / FRAME_MS).round() as usize {
            state.update(p, 0.1, pos, params);
        }
        state.status.speech
    }

    /// Speech holds through short pauses and a drop to 0.2, ends `hold_ms` after the last frame
    /// at `on_threshold`, also when a half speech-like sound follows; no speech below the level
    /// gate.
    #[test]
    fn hysteresis_hold_and_level_gate() {
        let params = HysteresisParams::new(&hysteresis_config());
        assert_eq!(params.hold_frames, 50);

        let mut state = Hysteresis::default();
        assert!(!feed(&mut state, &params, 0.6, 1000.0), "below on_threshold");
        assert!(feed(&mut state, &params, 1.0, 500.0));
        assert!(feed(&mut state, &params, 0.2, 400.0), "held at 0.2");
        // Syllables and pauses
        for _ in 0..10 {
            assert!(feed(&mut state, &params, 0.9, 150.0));
            assert!(feed(&mut state, &params, 0.05, 250.0));
        }
        assert!(feed(&mut state, &params, 0.9, 150.0));
        assert!(feed(&mut state, &params, 0.0, 400.0), "held during the pause");
        assert!(!feed(&mut state, &params, 0.0, 200.0), "ended after hold_ms");

        // Clapping after speech: about 0.45, never on_threshold
        assert!(feed(&mut state, &params, 1.0, 500.0));
        assert!(feed(&mut state, &params, 0.45, 400.0), "held after the speech");
        assert!(!feed(&mut state, &params, 0.45, 200.0), "a 0.45 sound does not keep it");

        let mut quiet = Hysteresis::default();
        for _ in 0..50 {
            quiet.update(1.0, 1e-4, [10.0, 10.0], &params);
        }
        assert!(!quiet.status.speech, "-80 dB is below the level gate");
    }

    /// Another source taking over the track ends speech at once; the speaker moving a little
    /// does not.
    #[test]
    fn hysteresis_ends_when_track_moves() {
        let params = HysteresisParams::new(&hysteresis_config());
        let mut state = Hysteresis::default();
        assert!(feed_at(&mut state, &params, 1.0, 500.0, [10.0, 10.0]));
        assert!(feed_at(&mut state, &params, 0.3, 200.0, [11.5, 11.0]), "within 3 cells");
        // Speech at the new position re-anchors
        assert!(feed_at(&mut state, &params, 1.0, 200.0, [12.0, 11.0]));
        assert!(!feed_at(&mut state, &params, 0.3, 20.0, [16.0, 11.0]), "snapping 4 cells away");
    }

    /// Bursts shorter than the onset (snapping, clapping) do not start speech, a syllable does.
    #[test]
    fn short_bursts_do_not_start_speech() {
        let params = HysteresisParams::new(&hysteresis_config());
        let mut state = Hysteresis::default();
        for _ in 0..10 {
            assert!(!feed(&mut state, &params, 1.0, 60.0));
            assert!(!feed(&mut state, &params, 0.0, 300.0));
        }
        assert!(feed(&mut state, &params, 1.0, 200.0));
    }

    /// A delayed, weaker copy correlates with the original; independent noise does not.
    #[test]
    fn correlation_finds_delayed_copy() {
        let original = noise(4800, 1, 0.1);
        let other = noise(4800, 2, 0.1);
        let mut copy = vec![0.0; 120];
        copy.extend(original[..4800 - 120].iter().map(|v| 0.3 * v));
        let copy: Vec<f32> = copy.iter().zip(noise(4800, 3, 0.01)).map(|(c, n)| c + n).collect();

        let echo = max_correlation(&copy, &original, 320);
        let independent = max_correlation(&other, &original, 320);
        assert!(echo > 0.9, "echo {echo}");
        assert!(independent < 0.1, "independent {independent}");
        // Cut to the shorter one at the start
        assert!(max_correlation(&copy[2000..], &original, 320) > 0.9);
    }

    /// A reflection of a speaker (weaker, 5 ms later) is not speech, the speaker is.
    #[test]
    fn reflection_is_not_speech() {
        let speech = speech_fixture();
        let mut reflection = vec![0.0; 80];
        reflection.extend(speech[..speech.len() - 80].iter().map(|v| 0.3 * v));
        let room = noise(reflection.len(), 5, 0.002);
        let reflection: Vec<f32> = reflection.iter().zip(room).map(|(r, n)| r + n).collect();

        let mut detector = SpeechDetector::load(&config()).unwrap();
        let (mut speaker_maps, mut echo_maps, mut maps) = (0, 0, 0);
        for (a, b) in speech.chunks(1707).zip(reflection.chunks(1707)) {
            // The beamformer map ranks the speaker above its reflection
            detector
                .update(&[track_with_id(0, 0.0, a), track_with_id(1, -6.0, b)])
                .unwrap();
            let speaker = detector.status(0).unwrap_or_default();
            let echo = detector.status(1).unwrap_or_default();
            assert!(!echo.speech, "reflection is speech in map {maps}");
            speaker_maps += usize::from(speaker.speech);
            echo_maps += usize::from(echo.echo_of == Some(0));
            maps += 1;
        }
        assert!(speaker_maps > maps / 2, "speaker is speech in {speaker_maps} of {maps} maps");
        assert!(echo_maps > 0, "the reflection was never speech-like");
    }

    /// Snapping while talking: the snapping track is louder and shares the snaps and the speech
    /// with the speaker's track (leakage both ways). The speaker stays speech, the snapping track
    /// (speech-like through the leaked speech) is its echo.
    #[test]
    fn snapping_does_not_make_speaker_an_echo() {
        let speech = speech_fixture();
        let hiss = noise(speech.len(), 9, 1.0);
        // A snap every 300 ms, decaying with 8 ms
        let snaps: Vec<f32> = hiss
            .iter()
            .enumerate()
            .map(|(n, v)| 0.5 * (-((n % 4800) as f32) / 128.0).exp() * v)
            .collect();
        let speaker: Vec<f32> = speech.iter().zip(&snaps).map(|(s, k)| s + 0.3 * k).collect();
        let snapping: Vec<f32> = speech.iter().zip(&snaps).map(|(s, k)| 0.1 * s + k).collect();

        let mut detector = SpeechDetector::load(&config()).unwrap();
        let (mut speaker_maps, mut snapping_maps, mut maps) = (0, 0, 0);
        for (a, b) in speaker.chunks(1707).zip(snapping.chunks(1707)) {
            // The snapping is louder in the map
            detector
                .update(&[track_with_id(0, -3.0, a), track_with_id(1, 0.0, b)])
                .unwrap();
            let status = detector.status(0).unwrap_or_default();
            assert_eq!(status.echo_of, None, "speaker is an echo in map {maps}");
            speaker_maps += usize::from(status.speech);
            snapping_maps += usize::from(detector.status(1).unwrap_or_default().speech);
            maps += 1;
        }
        assert!(speaker_maps > maps / 2, "speaker is speech in {speaker_maps} of {maps} maps");
        // The speech leaking into the snapping track is an echo of the speaker
        assert_eq!(snapping_maps, 0, "snapping is speech in {snapping_maps} of {maps} maps");
    }

    /// Two talkers at the same level, each leaking into the other's beam, both stay speech.
    #[test]
    fn two_talkers_are_both_speech() {
        let speech = speech_fixture();
        // The second talker says the same text 1.3 s later (independent signal)
        let offset = 20_800;
        let other: Vec<f32> =
            speech.iter().cycle().skip(offset).take(speech.len()).copied().collect();
        let a: Vec<f32> = speech.iter().zip(&other).map(|(s, o)| s + 0.3 * o).collect();
        let b: Vec<f32> = other.iter().zip(&speech).map(|(o, s)| o + 0.3 * s).collect();

        let mut detector = SpeechDetector::load(&config()).unwrap();
        let mut together = [0, 0];
        for (a, b) in a.chunks(1707).zip(b.chunks(1707)) {
            detector
                .update(&[track_with_id(0, 0.0, a), track_with_id(1, 0.0, b)])
                .unwrap();
            for id in [0, 1] {
                together[id] += usize::from(detector.status(id as u64).unwrap_or_default().speech);
            }
        }
        // While one talker pauses, its beam carries only the other's leaked speech: that is an
        // echo. Talking, each stays speech nearly as often as alone (78 % at on_threshold 0.95:
        // the other's leaked speech lowers the probability a little).
        for (id, alone) in [&speech, &other].into_iter().enumerate() {
            let alone = speech_per_map(alone).iter().filter(|&&s| s).count();
            assert!(
                together[id] * 10 >= alone * 7,
                "track {id}: speech in {} maps together, {alone} alone",
                together[id]
            );
        }
    }

    /// Synthesized speech (macOS `say`, 4.7 s) is detected as speech.
    #[test]
    fn speech_fixture_is_speech() {
        let flags = speech_per_map(&speech_fixture());
        let speech = flags.iter().filter(|&&s| s).count();
        assert!(speech > flags.len() / 2, "speech in {speech} of {} maps", flags.len());
    }

    /// White noise and a steady tone at speech level are not speech.
    #[test]
    fn noise_and_tone_are_not_speech() {
        let noise = noise(48_000, 1, 0.1);
        let tone: Vec<f32> = (0..48_000)
            .map(|n| 0.1 * (2.0 * std::f32::consts::PI * 1000.0 * n as f32 / 16_000.0).sin())
            .collect();

        for (name, audio) in [("noise", noise), ("tone", tone)] {
            let flags = speech_per_map(&audio);
            assert!(!flags.iter().any(|&s| s), "{name}: {flags:?}");
        }
    }
}
