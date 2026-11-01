//! Speech detection per tracked source: the audio of each track goes through the
//! voice-activity-detection classifier, smoothed with hysteresis per track.

use std::collections::HashMap;

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
    /// Smoothed probability at which a source counts as speech, in (0, 1]
    pub threshold: f32,
    /// A source stays speech while the smoothed probability is above `threshold - release`;
    /// below `threshold`, so that speech can end
    pub release: f32,
    /// Speech is held this long after the probability fell below `threshold - release`, in ms
    pub hangover_ms: f32,
    /// Classifier frames quieter than this count as non-speech; dB of the beam signal (full scale
    /// at the mic, referenced to 1 m)
    pub min_level_db: f32,
}

impl Default for SpeechConfig {
    fn default() -> Self {
        // Same values as the voice-activity-detection app
        Self {
            enabled: true,
            model: "model/speech_detector.json".to_string(),
            threshold: 0.9,
            release: 0.15,
            hangover_ms: 300.0,
            min_level_db: -55.0,
        }
    }
}

impl SpeechConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(self.threshold > 0.0 && self.threshold <= 1.0) {
            return Err("speech.threshold must be in (0, 1]".to_string());
        }
        if !(0.0..self.threshold).contains(&self.release) || self.hangover_ms < 0.0 {
            return Err("speech: release must be in [0, threshold), hangover_ms >= 0".to_string());
        }
        Ok(())
    }

    /// Hangover in classifier frames (one per `HOP` samples)
    fn hangover_frames(&self) -> usize {
        let frame_ms = 1000.0 * HOP as f32 / SAMPLE_RATE as f32;
        (self.hangover_ms / frame_ms).round() as usize
    }
}

/// Speech state of one track, for display
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpeechStatus {
    pub speech: bool,
    /// Smoothed speech probability in [0, 1]
    pub probability: f32,
}

/// Weight of a new frame's probability in the smoothed one (10 ms frames)
const SMOOTHING: f32 = 0.2;

/// Smoothing and hysteresis of the classifier output, as in the voice-activity-detection app:
/// speech starts when the smoothed probability reaches `threshold` and ends `hangover_frames`
/// after it fell below `threshold - release`.
#[derive(Default)]
struct Hysteresis {
    status: SpeechStatus,
    /// Frames left until speech ends
    hangover: usize,
}

impl Hysteresis {
    fn update(&mut self, probability: f32, rms: f32, config: &SpeechConfig, hangover_frames: usize) {
        let level_db = 20.0 * rms.max(1e-5).log10();
        let probability = if level_db < config.min_level_db {
            0.0
        } else {
            probability.clamp(0.0, 1.0)
        };
        let status = &mut self.status;
        status.probability += SMOOTHING * (probability - status.probability);

        if status.probability >= config.threshold {
            status.speech = true;
            self.hangover = hangover_frames;
        } else if status.speech {
            if status.probability >= config.threshold - config.release {
                self.hangover = hangover_frames;
            } else {
                self.hangover = self.hangover.saturating_sub(1);
                status.speech = self.hangover > 0;
            }
        }
    }
}

struct TrackSpeech {
    preprocessor: Preprocessor,
    hysteresis: Hysteresis,
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

    /// Change the threshold while running; applies from the next classifier frame. Must stay
    /// above `release` (see [`SpeechConfig::validate`]).
    pub fn set_threshold(&mut self, threshold: f32) {
        debug_assert!(threshold > self.config.release && threshold <= 1.0);
        self.config.threshold = threshold;
    }

    /// Classify the new audio of each track (`Track::samples`, 16 kHz); tracks that are gone are
    /// forgotten.
    pub fn update(&mut self, tracks: &[Track]) -> Result<(), BoxError> {
        self.tracks
            .retain(|id, _| tracks.iter().any(|track| track.id == *id));
        let hangover_frames = self.config.hangover_frames();

        for track in tracks {
            let state = self.tracks.entry(track.id).or_insert_with(|| TrackSpeech {
                preprocessor: Preprocessor::new(),
                hysteresis: Hysteresis::default(),
            });
            let frames = state.preprocessor.push(&track.samples);
            if frames.is_empty() {
                continue;
            }
            let rms: Vec<f32> = frames.iter().map(|frame| frame.rms).collect();
            let features = frames.into_iter().map(|frame| frame.features).collect();
            let probabilities = self.model.predict(features)?;
            for (probability, rms) in probabilities.into_iter().zip(rms) {
                state
                    .hysteresis
                    .update(probability, rms, &self.config, hangover_frames);
            }
        }
        Ok(())
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

#[cfg(test)]
mod test {
    use sound2view_core::tracking::Track;

    use super::{Hysteresis, SpeechConfig, SpeechDetector};

    fn config() -> SpeechConfig {
        SpeechConfig {
            model: concat!(env!("CARGO_MANIFEST_DIR"), "/../../model/speech_detector.json")
                .to_string(),
            ..SpeechConfig::default()
        }
    }

    /// Track 0 with `samples`
    fn track(samples: &[f32]) -> Track {
        Track {
            id: 0,
            col: 0.0,
            row: 0.0,
            level_db: 0.0,
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

    /// Hold during the hangover, end after it, and no speech below the level gate.
    #[test]
    fn hysteresis_hangover_and_level_gate() {
        let config = SpeechConfig::default();
        let frames = config.hangover_frames();
        assert_eq!(frames, 30);

        let mut state = Hysteresis::default();
        for _ in 0..20 {
            state.update(1.0, 0.1, &config, frames);
        }
        assert!(state.status.speech);
        for _ in 0..10 {
            state.update(0.0, 0.1, &config, frames);
        }
        assert!(state.status.speech, "held during the hangover");
        for _ in 0..frames {
            state.update(0.0, 0.1, &config, frames);
        }
        assert!(!state.status.speech, "ended after the hangover");

        let mut quiet = Hysteresis::default();
        for _ in 0..50 {
            quiet.update(1.0, 1e-4, &config, frames);
        }
        assert!(!quiet.status.speech, "-80 dB is below the level gate");
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
        // xorshift, uniform in [-0.1, 0.1]
        let mut x: u32 = 1;
        let noise: Vec<f32> = (0..48_000)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x as f32 / u32::MAX as f32 - 0.5) * 0.2
            })
            .collect();
        let tone: Vec<f32> = (0..48_000)
            .map(|n| 0.1 * (2.0 * std::f32::consts::PI * 1000.0 * n as f32 / 16_000.0).sin())
            .collect();

        for (name, audio) in [("noise", noise), ("tone", tone)] {
            let flags = speech_per_map(&audio);
            assert!(!flags.iter().any(|&s| s), "{name}: {flags:?}");
        }
    }
}
