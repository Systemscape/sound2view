//! GUI settings kept between runs in `sound2view-gui.toml` (working directory): everything that
//! can be changed in the settings menu. Written whenever a setting changes and applied on top of
//! `sound2view.toml` at the next start; delete the file to go back to `sound2view.toml`.

use std::fs;

use serde::{Deserialize, Serialize};

use crate::config::{BeamformerConfig, Config};

pub const GUI_STATE_PATH: &str = "sound2view-gui.toml";

/// Settings menu values; `None` (not in the file) keeps the value from `sound2view.toml`
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GuiState {
    pub beamformer: Option<BeamformerConfig>,
    pub f_lo_hz: Option<f64>,
    pub f_hi_hz: Option<f64>,
    pub skip_transients: Option<bool>,
    pub db_range: Option<f64>,
    pub tracks: Option<bool>,
    pub min_above_floor_db: Option<f64>,
    pub min_coherence_db: Option<f64>,
    pub speech: Option<bool>,
    pub speech_threshold: Option<f32>,
    /// Name of the chosen camera; not set: automatic (`camera.name` from the config)
    pub camera: Option<String>,
}

impl GuiState {
    /// All values of `config` (camera automatic)
    pub fn of_config(config: &Config) -> Self {
        Self {
            beamformer: Some(config.beamformer.clone()),
            f_lo_hz: Some(config.pipeline.f_lo_hz),
            f_hi_hz: Some(config.pipeline.f_hi_hz),
            skip_transients: Some(config.pipeline.skip_transients),
            db_range: Some(config.display.db_range),
            tracks: Some(config.peaks.enabled),
            min_above_floor_db: Some(config.peaks.min_above_floor_db),
            min_coherence_db: Some(config.peaks.min_coherence_db),
            speech: Some(config.speech.enabled),
            speech_threshold: Some(config.speech.on_threshold),
            camera: None,
        }
    }

    /// Only the values that differ from `base`. Saving only those keeps later edits of
    /// `sound2view.toml` in effect for everything not changed in the GUI.
    pub fn changes_from(&self, base: &Self) -> Self {
        fn changed<T: Clone + PartialEq>(value: &Option<T>, base: &Option<T>) -> Option<T> {
            (value != base).then(|| value.clone()).flatten()
        }
        Self {
            beamformer: changed(&self.beamformer, &base.beamformer),
            f_lo_hz: changed(&self.f_lo_hz, &base.f_lo_hz),
            f_hi_hz: changed(&self.f_hi_hz, &base.f_hi_hz),
            skip_transients: changed(&self.skip_transients, &base.skip_transients),
            db_range: changed(&self.db_range, &base.db_range),
            tracks: changed(&self.tracks, &base.tracks),
            min_above_floor_db: changed(&self.min_above_floor_db, &base.min_above_floor_db),
            min_coherence_db: changed(&self.min_coherence_db, &base.min_coherence_db),
            speech: changed(&self.speech, &base.speech),
            speech_threshold: changed(&self.speech_threshold, &base.speech_threshold),
            camera: changed(&self.camera, &base.camera),
        }
    }

    /// Read `GUI_STATE_PATH`; empty if it does not exist, with a message if it cannot be read
    pub fn load() -> Self {
        let Ok(text) = fs::read_to_string(GUI_STATE_PATH) else {
            return Self::default();
        };
        toml::from_str(&text).unwrap_or_else(|e| {
            eprintln!("{GUI_STATE_PATH} ignored: {e}");
            Self::default()
        })
    }

    /// Write to `GUI_STATE_PATH`; an error is printed, the GUI goes on
    pub fn save(&self) {
        let text = match toml::to_string(self) {
            Ok(text) => text,
            Err(e) => {
                eprintln!("cannot save GUI settings: {e}");
                return;
            }
        };
        let header = "# Written by the sound2view GUI; delete to use sound2view.toml again\n\n";
        if let Err(e) = fs::write(GUI_STATE_PATH, format!("{header}{text}")) {
            eprintln!("cannot write {GUI_STATE_PATH}: {e}");
        }
    }

    /// `config` with the saved values; unchanged if the result would be invalid
    pub fn apply(&self, config: &Config) -> Config {
        let mut applied = config.clone();
        if let Some(beamformer) = &self.beamformer {
            applied.beamformer = beamformer.clone();
        }
        let pipeline = &mut applied.pipeline;
        set(&mut pipeline.f_lo_hz, self.f_lo_hz);
        set(&mut pipeline.f_hi_hz, self.f_hi_hz);
        set(&mut pipeline.skip_transients, self.skip_transients);
        set(&mut applied.display.db_range, self.db_range);
        set(&mut applied.peaks.enabled, self.tracks);
        set(
            &mut applied.peaks.min_above_floor_db,
            self.min_above_floor_db,
        );
        set(&mut applied.peaks.min_coherence_db, self.min_coherence_db);
        set(&mut applied.speech.enabled, self.speech);
        set(&mut applied.speech.on_threshold, self.speech_threshold);

        match applied.validate() {
            Ok(()) => applied,
            Err(e) => {
                eprintln!("{GUI_STATE_PATH} ignored: {e}");
                config.clone()
            }
        }
    }
}

fn set<T>(target: &mut T, value: Option<T>) {
    if let Some(value) = value {
        *target = value;
    }
}

#[cfg(test)]
mod test {
    use super::GuiState;
    use crate::config::Config;

    /// Only changed values are kept, and applying them gives the changed config back.
    #[test]
    fn changes_round_trip() {
        let config = Config::default();
        let base = GuiState::of_config(&config);
        let mut current = base.clone();
        current.db_range = Some(5.0);
        current.camera = Some("USB Camera".to_string());

        let changes = current.changes_from(&base);
        assert_eq!(changes.db_range, Some(5.0));
        assert_eq!(changes.camera.as_deref(), Some("USB Camera"));
        assert_eq!(changes.f_lo_hz, None);
        assert_eq!(changes.beamformer, None);

        let text = toml::to_string(&changes).unwrap();
        let loaded: GuiState = toml::from_str(&text).unwrap();
        assert_eq!(loaded, changes);
        assert_eq!(loaded.apply(&config).display.db_range, 5.0);
    }

    /// Saved values that make the config invalid are ignored as a whole.
    #[test]
    fn invalid_values_ignored() {
        let config = Config::default();
        let state = GuiState {
            db_range: Some(-1.0),
            f_lo_hz: Some(1000.0),
            ..GuiState::default()
        };
        let applied = state.apply(&config);
        assert_eq!(applied.display.db_range, config.display.db_range);
        assert_eq!(applied.pipeline.f_lo_hz, config.pipeline.f_lo_hz);
    }
}
