//! Processing chain after the `FrameSource`: STFT, covariance, beamformer and map output.

use std::{
    collections::VecDeque,
    io,
    ops::{ControlFlow, RangeInclusive},
    time::Instant,
};

use serde::Deserialize;

use crate::{
    BoxError, FrameSource,
    dsp::{
        beam_signal::{BeamSignal, TRACK_SAMPLE_RATE, decimation_factor},
        beamformer::{Beamformer, Steering, coherent_and_received_power, source_fraction},
        covariance::SlidingCovariance,
        stft::{Stft, StftStream},
    },
    geometry::{camera::Camera, mic_array::MicArray, scan_grid::ScanGrid},
    peaks::{PeakConfig, find_peaks},
    tracking::{Track, Tracker, TrackerConfig},
};

/// Processing parameters, e.g. from a config file; the defaults are tested with the UMA-16 at 48 kHz
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PipelineConfig {
    /// STFT length N, hop is N/2
    pub fft_len: usize,
    /// Memory of the sliding covariance in STFT frames (40: about 0.43 s at 48 kHz)
    pub memory_frames: usize,
    /// One map every this many STFT frames (10: about 0.11 s at 48 kHz); `power_map` must take less
    pub map_every: usize,
    /// Lower end of the band summed in the power map, in Hz
    pub f_lo_hz: f64,
    /// Upper end of the band, in Hz
    pub f_hi_hz: f64,
    /// Scan grid width in grid pixels; the height follows from the camera aspect ratio
    pub grid_width: u32,
    /// Focus distance of the scan grid in meters
    pub grid_z: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            fft_len: 1024,
            memory_frames: 40,
            map_every: 10,
            f_lo_hz: 900.0,
            f_hi_hz: 4100.0,
            grid_width: 64,
            grid_z: 2.0,
        }
    }
}

/// Lower end of the upper part of the map band, in Hz: from here on the 12.6 cm array resolves
/// directions (a source 30 degrees off the beam is attenuated), below every beam carries every
/// source. [`Track::source_upper_db`] is measured here.
pub const UPPER_BAND_HZ: f64 = 2000.0;

/// The noise floor of the source check is learned over this many seconds of maps: per map the
/// largest coherent power over a coarse subgrid (every `FLOOR_STRIDE`th cell), of which the
/// lowest is the floor (minimum statistics), so that the quietest moment sets it: the pauses
/// between words of a talker are enough, whatever happens in the rest of the window. Power, not
/// the fraction of the received power:
/// a steady background source (fan) has a fraction near 1 in silence, so no fraction could be
/// above it, but a talker has more power. With uncorrelated noise the coherent power
/// fluctuates around zero, so a single point (such as the map maximum) is useless in dB; the
/// largest of about 30 points is a stable statistic.
pub const FLOOR_WINDOW_S: f64 = 60.0;
const FLOOR_STRIDE: usize = 8;

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("invalid pipeline config: {0}")]
    InvalidConfig(String),
    #[error("reading the source: {0}")]
    Source(#[from] io::Error),
    #[error("map handler: {0}")]
    MapHandler(#[source] BoxError),
}

impl PipelineConfig {
    /// Error if a value cannot work, e.g. an empty band or an odd FFT length.
    pub fn validate(&self, sample_rate: f64) -> Result<(), PipelineError> {
        let invalid = |message: &str| Err(PipelineError::InvalidConfig(message.to_string()));

        if self.fft_len < 2 || !self.fft_len.is_multiple_of(2) {
            return invalid("fft_len must be even and at least 2");
        }
        if self.memory_frames == 0 || self.map_every == 0 {
            return invalid("memory_frames and map_every must be at least 1");
        }
        if !(0.0 <= self.f_lo_hz && self.f_lo_hz < self.f_hi_hz) {
            return invalid("band: 0 <= f_lo_hz < f_hi_hz required");
        }
        if self.f_lo_hz >= sample_rate / 2.0 {
            return invalid("band: f_lo_hz must be below sample_rate/2");
        }
        if self.grid_width == 0 || self.grid_z <= 0.0 {
            return invalid("grid_width must be at least 1 and grid_z > 0");
        }
        Ok(())
    }
}

/// Everything after the `FrameSource`: STFT, covariance, beamformer, peaks, tracks and output
pub struct Pipeline<const M: usize, B: Beamformer<M>> {
    stream: StftStream<M>,
    cov: SlidingCovariance<M>,
    beamformer: B,
    /// Bins summed in the power map, and the ones of them from `UPPER_BAND_HZ` on (all of them
    /// if the band ends below)
    bins: RangeInclusive<usize>,
    upper_bins: RangeInclusive<usize>,
    /// Steering of the scan grid, for the source check of peaks and tracks
    steering: Steering<M>,
    /// Largest coherent power over the subgrid per map (linear, > 0), for the last
    /// `floor_window` maps, see `FLOOR_WINDOW_S`
    floor_samples: VecDeque<f64>,
    floor_window: usize,
    /// One map every this many STFT frames
    map_every: usize,
    /// Layout of the power map
    layout: MapLayout,
    peaks: PeakConfig,
    tracker: Tracker,
    /// Audio toward each track, by track id; empty if the sample rate has no integer
    /// decimation factor to `TRACK_SAMPLE_RATE`
    beams: Vec<(u64, BeamSignal<M>)>,
    track_audio: Option<TrackAudio<M>>,
}

/// What a new track needs to get its beam signal: grid cell -> point -> mic distances
struct TrackAudio<const M: usize> {
    mic_array: MicArray<M>,
    /// Camera at scan grid resolution, see [`ScanGrid::new`]
    grid_camera: Camera,
    grid_z: f64,
    fft_len: usize,
    stft_scale: f64,
    sample_rate: f64,
}

impl<const M: usize> TrackAudio<M> {
    /// Mic distances to the point of grid cell (col, row) at the focus distance
    fn distances(&self, track: &Track) -> [f64; M] {
        let point = self.grid_camera.unproject(track.col, track.row, self.grid_z);
        self.mic_array.get_distance_from(&point)
    }
}

#[derive(Clone)]
pub struct MapLayout {
    pub camera: Camera,
    pub grid_width: usize,
    pub grid_height: usize,
}

impl<B: Beamformer<16>> Pipeline<16, B> {
    /// UMA-16 pipeline for `sample_rate` in Hz; the scan grid covers the field of view of `camera`.
    ///
    /// `make_beamformer` builds the beamformer from the steering of this pipeline's scan grid.
    /// Each map's peaks (`peaks`) are followed as tracks (`tracking`), each with the audio from
    /// its direction if `sample_rate` is an integer multiple of [`TRACK_SAMPLE_RATE`].
    ///
    /// Error if `config` or `tracking` is invalid, see [`PipelineConfig::validate`].
    pub fn new(
        sample_rate: f64,
        config: &PipelineConfig,
        peaks: &PeakConfig,
        tracking: &TrackerConfig,
        camera: Camera,
        make_beamformer: impl FnOnce(Steering<16>) -> B,
    ) -> Result<Self, PipelineError> {
        config.validate(sample_rate)?;
        tracking.validate().map_err(PipelineError::InvalidConfig)?;

        let scan_grid = ScanGrid::new(&camera, config.grid_width, config.grid_z);

        let n_bins = config.fft_len / 2 + 1;
        let df = sample_rate / config.fft_len as f64;
        // The upper end is clipped to the last bin (f_s/2)
        let bin_lo = (config.f_lo_hz / df).round() as usize;
        let bin_hi = ((config.f_hi_hz / df).round() as usize).min(n_bins - 1);
        let bins = bin_lo..=bin_hi;
        let bin_upper = (UPPER_BAND_HZ / df).round() as usize;
        let upper_bins = if bin_upper > bin_lo && bin_upper <= bin_hi {
            bin_upper..=bin_hi
        } else {
            bins.clone()
        };

        println!(
            "Grid {}x{}, bins {bins:?}, memory {} frames, map every {} frames",
            scan_grid.width(),
            scan_grid.height(),
            config.memory_frames,
            config.map_every
        );

        let stft = Stft::new(config.fft_len);
        let track_audio = match decimation_factor(sample_rate) {
            Some(_) => Some(TrackAudio {
                mic_array: MicArray::uma16(),
                grid_camera: Camera {
                    intrinsics: camera.intrinsics.scaled(config.grid_width),
                    mount: camera.mount.clone(),
                },
                grid_z: config.grid_z,
                fft_len: config.fft_len,
                stft_scale: stft.scale(),
                sample_rate,
            }),
            None => {
                println!(
                    "No track audio: {sample_rate} Hz is no integer multiple of {TRACK_SAMPLE_RATE} Hz"
                );
                None
            }
        };

        let steering = Steering::new(&scan_grid, &MicArray::uma16(), sample_rate, config.fft_len);
        Ok(Self {
            stream: StftStream::new(stft),
            cov: SlidingCovariance::new(n_bins, config.memory_frames),
            beamformer: make_beamformer(steering.clone()),
            bins,
            upper_bins,
            steering,
            floor_samples: VecDeque::new(),
            floor_window: (FLOOR_WINDOW_S * sample_rate
                / (config.map_every * config.fft_len / 2) as f64)
                .ceil()
                .max(1.0) as usize,
            map_every: config.map_every,
            layout: MapLayout {
                camera,
                grid_width: scan_grid.width(),
                grid_height: scan_grid.height(),
            },
            peaks: peaks.clone(),
            tracker: Tracker::new(tracking.clone()),
            beams: Vec::new(),
            track_audio,
        })
    }
}

impl<const M: usize, B: Beamformer<M>> Pipeline<M, B> {
    /// Index of the grid cell nearest to (col, row)
    fn point(&self, col: f64, row: f64) -> usize {
        let [width, height] = [self.layout.grid_width, self.layout.grid_height];
        let col = (col.round().max(0.0) as usize).min(width - 1);
        let row = (row.round().max(0.0) as usize).min(height - 1);
        row * width + col
    }

    /// Power arriving coherently from `point` over the map band, see
    /// [`coherent_and_received_power`]
    fn coherent_power(&self, point: usize) -> f64 {
        coherent_and_received_power(self.cov.estimate(), self.bins.clone(), &self.steering, point).0
    }

    /// Coherent power from the cell nearest to (col, row) in dB above `floor`
    fn above_floor_db(&self, col: f64, row: f64, floor: f64) -> f64 {
        10.0 * (self.coherent_power(self.point(col, row)).max(1e-30) / floor).log10()
    }

    /// Noise floor (linear): the lowest of `floor_samples`, infinite without samples
    fn floor(&self) -> f64 {
        self.floor_samples.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Continue the tracks with the peaks of `power`, measure each track's coherent power above
    /// the noise floor, hand each track the audio of its beam since the previous map, then
    /// point the beams at the new track positions.
    fn update_tracks(&mut self, power: &[f64]) {
        let size = [self.layout.grid_width, self.layout.grid_height];
        let peaks = find_peaks(power, size, &self.peaks);
        // The current map is part of the floor window; no sample without any coherent power
        // (e.g. a source that delivers zeros)
        let [width, height] = size;
        let sample = (FLOOR_STRIDE / 2..height)
            .step_by(FLOOR_STRIDE)
            .flat_map(|row| {
                let cols = (FLOOR_STRIDE / 2..width).step_by(FLOOR_STRIDE);
                cols.map(move |col| row * width + col)
            })
            .map(|point| self.coherent_power(point))
            .fold(f64::NEG_INFINITY, f64::max);
        if sample > 0.0 {
            if self.floor_samples.len() >= self.floor_window {
                self.floor_samples.pop_front();
            }
            self.floor_samples.push_back(sample);
        }
        let floor = self.floor();
        self.tracker.update(&peaks);

        let source_dbs: Vec<(f64, f64)> = self
            .tracker
            .tracks()
            .iter()
            .map(|track| {
                let point = self.point(track.col, track.row);
                let cov = self.cov.estimate();
                let upper = source_fraction(cov, self.upper_bins.clone(), &self.steering, point);
                let above = self.above_floor_db(track.col, track.row, floor);
                (above, 10.0 * upper.max(1e-10).log10())
            })
            .collect();
        let tracks = self.tracker.tracks_mut();
        for (track, (source_db, source_upper_db)) in tracks.iter_mut().zip(source_dbs) {
            track.source_db = source_db;
            track.source_upper_db = source_upper_db;
        }

        self.beams.retain(|(id, _)| tracks.iter().any(|track| track.id == *id));
        let Some(audio) = &self.track_audio else {
            return;
        };
        for track in tracks {
            let distances = audio.distances(track);
            match self.beams.iter_mut().find(|(id, _)| *id == track.id) {
                Some((_, beam)) => {
                    track.samples = beam.take_output();
                    beam.steer(&distances);
                }
                None => {
                    track.samples = Vec::new();
                    let beam = BeamSignal::new(
                        audio.fft_len,
                        audio.stft_scale,
                        audio.sample_rate,
                        &distances,
                    );
                    self.beams.push((track.id, beam));
                }
            }
        }
    }
}

/// Read `source` through the pipeline; one power map every `map_every` STFT frames.
///
/// The covariance is a sliding average over about `memory_frames` frames, so consecutive maps
/// overlap.
/// Each map is passed to `on_map` together with its [`MapLayout`] and the current tracks (each
/// with its audio since the previous map). Stops when the source ends (`Ok(0)`), when `on_map`
/// returns `Break`, or on the first error from `source` or `on_map`.
pub fn run<S: FrameSource<M>, B: Beamformer<M>, const M: usize>(
    source: &mut S,
    mut pipeline: Pipeline<M, B>,
    mut on_map: impl FnMut(&[f64], &MapLayout, &[Track]) -> Result<ControlFlow<()>, BoxError>,
) -> Result<(), PipelineError> {
    let mut buf = vec![0.0; 1024 * M];
    let mut total_frames = 0;
    let mut out = vec![0.0; pipeline.beamformer.n_points()];
    let mut map_index = 0;
    // STFT frames since the last map
    let mut new_frames = 0;

    loop {
        let frames = source.read(&mut buf)?;
        if frames == 0 {
            break;
        }
        total_frames += frames;

        // Closures capture disjoint fields, so `stream`, `cov` and `beams` can be borrowed
        // together
        pipeline.stream.push(&buf[..frames * M], |snapshots, _l| {
            pipeline.cov.accumulate(snapshots);
            for (_, beam) in &mut pipeline.beams {
                beam.push_frame(snapshots);
            }
            new_frames += 1;
        });

        if new_frames >= pipeline.map_every {
            let start = Instant::now();
            pipeline
                .beamformer
                .power_map(pipeline.cov.estimate(), pipeline.bins.clone(), &mut out);
            println!(
                "map {map_index}: power_map {:.1} ms",
                start.elapsed().as_secs_f64() * 1e3
            );

            new_frames = 0;
            map_index += 1;

            pipeline.update_tracks(&out);
            let flow = on_map(&out, &pipeline.layout, pipeline.tracker.tracks())
                .map_err(PipelineError::MapHandler)?;
            if flow.is_break() {
                break;
            }
        }
    }

    println!(
        "read {total_frames} frames ({:.2} s, {M} channels), {map_index} maps",
        total_frames as f64 / source.sample_rate()
    );
    Ok(())
}

#[cfg(test)]
mod test {
    use std::{f64::consts::PI, io, ops::ControlFlow};

    use super::{Pipeline, PipelineConfig, run};
    use crate::{
        FrameSource, Point3,
        dsp::{
            beam_signal::TRACK_SAMPLE_RATE,
            beamformer::clean_sc::CleanSc,
        },
        geometry::mic_array::MicArray,
        peaks::PeakConfig,
        source::simulator::{
            Simulator,
            scene::{MicNoiseParams, Scene, Source},
            signal::{Signal, Sine},
        },
        test_util::{SAMPLE_RATE, test_camera},
        tracking::{Track, TrackerConfig},
    };

    /// Amplitude of the component at `freq` (least squares fit of sine and cosine)
    fn amplitude_at(signal: &[f32], freq: f64) -> f64 {
        let (mut c, mut s) = (0.0, 0.0);
        for (n, &y) in signal.iter().enumerate() {
            let phase = 2.0 * PI * freq * n as f64 / TRACK_SAMPLE_RATE;
            c += y as f64 * phase.cos();
            s += y as f64 * phase.sin();
        }
        2.0 * (c * c + s * s).sqrt() / signal.len() as f64
    }

    /// Mic noise alone at `level_db` for `duration_s`
    fn noise_scene(level_db: f64, duration_s: f64, seed: u64) -> Scene {
        Scene {
            sample_rate: SAMPLE_RATE,
            duration_s,
            mic_noise_params: MicNoiseParams { seed, level_db },
            sources: Vec::new(),
        }
    }

    /// `0` until it ends, then `1`
    struct Chain(Simulator<16>, Simulator<16>);

    impl FrameSource<16> for Chain {
        fn sample_rate(&self) -> f64 {
            self.0.sample_rate()
        }

        fn read(&mut self, buf: &mut [f64]) -> io::Result<usize> {
            match self.0.read(buf)? {
                0 => self.1.read(buf),
                frames => Ok(frames),
            }
        }
    }

    /// Two sines on the focus plane after two quiet seconds (the noise floor needs a quiet map):
    /// two confirmed tracks at the sources, each with audio in which its own source dominates.
    /// CLEAN-SC, because the broad DAS main lobes of the two sources overlap and pull both peaks
    /// about two cells toward each other.
    #[test]
    fn tracks_follow_sources_with_their_audio() {
        let config = PipelineConfig::default();
        let sources = [
            (Point3::new(0.6, 0.2, config.grid_z), 3000.0),
            (Point3::new(-0.6, 0.2, config.grid_z), 3500.0),
        ];
        let scene = Scene {
            sample_rate: SAMPLE_RATE,
            duration_s: 2.0,
            mic_noise_params: MicNoiseParams {
                seed: 1,
                level_db: 20.0,
            },
            sources: sources
                .iter()
                .map(|&(pos, freq)| Source {
                    pos,
                    signal: Signal::Sine(Sine::new(1.0, freq, 0.0)),
                })
                .collect(),
        };
        let quiet = Simulator::new(noise_scene(20.0, 2.0, 3), MicArray::uma16()).unwrap();
        let mut sim = Chain(quiet, Simulator::new(scene, MicArray::uma16()).unwrap());
        let camera = test_camera();
        let pipeline = Pipeline::new(
            SAMPLE_RATE,
            &config,
            &PeakConfig::default(),
            &TrackerConfig::default(),
            camera.clone(),
            |steering| CleanSc::new(steering, 0.9, 10, 20.0, 1.5, true, true),
        )
        .unwrap();

        // Last map's tracks, and all audio per track id
        let mut last: Vec<Track> = Vec::new();
        let mut audio: Vec<(u64, Vec<f32>)> = Vec::new();
        run(&mut sim, pipeline, |_power, layout, tracks| {
            for track in tracks {
                match audio.iter_mut().find(|(id, _)| *id == track.id) {
                    Some((_, samples)) => samples.extend_from_slice(&track.samples),
                    None => audio.push((track.id, track.samples.clone())),
                }
            }
            last = tracks.to_vec();
            assert_eq!(layout.grid_width, config.grid_width as usize);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();

        let min_db = PeakConfig::default().min_above_floor_db;
        let confirmed: Vec<&Track> =
            last.iter().filter(|t| t.confirmed && t.source_db >= min_db).collect();
        assert_eq!(confirmed.len(), 2, "tracks: {last:?}");

        let intrinsics = &camera.intrinsics;
        let grid_width = config.grid_width as f64;
        let grid_height = (grid_width * intrinsics.height as f64 / intrinsics.width as f64).round();
        for (pos, freq) in sources {
            // Source in grid cells, as `Intrinsics::scaled`
            let (u, v) = camera.project(&pos);
            let col = (u + 0.5) * grid_width / intrinsics.width as f64 - 0.5;
            let row = (v + 0.5) * grid_height / intrinsics.height as f64 - 0.5;
            let track = confirmed
                .iter()
                .min_by(|a, b| {
                    let d = |t: &Track| (t.col - col).hypot(t.row - row);
                    d(a).total_cmp(&d(b))
                })
                .unwrap();
            let dist = (track.col - col).hypot(track.row - row);
            assert!(dist < 1.0, "track at ({}, {}), source at ({col}, {row})", track.col, track.row);

            // Each source gets about half of the received power, far above the quiet maps
            let above = track.source_db;
            assert!(above > 10.0, "track {}: {above} dB above the floor", track.id);
            let upper = track.source_upper_db;
            assert!(upper > -5.0, "track {}: {upper} dB coherent above 2 kHz", track.id);

            let (_, samples) = audio.iter().find(|(id, _)| *id == track.id).unwrap();
            // At least a second of audio
            assert!(samples.len() > TRACK_SAMPLE_RATE as usize, "{} samples", samples.len());
            let other = sources.iter().find(|(p, _)| *p != pos).unwrap().1;
            let own_db = 20.0 * (amplitude_at(samples, freq) / amplitude_at(samples, other)).log10();
            // About 30 dB and 12 dB: depends on where the other source falls in this beam's
            // sidelobe pattern (near a null at 3.5 kHz, not at 3 kHz)
            assert!(own_db > 8.0, "track {}: own source {own_db:.1} dB above the other", track.id);
        }
    }

    /// Mic noise alone, loud: every map has a maximum, but the coherent power from it stays at
    /// the noise floor, so no track would be shown.
    #[test]
    fn noise_alone_gives_no_tracks() {
        let mut sim = Simulator::new(noise_scene(80.0, 6.0, 2), MicArray::uma16()).unwrap();
        let pipeline = Pipeline::new(
            SAMPLE_RATE,
            &PipelineConfig::default(),
            &PeakConfig::default(),
            &TrackerConfig::default(),
            test_camera(),
            |steering| CleanSc::new(steering, 0.9, 10, 20.0, 1.5, true, true),
        )
        .unwrap();

        let min_db = PeakConfig::default().min_above_floor_db;
        let mut maps = 0;
        run(&mut sim, pipeline, |_power, _layout, tracks| {
            assert!(tracks.iter().all(|t| t.source_db < min_db), "map {maps}: {tracks:?}");
            maps += 1;
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
        assert!(maps > 50, "{maps} maps");
    }

    /// A steady source from the start is the noise floor (no dot: a fan in a real room), a
    /// source 14 dB louder that appears later gets a track.
    #[test]
    fn steady_background_is_the_floor() {
        let fan = Point3::new(-0.5, -0.3, 2.0);
        let talker = Point3::new(0.6, 0.2, 2.0);
        let source = |pos, amplitude| Source {
            pos,
            signal: Signal::Sine(Sine::new(amplitude, 3000.0, 0.0)),
        };
        let scene = |sources: Vec<Source>, seed| Scene {
            sample_rate: SAMPLE_RATE,
            duration_s: 2.0,
            mic_noise_params: MicNoiseParams {
                seed,
                level_db: 40.0,
            },
            sources,
        };
        let only_fan = Simulator::new(scene(vec![source(fan, 0.2)], 4), MicArray::uma16()).unwrap();
        let both = scene(vec![source(fan, 0.2), source(talker, 1.0)], 5);
        let mut sim = Chain(only_fan, Simulator::new(both, MicArray::uma16()).unwrap());
        let pipeline = Pipeline::new(
            SAMPLE_RATE,
            &PipelineConfig::default(),
            &PeakConfig::default(),
            &TrackerConfig::default(),
            test_camera(),
            |steering| CleanSc::new(steering, 0.9, 10, 20.0, 1.5, true, true),
        )
        .unwrap();

        let min_db = PeakConfig::default().min_above_floor_db;
        let (mut maps, mut confirmed) = (0, Vec::new());
        run(&mut sim, pipeline, |_power, _layout, tracks| {
            // The first second is the fan alone (18 maps); then the talker starts
            if maps < 17 {
                assert!(tracks.iter().all(|t| t.source_db < min_db), "map {maps}: {tracks:?}");
            }
            confirmed = tracks
                .iter()
                .filter(|t| t.confirmed && t.source_db >= min_db)
                .cloned()
                .collect();
            maps += 1;
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
        assert_eq!(confirmed.len(), 1, "{confirmed:?}");
        let track = &confirmed[0];
        // +x is left in the image (`Mount` turns the camera 180 degrees about z): the talker at
        // x = 0.6 is at about column 25, the fan at -0.5 at about 38
        assert!(track.col < 30.0, "track at col {}", track.col);
        assert!(track.source_db > 10.0, "{} dB above the floor", track.source_db);
    }
}
