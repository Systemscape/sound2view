# UMA-16 acoustic camera (Rust)

## Working mode

- I implement the code myself. Point me in the right direction, review, explain; don't write whole modules unless I ask.
- Keep answers short.
- In German text: "die Crate".
- English in code, comments, docs and this file: American spelling (center, normalize, meter, neighbor, visualization).
- No Unicode Greek symbols in code or comments: write `tau_m`, `sigma`, `omega`, `phi`, `PI`, `l` (frame index), `sum`. Greek is fine in this file and the blog. Open: maybe allow Unicode in doc comments (`power_map` already uses it).
- `Point3` for positions, `Vector3` for directions, offsets and differences.

## Goal

Acoustic camera with the miniDSP UMA-16 v2 microphone array. Processing in Rust on a Mac mini (Linux or macOS, not decided). Start with a **simulator**, then real capture, then beamforming, then camera overlay. Also documented as a blog series on systemscape.com (Zola).

## Hardware facts (UMA-16 v2)

- 16 Knowles SPH1668LM4H MEMS mics, 4×4 uniform rectangular array, **42 mm pitch** (aperture ≈ 126 mm).
- XMOS USB interface (MCHStreamer Lite) does PDM→PCM. USB Audio Class 2, class-compliant: no driver on Linux (`snd-usb-audio`).
- 24 bit; 8, 11.025, 12, 16, 32, 44.1, 48 kHz. 2 output channels show up but are not physically connected (ignore).
- Check on hardware: `arecord -l`, `cat /proc/asound/card*/stream0` (S32_LE vs S24_3LE). With S32, the 24 bits are left-justified (`x >> 8`).
- **Mic → USB channel mapping** verified on hardware with the camera overlay (x and y). It equals Acoular `minidsp_uma-16_mirrored.xml`; the newer `minidsp_uma-16.xml` (≥ 24.03, "corrected in x") is mirrored in x relative to our convention (row 0 = top, col 0 = left, viewed from the source side).
- **Camera:** icspring USB camera, mounted in the center hole of the array (default `Mount`). Selected by name (`camera.name` in `sound2view.toml`, `CameraSource::open_by_name` tries every device whose name matches; the V4L2 index, e.g. 6, changes on replug). Formats: MJPEG 640×480, 800×600, 1280×720, 1920×1080 all @60; YUYV only at low fps (1080p@5).
  - **Measured (coarse, tape on a wall at D = 1.66 m, 1920×1080):** W ≈ 2.9 m → HFOV ≈ 82°, re-measured W = 2.83 m at D = 1.6 m → 83°; H ≈ 1.65 m → VFOV ≈ 53°. Consistent with square pixels: fx = fy ≈ 1085 px. `HFOV_DEG = 83`, principal point centered.
  - Existing OpenCV calibration (rational model, 0.23 px reprojection error: fx = 1918.95, fy = 1888.03, cx = 762.08, cy = 715.41, HFOV ≈ 53°) is **wrong**: contradicts the measurement; principal point ≈ (−197, +176) px off center, distortion coefficients large and nearly canceling (k1 ≈ k4, k2 ≈ k5), probably overfit. Re-calibrate (checkerboard into the corners, standard 5-coefficient model) before relying on distortion or principal point.
  - Distortion ignored for now. Later: undistort per scan-grid pixel once (map in raw image coordinates, no per-frame remap).
  - Beamforming in the visible region: at 83° HFOV the DAS main lobe at 3 kHz (≈ 34°) covers ≈ 40 % of the image width. Grating lobes of in-image sources stay outside the image up to ≈ 6 kHz; above ≈ 5 kHz strong sources outside the image can alias in.
- Physics limits: aliasing-free below c/(2·pitch) ≈ 4.1 kHz; resolution ≈ λ/D (≈ 39° at 4 kHz); far field from ≈ 0.74 m at 4 kHz; max inter-mic delay ≈ 0.52 ms ≈ 25 samples at 48 kHz.

## Decisions

- **Capture:** `alsa` crate on Linux (`hw:` device, RWInterleaved). If the Mac mini runs macOS → `cpal` (CoreAudio). Keep capture behind the `FrameSource` trait so both are swappable.
- **Data types:** `i32` at the capture boundary, `f64` from the first processing step (`FrameSource::read` fills `f64`). `f64` inside the simulator.
- **Linear algebra:** `nalgebra` (not faer). Static sizes with const generic `M`: `SVector<Complex64, M>`, `SMatrix<Complex64, M, M>`. Type aliases in one module.
- **Later crates:** `realfft` (STFT), `hound` (WAV dump), `rand` (chacha feature) + `rand_distr` (seeded noise, already in use), `rerun` (3D debug viewer, optional).

## Interfaces (`src/lib.rs`)

```rust
pub trait FrameSource<const M: usize> {
    fn sample_rate(&self) -> f64;
    fn channels(&self) -> usize { M }
    /// Interleaved, len = frames * channels; returns frames read.
    fn read(&mut self, buf: &mut [f64]) -> std::io::Result<usize>;
}

pub trait ImageSource {
    type Error: std::error::Error + Send + Sync + 'static;
    fn next_frame(&mut self) -> Result<VideoFrame, Self::Error>;
}

pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    /// RGBA8, row-major, origin top-left; len = width * height * 4.
    pub pixels: Vec<u8>,
}

```

- Static dispatch only: the source (mic, WAV, simulator) and the beamformer are chosen once; a `match` calls the generic `run<S: FrameSource<M>, B: Beamformer<M>, const M>`. Switching the beamformer from the GUI restarts `run` with the other type (source is borrowed, survives); no enum, no `dyn`.
- M is a compile-time constant on the trait (matches `SMatrix<_, M, M>` in the pipeline). `sample_rate` is `f64`; hardware APIs use integer rates, convert at the boundary.
- `read` returns frames, not bytes. `Ok(0)` = end of stream (`std::io::Read` convention). `buf.len()` must be a multiple of M (`debug_assert`).
- No timestamp on `VideoFrame`: live view shows the latest frame. Add a `Duration` since stream start (not `Instant`) when recording or syncing. Audio map lags by roughly half the averaging window.

## Current code state

Crate and repo name: `sound2view` (was `audio2video-rs`; local folder may still have the old name). Library crate owns all modules. Always use `--release`: debug is too slow for live capture (power map ≈ 3 s vs 85 ms) and for Q = 200 noise.

Workspace (root `Cargo.toml`, resolver 3, `default-members = ["crates/app"]`, so `cargo run --release -- mic` works from the root; `cargo test --workspace` for all tests):
- `crates/core` (package `sound2view-core`, lib `sound2view_core`): everything below except `main.rs`. No GUI dependencies.
- `crates/app` (package `sound2view-app`, binary `sound2view`): `main.rs` (CLI), `config.rs`, `gui.rs`, `speech.rs`, `theme.rs`. Depends on the private `voice-activity-detection` crate (git, SSH, pinned `rev`; `.cargo/config.toml` sets `net.git-fetch-with-cli` so the SSH agent is used). Its package also builds burn and eframe 0.36, which is why speech detection lives in the app, not in core. CI needs SSH read access to that repo (not set up yet).

Module layout of `crates/core/src` (`foo.rs` + `foo/` style, no `mod.rs`); `main.rs` lives in `crates/app/src`:

```
src/
  lib.rs, peaks.rs, pipeline.rs, render.rs, test_util.rs, tracking.rs
  geometry/   camera, mic_array, scan_grid
  source/     capture, simulator/ (scene, signal, reference), video/ (v4l on Linux, avfoundation on macOS)
  dsp/        stft, covariance, beam_signal, beamformer/ (delay_and_sum, functional, clean_sc)
```

Logo: `assets/` (`logo.svg`, `logo-small.svg` for ≤ 32 px, `logo-wordmark.svg`, PNGs rendered with resvg; `logo-256.png` is the GUI window icon via `include_bytes!` in `main.rs`, see `assets/README.md`).

Dependencies point one way: `geometry` ← `dsp` ← `pipeline` → `render`; `source` only needs `geometry`.

- `main.rs`: CLI. `sound2view [scene.toml]` (default `config.toml` = simulator scene), `mic [maps]` (UMA-16 live, `MIC_MAPS` 20 PNGs unless given), `probe`, `camprobe`, `camshot`, `gui`. Loads `sound2view.toml` (`config::CONFIG_PATH`) for all modes except the probes. `type BoxError = Box<dyn Error + Send + Sync>`.
  - `run_configured(&mut source, sample_rate, &config, on_map) -> Result<(), PipelineError>`: builds the `Pipeline` with `config.pipeline`, `config.camera.camera()` and the configured beamformer (`match` on `Method`: `Das::new(.., true, TrueLocation)`, `Functional::new(.., nu, functional_noise_removal, functional_equal_bins)` or `CleanSc::new(.., clean_gain, clean_max_iter, clean_stop_db, clean_beam_cells, true)`), then `run`. One `run` instantiation per method. Used by CLI and GUI; the CLI also uses the configured method (default CLEAN-SC). Core errors convert into `main`'s `Box<dyn Error>` with plain `?`.
- `config.rs` (app): `Config { audio, camera, pipeline: PipelineConfig (core), beamformer, display, peaks, tracking, speech }`, serde with `#[serde(default, deny_unknown_fields)]` everywhere: file, sections and entries optional, typos are errors. `Config::load(path)`: missing file → defaults (prints a note). `AudioConfig { device, sample_rate }`, `CameraConfig { name, width, height, fps, hfov_deg }` + `camera()` (pinhole via `from_hfov`, default `Mount`), `Method` (`das`/`functional`, `ALL`, `label`), `BeamformerConfig { method, nu }`, `DisplayConfig { db_range }`. Example with all defaults: `sound2view.toml` in the root.
- `gui.rs` (eframe 0.37: `App::ui(&mut self, ui, frame)`, panels via `egui::Panel::top(..).show(ui, ..)` and `CentralPanel::default().show(ui, ..)`; `show_inside` is deprecated). `sound2view gui`, window starts maximized (`ViewportBuilder::with_maximized`). `App::new(cc, config)`.
  - Camera thread (`spawn_camera`): opens the camera selected in the GUI (`CameraChoice::Auto` = `CameraSource::open_by_name(camera.name)`, or `CameraChoice::Index(i)` = `CameraSource::new(i)`) inside the thread, `sync_channel(1)` + `try_send` (drops frames), `request_repaint` per frame. Refreshes `list_cameras()` on every open attempt (also while missing, so plugged-in cameras show up). Reopens immediately when the selection changes (old camera dropped first), retries every 2 s (`RETRY`; `wait_for_other_choice` polls the selection every 100 ms meanwhile), reopens after 10 consecutive frame errors (`CAMERA_MAX_FAILED_FRAMES`).
  - `CameraShared { choice: Mutex<CameraChoice>, status: Mutex<DeviceStatus>, cameras: Mutex<Vec<CameraEntry>> }` in an `Arc`, shared by GUI and camera thread. `camera_control()` in the top bar: "Camera" `ComboBox` with "Auto (<camera.name>)" plus every listed camera as "name (index)"; a change sets the choice and status `Opening`. Unfiltered: on Linux the icspring metadata node is listed too (selecting it shows the open error). `hfov_deg` belongs to the configured camera; other cameras only fit the overlay with the same field of view.
  - `DeviceStatus` (`Opening`, `Missing(error)`, `Running`, `Lost(error)`) for camera and mic, each in an `Arc<Mutex<..>>`, set by its thread via `set_status` (with `request_repaint`). Top bar: dot + short text per device, full message as tooltip. Camera message centered over the dimmed last frame (or alone before the first frame); the map overlay is hidden while the mic is not `Running` (stale map).
  - Audio thread (`spawn_audio`): outer loop opens the mic (`Missing` + retry every 2 s on failure). Inner loop: read `PipelineSettings { beamformer, pipeline }` from an `Arc<Mutex<..>>`, put them into its `Config` copy, `run_configured`. `on_map` runs the speech detector (loaded once per thread, reset per run) and sends `MapData { size, power, tracks, speech }` (raw power, tracks without audio), returns `Break` when the settings changed (restart with the mic kept open, ALSA `hw:` is exclusive) or the GUI closed (thread ends). A pipeline error (e.g. mic unplugged: `CaptureError` via the source) drops the mic, sets `Lost(error)` and reopens after 2 s.
  - `theme.rs` (app): dark theme applied in `App::new` (`theme::apply`, forces `Theme::Dark`, `all_styles_mut`). Palette constants: `BACKGROUND` #0f1113 (behind the image), `PANEL` #181b1f (top bar), `TEXT`, `TEXT_WEAK`, `ACCENT` amber #f5a524 (matches the overlay), `OK`, `ERROR`; 6 px radius, larger text (body 15) and padding.
  - `ui`: top bar (`Frame` with `PANEL`): title, `controls()`: beamformer `ComboBox`, the selected method's parameter (functional: nu slider 1–100, log; CLEAN-SC: "Sources" = `clean_max_iter` 1–30; DAS: none), band sliders f_lo/f_hi (100–6000 Hz, log, each limited by the other, min width 100 Hz), overlay dB range slider (1–20 dB); mic and camera status right-aligned. Pipeline settings are copied to the shared settings only while no pointer button is down (dragging does not restart the pipeline on every step). The overlay RGBA is computed in the GUI (`render::map_rgba` with the current dB range) from the last `MapData`, so the dB range applies immediately, also between maps. Central panel (`BACKGROUND`, 12 px margin) paints camera and map texture (`TextureOptions::LINEAR`) with `painter.image` on the same rect: largest rect with the image aspect ratio, centered in the panel. GUI changes are not written back to the file.
- `speech.rs` (app): speech detection per track with voice-activity-detection (`Preprocessor::push(&[f32])` at 16 kHz, 62 features per 10 ms frame, `Model::predict`, an MLP on burn). `SpeechConfig { enabled, model "model/speech_detector.json" (copy of the classifier's model, relative to the working directory), on_threshold 0.95, hold_ms 500 (was 2000: snapping after speech created tracks that briefly scored as speech and stayed green for 2 s), smoothing_ms 50, min_onset_ms 80, echo_correlation 0.2, echo_margin_db 6, max_shift_cells 3, gain_db 36, min_level_db -55 }` (`[speech]`, `validate`: 0 < on ≤ 1). `SpeechDetector::load`, `reset()` per pipeline run (ids start over), `update(&tracks)` (one `Preprocessor` and `Hysteresis` per track id, gone ids forgotten), `status(id)`/`statuses()` -> `SpeechStatus { speech, probability }`. Track audio × `gain_db` before the classifier: the model and the −55 dB gate expect microphone levels (speech ≈ −25 dBFS), UMA-16 speech at 1–2 m is ≈ −60 dBFS; without the gain every frame was gated (p = 0), measured on a recording: mean p 0.68 raw (gate off), 0.91 at +36 dB, no frame gated. `Hysteresis` (constants in `HysteresisParams`): level gate (p = 0 below `min_level_db`), smoothed p += w·(p − smoothed) with w = 1 − exp(−10 ms/τ), speech starts when the smoothed p has been ≥ `on_threshold` for `min_onset_ms` and the frame's own p is too (own p, so the smoothing tail cannot restart it; onset: synthetic snaps reach 0.7 for one 10 ms frame at most, never smoothed, speech stays above for 50 ms to over 1 s), ends `hold_ms` after the last frame whose own (unsmoothed) p ≥ `on_threshold` (pauses between words stay green). Only such frames re-arm the hold: with an `off_threshold` (0.2) re-arming it, clapping or snapping after speech (same track, p ≈ 0.3–0.5) kept the track green indefinitely. The anchor (track position at the last such frame; unsmoothed, because the smoothing tail would anchor the source that takes over) ends speech at once when the track moves more than `max_shift_cells` from it: snapping next to the speaker was taken over by the speaker's track (tracker gate 6 cells) and inherited the green. Echo check after each update (`echo_source`, decided on the speech flags before any is cleared): a speech track is an echo of another speech track that has ≥ `echo_margin_db` more speech energy (`TrackSpeech::speech_energy` = Σ p·rms² over the classifier frames of the last 0.3 s, `ECHO_WINDOW`) and whose audio correlates ≥ `echo_correlation` (`max_correlation`, normalized, ±20 ms `ECHO_MAX_LAG`, raw samples in `TrackSpeech::recent`): speech off, hold 0, `SpeechStatus::echo_of = Some(id)`, GUI label "echo of #N". Why speech energy: below ≈ 1.5 kHz (most speech energy) the 12.6 cm array has almost no directivity, so every beam carries much of every talker; a snapping track next to the speaker is speech-like through the leaked speech, louder (map level and beam power: an earlier version ranked by those and made the speaker the echo of the snapping) and correlates only ≈ 0.35 with the speaker (snaps dilute it; correlating only in speech-like frames did not help, the snapping track is speech-like in most frames), but its speech energy is 9–18 dB below the speaker's. Two talkers at similar levels both stay speech; during one talker's pause its beam carries only leaked speech and counts as an echo. `main::load_speech` (`None` with a note if disabled or the model fails). GUI: green dot (`theme::SPEECH`) and probability in the label, "Speech" `on_threshold` slider (0.3 to 0.99, only with `speech.enabled`) passed to the audio thread as f32 bits in an `Arc<AtomicU32>` and applied per map via `set_on_threshold` (no pipeline restart); CLI: green dot in the PNG (`map_image(.., speech_ids)`) and a line per speech start/stop. Tests: hysteresis (below on, hold at 0.2, syllables and pauses, end after `hold_ms`, a 0.45 sound after speech ends it, a shift of 4 cells ends it at once, 1.8 cells does not), short bursts (60 ms) do not start speech, a delayed 0.3× copy correlates > 0.9 and independent noise < 0.1, a reflection of the speech fixture (5 ms, 0.3×) is never speech while the speaker is, speaker plus louder snapping with mutual leakage: speaker never an echo, snapping never speech, two talkers with 0.3 mutual leakage each speech in ≥ 70 % of the maps they are speech alone (78 % at on_threshold 0.95, ≥ 80 % at 0.7), level gate; `testdata/speech_16k.wav` (macOS `say`, 4.7 s) is speech in 40 of 44 maps; white noise and a 1 kHz tone stay below 0.1 smoothed probability.
- `pipeline.rs`: `PipelineConfig { fft_len 1024, memory_frames 40, map_every 10, f_lo_hz 900, f_hi_hz 4100, grid_width 64, grid_z 2.0 }` (serde, `Default` = these values; memory 40 ≈ 0.43 s, one map per 10 frames ≈ 0.11 s ≈ 9 maps/s, `power_map` ≈ 85 ms must fit).
  - `Pipeline<M, B: Beamformer<M>> { stream, cov, beamformer, bins, map_every, layout }` (private fields). `Pipeline::<16, B>::new(sample_rate, &PipelineConfig, &PeakConfig, &TrackerConfig, camera: Camera, make_beamformer: impl FnOnce(Steering<16>) -> B)` builds scan grid, bins (upper end clipped to the last bin) and `Steering`, then the beamformer from it. The closure guarantees grid and beamformer match. Asserts f_lo < f_hi and map_every > 0 (TODO: `Result` with thiserror).
  - `run(&mut source, pipeline, on_map)`: read → `StftStream::push` → `SlidingCovariance::accumulate`; every `map_every` frames `power_map` on the current estimate (timed, printed) → `update_tracks` (`find_peaks`, `Tracker::update`, each track gets its beam's samples since the previous map, beams of dropped tracks removed, new tracks get a `BeamSignal`, all re-steered to the grid cell at `grid_z` via the scaled camera; `TrackAudio` is `None` without integer decimation) → `on_map(&power, &layout, &tracks) -> Result<ControlFlow<()>, Box<dyn Error + Send + Sync>>`. Every STFT frame also goes to all beams. Test: two sines with CLEAN-SC give two confirmed tracks within 0.34 cells, own source 30/12 dB above the other in the track audio (DAS pulls the two peaks about 2 cells together). Stops at `Ok(0)`, on `Break`, or on the first error. CLI: `save_maps(src_pos, max_maps, show_tracks)` in `main.rs` writes PNGs and, via `TrackRecorder` (hound), `track_NNN.wav` per confirmed track (16 kHz mono f32, from confirmation on); GUI: `spawn_audio` in `gui.rs` sends raw power maps. Takes `&mut S` so `run_mic` can print `dropped()` afterwards.
- `peaks.rs`: `PeakConfig { enabled, range_db 6, min_dist_cells 4, max_count 5 }` (serde, `[peaks]` in `sound2view.toml`), `Peak { col, row (grid cells, sub-cell), level_db }`, `find_peaks(power, [w, h], config)`: local maxima (8 neighbors, plateau gives its first cell), within `range_db` of the maximum, greedy minimum distance (stronger kept), at most `max_count`, strongest first; per axis a parabola through the log power (exact for a Gaussian), none on the border. Computed in the pipeline; drawn as tracks (see `tracking.rs`): GUI (`draw_tracks`, sizes relative to the drawn image height with minimums): hollow blue ring (`theme::PEAK`) on a dark fill, speech: larger filled green dot (`theme::SPEECH`) with a halo; label "#id level dB probability" on a pill (dark, or green with dark text and "speech"), "Tracks" checkbox (`peaks.enabled`); CLI PNGs blue dot. Also the tool for the `Mount` alignment (sub-cell peak positions). Tests: Gaussian at (10.3, 7.6) exact, two peaks with levels, range/distance/count, plateau and empty map.
- `tracking.rs`: `TrackerConfig { max_dist_cells 6, min_hits 3, max_missed 5, smoothing 0.5 }` (serde, `[tracking]`, `validate`), `Track { id, col, row (smoothed, grid cells), level_db, hits, missed, confirmed, samples: Vec<f32> }`, `Tracker::update(&peaks)`: greedy nearest (track, peak) pairs within the gate, position += smoothing·(peak − position), unmatched peaks start tracks (ids never reused), unmatched tracks count a miss and are dropped after more than `max_missed`; `confirmed` after `min_hits` maps in a row, sticky. Only confirmed tracks are drawn (dot persistence against flicker). Tests: moving peak keeps id, jump starts a track, confirmation/drop, nearest pairs first.
- `dsp/beam_signal.rs`: audio from one direction for the speech classifier (`/Users/onmcu/Git/voice-activity-detection`, `Preprocessor::push(&[f32])` at 16 kHz, next step). `TRACK_SAMPLE_RATE` 16 kHz, `decimation_factor(f_s)` (integer only, 44.1 kHz gives none). `BeamSignal<M>::new(fft_len, stft_scale, f_s, distances)`, `steer(distances)`: w_k = a·e^{jωτ_ref}/(aᴴa) per bin (all N/2 + 1), τ_ref = mean mic distance / c, `push_frame(snapshots)`: y = wᴴx, real DC/Nyquist, inverse FFT, overlap-add with gain 1/(N·scale) (periodic Hann sums to 1 at hop N/2; the weights shift each mic's frame circularly only by (r_m − r_ref)/c ≤ about 13 samples, where the window is ≈ 0. Removing the whole delay r_m/c, ≈ 280 samples at 2 m, wrapped the window around the frame: error at every hop, heard as a 94 Hz buzz, fixed), decimated by a Blackman windowed-sinc FIR (32·factor − 1 taps, cutoff 0.4·16 kHz), `take_output()`. A source at the point comes out as s(t − τ_ref) at 1 m. Tests: band noise at off-bin frequencies reconstructed to 1e-7 (factor 1; a bin-centered sine is N-periodic and hides a wrong circular shift, 2e-2 with the old weights), decimated amplitude, about 17 dB suppression of a source 62° away at 3.4 kHz, decimator pass/stopband with uneven chunks.
- `render.rs`: `map_image(power, layout, src_pos, tracks, speech_ids) -> RgbImage` (dot per confirmed track, green if its id is in `speech_ids`), independent of `Pipeline` (reusable for the GUI). Grid image → `imageops::resize` (Nearest) to camera size → red cross per true source (`camera.project`, skipped if z ≤ 0 or outside). `power_to_rgb`: gray, 0 dB = white, −15 dB and below = black (hard-coded `15.0`; to become a `DB_RANGE` constant). Sim and mic files overwrite each other.
- `source/capture.rs` (cpal 0.18.2; `device.id()`/`description()`, not `name()`):
  - `probe()`: per host, matching input devices (`UMA16_PATTERNS` "uma"/"minidsp") with their configs; devices with > 20 configs are plug devices, skipped. Result on Linux: `alsa:hw:CARD=UMA16v2,DEV=0`, 16 ch, I32, fixed rates 8000/11025/12000/16000/32000/44100/48000. Use the card name, not the index.
  - `CpalSource<M> { _stream, rx, pending: VecDeque<f64>, dropped: Arc<AtomicUsize>, stream_error: Arc<Mutex<Option<cpal::Error>>>, sample_rate }`. `new(pattern, sample_rate) -> Result<Self, CaptureError>`: first input device whose id or name contains the pattern, `StreamConfig` with M channels, default buffer. Callback converts samples to f64 (see below) and `try_send`s on a `sync_channel(64)`; when full the chunk is dropped and counted (audio thread never blocks). Error callback: `Xrun`, `RealtimeDenied`, `DeviceChanged` are printed (stream keeps running); any other `cpal::ErrorKind` (e.g. `DeviceNotAvailable` on unplug) is stored as the first fatal error. `dropped()` getter.
  - `CaptureError` (thiserror): `NotFound(pattern)`, `Device(cpal::Error)` (from), `Stream(cpal::Error)`, `NoData(Duration)`, `Closed`, `UnsupportedFormat(SampleFormat)`. cpal 0.18 has a single `cpal::Error` with `kind() -> ErrorKind`.
  - Device selection and format (for macOS): `new(pattern, ..)` matches the pattern against id or name (`description()`). The stream type follows `default_input_config().sample_format()`: I32 (UMA-16 on ALSA), F32 (CoreAudio reports inputs only as F32), I16; else `CaptureError::UnsupportedFormat`. Generic `build_stream::<T>` with `Callbacks { tx, dropped, stream_error }`, conversion via `Sample::to_sample::<f64>()` (integers / full scale, e.g. 2^31 for I32). The app's default `audio.device` is per platform: `hw:CARD=UMA16v2` on Linux (ALSA lists the UMA-16 several times), `UMA16` on macOS.
  - `read`: when `pending` is empty, `next_chunk` waits with `recv_timeout` (200 ms polls): error on a stored stream error, `Closed` (callback dropped), or `NoData` after 2 s without data (unplug without error callback). Errors go out as `io::Error::other(CaptureError)` (`FrameSource::read` stays `io::Result`). No draining with `try_recv`, so a slow reader drops chunks instead of growing latency. Copies `min(buf.len(), pending.len())` samples via `as_slices` + `copy_from_slice`, then `drain(..total)`. cpal chunks are whole frames (asserted), so `pending` stays a multiple of M.
  - Measured: chunks ≈ 180 frames (≈ 4 ms), queue ≈ 0.25 s. Release keeps up, 0 dropped.
  - Threads, not async: cpal owns the audio thread, the channel is the boundary. With egui: worker thread runs `run` and sends maps to the GUI thread (`try_recv` per frame or `Arc<Mutex<latest map>>`).

- `source/video.rs` (+ `video/v4l.rs`, `video/avfoundation.rs`): camera capture behind the same API on both platforms, backend chosen with `#[cfg(target_os)]` (`pub use` of `CameraSource`, `list_cameras`, `initialize`, `is_authorized`, `probe`). Shared: `VideoError` (`NotFound(name)`, `NoSuchIndex`, `NotAuthorized`, `Camera` (Linux: `NokhwaError`, macOS: `String` from `NSError`), `NoFrame(Duration)`, `Closed`), `CameraEntry { index, name }`, `CameraSource::open_by_name` (case-insensitive substring, tries every match in order).
  - `v4l.rs` (Linux): nokhwa 0.10 with V4L2. `new(index = /dev/videoN, w, h, fps)`: MJPEG `Exact`, else `Closest`; RGBA via nokhwa's decoder. `initialize` no-op, `is_authorized` always true (`video` group, see README). `probe` opens each camera to list its formats.
  - `avfoundation.rs` (macOS): own backend on objc2 (nokhwa's AVFoundation bindings are broken, see the macOS section). `devices()` = `AVCaptureDeviceDiscoverySession` with `AVCaptureDeviceTypeExternal` (macOS 14+) and `BuiltInWideAngleCamera`; index = position in that list, name = `localizedName`. `new`: pick the (format, frame rate range) pair closest to the request (resolution mismatch in pixels first, then fps outside the range; `FormatOption::distance`), so format and frame rate always come from the same entry; `AVCaptureSession` + `AVCaptureDeviceInput` + `AVCaptureVideoDataOutput` with `videoSettings` = 32BGRA (AVFoundation converts MJPEG/YUV), `setActiveFormat` and min = max frame duration (`frame_duration`: the range's own CMTime at its ends, else 1/fps) under `lockForConfiguration` inside begin/commitConfiguration, then `startRunning`. Delegate `FrameDelegate` (`define_class!`, ivars = `SyncSender<VideoFrame>`) on a serial dispatch queue: `to_rgba` locks the CVPixelBuffer, copies BGRA → RGBA row by row (stride), `try_send` (drops while the reader is behind). `next_frame` = `recv_timeout(2 s)`. `Drop` stops the session. `initialize` = `requestAccessForMediaType`, `is_authorized` = `authorizationStatusForMediaType == Authorized`. `probe` lists devices with every format and fps range (`1920x1080 @ 5-30 FPS, 420v`) without opening a session. Everything that returns autoreleased objects runs in `autoreleasepool`.
- `lib.rs`: `Vector3 = nalgebra::Vector3<f64>`, `Point3 = nalgebra::Point3<f64>`, `Complex64 = nalgebra::Complex<f64>` (same type as `realfft::num_complex::Complex<f64>`, single num-complex 0.4 in the lock file), `C_SOUND = 343.0` (TODO: rename to `SPEED_OF_SOUND`), `distance_to_delay(d) = d / C_SOUND`, the traits above.
- `geometry/mic_array.rs`: `pub struct MicArray<const M: usize> { pub mic_pos: [Point3; M] }`, indexed by USB channel.
  - `uniform_rect(cols, rows, pitch_x, pitch_y, channel_of: impl Fn(row, col) -> usize)`: asserts `rows * cols == M` and each channel assigned once. Row 0 = top = +y, col 0 = left = −x, viewed from the sound-source side (z into the scene).
  - `get_distance_from(&Point3) -> [f64; M]`: r_mk for all mics.
  - Steering vector a_m = e^{−j2πfτ_m}/r_m, not normalized (includes 1/r). Split into geometry and frequency: `steering_from_distances(r: &[f64; M], f) -> SVector` (associated fn, no `self`) and the wrapper `steering(&self, pos, f)`. Callers compute distances once per position (per source in `reference.rs`, per grid point in the beamformer) and only redo the frequency part per bin/component.
  - `impl MicArray<16> { uma16() }` uses `UMA16_CHANNELS[row][col]` = `[[7, 6, 9, 8], [5, 4, 11, 10], [3, 2, 13, 12], [1, 0, 15, 14]]` (verified on hardware). `test_uma16_geometry` checks all 16 positions against the Acoular `minidsp_uma-16.xml` table with x negated.
  - Tests: geometry (ch0, ch15), relative delay vs far-field reference at 100 m, all mics, tol 1e-6 s. `f64::EPSILON` and the long comment in `uniform_rect` are kept on purpose.
- `geometry/camera.rs`: `Camera { intrinsics, mount }`, serde. Pure projection math, not video capture.
  - `Intrinsics { width, height: u32; fx, fy, cx, cy: f64 }`, `from_hfov(width, height, hfov_rad)`: fx = fy, c = (n − 1)/2 (pixel-center convention).
  - `Intrinsics::unproject(u, v, z) -> Point3` / `project(&Point3) -> (u, v)`: plain pinhole in the camera frame, no signs; both assert z > 0.
  - `Intrinsics::scaled(w_target)`: same FoV at lower resolution. h' = max(1, round(w'·h/w)), separate s_x, s_y; f·s, c' = (c + 0.5)·s − 0.5.
  - `Mount { pub offset: Vector3 (m, array frame), pub rotation: UnitQuaternion }`; `Default` / `from_offset` use 180° about z. `pose() -> Isometry3`: p_array = R·p_cam + t.
  - `Camera::unproject(u, v, z)` = pose ∘ intrinsics; `project(&Point3)` = intrinsics ∘ pose⁻¹. z is the depth from the camera, not the array z.
  - Tests: round trip, image center on axis, top-left pixel is +x/+y (pins the rotation), offset mount (round trip with depth z − offset.z, center maps to offset + z·ẑ).
- `geometry/scan_grid.rs`: `ScanGrid { width, height: usize, points: Vec<Point3> }` (private, getters). `new(camera, width, z)` unprojects every pixel center of the `scaled` camera at depth z, so the grid covers exactly the camera FoV. Row-major, u = col, v = row; `index(row, col)`.
  - Tests: len = w·h, all z exact, corners (project with full-res camera lands on (i + 0.5)/s − 0.5), height (64 → 36), left/right symmetry.
- `dsp/beamformer.rs`: shared parts of all beamformers.
  - `Steering<M> { distances: Vec<[f64; M]>, df, grid_size }`, `new(scan_grid, mic_array, f_s, fft_len)`; neither grid nor array is stored. `n_points()`, `grid_size() -> [width, height]` (for CLEAN-SC's clean beam), `vector(point, bin)` = a(r, f_k) via `steering_from_distances`, `gains(point)` = g_m = 1/r_m². Place for steering-vector caching later.
  - `trait Beamformer<const M> { fn n_points(&self); fn power_map(&self, cov, bins, out) }`. `&self` (keeps `rayon` over points possible; scratch allocations per map are negligible).
  - `Normalization` enum lives here (DAS and later functional beamforming use it).
- `dsp/beamformer/delay_and_sum.rs`: `Das<M> { normalization: Vec<f64>, remove_diag, steering }`, `new(steering, remove_diag, normalization)` precomputes c per point from `steering.gains(i)`. `impl Beamformer`. Siblings: `functional.rs`, `clean_sc.rs`.
- `dsp/beamformer/functional.rs`: `Functional<M> { steering, nu, noise_removal, equal_bins, inv_norm2 }` (Dougherty 2014), `new(steering, nu, noise_removal, equal_bins)`. Per bin (bins in parallel) R^{1/ν} = U diag(λ'^{1/ν}) Uᴴ via `symmetric_eigen` + `recompose`, then P_k = (wᴴR^{1/ν}w)^ν / M with w = a/‖a‖. Single source without noise: P = S·‖a_0‖²/M at the source (same as DAS `TrueLocation`); ν = 1 is DAS without diagonal removal. `noise_removal`: λ' = λ − median eigenvalue (noise estimate, assumes < M/2 sources; median, not min, because the noise eigenvalues of an L-frame estimate spread by (1 ± √(M/L))²). λ' below 1e-12·λ_max is set to 0 (`EIGEN_ROUNDING`: the 1/ν power would lift rounding noise to a floor, 1e-16^(1/20) = 0.16). `equal_bins`: each bin map divided by its maximum before summing, bins more than 20 dB (`EQUAL_BINS_GATE_DB`) below the strongest left out; without it low bins (most speech power, no resolution) dominate. Defaults (app): both on, GUI checkboxes. The `Beamformer` impl needs `Const<M>: DimSub<U1>` and `DefaultAllocator: Allocator<DimDiff<Const<M>, U1>>` (bounds of `symmetric_eigen`). Tests: hand-built R peak and level at p_0; with 10 dB noise peak at p_0 and fewer points within 3 dB than ν = 1; noise removal gives the noise-free map; equal bins gives a −15 dB source the same peak and drops a −30 dB bin.
- `dsp/beamformer/clean_sc.rs`: `CleanSc<M> { steering, gain, max_iter, stop_ratio, beam_sigma, remove_diag, weight_scale }` (Sijtsma 2007), `new(steering, gain, max_iter, stop_db, beam_sigma, remove_diag)` (asserts ranges; app passes `remove_diag = true`). Weights w = a/(‖a‖√M), √(M − 1) with diagonal removal (levels as DAS `TrueLocation`); G = R, or R with zeroed diagonal; dirty map P_j = wᴴGw. Two passes: all dirty maps first (`DirtyBin`, bins in parallel), threshold = largest dirty value of all bins · 10^(−stop_db/10); then per bin: j_max, source vector h with P_max (h hᴴ − H) w_max = G w_max, H = diag(|h_m|²) (0 without diagonal removal, then h = R w_max / P_max), found by 20 fixed-point steps h = (Gw/P_max + Hw)/√(1 + wᴴHw) (`H_ITER`, as Acoular); clean[j_max] += gain·P_max, G −= gain·P_max (h hᴴ − H), dirty map updated incrementally P_j −= gain·P_max (|w_jᴴh|² − w_jᴴHw_j) (O(points·M)); stops after `max_iter` or below the threshold. The cross-bin threshold keeps bins without a source out (a per-bin threshold added their noise or reflection maximum). Clean maps summed, then separable Gaussian clean beam (`gaussian_blur`, peak 1, σ in grid cells, 0 = single points). Defaults (app): gain 0.9, 10 sources, 20 dB, σ 1.5 cells. Tests: single source with gain 1 is exactly one point (‖a_0‖²/M, with diagonal removal (‖a_0‖⁴ − Σ|a_0m|⁴)/((M − 1)‖a_0‖²)); diagonal removal ignores σ²I 10 dB above the source; a bin 30 dB below the strongest adds nothing at 20 dB but does at 40 dB; two incoherent sources at 1/6 and 5/6 of the width (bin 80, 3840 Hz) give the maximum of each half within one cell; blur keeps the peak and is symmetric.
  - `Normalization` (Sarradj 2012, formulations III and IV), P = c·aᴴRa with g_m = 1/r_m², Σg = aᴴa:
    - `TrueLevel`: c = 1/(Σg)², with diagonal removal 1/((Σg)² − Σg²). P = S exactly at the source, but P(p) = cos²θ·‖a_0‖²/‖a‖²: farther points can exceed S, the peak drifts outward off-axis.
    - `TrueLocation`: c = 1/(M·Σg), with diagonal removal 1/((M − 1)·Σg) (Acoular's M/(M − 1)). Peak exactly at the source if it lies on the grid plane; symmetric lobes for symmetric sources. Used in `main`.
  - Diagonal removal subtracts Σ_m |a_m|²·R_mm (= Σ g_m·R_mm, computed from `a` directly), removing σ²I in expectation. P can be negative: clip only in the display (dB floor), never per bin.
  - `power_map(cov: &[SMatrix] (full spectrum, N/2 + 1), bins: RangeInclusive<usize>, out: &mut [f64])`: bins outer, points inner (`enumerate` over normalization and `out`), `steering.vector(i, k)` per (bin, point). Overwrites `out`.
  - Speed (release, 64×36 grid, ≈ 70 bins, M = 16): ≈ 85 ms per map single-threaded, budget ≈ 0.11 s (`map_every` 10). DAS and functional now run the point loop with `rayon` (`par_iter().zip(out.par_iter_mut()).enumerate()`, bins stay sequential); not measured yet. Linear in points and bins. Further reserve: cache steering vectors per bin (saves the 16 `exp` per (bin, point), ≈ 67 MB).
  - Tests (all with `false, TrueLevel`): `hand_built_r_peak_at_source` (R = a_0a_0ᴴ near the corner: P(p_0) = 1, Cauchy-Schwarz P·‖a‖² ≤ ‖a_0‖²), `oracle_r_peak_at_source` (source on the center grid point: argmax at source, P = `sine_bin_power`), `stream_r_peak_at_source` (90 dB mic noise, 0.5 s: argmax at source, P within 2 % of oracle; 110 dB would make the argmax flaky, the map is flat on grid scale). Skeletons with `todo!()` (deferred): `diag_removal_hand_built_r_at_source`, `diag_removal_noise_only_is_zero`, `diag_removal_stream_r_peak_at_source`.
- `source/simulator.rs`: `P_REF = 20e-6` (private), `level_to_pressure(level_db) -> p_rms` (dB SPL → Pa). `Simulator<const M> { scene, sim_sources: Vec<SimSource<M>>, mic_noise, n: u64, n_total: u64 }` implementing `FrameSource<M>`.
  - `SimSource<M> { signal, gain: [f64; M] (1/r_mk), delay: [f64; M] (τ_mk, s) }`, built in `new(scene, mic_array)`; `Scene` stays untouched as ground truth.
  - `read`: frames = min(buf frames, n_total − n); loop frame → mic → Σ sources gain·eval(t − delay) + one noise draw. `Ok(0)` at end.
  - `mic_array` is not stored; only `new` uses it.
  - Tests use `test_util`. `single_source_propagation`: sine at (0.1, 0.2, 1.0), noise −1000 dB, expected value computed independently from `mic_pos`, tol 1e-6, also checks total frames (EOF). `chunked_read_matches_single_read`: chunks of 7 vs one read, noise 54 dB, bit-identical. Both pass.
- `source/simulator/signal.rs`:
  - Pattern: `*Params` structs are the config form (dB, serde); runtime structs are linear and built via `#[serde(from = "...Params")]` + `From`.
  - `Signal` enum, `#[serde(tag = "type", rename_all = "snake_case")]`, `eval(t)` dispatches. Valid for all t, including t < 0.
  - `SineParams { level_db, freq_hz, phase_rad }` → `Sine { amplitude, freq_hz, phase_rad }`, A = √2·p_rms, s(t) = A·sin(2π f t + φ). `Sine::new(amplitude, ...)` takes linear amplitude.
  - `NoiseParams { seed, n_components (Q), f_lo_hz, f_hi_hz (both inclusive), level_db }` → `Noise { _params, components: Vec<Sine> }` (private). Asserts Q > 0, f_lo < f_hi. `ChaCha8Rng::seed_from_u64` (not `StdRng`, not portable), frequencies uniform in [f_lo, f_hi], phases in [0, 2π), a = √(2/Q)·p_rms.
  - Tests: sine values and phase, TOML parse, level round trip + 0 dB anchor, noise band, same seed, different seed, power. Power: exact Σ a²/2 = p_rms² (tol 1e-12) plus time-domain mean square (seed 4234, 2 %, seed-dependent because close components leave cross-term residuals ~2/(Q π Δf T)).
- `source/simulator/scene.rs`:
  - `Scene { sample_rate: f64, duration_s, mic_noise_params, sources: Vec<Source> }`, plain data and ground truth, no RNG state.
  - `Scene::from_toml(&str) -> Result<Scene, SceneError>` parses only; `validate(&self) -> Result<(), SceneError>` rejects duration ≤ 0, sample_rate ≤ 0, z ≤ 0, `max_freq ≥ f_s/2`. `Simulator::new` calls it and returns `Result<Self, SceneError>` (tests `.unwrap()`). `SceneError` (thiserror): `Parse(toml::de::Error)`, `Invalid(String)`. Array-dependent checks (min r_mk, 4.1 kHz warning) belong in `Simulator::new`, not done yet.
  - `Signal::max_freq()`: sine `freq_hz`, noise max over drawn components.
  - `Source { pos: Point3 (m, array frame, z into scene), signal }`.
  - `MicNoiseParams { seed, level_db }` → `MicNoise { rng: ChaCha8Rng, distribution: Normal(0, σ = p_rms) }`, `get_sample(&mut self)`. White Gaussian, not a `Source` (no position, independent per mic, σ²I in R). Owned by the `Simulator`; one RNG for all mics, drawn in order sample n → mics 0…M−1.
  - Tests: `parse_scene` (inline TOML fixture, independent of `config.toml`), `mic_noise` power with tolerance 5·σ²·√(2/N).
- `config.toml`: example scene at the root (sine + noise source, mic noise). Will move to separate scene files. No test that shipped configs parse yet.
- `dsp/stft.rs`:
  - `Stft` (one channel): plan, scratch/input/output buffers, periodic Hann `window` (divisor N), `scale` = 1/√(Σw²), `weights` = window·scale. `new(fft_len)` asserts N even. `transform(&mut self, frame) -> &[Complex64]` windows in place into `input_buf`, no allocation, returns N/2 + 1 bins.
  - Scaling 1/√(Σw²): white noise σ² → E|x(f)|² = σ² in every bin (σ²I in R needs no factor); sine power P on a bin center → P·N/3 (Hann); band noise P over B → P·f_s/(2B). In general E|x|² = two-sided PSD · f_s. Normalized dB maps don't care; ‖R̂ − R‖ does.
  - `StftStream<const M>` (`new(stft)`): per-channel `buf: [Vec<f64>; M]`, `fill`, `snapshots: Vec<SVector<Complex64, M>>` (N/2 + 1, reused), `frame_count` ℓ. `push(chunk, on_frame: impl FnMut(&[SVector<Complex64, M>], u64))`: first frame after N samples, then every N/2; shift with `copy_within` (FFT needs contiguous input, so a ring buffer saves nothing). ℓ starts at 0; frame center at (ℓ·N/2 + N/2)/f_s.
  - Tests: `scale` ≈ √(8/(3N)); COLA w[n] + w[n + N/2] = 1; `sine_peak` (bin 50: (A²/2)·N/3, neighbors 1/4, rest 0); `stream_frame_count` ((F − N)/(N/2) + 1, ℓ gapless); `stream_chunk_independence` (7 vs one push, bit-identical); `phase` (`test_scene` → STFT with N = 1000, x_m/x_0 = (r_0/r_m)·e^{−j2πf(τ_m − τ_0)} on bin 50, tol 1e-9, compare complex numbers not angles). All pass. Validation step 5 (phase part) done.
- `dsp/covariance.rs`: `Covariance<const M> { sum: Vec<SMatrix<Complex64, M, M>>, count }`. `new(n_bins)`, `accumulate(&[SVector])` (asserts bin count, adds x xᴴ), `count()` = L, `estimate()` = sum/L (panics at L = 0, does not reset), `reset()`.
  - `SlidingCovariance<M> { estimate, alpha, count }`, `new(n_bins, memory)` with alpha = 1 − 1/memory. `accumulate`: R̂ ← R̂ + w·(x xᴴ − R̂), w = max(1 − alpha, 1/l): plain mean during start-up (no bias toward zero), then exponential average. Never reset; `estimate() -> &[SMatrix]` (panics before the first frame). Used by `pipeline` for live maps. Tests: equals the plain mean during start-up, forgets old frames (0.9^200), panics without frames.
  - Tests: single frame = outer product, average over frames, Hermitian with real nonnegative diagonal, panics (no frames, after reset, wrong bin count).
- `source/simulator/reference.rs`: `reference_covariance(scene, mic_array, stft) -> Vec<SMatrix>` (N/2 + 1 bins), the expectation R = σ²I + Σ_q S_q(k)·a aᴴ.
  - Per sine component q (a noise source contributes Q of them, via `Signal::components()`): a = `steering_from_distances` at the component's own frequency f_q, distances once per source; S_q(k) = (A/2)²·|W_k|², W_k = Σ_n weights[n]·e^{j2π(f_q/f_s − k/N)n}. Uses the `stft` scaling.
  - Ignores the −f image; only ±8 bins (`BINS_AROUND`) around each component.
  - Lives under `simulator` because it depends on simulated ground truth; the simulator itself does not use it.
  - Tests use `test_util::stream_covariance` and a local `relative_error` = √(Σ_k ‖R̂_k − R_k‖²_F / Σ_k ‖R_k‖²_F) over all bins (per-bin relative error blows up in zero bins).
  - `one_sine_no_noise`: bin-center sine, R̂ = R to 1e-9 for every L (frame phase cancels in x xᴴ; image is exactly 0 near the bin).
  - `one_sine_with_mic_noise`: 110 dB mic noise (σ² ≈ 40 Pa² vs ≈ 150 Pa² sine in bin 50), 0.5 s vs 2 s; error ratio ≈ √(L_long/L_short) ± 0.5. Detects bias in R (error floor → ratio → 1); verified by dropping σ² → σ. Expected size ≈ √(M/L) for noise alone. Avoid 94 dB (σ = 1 Pa, σ = σ²).
- `test_util.rs` (`#[cfg(test)] pub(crate)`): `SAMPLE_RATE` 48 kHz, `FFT_LEN` 1000, `BIN` 50, `F_SINE` = bin center (2400 Hz), `test_pos()` (0.1, 0.2, 1.0), `test_sine()` (A = 1), `test_scene(mic_noise_db, duration_s)`, `read_all(sim, chunk_frames)` (reads to `Ok(0)`), `stream_covariance(scene, mic_array) -> (R̂, L)` (simulator → `StftStream` → `Covariance`), `test_camera()` (1920×1080, 110° HFOV, default mount), `test_scan_grid()` (192 wide, z = 1 m), `sine_bin_power(A, N)` = (A²/2)·N/3. Only helpers used by ≥ 2 modules; module-specific ones stay local. Plain `16`s in tests still to replace.
- Cargo: `nalgebra` (std, serde-serialize), `serde` (derive), `toml` (std, serde, parse), `rand` 0.10 (chacha), `rand_distr` 0.6, `realfft` 3.5, `cpal` 0.18.2, `image` 0.25 (png only), `nokhwa` 0.10.11 (Linux only), macOS only: `objc2` 0.6, `objc2-foundation`/`objc2-av-foundation`/`objc2-core-media`/`objc2-core-video`/`objc2-core-foundation` 0.3, `block2` 0.6, `dispatch2` 0.3; `rayon` 1.12, `thiserror` 2.0. App: `eframe` 0.37, `serde`, `toml`, `image`.
- `channel_of` is not stored; the mapping is baked into the order of `mic_pos`. Store the inverse `grid_pos` only when needed.

Status: simulator and live UMA-16 both run end to end (source → STFT → R̂ → DAS → PNG). Simulated maps match the markers (sources at z = 2 m on the grid plane, 3.4/3.6 kHz). Live with GUI: camera image plus map overlay; orientation verified (x was flipped, fixed).

Hardware findings (tone test indoors): a pure tone gives several bright spots. Causes: room reflections (coherent with a sine; band noise averages them out), DAS sidelobes of the uniform 4×4 at ≈ −13 dB (visible with a −15 dB display range; −6 dB shows the main lobe only), and a wide band summing noise-only bins for a single tone (narrow the band to the tone). Narrow band + smaller dB range made it much better.

Next:
0. **Speech hysteresis: test on hardware.** `on_threshold` with a 500 ms hold after the last frame above it (see `speech.rs`). Check whether speech stays green through pauses and other sounds turn blue again; if not, tune `hold_ms`/`smoothing_ms`, or decide over a window (fraction of the last 1–2 s of frames above `on_threshold`).
   - CI: the private classifier dependency needs a deploy key on voice-activity-detection and an SSH agent step (e.g. `webfactory/ssh-agent`) in `.github/workflows/ci.yml`; not set up, CI fails at the fetch until then.
1. Test on hardware: CLEAN-SC with two incoherent sources, `rayon` speedup (map times), mic unplug/replug (status `Lost` → reconnect), camera unplug/replug.
1a. macOS: see "macOS (Mac mini): status, continue here" below.
2. Review the Claude-implemented parts (overlay, sliding covariance, functional, CLEAN-SC, GUI, config, errors).
3. Camera calibration (principal point, distortion) once the spots are small enough to matter.
4. Small things: rename the scene `config.toml` → `scene.toml`, "Save" button for GUI settings, deferred tests (diagonal removal, `TrueLocation`).

Error handling (thiserror 2 in core): `CaptureError` (capture), `VideoError` (video), `SceneError` (scene), `PipelineError` (pipeline: `InvalidConfig(String)`, `Source(io::Error)`, `MapHandler(BoxError)`). `pub type BoxError = Box<dyn Error + Send + Sync>` in `lib.rs` for caller-defined parts (map handlers). All core errors are `Send + Sync` (worker thread → GUI). `ImageSource { type Error; fn next_frame(&mut self) -> Result<VideoFrame, Self::Error> }`. Config errors are `Result`s: `PipelineConfig::validate(sample_rate)` (called by `Pipeline::new`), app `Config::validate` (called by `Config::load`). Remaining asserts are internal invariants (bin counts, buffer lengths) plus `CleanSc::new`'s parameter ranges (the app validates them first). The probes still return `Box<dyn Error>` (CLI only).

Open: diag-removal tests and a `TrueLocation` hand-built test; `DB_RANGE` constant, `FilterType::Triangle`; tests for `max_freq`/`validate`, min r_mk check in `Simulator::new`, `Covariance` doc comments, off-center sine reference test; replace plain `16`s in tests; Unicode in doc comments.

## macOS (Mac mini): status, continue here

Target: the same app on a Mac mini (macOS 14+, the backend uses `AVCaptureDeviceTypeExternal`). Build and run as on Linux: `cargo run --release -- <mode>`; Xcode command line tools are the only system dependency. Permissions (mic and camera belong to the terminal app) are in the README.

Done for macOS (compiles in CI on `macos-latest`, `.github/workflows/ci.yml`: fmt on Linux, clippy + release tests on ubuntu-latest and macos-latest; not yet run on the Mac at the time of writing):
- Audio: default `audio.device` = `UMA16` on macOS (`DEFAULT_AUDIO_DEVICE`, `#[cfg]` in `config.rs`; ALSA keeps `hw:CARD=UMA16v2`); `CpalSource::new` matches id or name. CoreAudio reports inputs only as F32, so the stream type follows `default_input_config().sample_format()` (I32/F32/I16, generic `build_stream::<T>`). Without microphone access CoreAudio delivers silence: "Mic live", but an empty overlay.
- Camera: own AVFoundation backend (`source/video/avfoundation.rs`, see above). `main` calls `video::initialize()` for `camprobe`/`camshot`/`gui`; the CLI modes wait up to 10 s for the permission answer, the GUI camera thread retries (`VideoError::NotAuthorized` as status).

Why not nokhwa on macOS: `nokhwa-bindings-macos` 0.2.4 `AVCaptureDevice::set_all` (lib.rs ≈ 1031–1085) takes `selected_format` from the *last* format with the requested resolution but `selected_range` from the format where the fps matched, ignores the pixel format, and `lock(&self)` never sets `locked`. Setting `activeVideoMaxFrameDuration` (lib.rs:1080) then throws an Objective-C exception, which aborts the process ("panic in a function that cannot unwind"); seen with `camprobe` (`Camera::new` with `RequestedFormatType::None`) and when opening the camera in the GUI. nokhwa is barely maintained, so it is only used on Linux.

First steps on the Mac:
1. `cargo run --release -- probe`: is the UMA-16 listed, with 16 channels, F32, 48 kHz? Does `UMA16` match its id or name (else set `audio.device`)?
2. `cargo run --release -- camprobe`: camera name (macOS name may differ from Linux "icspring camera"; set `camera.name` if needed) and its formats/fps ranges. Is 1920x1080 listed, at which fps?
3. `cargo run --release -- camshot`: one frame as `camshot.png` and the frame time (BGRA → RGBA copy on the CPU). Check orientation and colors (BGRA order) and that the image covers the measured 83° field of view.
4. `cargo run --release -- mic` (PNG maps), then `gui`: overlay alignment (same camera, same `hfov_deg`), camera dropdown, method switching, mic/camera unplug and replug.

Known risks / likely fixes:
- Compile errors in `avfoundation.rs` that CI did not catch can only be fixed on the Mac (objc2 signatures were checked against objc2 0.6.5 / objc2-av-foundation 0.3.2 source).
- If the delegate never fires: the output may need `alwaysDiscardsLateVideoFrames` off, or the session a preset; check `session.isRunning()`.
- If the active format is reset after `startRunning`: unlock later, or set the format after `startRunning`.
- Frames from AVFoundation may arrive mirrored for some cameras (`AVCaptureConnection` mirroring); compare with the Linux image.
- macOS < 14: `AVCaptureDeviceTypeExternal` does not exist (strong link); would need `AVCaptureDeviceTypeExternalUnknown`.

## Simulator plan

Equation (spherical waves, near and far field, no separate plane-wave model):

x_m[n] = Σ_k (1/r_mk) · s_k(n/f_s − τ_mk) + v_m[n],  τ_mk = ‖r_k − p_m‖ / c,  c = 343 m/s

- **Analytic sources:** `Signal::eval(t: f64) -> f64`, so fractional delays are exact. Recorded sources later need windowed-sinc fractional delay.
- **`Signal` is an enum** (serde-friendly for TOML scenes, no `dyn`). Start with `Sine` and `Noise`; `Burst` later for step 5 validation only. No `Chirp`, no `Sum` (a scene holds a list of sources).
  - `Sine`: energy in one bin; reference R power A²/2. Resolution vs. frequency.
  - `Noise`: band-limited sum of sines, frequencies and phases drawn once at construction from a seeded RNG. Independent noise sources are uncorrelated.
- `eval` is a pure function of t (`&self`, no state, no RNG calls). Define s(t) for t < 0 and document it.
- `Source { pos, signal }`: position stays out of the signal. Amplitude referenced to 1 m; Hz; rad.
- Optional methods: `max_freq()` (assert < f_s/2, warn > 4.1 kHz), power or spectrum for the reference R.
- Same-frequency sources are coherent (MVDR fails); different frequencies never share a bin. Choose deliberately per benchmark.
- Compute t = n/f_s from a `u64` sample index in `f64`, never by accumulation.
- Smooth envelopes, everything below f_s/2. Test tones < 4 kHz first.
- Precompute r_mk, τ_mk for static sources. Moving sources (Doppler) and image sources (reflections) come later.
- Independent noise per mic, seeded RNG.
- Units: meters, seconds, radians, documented on every field.

Build order and checks:
1. Geometry: 16 positions, centroid 0, neighbor distance = pitch.
2. Delays: far-field oracle Δτ_m = −p_m·u_k/c for a source 100 m away.
3. Signals: known values.
4. Simulator implementing `FrameSource`.
5. Validation: single-bin DFT phase per channel vs ω·Δτ_m (pins the sign convention); burst + cross-correlation with parabolic interpolation vs τ_mk − τ_jk; source top-left → shortest delay at `UMA16_CHANNELS[0][0]`.
6. WAV dump (`hound`, 16 ch, f32).

Testing approach:
- Tests prove the algorithms work and act as a benchmark: how well does the output match the truth. For now judged visually (see below).
- Everything runs through the stream path (`FrameSource` → STFT → R̂ → beamformer). No reverberation; recorded WAVs later.
- **Covariance oracle:** R(f) = A(f) S Aᴴ(f) + σ²I from the true `src_pos`. Feeding R instead of R̂ gives the algorithm-only result, i.e. the best achievable map. The gap to the stream result is pipeline error. Also check ‖R̂ − R‖_F / ‖R‖_F.
- The simulator exposes ground truth (`src_pos`, source powers) next to the stream.

## Visualization and camera

- **egui** (fallback: `rerun`). Power map as `egui::ColorImage` with alpha over the camera frame (texture), markers via `ui.painter()`. No `egui_plot`.
- Map in dB, normalized to its max, clipped (e.g. −15 dB). True sources and estimated peaks as distinct markers. Oracle R and stream R̂ side by side.
- Shown **as the camera sees it**. Array geometry stays as is; map to the image by calculation.
- Camera sits in the hole at the array center: t ≈ 0 (small z offset for the lens, negligible), rotation = 180° about z.
- Array → pixel: x_c = −x, y_c = −y, z_c = z; u = c_x + f_x·x_c/z_c, v = c_y + f_y·y_c/z_c.
- Pixel → array (scan grid at focus distance z): x = −(u − c_x)·z/f_x, y = −(v − c_y)·z/f_y.
- One `project`/`unproject` pair with a round-trip test (done, signs via `Mount` rotation). Truth markers use the same `project`.
- Simulator uses a virtual pinhole camera (e.g. 640×480, ~90° HFOV), no distortion. Real intrinsics and distortion: calibrate once the camera exists.

## Processing pipeline (later)

STFT (e.g. 1024, Hann, 50 %) → covariance per bin averaged over L frames → steering vectors for a grid (near field, focus distance) → map per bin → sum over band → overlay.

- **Baseline:** delay-and-sum on R with diagonal removal. P = aᴴRa / (aᴴa)².
- **Then:** functional beamforming or CLEAN-SC (most useful for this small array).
- **MVDR:** w = R⁻¹a / (aᴴR⁻¹a), P = 1/(aᴴR⁻¹a). Needs diagonal loading (1–10 % of tr(R)/M), ≥ 32 snapshots, and no diagonal removal; fails with coherent sources. Use `lu().solve()`, not an explicit inverse.
- MUSIC and sparse methods possible; neural methods are research stage.
- Reference implementation for comparison: Acoular (Python).
- Calibrate per-mic gain and phase before adaptive methods.

## Notation (full version in the blog post `notation.md`)

- Bold lowercase = vector, bold uppercase = matrix, regular = scalar, regular uppercase = count. Hat = estimate. (·)ᴴ = conjugate transpose.
- Index pairs: m = 1…M mics, k = 1…K sources, n = 1…N samples.
- p_m mic position; r_k source position; r_k = ‖r_k‖; u_k = r_k/r_k; r_mk = r_k − p_m; r_mk = ‖r_mk‖; τ_mk = r_mk/c.
- s_k(t) source signal (referenced to 1 m); x_m mic signal; v noise (not n); y beamformer output.
- a(r, f), with a_m = e^{−jω‖r−p_m‖/c} / ‖r−p_m‖, steering vector for candidate r; A (M×K) with A_mk = a_m(r_k, f) exists only in model and simulator. x = As + v.
- w weights, y = wᴴx; R = E[xxᴴ]; P(r) = wᴴRw.
- Sign convention: delay = e^{−jωτ}.
- Avoided clashes: noise v (not n), no wavevector k (use ωτ), spacing called "pitch" (not d), mic position p (not m).
- Rust names: `mic_pos[m]`, `src_pos[k]`, `mic_to_src`, `range`, `delay`, `steering`, `steering_matrix`, `weights`, `covariance`, `power_map`, `SPEED_OF_SOUND`, `sample_rate`; estimates get suffix `_est`. Symbol goes in the doc comment (`/// r_mk`).

**Proposed, not yet in `notation.md`:**
- Mark the domain only by the argument, never by case: x_m(t), x_m[n], x_m(f), x_m(ℓ, f). Drop X_m(f).
- N = STFT length; frames ℓ = 1…L. R̂(f) = (1/L) Σ_ℓ x(ℓ,f) xᴴ(ℓ,f).
- Noise components q = 1…Q (`n_components`) within one noise source: s_k(t) = Σ_q a·sin(2π f_q t + φ_q), a = √(2/Q) · p_rms.
- The array-processing section states "all quantities per (ℓ, f); arguments dropped".

## Blog (Zola, systemscape.com)

- The post must be a page bundle: `content/blog/2026/acoustic-camera/index.md`, not `_index.md` (sections don't accept `date`).
- Math is escaped in backticks: `` `$…$` ``, `` `$$…$$` ``.
- Figure: `geometry.typ`, CeTZ **0.5.2**, `ortho(x: 20deg, y: -55deg)`, `length: 1.6cm`, 9 pt text; points drawn after the projection so they stay round. Compile with `typst compile geometry.typ geometry.svg`. Planned second figure: top-down far field (plane wavefronts, u_k, projection p_m·u_k).