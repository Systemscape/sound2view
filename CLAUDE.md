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

pub trait ImageSource { fn next_frame(&mut self) -> Option<VideoFrame>; }

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
- `crates/app` (package `sound2view-app`, binary `sound2view`): `main.rs` (CLI), `config.rs`, `gui.rs`, `theme.rs`.

Module layout of `crates/core/src` (`foo.rs` + `foo/` style, no `mod.rs`); `main.rs` lives in `crates/app/src`:

```
src/
  lib.rs, pipeline.rs, render.rs, test_util.rs
  geometry/   camera, mic_array, scan_grid
  source/     capture, simulator/ (scene, signal, reference)
  dsp/        stft, covariance, beamformer/ (delay_and_sum, functional, clean_sc)
```

Dependencies point one way: `geometry` ← `dsp` ← `pipeline` → `render`; `source` only needs `geometry`.

- `main.rs`: CLI. `sound2view [scene.toml]` (default `config.toml` = simulator scene), `mic` (UMA-16 live, `MIC_MAPS` 20 PNGs), `probe`, `camprobe`, `camshot`, `gui`. Loads `sound2view.toml` (`config::CONFIG_PATH`) for all modes except the probes. `type BoxError = Box<dyn Error + Send + Sync>`.
  - `run_configured(&mut source, sample_rate, &config, on_map) -> Result<(), PipelineError>`: builds the `Pipeline` with `config.pipeline`, `config.camera.camera()` and the configured beamformer (`match` on `Method`: `Das::new(.., true, TrueLocation)`, `Functional::new(.., nu)` or `CleanSc::new(.., clean_gain, clean_max_iter, clean_stop_db, clean_beam_cells)`), then `run`. One `run` instantiation per method. Used by CLI and GUI; the CLI also uses the configured method (default functional). Core errors convert into `main`'s `Box<dyn Error>` with plain `?`.
- `config.rs` (app): `Config { audio, camera, pipeline: PipelineConfig (core), beamformer, display }`, serde with `#[serde(default, deny_unknown_fields)]` everywhere: file, sections and entries optional, typos are errors. `Config::load(path)`: missing file → defaults (prints a note). `AudioConfig { device, sample_rate }`, `CameraConfig { name, width, height, fps, hfov_deg }` + `camera()` (pinhole via `from_hfov`, default `Mount`), `Method` (`das`/`functional`, `ALL`, `label`), `BeamformerConfig { method, nu }`, `DisplayConfig { db_range }`. Example with all defaults: `sound2view.toml` in the root.
- `gui.rs` (eframe 0.37: `App::ui(&mut self, ui, frame)`, panels via `egui::Panel::top(..).show(ui, ..)` and `CentralPanel::default().show(ui, ..)`; `show_inside` is deprecated). `sound2view gui`, window starts maximized (`ViewportBuilder::with_maximized`). `App::new(cc, config)`.
  - Camera thread (`spawn_camera`): `CameraSource::open_by_name` with the `CameraConfig` inside the thread, `sync_channel(1)` + `try_send` (drops frames), `request_repaint` per frame. Retries opening every 2 s (`RETRY`, polling), reopens after 10 consecutive frame errors (`CAMERA_MAX_FAILED_FRAMES`, e.g. unplugged).
  - `DeviceStatus` (`Opening`, `Missing(error)`, `Running`, `Lost(error)`) for camera and mic, each in an `Arc<Mutex<..>>`, set by its thread via `set_status` (with `request_repaint`). Top bar: dot + short text per device, full message as tooltip. Camera message centered over the dimmed last frame (or alone before the first frame); the map overlay is hidden while the mic is not `Running` (stale map).
  - Audio thread (`spawn_audio`): outer loop opens the mic (`Missing` + retry every 2 s on failure). Inner loop: read `PipelineSettings { beamformer, pipeline }` from an `Arc<Mutex<..>>`, put them into its `Config` copy, `run_configured`. `on_map` sends `MapData { size, power }` (raw power), returns `Break` when the settings changed (restart with the mic kept open, ALSA `hw:` is exclusive) or the GUI closed (thread ends). A pipeline error (e.g. mic unplugged: `CaptureError` via the source) drops the mic, sets `Lost(error)` and reopens after 2 s.
  - `theme.rs` (app): dark theme applied in `App::new` (`theme::apply`, forces `Theme::Dark`, `all_styles_mut`). Palette constants: `BACKGROUND` #0f1113 (behind the image), `PANEL` #181b1f (top bar), `TEXT`, `TEXT_WEAK`, `ACCENT` amber #f5a524 (matches the overlay), `OK`, `ERROR`; 6 px radius, larger text (body 15) and padding.
  - `ui`: top bar (`Frame` with `PANEL`): title, `controls()`: beamformer `ComboBox`, the selected method's parameter (functional: nu slider 1–100, log; CLEAN-SC: "Sources" = `clean_max_iter` 1–30; DAS: none), band sliders f_lo/f_hi (100–6000 Hz, log, each limited by the other, min width 100 Hz), overlay dB range slider (1–20 dB); mic and camera status right-aligned. Pipeline settings are copied to the shared settings only while no pointer button is down (dragging does not restart the pipeline on every step). The overlay RGBA is computed in the GUI (`render::map_rgba` with the current dB range) from the last `MapData`, so the dB range applies immediately, also between maps. Central panel (`BACKGROUND`, 12 px margin) paints camera and map texture (`TextureOptions::LINEAR`) with `painter.image` on the same rect: largest rect with the image aspect ratio, centered in the panel. GUI changes are not written back to the file.
- `pipeline.rs`: `PipelineConfig { fft_len 1024, memory_frames 40, map_every 10, f_lo_hz 900, f_hi_hz 4100, grid_width 64, grid_z 2.0 }` (serde, `Default` = these values; memory 40 ≈ 0.43 s, one map per 10 frames ≈ 0.11 s ≈ 9 maps/s, `power_map` ≈ 85 ms must fit).
  - `Pipeline<M, B: Beamformer<M>> { stream, cov, beamformer, bins, map_every, layout }` (private fields). `Pipeline::<16, B>::new(sample_rate, &PipelineConfig, camera: Camera, make_beamformer: impl FnOnce(Steering<16>) -> B)` builds scan grid, bins (upper end clipped to the last bin) and `Steering`, then the beamformer from it. The closure guarantees grid and beamformer match. Asserts f_lo < f_hi and map_every > 0 (TODO: `Result` with thiserror).
  - `run(&mut source, pipeline, on_map)`: read → `StftStream::push` → `SlidingCovariance::accumulate`; every `map_every` frames `power_map` on the current estimate (timed, printed) → `on_map(&power, &layout) -> Result<ControlFlow<()>, Box<dyn Error + Send + Sync>>`. Stops at `Ok(0)`, on `Break`, or on the first error. CLI: `save_maps(src_pos, max_maps)` in `main.rs` writes PNGs; GUI: `spawn_audio` in `gui.rs` sends raw power maps. Takes `&mut S` so `run_mic` can print `dropped()` afterwards.
- `render.rs`: `map_image(power, grid_width, grid_height, camera, src_pos) -> RgbImage`, independent of `Pipeline` (reusable for the GUI). Grid image → `imageops::resize` (Nearest) to camera size → red cross per true source (`camera.project`, skipped if z ≤ 0 or outside). `power_to_rgb`: gray, 0 dB = white, −15 dB and below = black (hard-coded `15.0`; to become a `DB_RANGE` constant). Sim and mic files overwrite each other.
- `source/capture.rs` (cpal 0.18.2; `device.id()`/`description()`, not `name()`):
  - `probe()`: per host, matching input devices (`UMA16_PATTERNS` "uma"/"minidsp") with their configs; devices with > 20 configs are plug devices, skipped. Result on Linux: `alsa:hw:CARD=UMA16v2,DEV=0`, 16 ch, I32, fixed rates 8000/11025/12000/16000/32000/44100/48000. Use the card name, not the index.
  - `CpalSource<M> { _stream, rx, pending: VecDeque<f64>, dropped: Arc<AtomicUsize>, stream_error: Arc<Mutex<Option<cpal::Error>>>, sample_rate }`. `new(pattern, sample_rate) -> Result<Self, CaptureError>`: first input device whose id or name contains the pattern, `StreamConfig` with M channels, default buffer. Callback converts samples to f64 (see below) and `try_send`s on a `sync_channel(64)`; when full the chunk is dropped and counted (audio thread never blocks). Error callback: `Xrun`, `RealtimeDenied`, `DeviceChanged` are printed (stream keeps running); any other `cpal::ErrorKind` (e.g. `DeviceNotAvailable` on unplug) is stored as the first fatal error. `dropped()` getter.
  - `CaptureError` (thiserror): `NotFound(pattern)`, `Device(cpal::Error)` (from), `Stream(cpal::Error)`, `NoData(Duration)`, `Closed`, `UnsupportedFormat(SampleFormat)`. cpal 0.18 has a single `cpal::Error` with `kind() -> ErrorKind`.
  - Device selection and format (for macOS): `new(pattern, ..)` matches the pattern against id or name (`description()`). The stream type follows `default_input_config().sample_format()`: I32 (UMA-16 on ALSA), F32 (CoreAudio reports inputs only as F32), I16; else `CaptureError::UnsupportedFormat`. Generic `build_stream::<T>` with `Callbacks { tx, dropped, stream_error }`, conversion via `Sample::to_sample::<f64>()` (integers / full scale, e.g. 2^31 for I32). The app's default `audio.device` is per platform: `hw:CARD=UMA16v2` on Linux (ALSA lists the UMA-16 several times), `UMA16` on macOS.
  - `read`: when `pending` is empty, `next_chunk` waits with `recv_timeout` (200 ms polls): error on a stored stream error, `Closed` (callback dropped), or `NoData` after 2 s without data (unplug without error callback). Errors go out as `io::Error::other(CaptureError)` (`FrameSource::read` stays `io::Result`). No draining with `try_recv`, so a slow reader drops chunks instead of growing latency. Copies `min(buf.len(), pending.len())` samples via `as_slices` + `copy_from_slice`, then `drain(..total)`. cpal chunks are whole frames (asserted), so `pending` stays a multiple of M.
  - Measured: chunks ≈ 180 frames (≈ 4 ms), queue ≈ 0.25 s. Release keeps up, 0 dropped.
  - Threads, not async: cpal owns the audio thread, the channel is the boundary. With egui: worker thread runs `run` and sends maps to the GUI thread (`try_recv` per frame or `Arc<Mutex<latest map>>`).

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
- `dsp/beamformer/functional.rs`: `Functional<M> { steering, nu, inv_norm2 }` (Dougherty 2014), `new(steering, nu)`. Per bin R^{1/ν} = U diag(max(λ, 0)^{1/ν}) Uᴴ via `symmetric_eigen` + `recompose`, then P += (wᴴR^{1/ν}w)^ν / M with w = a/‖a‖. Single source without noise: P = S·‖a_0‖²/M at the source (same as DAS `TrueLocation`); ν = 1 is DAS without diagonal removal. No diagonal removal. The `Beamformer` impl needs `Const<M>: DimSub<U1>` and `DefaultAllocator: Allocator<DimDiff<Const<M>, U1>>` (bounds of `symmetric_eigen`). Tests: hand-built R peak and level at p_0; with 10 dB noise peak at p_0 and fewer points within 3 dB than ν = 1. GUI default (`make_functional`, `FUNCTIONAL_NU` = 20 in `main.rs`); CLI modes still use DAS. Chosen over raising the band (male voices need the low frequencies).
- `dsp/beamformer/clean_sc.rs`: `CleanSc<M> { steering, gain, max_iter, stop_ratio, beam_sigma, weight_scale }` (Sijtsma 2007), `new(steering, gain, max_iter, stop_db, beam_sigma)` (asserts ranges). Per bin with true-location weights w = a/(‖a‖√M) (levels as DAS `TrueLocation`): dirty map P_j = wᴴRw; loop: j_max, h = R w_max / P_max, clean[j_max] += gain·P_max, R −= gain·P_max h hᴴ, dirty map updated incrementally P_j −= gain·P_max|w_jᴴh|² (O(points·M), so CLEAN-SC costs about DAS plus the iterations); stops after `max_iter` or when P_max < first max · 10^(−stop_db/10). Bins in parallel (`into_par_iter().fold(..).reduce(..)`), clean maps summed, then separable Gaussian clean beam (`gaussian_blur`, peak 1, σ in grid cells, 0 = single points). No diagonal removal. Defaults (app): gain 0.9, 10 sources, 20 dB, σ 1.5 cells. Tests: single source with gain 1 is exactly one point (P = ‖a_0‖²/M); two incoherent sources at 1/6 and 5/6 of the width (bin 80, 3840 Hz) give the maximum of each half within one cell; blur keeps the peak and is symmetric.
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
- Cargo: `nalgebra` (std, serde-serialize), `serde` (derive), `toml` (std, serde, parse), `rand` 0.10 (chacha), `rand_distr` 0.6, `realfft` 3.5, `cpal` 0.18.2, `image` 0.25 (png only), `nokhwa` 0.10.11, `rayon` 1.12, `thiserror` 2.0. App: `eframe` 0.37, `serde`, `toml`, `image`.
- `channel_of` is not stored; the mapping is baked into the order of `mic_pos`. Store the inverse `grid_pos` only when needed.

Status: simulator and live UMA-16 both run end to end (source → STFT → R̂ → DAS → PNG). Simulated maps match the markers (sources at z = 2 m on the grid plane, 3.4/3.6 kHz). Live with GUI: camera image plus map overlay; orientation verified (x was flipped, fixed).

Hardware findings (tone test indoors): a pure tone gives several bright spots. Causes: room reflections (coherent with a sine; band noise averages them out), DAS sidelobes of the uniform 4×4 at ≈ −13 dB (visible with a −15 dB display range; −6 dB shows the main lobe only), and a wide band summing noise-only bins for a single tone (narrow the band to the tone). Narrow band + smaller dB range made it much better.

Next:
1. Test on hardware: CLEAN-SC with two incoherent sources, `rayon` speedup (map times), mic unplug/replug (status `Lost` → reconnect), camera unplug/replug.
1a. macOS (Mac mini): untested. Prepared: platform default audio device, F32 capture, camera access via `video::initialize()` (nokhwa `nokhwa_initialize`, no-op on Linux) and `is_authorized()` (`CameraSource::new` returns `VideoError::NotAuthorized` until granted; CLI camera modes wait up to 10 s, the GUI camera thread retries), MJPEG `Exact` with `Closest` fallback (AVFoundation often lists only NV12/YUYV; a different resolution is printed). README has the macOS permission steps. CI (`.github/workflows/ci.yml`): fmt on Linux, clippy + release tests on ubuntu-latest and macos-latest. Risk: nokhwa's AVFoundation backend is less mature than V4L2.
2. Review the Claude-implemented parts (overlay, sliding covariance, functional, CLEAN-SC, GUI, config, errors).
3. Camera calibration (principal point, distortion) once the spots are small enough to matter.
4. Small things: rename the scene `config.toml` → `scene.toml`, "Save" button for GUI settings, deferred tests (diagonal removal, `TrueLocation`), unused `Stft::scale`.

Error handling (thiserror 2 in core): `CaptureError` (capture), `VideoError` (video), `SceneError` (scene), `PipelineError` (pipeline: `InvalidConfig(String)`, `Source(io::Error)`, `MapHandler(BoxError)`). `pub type BoxError = Box<dyn Error + Send + Sync>` in `lib.rs` for caller-defined parts (map handlers). All core errors are `Send + Sync` (worker thread → GUI). `ImageSource { type Error; fn next_frame(&mut self) -> Result<VideoFrame, Self::Error> }`. Config errors are `Result`s: `PipelineConfig::validate(sample_rate)` (called by `Pipeline::new`), app `Config::validate` (called by `Config::load`). Remaining asserts are internal invariants (bin counts, buffer lengths) plus `CleanSc::new`'s parameter ranges (the app validates them first). The probes still return `Box<dyn Error>` (CLI only).

Open: diag-removal tests and a `TrueLocation` hand-built test; unused `Stft::scale`; `DB_RANGE` constant, `FilterType::Triangle`; tests for `max_freq`/`validate`, min r_mk check in `Simulator::new`, `Covariance` doc comments, off-center sine reference test; replace plain `16`s in tests; Unicode in doc comments.

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