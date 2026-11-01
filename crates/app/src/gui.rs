use std::{
    collections::HashMap,
    ops::{ControlFlow, RangeInclusive},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU32, Ordering},
        mpsc::{Receiver, TrySendError, sync_channel},
    },
    thread,
    time::Duration,
};

use eframe::egui::{
    self, Align, Align2, Color32, ColorImage, FontId, Layout, Margin, Rect, RichText, Slider,
    Stroke, TextureHandle, TextureOptions, pos2,
};
use sound2view_core::{
    BoxError, ImageSource, VideoFrame,
    peaks::PeakConfig,
    pipeline::{MapLayout, PipelineConfig},
    render,
    source::{
        capture::CpalSource,
        video::{CameraEntry, CameraSource, list_cameras},
    },
    tracking::{Track, TrackerConfig},
};

use crate::{
    config::{BeamformerConfig, CameraConfig, Config, Method},
    load_speech, run_configured,
    speech::SpeechStatus,
    theme,
};

/// State of the camera or the mic, set by its thread and shown in the GUI
#[derive(Clone, Debug, PartialEq)]
enum DeviceStatus {
    Opening,
    /// Opening failed, with the error; retried every `RETRY`
    Missing(String),
    Running,
    /// Failed while running, with the error; reopening
    Lost(String),
}

impl DeviceStatus {
    /// Full message, e.g. over the image; `device` is "Camera" or "Mic"
    fn message(&self, device: &str) -> String {
        let retry = RETRY.as_secs();
        match self {
            DeviceStatus::Opening => format!("Opening {}", device.to_lowercase()),
            DeviceStatus::Missing(e) => {
                format!("{device} not found, retrying every {retry} s\n{e}")
            }
            DeviceStatus::Running => format!("{device} live"),
            DeviceStatus::Lost(e) => format!("{device} lost, reopening\n{e}"),
        }
    }

    /// One or two words for the top bar
    fn short(&self, device: &str) -> String {
        match self {
            DeviceStatus::Opening => format!("{device} opening"),
            DeviceStatus::Missing(_) => format!("{device} not found"),
            DeviceStatus::Running => format!("{device} live"),
            DeviceStatus::Lost(_) => format!("{device} lost"),
        }
    }

    fn color(&self) -> Color32 {
        match self {
            DeviceStatus::Running => theme::OK,
            DeviceStatus::Opening | DeviceStatus::Missing(_) => theme::ACCENT,
            DeviceStatus::Lost(_) => theme::ERROR,
        }
    }
}

/// Status dot with short text; the full message as tooltip
fn status_indicator(ui: &mut egui::Ui, status: &DeviceStatus, device: &str) {
    ui.label(RichText::new(status.short(device)).color(theme::TEXT_WEAK))
        .on_hover_text(status.message(device));
    let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
    ui.painter()
        .circle_filled(rect.center(), 4.0, status.color());
}

/// Wait between attempts to open the camera or the mic
const RETRY: Duration = Duration::from_secs(2);
/// Consecutive failed camera frames after which the camera counts as lost and is reopened
const CAMERA_MAX_FAILED_FRAMES: usize = 10;

/// Font size of the camera status text on the image
const STATUS_FONT_SIZE: f32 = 28.0;

/// Tooltips of the controls, shown on the label and the control
const NU_HELP: &str = "Exponent of functional beamforming. Larger values make sources sharper \
    and suppress sidelobes, but also amplify model errors (uncalibrated mics, wrong focus \
    distance): a source can then fade or split. 1 is delay and sum.";
const NOISE_REMOVAL_HELP: &str = "Subtracts an estimate of the uncorrelated noise (mic \
    self-noise, diffuse room sound) in every frequency bin before beamforming. Without it, \
    noise widens the source zones. Assumes fewer than 8 sources per bin.";
const EQUAL_BINS_HELP: &str = "Gives every frequency bin the same weight. Otherwise the low \
    bins, where speech has most of its energy but the array has almost no resolution, dominate \
    and the zones get wide. Bins more than 20 dB below the strongest are left out.";
const BAND_HELP: &str = "Frequency range summed into the map. Lower frequencies give wider \
    zones (the array is only 12.6 cm wide); above about 4 kHz ghost sources (grating lobes) can \
    appear. For a single tone, narrow the band around it.";
const OVERLAY_HELP: &str = "Dynamic range of the overlay: everything more than this many dB \
    below the loudest point is transparent. Smaller values show only the strongest sources.";
/// Radius of the dot per estimated source in screen points, and size of its level label
const PEAK_RADIUS: f32 = 7.0;
const PEAK_FONT_SIZE: f32 = 14.0;

/// Slider limits
const NU_RANGE: RangeInclusive<f64> = 1.0..=100.0;
const CLEAN_MAX_ITER_RANGE: RangeInclusive<usize> = 1..=30;
const DB_RANGE_RANGE: RangeInclusive<f64> = 1.0..=20.0;
/// Speech threshold limits; the lower one is also kept above `speech.release`, so speech can end
const SPEECH_THRESHOLD_MIN: f32 = 0.3;
const SPEECH_THRESHOLD_MAX: f32 = 0.99;
/// Band limits in Hz; above about 6 kHz grating lobes appear in the image
const F_MIN_HZ: f64 = 100.0;
const F_MAX_HZ: f64 = 6000.0;
/// Smallest band width in Hz
const MIN_BAND_HZ: f64 = 100.0;

/// Settings whose change restarts the pipeline in the audio thread
#[derive(Clone, Debug, PartialEq)]
struct PipelineSettings {
    beamformer: BeamformerConfig,
    pipeline: PipelineConfig,
}

/// Power map from the audio thread, row-major like the scan grid
struct MapData {
    /// [width, height] in grid pixels
    size: [usize; 2],
    power: Vec<f64>,
    /// Tracks of this map, without their audio
    tracks: Vec<Track>,
    /// Speech state per track id; empty without speech detection
    speech: HashMap<u64, SpeechStatus>,
}

pub struct App {
    rx: Receiver<VideoFrame>,
    texture: Option<TextureHandle>,
    map_rx: Receiver<MapData>,
    map_texture: Option<TextureHandle>,
    /// Kept to redraw the overlay when the dB range changes between maps
    last_map: Option<MapData>,
    /// Overlay range in dB below the map maximum; applies immediately
    db_range: f64,
    /// Peak search of the pipeline, for the tooltip; `enabled` (show the dots) is toggled in the
    /// GUI
    peaks: PeakConfig,
    /// Tracking of the pipeline, for the tooltip
    tracking: TrackerConfig,
    /// Speech detection on (`speech.enabled`); the threshold slider is shown only then
    speech_enabled: bool,
    /// `speech.release`, lower limit of the threshold slider
    speech_release: f32,
    /// Speech threshold edited in the GUI
    speech_threshold: f32,
    /// Read by the audio thread once per map, as f32 bits; no pipeline restart
    shared_speech_threshold: Arc<AtomicU32>,
    /// Settings edited in the GUI
    settings: PipelineSettings,
    /// Settings read by the audio thread; updated from `settings` when no slider is dragged
    shared_settings: Arc<Mutex<PipelineSettings>>,
    /// Camera selection, status and list, shared with the camera thread
    camera: Arc<CameraShared>,
    /// `camera.name` from the config, shown for `CameraChoice::Auto`
    camera_name: String,
    /// Written by the audio thread
    audio_status: Arc<Mutex<DeviceStatus>>,
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut eframe::egui::Ui, _frame: &mut eframe::Frame) {
        if let Some(frame) = self.rx.try_iter().last() {
            let image = ColorImage::from_rgba_unmultiplied(
                [frame.width as _, frame.height as _],
                &frame.pixels,
            );

            if let Some(texture) = &mut self.texture {
                texture.set(image, Default::default());
            } else {
                self.texture = Some(ui.ctx().load_texture("camera", image, Default::default()));
            }
        }

        let mut redraw_map = false;
        if let Some(map) = self.map_rx.try_iter().last() {
            self.last_map = Some(map);
            redraw_map = true;
        }

        let camera_status = self.camera.status.lock().unwrap().clone();
        let audio_status = self.audio_status.lock().unwrap().clone();

        let top_frame = egui::Frame::new()
            .fill(theme::PANEL)
            .inner_margin(Margin::symmetric(16, 10));
        egui::Panel::top("controls")
            .frame(top_frame)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("sound2view").strong().size(18.0));
                    ui.add_space(24.0);
                    self.camera_control(ui);
                    ui.add_space(16.0);
                    redraw_map |= self.controls(ui);

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        status_indicator(ui, &camera_status, "Camera");
                        ui.add_space(12.0);
                        status_indicator(ui, &audio_status, "Mic");
                    });
                });
            });

        // Apply pipeline settings only when no slider is dragged: each change restarts the
        // pipeline, and the sliding covariance needs a few frames for the first map
        let pointer_down = ui.input(|i| i.pointer.any_down());
        if !pointer_down {
            let mut shared = self.shared_settings.lock().unwrap();
            if *shared != self.settings {
                *shared = self.settings.clone();
            }
        }

        if redraw_map && let Some(map) = &self.last_map {
            let rgba = render::map_rgba(&map.power, self.db_range);
            let image = ColorImage::from_rgba_unmultiplied(map.size, &rgba);
            // Linear filtering interpolates the coarse grid smoothly to the image size
            if let Some(texture) = &mut self.map_texture {
                texture.set(image, TextureOptions::LINEAR);
            } else {
                self.map_texture =
                    Some(ui.ctx().load_texture("map", image, TextureOptions::LINEAR));
            }
        }

        let central_frame = egui::Frame::new()
            .fill(theme::BACKGROUND)
            .inner_margin(Margin::same(12));
        egui::CentralPanel::default()
            .frame(central_frame)
            .show(ui, |ui| {
                let available = ui.available_rect_before_wrap();
                let painter = ui.painter();
                let font = FontId::proportional(STATUS_FONT_SIZE);
                let camera_message = camera_status.message("Camera");

                let Some(texture) = &self.texture else {
                    let color = ui.visuals().text_color();
                    painter.text(
                        available.center(),
                        Align2::CENTER_CENTER,
                        camera_message,
                        font,
                        color,
                    );
                    return;
                };

                // Largest rectangle with the image aspect ratio, centered in the panel
                let size = texture.size_vec2();
                let scale = (available.width() / size.x).min(available.height() / size.y);
                let rect = Rect::from_center_size(available.center(), size * scale);
                let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));

                painter.image(texture.id(), rect, uv, Color32::WHITE);
                // The scan grid covers exactly the camera field of view, so the map goes on the
                // same rectangle as the image. Without a running mic the last map would be stale.
                if audio_status == DeviceStatus::Running
                    && let Some(map_texture) = &self.map_texture
                {
                    painter.image(map_texture.id(), rect, uv, Color32::WHITE);
                    if self.peaks.enabled
                        && let Some(map) = &self.last_map
                    {
                        draw_tracks(painter, rect, map.size, &map.tracks, &map.speech);
                    }
                }

                // The last frame stays visible but dimmed while the camera is not streaming
                if camera_status != DeviceStatus::Running {
                    painter.rect_filled(rect, 0.0, Color32::from_black_alpha(160));
                    painter.text(
                        rect.center(),
                        Align2::CENTER_CENTER,
                        camera_message,
                        font,
                        Color32::WHITE,
                    );
                }
            });
    }
}

impl App {
    pub fn new(cc: &eframe::CreationContext, config: Config) -> Self {
        theme::apply(&cc.egui_ctx);

        let settings = PipelineSettings {
            beamformer: config.beamformer.clone(),
            pipeline: config.pipeline.clone(),
        };
        let shared_settings = Arc::new(Mutex::new(settings.clone()));
        let camera = Arc::new(CameraShared {
            choice: Mutex::new(CameraChoice::Auto),
            status: Mutex::new(DeviceStatus::Opening),
            cameras: Mutex::new(Vec::new()),
        });
        let audio_status = Arc::new(Mutex::new(DeviceStatus::Opening));
        let shared_speech_threshold = Arc::new(AtomicU32::new(config.speech.threshold.to_bits()));

        Self {
            rx: spawn_camera(
                cc.egui_ctx.clone(),
                config.camera.clone(),
                Arc::clone(&camera),
            ),
            texture: None,
            map_rx: spawn_audio(
                cc.egui_ctx.clone(),
                config.clone(),
                Arc::clone(&shared_settings),
                Arc::clone(&audio_status),
                Arc::clone(&shared_speech_threshold),
            ),
            map_texture: None,
            last_map: None,
            db_range: config.display.db_range,
            peaks: config.peaks.clone(),
            tracking: config.tracking.clone(),
            speech_enabled: config.speech.enabled,
            speech_release: config.speech.release,
            speech_threshold: config.speech.threshold,
            shared_speech_threshold,
            settings,
            shared_settings,
            camera,
            camera_name: config.camera.name.clone(),
            audio_status,
        }
    }

    /// Camera dropdown: "Auto" (first camera matching `camera.name`) or one listed camera.
    /// The field of view in the config belongs to the configured camera, so the overlay only
    /// fits other cameras with the same field of view.
    fn camera_control(&self, ui: &mut egui::Ui) {
        let cameras = self.camera.cameras.lock().unwrap().clone();
        let current = self.camera.choice.lock().unwrap().clone();
        let auto_label = format!("Auto ({})", self.camera_name);
        let entry_label = |entry: &CameraEntry| format!("{} ({})", entry.name, entry.index);

        let selected_text = match &current {
            CameraChoice::Auto => auto_label.clone(),
            CameraChoice::Index(index) => cameras
                .iter()
                .find(|entry| entry.index == *index)
                .map_or_else(|| format!("Camera {index}"), entry_label),
        };

        let mut choice = current.clone();
        label(ui, "Camera").on_hover_text(format!(
            "Auto opens the first camera whose name contains \"{}\" (camera.name). The overlay \
             only lines up with cameras that match the configured field of view or calibration.",
            self.camera_name
        ));
        egui::ComboBox::from_id_salt("camera")
            .selected_text(selected_text)
            .width(CAMERA_COMBO_WIDTH)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut choice, CameraChoice::Auto, auto_label);
                for entry in &cameras {
                    let label = entry_label(entry);
                    ui.selectable_value(&mut choice, CameraChoice::Index(entry.index), label);
                }
            });

        if choice != current {
            *self.camera.choice.lock().unwrap() = choice;
            set_status(ui.ctx(), &self.camera.status, DeviceStatus::Opening);
        }
    }

    /// Beamformer and its parameters, band, overlay range and peak toggle; returns true if the
    /// overlay needs a redraw
    fn controls(&mut self, ui: &mut egui::Ui) -> bool {
        let beamformer = &mut self.settings.beamformer;
        let pipeline = &mut self.settings.pipeline;

        let method_help = beamformer.method.description();
        label(ui, "Beamformer").on_hover_text(method_help);
        egui::ComboBox::from_id_salt("method")
            .selected_text(beamformer.method.label())
            .show_ui(ui, |ui| {
                for method in Method::ALL {
                    ui.selectable_value(&mut beamformer.method, method, method.label())
                        .on_hover_text(method.description());
                }
            })
            .response
            .on_hover_text(method_help);

        // Parameter of the selected method
        match beamformer.method {
            Method::Das => {}
            Method::Functional => {
                ui.add_space(16.0);
                label(ui, "nu").on_hover_text(NU_HELP);
                ui.add(
                    Slider::new(&mut beamformer.nu, NU_RANGE)
                        .logarithmic(true)
                        .max_decimals(0),
                )
                .on_hover_text(NU_HELP);
                ui.checkbox(&mut beamformer.functional_noise_removal, "Noise removal")
                    .on_hover_text(NOISE_REMOVAL_HELP);
                ui.checkbox(&mut beamformer.functional_equal_bins, "Equal bins")
                    .on_hover_text(EQUAL_BINS_HELP);
            }
            Method::CleanSc => {
                ui.add_space(16.0);
                let sources_help = format!(
                    "At most this many sources are removed per frequency bin. More finds weaker \
                     sources, but also more reflections. Bins whose strongest point is more than \
                     {} dB below the strongest bin add nothing.",
                    beamformer.clean_stop_db
                );
                label(ui, "Sources").on_hover_text(&sources_help);
                ui.add(Slider::new(
                    &mut beamformer.clean_max_iter,
                    CLEAN_MAX_ITER_RANGE,
                ))
                .on_hover_text(sources_help);
            }
        }

        ui.add_space(16.0);
        label(ui, "Band").on_hover_text(BAND_HELP);
        // Each end is limited by the other, so the band never collapses
        ui.add(
            Slider::new(
                &mut pipeline.f_lo_hz,
                F_MIN_HZ..=pipeline.f_hi_hz - MIN_BAND_HZ,
            )
            .logarithmic(true)
            .suffix(" Hz")
            .max_decimals(0),
        )
        .on_hover_text(BAND_HELP);
        ui.add(
            Slider::new(
                &mut pipeline.f_hi_hz,
                pipeline.f_lo_hz + MIN_BAND_HZ..=F_MAX_HZ,
            )
            .logarithmic(true)
            .suffix(" Hz")
            .max_decimals(0),
        )
        .on_hover_text(BAND_HELP);

        ui.add_space(16.0);
        label(ui, "Overlay").on_hover_text(OVERLAY_HELP);
        let db_range_changed = ui
            .add(
                Slider::new(&mut self.db_range, DB_RANGE_RANGE)
                    .suffix(" dB")
                    .max_decimals(1),
            )
            .on_hover_text(OVERLAY_HELP)
            .changed();

        ui.add_space(16.0);
        ui.checkbox(&mut self.peaks.enabled, "Tracks")
            .on_hover_text(format!(
                "Dots at the tracked sources: local maxima of the map within {} dB of the \
                 loudest point, at least {} grid cells apart, at most {}, followed from map to \
                 map. A dot appears after {} maps in a row and stays up to {} maps after its \
                 peak is gone. Green: the audio from its direction is classified as speech. The \
                 label is the track number, the level relative to the loudest point and the \
                 speech probability.",
                self.peaks.range_db,
                self.peaks.min_dist_cells,
                self.peaks.max_count,
                self.tracking.min_hits,
                self.tracking.max_missed
            ));

        if self.speech_enabled {
            ui.add_space(16.0);
            let help = "Speech probability (smoothed) at which a track turns green. Raise it if \
                noise or music turns green, lower it if speech stays blue. Applies immediately.";
            label(ui, "Speech").on_hover_text(help);
            let min = SPEECH_THRESHOLD_MIN.max(self.speech_release + 0.05);
            let changed = ui
                .add(
                    Slider::new(&mut self.speech_threshold, min..=SPEECH_THRESHOLD_MAX)
                        .max_decimals(2),
                )
                .on_hover_text(help)
                .changed();
            if changed {
                self.shared_speech_threshold
                    .store(self.speech_threshold.to_bits(), Ordering::Relaxed);
            }
        }
        db_range_changed
    }
}

/// Dot with id and level per confirmed track; `rect` is where the map is drawn, `size` its grid
/// size
fn draw_tracks(
    painter: &egui::Painter,
    rect: Rect,
    [width, height]: [usize; 2],
    tracks: &[Track],
    speech: &HashMap<u64, SpeechStatus>,
) {
    let font = FontId::proportional(PEAK_FONT_SIZE);
    for track in tracks.iter().filter(|track| track.confirmed) {
        // Grid cell centers at (col + 0.5) / width of the image
        let x = rect.left() + (track.col as f32 + 0.5) / width as f32 * rect.width();
        let y = rect.top() + (track.row as f32 + 0.5) / height as f32 * rect.height();
        let center = pos2(x, y);
        let status = speech.get(&track.id);
        let fill = if status.is_some_and(|s| s.speech) {
            theme::SPEECH
        } else {
            theme::PEAK
        };
        painter.circle(center, PEAK_RADIUS, fill, Stroke::new(2.0, Color32::WHITE));
        let mut text = format!("#{} {:.1} dB", track.id, track.level_db);
        if let Some(status) = status {
            text += &format!(" {:.0} %", 100.0 * status.probability);
        }
        painter.text(
            pos2(x + PEAK_RADIUS + 4.0, y),
            Align2::LEFT_CENTER,
            text,
            font.clone(),
            Color32::WHITE,
        );
    }
}

/// Muted label in front of a control
fn label(ui: &mut egui::Ui, text: &str) -> egui::Response {
    ui.label(RichText::new(text).color(theme::TEXT_WEAK))
}

/// Set `status` and repaint, so the GUI shows it without waiting for the next frame
fn set_status(ctx: &egui::Context, status: &Mutex<DeviceStatus>, new: DeviceStatus) {
    *status.lock().unwrap() = new;
    ctx.request_repaint();
}

/// Which camera the camera thread opens
#[derive(Clone, Debug, PartialEq)]
enum CameraChoice {
    /// First camera whose name contains `camera.name` from the config
    Auto,
    /// Camera with this index, see `CameraEntry`
    Index(u32),
}

/// Camera state shared between the GUI and the camera thread
struct CameraShared {
    /// Set by the GUI; the thread reopens when it changes
    choice: Mutex<CameraChoice>,
    /// Set by the thread
    status: Mutex<DeviceStatus>,
    /// Cameras for the dropdown; refreshed by the thread on every open attempt
    cameras: Mutex<Vec<CameraEntry>>,
}

/// Width of the camera dropdown; camera names are long
const CAMERA_COMBO_WIDTH: f32 = 260.0;
/// The camera thread checks for another selection this often while it waits
const CHOICE_POLL: Duration = Duration::from_millis(100);

/// Camera thread: sends frames, drops them while the GUI has not taken the previous one.
///
/// Opens the camera selected in `shared.choice` and reopens when the selection changes. Retries
/// every `RETRY` until the camera is there, and reopens it when it fails (e.g. unplugged). A
/// single bad frame does not count as lost.
fn spawn_camera(
    ctx: egui::Context,
    config: CameraConfig,
    shared: Arc<CameraShared>,
) -> Receiver<VideoFrame> {
    let (tx, rx) = sync_channel(1);

    thread::spawn(move || {
        loop {
            let choice = shared.choice.lock().unwrap().clone();
            // Also while no camera can be opened, so a newly plugged one shows up
            if let Ok(cameras) = list_cameras() {
                *shared.cameras.lock().unwrap() = cameras;
            }

            let (width, height, fps) = (config.width, config.height, config.fps);
            let camera = match &choice {
                CameraChoice::Auto => CameraSource::open_by_name(&config.name, width, height, fps),
                CameraChoice::Index(index) => CameraSource::new(*index, width, height, fps),
            };
            let mut camera = match camera {
                Ok(camera) => {
                    set_status(&ctx, &shared.status, DeviceStatus::Running);
                    camera
                }
                Err(e) => {
                    set_status(&ctx, &shared.status, DeviceStatus::Missing(e.to_string()));
                    wait_for_other_choice(&shared.choice, &choice, RETRY);
                    continue;
                }
            };

            let mut failed = 0;
            let mut last_error = String::new();
            let mut size_checked = false;
            while failed < CAMERA_MAX_FAILED_FRAMES {
                // Another camera selected: drop this one (end of the loop body), then open it
                if *shared.choice.lock().unwrap() != choice {
                    break;
                }
                let frame = match camera.next_frame() {
                    Ok(frame) => frame,
                    Err(e) => {
                        failed += 1;
                        last_error = e.to_string();
                        continue;
                    }
                };
                failed = 0;
                // The overlay is stretched over the frame; intrinsics only fit the configured size
                if !size_checked {
                    size_checked = true;
                    if (frame.width, frame.height) != (width, height) {
                        eprintln!(
                            "camera delivers {}x{}, config expects {width}x{height}: \
                             overlay alignment is off",
                            frame.width, frame.height
                        );
                    }
                }
                match tx.try_send(frame) {
                    Ok(()) => ctx.request_repaint(),
                    Err(TrySendError::Full(_)) => {}
                    Err(TrySendError::Disconnected(_)) => return,
                }
            }
            if failed >= CAMERA_MAX_FAILED_FRAMES {
                set_status(&ctx, &shared.status, DeviceStatus::Lost(last_error));
            }
        }
    });

    rx
}

/// Sleep up to `timeout`, but return as soon as the GUI selects another camera than `current`
fn wait_for_other_choice(choice: &Mutex<CameraChoice>, current: &CameraChoice, timeout: Duration) {
    let mut waited = Duration::ZERO;
    while waited < timeout && *choice.lock().unwrap() == *current {
        thread::sleep(CHOICE_POLL);
        waited += CHOICE_POLL;
    }
}

/// Audio thread: UMA-16 capture and pipeline, sends each power map to the GUI.
///
/// Opens the mic with retries every `RETRY`. While it runs, a settings change stops `run` after
/// the next map and restarts it with a new pipeline (possibly of another beamformer type); the
/// mic stays open (ALSA `hw:` devices are exclusive). A pipeline error (e.g. mic unplugged)
/// closes the mic and reopens it. Source and pipeline are created inside the thread
/// (`cpal::Stream` is not `Send` everywhere).
fn spawn_audio(
    ctx: egui::Context,
    mut config: Config,
    settings: Arc<Mutex<PipelineSettings>>,
    status: Arc<Mutex<DeviceStatus>>,
    speech_threshold: Arc<AtomicU32>,
) -> Receiver<MapData> {
    let (tx, rx) = sync_channel(1);

    thread::spawn(move || {
        let sample_rate = config.audio.sample_rate as f64;
        // Loaded once; reset for every pipeline run, since track ids start over
        let mut speech = load_speech(&config.speech);

        loop {
            let mut mic =
                match CpalSource::<16>::new(&config.audio.device, config.audio.sample_rate) {
                    Ok(mic) => mic,
                    Err(e) => {
                        set_status(&ctx, &status, DeviceStatus::Missing(e.to_string()));
                        thread::sleep(RETRY);
                        continue;
                    }
                };
            set_status(&ctx, &status, DeviceStatus::Running);

            // Restart loop for settings changes; ends with the error that closes the mic
            let error = loop {
                let current = settings.lock().unwrap().clone();
                config.beamformer = current.beamformer.clone();
                config.pipeline = current.pipeline.clone();
                let mut gui_closed = false;
                if let Some(detector) = &mut speech {
                    detector.reset();
                }

                let on_map = |power: &[f64],
                              layout: &MapLayout,
                              tracks: &[Track]|
                 -> Result<ControlFlow<()>, BoxError> {
                        let speech = match &mut speech {
                            Some(detector) => {
                                let threshold = speech_threshold.load(Ordering::Relaxed);
                                detector.set_threshold(f32::from_bits(threshold));
                                detector.update(tracks)?;
                                detector.statuses()
                            }
                            None => HashMap::new(),
                        };
                        let tracks = tracks
                            .iter()
                            .map(|track| Track {
                                samples: Vec::new(),
                                ..track.clone()
                            })
                            .collect();
                        let map = MapData {
                            size: [layout.grid_width, layout.grid_height],
                            power: power.to_vec(),
                            tracks,
                            speech,
                        };
                        match tx.try_send(map) {
                            Ok(()) => ctx.request_repaint(),
                            Err(TrySendError::Full(_)) => {}
                            Err(TrySendError::Disconnected(_)) => {
                                gui_closed = true;
                                return Ok(ControlFlow::Break(()));
                            }
                        }
                        if *settings.lock().unwrap() != current {
                            return Ok(ControlFlow::Break(()));
                        }
                        Ok(ControlFlow::Continue(()))
                    };

                let result = run_configured(&mut mic, sample_rate, &config, on_map);

                if gui_closed {
                    return;
                }
                match result {
                    Err(e) => break e.to_string(),
                    // Settings changed: restart with the new ones
                    Ok(()) if *settings.lock().unwrap() != current => continue,
                    // A live source does not end; treat it like an error
                    Ok(()) => break "audio stream ended".to_string(),
                }
            };

            drop(mic);
            set_status(&ctx, &status, DeviceStatus::Lost(error));
            thread::sleep(RETRY);
        }
    });

    rx
}
