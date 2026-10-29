use std::{
    ops::ControlFlow,
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, TrySendError, sync_channel},
    },
    thread,
    time::Duration,
};

use eframe::egui::{
    self, Align, Align2, Color32, ColorImage, FontId, Layout, Margin, Rect, RichText, Slider,
    TextureHandle, TextureOptions, pos2,
};
use sound2view_core::{
    ImageSource, VideoFrame,
    pipeline::{MapLayout, PipelineConfig},
    render,
    source::{capture::CpalSource, video::CameraSource},
};

use crate::{
    BoxError,
    config::{BeamformerConfig, CameraConfig, Config, Method},
    run_configured, theme,
};

/// Camera state, set by the camera thread and shown in the GUI
#[derive(Clone, Debug, PartialEq)]
enum CameraStatus {
    Opening,
    /// Opening failed, with the error; retried every `CAMERA_RETRY`
    Missing(String),
    Streaming,
    /// Too many failed frames, reopening
    Lost,
}

impl CameraStatus {
    fn message(&self) -> String {
        match self {
            CameraStatus::Opening => "Opening camera".to_string(),
            CameraStatus::Missing(e) => {
                format!("Camera not found, retrying every {} s\n{e}", CAMERA_RETRY.as_secs())
            }
            CameraStatus::Streaming => "Camera streaming".to_string(),
            CameraStatus::Lost => "Camera lost, reopening".to_string(),
        }
    }

    /// One or two words for the top bar
    fn short(&self) -> &'static str {
        match self {
            CameraStatus::Opening => "Camera opening",
            CameraStatus::Missing(_) => "Camera not found",
            CameraStatus::Streaming => "Camera live",
            CameraStatus::Lost => "Camera lost",
        }
    }

    fn color(&self) -> Color32 {
        match self {
            CameraStatus::Streaming => theme::OK,
            CameraStatus::Opening | CameraStatus::Missing(_) => theme::ACCENT,
            CameraStatus::Lost => theme::ERROR,
        }
    }
}

/// Small filled circle, vertically centered in the current row
fn status_dot(ui: &mut egui::Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 4.0, color);
}

/// Font size of the camera status text on the image
const STATUS_FONT_SIZE: f32 = 28.0;

/// Slider limits
const NU_RANGE: std::ops::RangeInclusive<f64> = 1.0..=100.0;
const DB_RANGE_RANGE: std::ops::RangeInclusive<f64> = 1.0..=20.0;
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
    /// Settings edited in the GUI
    settings: PipelineSettings,
    /// Settings read by the audio thread; updated from `settings` when no slider is dragged
    shared_settings: Arc<Mutex<PipelineSettings>>,
    /// Written by the camera thread
    camera_status: Arc<Mutex<CameraStatus>>,
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

        let status = self.camera_status.lock().unwrap().clone();

        let top_frame = egui::Frame::new()
            .fill(theme::PANEL)
            .inner_margin(Margin::symmetric(16, 10));
        egui::Panel::top("controls").frame(top_frame).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("sound2view").strong().size(18.0));
                ui.add_space(24.0);
                redraw_map |= self.controls(ui);

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(RichText::new(status.short()).color(theme::TEXT_WEAK));
                    status_dot(ui, status.color());
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
        egui::CentralPanel::default().frame(central_frame).show(ui, |ui| {
            let available = ui.available_rect_before_wrap();
            let painter = ui.painter();
            let font = FontId::proportional(STATUS_FONT_SIZE);

            let Some(texture) = &self.texture else {
                let color = ui.visuals().text_color();
                painter.text(available.center(), Align2::CENTER_CENTER, status.message(), font, color);
                return;
            };

            // Largest rectangle with the image aspect ratio, centered in the panel
            let size = texture.size_vec2();
            let scale = (available.width() / size.x).min(available.height() / size.y);
            let rect = Rect::from_center_size(available.center(), size * scale);
            let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));

            painter.image(texture.id(), rect, uv, Color32::WHITE);
            // The scan grid covers exactly the camera field of view, so the map
            // goes on the same rectangle as the image
            if let Some(map_texture) = &self.map_texture {
                painter.image(map_texture.id(), rect, uv, Color32::WHITE);
            }

            // The last frame stays visible but dimmed while the camera is not streaming
            if status != CameraStatus::Streaming {
                painter.rect_filled(rect, 0.0, Color32::from_black_alpha(160));
                painter.text(rect.center(), Align2::CENTER_CENTER, status.message(), font, Color32::WHITE);
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
        let camera_status = Arc::new(Mutex::new(CameraStatus::Opening));

        Self {
            rx: spawn_camera(
                cc.egui_ctx.clone(),
                config.camera.clone(),
                Arc::clone(&camera_status),
            ),
            texture: None,
            map_rx: spawn_audio(
                cc.egui_ctx.clone(),
                config.clone(),
                Arc::clone(&shared_settings),
            ),
            map_texture: None,
            last_map: None,
            db_range: config.display.db_range,
            settings,
            shared_settings,
            camera_status,
        }
    }

    /// Beamformer, nu, band and overlay range; returns true if the overlay needs a redraw
    fn controls(&mut self, ui: &mut egui::Ui) -> bool {
        let beamformer = &mut self.settings.beamformer;
        let pipeline = &mut self.settings.pipeline;

        label(ui, "Beamformer");
        egui::ComboBox::from_id_salt("method")
            .selected_text(beamformer.method.label())
            .show_ui(ui, |ui| {
                for method in Method::ALL {
                    ui.selectable_value(&mut beamformer.method, method, method.label());
                }
            });

        ui.add_space(16.0);
        label(ui, "nu");
        let is_functional = beamformer.method == Method::Functional;
        ui.add_enabled(
            is_functional,
            Slider::new(&mut beamformer.nu, NU_RANGE)
                .logarithmic(true)
                .max_decimals(0),
        )
        .on_hover_text("Exponent of functional beamforming: larger is sharper");

        ui.add_space(16.0);
        label(ui, "Band");
        // Each end is limited by the other, so the band never collapses
        ui.add(
            Slider::new(&mut pipeline.f_lo_hz, F_MIN_HZ..=pipeline.f_hi_hz - MIN_BAND_HZ)
                .logarithmic(true)
                .suffix(" Hz")
                .max_decimals(0),
        );
        ui.add(
            Slider::new(&mut pipeline.f_hi_hz, pipeline.f_lo_hz + MIN_BAND_HZ..=F_MAX_HZ)
                .logarithmic(true)
                .suffix(" Hz")
                .max_decimals(0),
        );

        ui.add_space(16.0);
        label(ui, "Overlay");
        ui.add(
            Slider::new(&mut self.db_range, DB_RANGE_RANGE)
                .suffix(" dB")
                .max_decimals(1),
        )
        .on_hover_text("Range below the map maximum that is shown")
        .changed()
    }
}

/// Muted label in front of a control
fn label(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).color(theme::TEXT_WEAK));
}

/// Wait between attempts to open the camera
const CAMERA_RETRY: Duration = Duration::from_secs(2);
/// Consecutive failed frames after which the camera counts as lost and is reopened
const CAMERA_MAX_FAILED_FRAMES: usize = 10;

/// Camera thread: sends frames, drops them while the GUI has not taken the previous one.
///
/// Retries opening every `CAMERA_RETRY` until the camera is there, and reopens it when it
/// fails (e.g. unplugged). A single bad frame does not count as lost. Writes the current
/// state to `status`.
fn spawn_camera(
    ctx: egui::Context,
    config: CameraConfig,
    status: Arc<Mutex<CameraStatus>>,
) -> Receiver<VideoFrame> {
    let (tx, rx) = sync_channel(1);

    thread::spawn(move || {
        let set_status = |new: CameraStatus| {
            *status.lock().unwrap() = new;
            ctx.request_repaint();
        };

        loop {
            let camera =
                CameraSource::open_by_name(&config.name, config.width, config.height, config.fps);
            let mut camera = match camera {
                Ok(camera) => {
                    set_status(CameraStatus::Streaming);
                    camera
                }
                Err(e) => {
                    set_status(CameraStatus::Missing(e.to_string()));
                    thread::sleep(CAMERA_RETRY);
                    continue;
                }
            };

            let mut failed = 0;
            while failed < CAMERA_MAX_FAILED_FRAMES {
                let Some(frame) = camera.next_frame() else {
                    failed += 1;
                    continue;
                };
                failed = 0;
                match tx.try_send(frame) {
                    Ok(()) => ctx.request_repaint(),
                    Err(TrySendError::Full(_)) => {}
                    Err(TrySendError::Disconnected(_)) => return,
                }
            }
            set_status(CameraStatus::Lost);
        }
    });

    rx
}

/// Audio thread: UMA-16 capture and pipeline, sends each power map to the GUI.
///
/// The mic stays open; when `settings` change, `run` stops after the next map and restarts with
/// a new pipeline (possibly of another beamformer type). Source and pipeline are created inside
/// the thread (`cpal::Stream` is not `Send` everywhere).
fn spawn_audio(
    ctx: egui::Context,
    mut config: Config,
    settings: Arc<Mutex<PipelineSettings>>,
) -> Receiver<MapData> {
    let (tx, rx) = sync_channel(1);

    thread::spawn(move || {
        let audio = &config.audio;
        let mut mic = match CpalSource::<16>::new(&audio.device, audio.sample_rate) {
            Ok(mic) => mic,
            Err(e) => {
                eprintln!("audio capture: {e}");
                return;
            }
        };
        let sample_rate = audio.sample_rate as f64;

        loop {
            let current = settings.lock().unwrap().clone();
            config.beamformer = current.beamformer.clone();
            config.pipeline = current.pipeline.clone();
            let mut gui_closed = false;

            let on_map = |power: &[f64], layout: &MapLayout| -> Result<ControlFlow<()>, BoxError> {
                let map = MapData {
                    size: [layout.grid_width, layout.grid_height],
                    power: power.to_vec(),
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

            if let Err(e) = result {
                eprintln!("audio pipeline: {e}");
                break;
            }
            // Restart only on a settings change; GUI closed or end of stream stops the thread
            if gui_closed || *settings.lock().unwrap() == current {
                break;
            }
        }
    });

    rx
}
