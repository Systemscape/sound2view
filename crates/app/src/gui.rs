use std::{
    error::Error,
    ops::ControlFlow,
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, TrySendError, sync_channel},
    },
    thread,
    time::Duration,
};

use eframe::egui::{
    self, Align, Align2, Color32, ColorImage, FontId, Layout, Margin, Rect, RichText, TextureHandle,
    TextureOptions, pos2,
};
use sound2view_core::{
    ImageSource, VideoFrame,
    pipeline::{MapLayout, Pipeline, run},
    render,
    source::{capture::CpalSource, video::CameraSource},
};

use crate::{MIC_DEVICE, MIC_SAMPLE_RATE, make_das, make_functional, theme};

/// Overlay range in dB below the map maximum; lower values are transparent
const OVERLAY_DB_RANGE: f64 = 3.0;

/// Beamforming method, selectable in the GUI
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Method {
    Das,
    Functional,
}

impl Method {
    const ALL: [Method; 2] = [Method::Das, Method::Functional];

    fn label(self) -> &'static str {
        match self {
            Method::Das => "Delay and sum",
            Method::Functional => "Functional",
        }
    }
}

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

/// Power map as RGBA at grid resolution, ready for a texture
struct MapImage {
    /// [width, height] in grid pixels
    size: [usize; 2],
    /// Unmultiplied RGBA, row-major
    rgba: Vec<u8>,
}

pub struct App {
    rx: Receiver<VideoFrame>,
    texture: Option<TextureHandle>,
    map_rx: Receiver<MapImage>,
    map_texture: Option<TextureHandle>,
    /// Method shown in the GUI
    method: Method,
    /// Method read by the audio thread, which restarts the pipeline when it changes
    shared_method: Arc<Mutex<Method>>,
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

        if let Some(map) = self.map_rx.try_iter().last() {
            let image = ColorImage::from_rgba_unmultiplied(map.size, &map.rgba);
            // Linear filtering interpolates the coarse grid smoothly to the image size
            if let Some(texture) = &mut self.map_texture {
                texture.set(image, TextureOptions::LINEAR);
            } else {
                self.map_texture =
                    Some(ui.ctx().load_texture("map", image, TextureOptions::LINEAR));
            }
        }

        let status = self.camera_status.lock().unwrap().clone();

        let previous = self.method;
        let top_frame = egui::Frame::new()
            .fill(theme::PANEL)
            .inner_margin(Margin::symmetric(16, 10));
        egui::Panel::top("controls").frame(top_frame).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("sound2view").strong().size(18.0));
                ui.add_space(24.0);
                ui.label(RichText::new("Beamformer").color(theme::TEXT_WEAK));
                egui::ComboBox::from_id_salt("method")
                    .selected_text(self.method.label())
                    .show_ui(ui, |ui| {
                        for method in Method::ALL {
                            ui.selectable_value(&mut self.method, method, method.label());
                        }
                    });

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(RichText::new(status.short()).color(theme::TEXT_WEAK));
                    status_dot(ui, status.color());
                });
            });
        });
        if self.method != previous {
            *self.shared_method.lock().unwrap() = self.method;
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
    pub fn new(cc: &eframe::CreationContext, camera_idx: u32) -> Self {
        theme::apply(&cc.egui_ctx);

        let method = Method::Functional;
        let shared_method = Arc::new(Mutex::new(method));
        let camera_status = Arc::new(Mutex::new(CameraStatus::Opening));
        Self {
            rx: spawn_camera(
                cc.egui_ctx.clone(),
                camera_idx,
                Arc::clone(&camera_status),
            ),
            texture: None,
            map_rx: spawn_audio(cc.egui_ctx.clone(), Arc::clone(&shared_method)),
            map_texture: None,
            method,
            shared_method,
            camera_status,
        }
    }
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
    camera_idx: u32,
    status: Arc<Mutex<CameraStatus>>,
) -> Receiver<VideoFrame> {
    let (tx, rx) = sync_channel(1);

    thread::spawn(move || {
        let set_status = |new: CameraStatus| {
            *status.lock().unwrap() = new;
            ctx.request_repaint();
        };

        loop {
            let mut camera = match CameraSource::new(camera_idx) {
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

/// Audio thread: UMA-16 capture and pipeline, sends each map as RGBA overlay.
///
/// The mic stays open; when `method` changes, `run` stops after the next map and restarts with a
/// new pipeline of the other beamformer type. Source and pipeline are created inside the thread
/// (`cpal::Stream` is not `Send` everywhere).
fn spawn_audio(ctx: egui::Context, method: Arc<Mutex<Method>>) -> Receiver<MapImage> {
    let (tx, rx) = sync_channel(1);

    thread::spawn(move || {
        let mut mic = match CpalSource::<16>::new(MIC_DEVICE, MIC_SAMPLE_RATE) {
            Ok(mic) => mic,
            Err(e) => {
                eprintln!("audio capture: {e}");
                return;
            }
        };
        let sample_rate = MIC_SAMPLE_RATE as f64;

        loop {
            let current = *method.lock().unwrap();
            let mut gui_closed = false;

            let on_map = |power: &[f64],
                          layout: &MapLayout|
             -> Result<ControlFlow<()>, Box<dyn Error + Send + Sync>> {
                let map = MapImage {
                    size: [layout.grid_width, layout.grid_height],
                    rgba: render::map_rgba(power, OVERLAY_DB_RANGE),
                };
                match tx.try_send(map) {
                    Ok(()) => ctx.request_repaint(),
                    Err(TrySendError::Full(_)) => {}
                    Err(TrySendError::Disconnected(_)) => {
                        gui_closed = true;
                        return Ok(ControlFlow::Break(()));
                    }
                }
                if *method.lock().unwrap() != current {
                    return Ok(ControlFlow::Break(()));
                }
                Ok(ControlFlow::Continue(()))
            };

            // One arm per beamformer type: static dispatch, no `dyn`
            let result = match current {
                Method::Das => run(&mut mic, Pipeline::new(sample_rate, make_das), on_map),
                Method::Functional => {
                    run(&mut mic, Pipeline::new(sample_rate, make_functional), on_map)
                }
            };

            if let Err(e) = result {
                eprintln!("audio pipeline: {e}");
                break;
            }
            // Restart only on a method change; GUI closed or end of stream stops the thread
            if gui_closed || *method.lock().unwrap() == current {
                break;
            }
        }
    });

    rx
}
