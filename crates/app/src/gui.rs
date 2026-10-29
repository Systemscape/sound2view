use std::{
    ops::ControlFlow,
    sync::mpsc::{Receiver, TrySendError, sync_channel},
    thread,
};

use eframe::egui::{self, Color32, ColorImage, Rect, TextureHandle, TextureOptions, pos2};
use sound2view_core::{
    ImageSource, VideoFrame,
    pipeline::{Pipeline, run},
    render,
    source::{capture::CpalSource, video::CameraSource},
};

use crate::{MIC_DEVICE, MIC_SAMPLE_RATE, make_functional};

/// Overlay range in dB below the map maximum; lower values are transparent
const OVERLAY_DB_RANGE: f64 = 3.0;

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

        match &self.texture {
            Some(texture) => {
                let rect = ui.add(egui::Image::new(texture).shrink_to_fit()).rect;
                // The scan grid covers exactly the camera field of view, so the map
                // goes on the same rectangle as the image
                if let Some(map_texture) = &self.map_texture {
                    let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
                    ui.painter()
                        .image(map_texture.id(), rect, uv, Color32::WHITE);
                }
            }
            None => {
                ui.label("waiting for camera");
            }
        }
    }
}

impl App {
    pub fn new(cc: &eframe::CreationContext, camera_idx: u32) -> Self {
        Self {
            rx: spawn_camera(cc.egui_ctx.clone(), camera_idx),
            texture: None,
            map_rx: spawn_audio(cc.egui_ctx.clone()),
            map_texture: None,
        }
    }
}

/// Camera thread: sends frames, drops them while the GUI has not taken the previous one.
fn spawn_camera(ctx: egui::Context, camera_idx: u32) -> Receiver<VideoFrame> {
    let (tx, rx) = sync_channel(1);

    thread::spawn(move || {
        // TODO(Julian): Add proper error handling here, show it on the GUI
        let mut camera = CameraSource::new(camera_idx).unwrap();

        loop {
            if let Some(frame) = camera.next_frame() {
                match tx.try_send(frame) {
                    Ok(()) => ctx.request_repaint(),
                    Err(TrySendError::Full(_)) => {}
                    Err(TrySendError::Disconnected(_)) => break,
                }
            }
        }
    });

    rx
}

/// Audio thread: UMA-16 capture and pipeline, sends each map as RGBA overlay.
///
/// Source and pipeline are created inside the thread (`cpal::Stream` is not `Send` everywhere).
fn spawn_audio(ctx: egui::Context) -> Receiver<MapImage> {
    let (tx, rx) = sync_channel(1);

    thread::spawn(move || {
        let mut mic = match CpalSource::<16>::new(MIC_DEVICE, MIC_SAMPLE_RATE) {
            Ok(mic) => mic,
            Err(e) => {
                eprintln!("audio capture: {e}");
                return;
            }
        };
        let pipeline = Pipeline::new(MIC_SAMPLE_RATE as f64, make_functional);

        let result = run(&mut mic, pipeline, |power, layout| {
            let map = MapImage {
                size: [layout.grid_width, layout.grid_height],
                rgba: render::map_rgba(power, OVERLAY_DB_RANGE),
            };
            match tx.try_send(map) {
                Ok(()) => ctx.request_repaint(),
                Err(TrySendError::Full(_)) => {}
                Err(TrySendError::Disconnected(_)) => return Ok(ControlFlow::Break(())),
            }
            Ok(ControlFlow::Continue(()))
        });
        if let Err(e) = result {
            eprintln!("audio pipeline: {e}");
        }
    });

    rx
}
