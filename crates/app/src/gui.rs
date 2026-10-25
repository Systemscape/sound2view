use std::{
    sync::mpsc::{Receiver, TrySendError, sync_channel},
    thread,
};

use eframe::egui::{self, ColorImage, TextureHandle};
use sound2view_core::{ImageSource, VideoFrame, source::video::CameraSource};

pub struct App {
    rx: Receiver<VideoFrame>,
    texture: Option<TextureHandle>,
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

            ()
        }
        match &self.texture {
            Some(texture) => {
                ui.add(egui::Image::new(texture).shrink_to_fit());
            }
            None => {
                ui.label("waiting for camera");
            }
        }
    }
}

impl App {
    pub fn new(cc: &eframe::CreationContext, camera_idx: u32) -> Self {
        let (tx, rx) = sync_channel(1);
        let ctx = cc.egui_ctx.clone();

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

        Self { rx, texture: None }
    }
}
