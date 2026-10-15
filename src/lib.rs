pub mod camera;
pub mod mic_array;

pub type Vector3 = nalgebra::Vector3<f64>;

/// Speed of sound
// TODO(julian): Notation says SPEED_OF_SOUND, align it 
pub const C_SOUND: f64 = 343.0;

pub fn distance_to_delay(distance: f64) -> f64 {
    distance / C_SOUND
}

pub trait FrameSource {
    fn sample_rate(&self) -> u32;
    fn channels(&self) -> usize;
    /// Interleaved, len = frames * channels; returns frames read.
    fn read(&mut self, buf: &mut [f32]) -> std::io::Result<usize>;
}

pub trait ImageSource {
    fn next_frame(&mut self) -> Option<VideoFrame>;
}

pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    /// RGBA8, row-major, origin top-left; len = width * height * 4.
    pub pixels: Vec<u8>,
}

pub trait Beamformer {}
