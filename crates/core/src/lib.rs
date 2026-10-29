pub mod dsp;
pub mod geometry;
pub mod pipeline;
pub mod render;
pub mod source;

#[cfg(test)]
pub(crate) mod test_util;

pub type Vector3 = nalgebra::Vector3<f64>;
pub type Point3 = nalgebra::Point3<f64>;
pub type Complex64 = nalgebra::Complex<f64>;

/// Speed of sound
// TODO(julian): Notation says SPEED_OF_SOUND, align it
pub const C_SOUND: f64 = 343.0;

pub fn distance_to_delay(distance: f64) -> f64 {
    distance / C_SOUND
}

pub trait FrameSource<const M: usize> {
    fn sample_rate(&self) -> f64;
    fn channels(&self) -> usize {
        M
    }
    /// Interleaved, len = frames * channels; returns frames read.
    fn read(&mut self, buf: &mut [f64]) -> std::io::Result<usize>;
}

pub trait ImageSource {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Blocks until the next frame is available.
    fn next_frame(&mut self) -> Result<VideoFrame, Self::Error>;
}

/// Boxed error for caller-defined parts (e.g. map handlers); `Send + Sync` to cross threads
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    /// RGBA8, row-major, origin top-left; len = width * height * 4.
    pub pixels: Vec<u8>,
}
