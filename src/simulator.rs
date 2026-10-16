use crate::FrameSource;

pub mod signal;

pub struct Simulator;

impl FrameSource for Simulator {
    fn sample_rate(&self) -> u32 {
        todo!()
    }

    fn channels(&self) -> usize {
        todo!()
    }

    fn read(&mut self, buf: &mut [f32]) -> std::io::Result<usize> {
        todo!()
    }
}
