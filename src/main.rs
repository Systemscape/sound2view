pub type Vector3 = nalgebra::Vector3<f64>;

/// A Microphone array with `M` elements.
struct MicArray<const M: usize> {
    /// Microphone positions, indexed by channels from the audio stream
    pub mic_pos: [Vector3; M],
}

impl<const M: usize> MicArray<M> {
    /// Uniform rectangular array in the x-y plane, centred on the origin.
    ///
    /// ## Inputs
    /// - `cols`: number of columns
    /// - `rows`: number of rows
    /// - `pitch_x`: pitch in x-direction in meters
    /// - `pitch_y`: pitch in y-direction in meters
    /// - `channel_of`: function returning the channel number in the audio stream for a microphone at (row-index, column-index)
    pub fn uniform_rect(
        cols: usize,
        rows: usize,
        pitch_x: f64,
        pitch_y: f64,
        channel_of: impl Fn(usize, usize) -> usize,
    ) -> MicArray<M> {
        // Initialize to Option::None so that we can keep track of which mics have been assigned
        let mut pos = [None; M];

        for row in 0..rows {
            for col in 0..cols {
                let ch = channel_of(row, col);

                let p = Vector3::new(
                    // Row and col are zero-based indices.
                    // To map them to an x/y coordinate we have to
                    //
                    // For the first column, i.e., column 0, we have to move half the columns minus 1
                    // to the left.
                    // E.g. if we have 5 elements in the row, the 3rd element sits at the center.
                    // Then the 1st element must be exactly at -2*pitch_x
                    // The 2nd element must be 1*pitch_x further to the right, i.e., -1*pitch_x.
                    //
                    // If we have an even number or array elements in the row, there is no element at the center.
                    // E.g. if we have 4 elements in the row, the 1st element sits at -1.5*pitch_x and
                    // the 2nd element at -0.5*pitch_x.
                    //
                    // So we get for the position of the 1st element:
                    // -(cols-1)/2 * pitch_x
                    // And move to the nth element by adding col*pitch_x
                    //
                    // Remember: indices are zero-based, so column 1 move 0*pitch_x to the right.
                    (col as f64 - (cols - 1) as f64 / 2.0) * pitch_x,
                    (row as f64 - (rows - 1) as f64 / 2.0) * pitch_y,
                    0.0,
                );
                assert!(pos[ch].replace(p).is_none(), "channel {ch} assigned twice");
            }
        }

        Self {
            mic_pos: pos.map(|elem| elem.unwrap()),
        }
    }
}

impl MicArray<16> {
    /// Helper to create the UMA-16 Mic array
    pub fn uma16() -> MicArray<16> {
        MicArray::<16>::uniform_rect(4, 4, 0.042, 0.042, |x, y| x + 4 * y)
    }
}

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod test {
    use crate::{MicArray, Vector3};

    #[test]
    fn test_uma16() {
        let uma16 = MicArray::uma16();

        // The position should be -1.5 * 42mm in both, x and y.
        let expected_pos_ch0 = Vector3::new(-1.5 * 0.042, -1.5 * 0.042, 0.0);

        eprintln!("{:#?}", uma16.mic_pos[0]);
        eprintln!("expected: {:#?}", expected_pos_ch0);

        let norm = (uma16.mic_pos[0] - expected_pos_ch0).norm();
        eprintln!("norm: {norm}");

        // Assert the norm is almost zero (ignored float imprecisions)
        assert!(norm < 0.00001)
    }
}
