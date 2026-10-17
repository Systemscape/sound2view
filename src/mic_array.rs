use crate::Vector3;

/// A Microphone array with `M` elements.
#[derive(Clone, Debug)]
pub struct MicArray<const M: usize> {
    /// Microphone positions, indexed by channels from the audio stream
    pub mic_pos: [Vector3; M],
}

impl<const M: usize> MicArray<M> {
    /// Uniform rectangular array in the x-y plane, centered on the origin.
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
        assert_eq!(rows * cols, M);

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
                    // For the y-coordinate we have to swap it because row 0 is at the top, i.e., +y
                    //
                    // Remember: indices are zero-based, so column 1 move 0*pitch_x to the right.
                    (col as f64 - (cols - 1) as f64 / 2.0) * pitch_x,
                    ((rows - 1) as f64 / 2.0 - row as f64) * pitch_y,
                    0.0,
                );
                assert!(pos[ch].replace(p).is_none(), "channel {ch} assigned twice");
            }
        }

        Self {
            mic_pos: pos.map(|elem| elem.unwrap()),
        }
    }

    pub fn get_distance_from(&self, r: &Vector3) -> [f64; M] {
        self.mic_pos.map(|p_m| (r - p_m).norm())
    }
}

impl MicArray<16> {
    /// Helper to create the UMA-16 Mic array
    pub fn uma16() -> MicArray<16> {
        MicArray::<16>::uniform_rect(4, 4, 0.042, 0.042, |row, column| row + 4 * column)
    }
}

#[cfg(test)]
mod test {
    use crate::{Vector3, distance_to_delay, mic_array::MicArray};

    #[test]
    fn test_uma16_geometry() {
        let uma16 = MicArray::uma16();

        // The position should be -1.5 * 42mm in x and +1.5 * 42mm in y.
        //
        // TODO(julian): Once we know the real channel mapping, update this test
        let expected_pos_ch0 = Vector3::new(-1.5 * 0.042, 1.5 * 0.042, 0.0);
        let expected_pos_ch15 = Vector3::new(1.5 * 0.042, -1.5 * 0.042, 0.0);

        eprintln!("{:#?}", uma16.mic_pos[0]);
        eprintln!("expected: {:#?}", expected_pos_ch0);

        let norm_ch0 = (uma16.mic_pos[0] - expected_pos_ch0).norm();
        let norm_ch15 = (uma16.mic_pos[15] - expected_pos_ch15).norm();

        // Sanity check: use a different mic and computeh against ch15's expected position
        let norm_ch14_false = (uma16.mic_pos[14] - expected_pos_ch15).norm();

        // Assert the norm is almost zero (ignored float imprecisions)
        assert!(norm_ch0 < f64::EPSILON);
        assert!(norm_ch15 < f64::EPSILON);
        assert!(norm_ch14_false > f64::EPSILON);
    }

    #[test]
    fn test_relative_delay() {
        // Distance of source from origin in meters
        const DISTANCE: f64 = 100.0;

        let uma16 = MicArray::uma16();
        let p = uma16.mic_pos;

        let direction = Vector3::new(-1.0, 1.0, 2.0);

        // Unit vector in direction
        let u_k = direction / direction.norm();
        // Rescale to be DISTANCE away
        let r_k = u_k * DISTANCE;

        let r_mk = uma16.get_distance_from(&r_k);
        let tau_mk = r_mk.map(distance_to_delay);

        let tau_mk_approx = p
            .map(|p_m| p_m.dot(&u_k))
            .map(distance_to_delay)
            .map(|x| -1.0 * x + distance_to_delay(DISTANCE));

        println!("tau_mk: {tau_mk:#?}, tau_mk_approx: {tau_mk_approx:#?}");

        for m in 0..p.len() {
            assert!(
                (tau_mk[m] - tau_mk_approx[m]).abs() < 1e-6,
                "mic {m}: {} vs {}",
                tau_mk[m],
                tau_mk_approx[m]
            );
        }
    }
}
