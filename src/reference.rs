use std::{cmp, f64::consts::PI};

use nalgebra::{SMatrix, SVector};

use crate::{
    Complex64, distance_to_delay,
    mic_array::MicArray,
    simulator::{level_to_pressure, scene::Scene},
    stft::Stft,
};

/// Calculate expected covariance matrix for a [Scene]
///
/// - returns the expectation E(x * x^H) per FFT bin (vec[bin])
/// - ignores the negative frequency image
/// - considers only +/- 8 bins around each component
/// - uses the scaling from the `stft`
pub fn reference_covariance<const M: usize>(
    scene: &Scene,
    mic_array: &MicArray<M>,
    stft: &Stft,
) -> Vec<SMatrix<Complex64, M, M>> {
    // Allocate N/2+1 Matrices (one for each FFT bin)
    //
    // They are the covariance matrices of iid white gaussian noise,
    // i.e., identity matrices by the noise variance: sigma^2 * I
    let sigma_squared = level_to_pressure(scene.mic_noise_params.level_db).powi(2);
    let mut r_ref: Vec<SMatrix<Complex64, M, M>> =
        vec![
            SMatrix::from_diagonal_element(Complex64::from(sigma_squared));
            stft.fft_len() / 2 + 1
        ];

    let weights = stft.weights();
    let fft_len = stft.fft_len();
    let f_s = scene.sample_rate;

    for source in &scene.sources {
        let r = mic_array.get_distance_from(&source.pos);

        for component in source.signal.components() {
            let mut a: SVector<Complex64, M> = SVector::zeros();

            for m in 0..M {
                let r_m = r[m];
                let tau_m = distance_to_delay(r_m);
                // Steering vector for that component's frequency
                a[m] = Complex64::from_polar(1.0 / r_m, -2.0 * PI * component.freq_hz * tau_m);
            }

            let a_ah = a * a.adjoint();

            // Number of bins around the "real" one to catch leakage
            // We inspect bin `k +/- BINS_AROUND`
            const BINS_AROUND: usize = 8;

            let bin = (component.freq_hz * fft_len as f64 / f_s).round() as usize;
            let bin_lower_range = bin.saturating_sub(BINS_AROUND);
            let bin_upper_range = cmp::min(fft_len / 2, bin + BINS_AROUND);

            for k in bin_lower_range..=bin_upper_range {
                // distance to bin k, in cycles per sample
                let offset = component.freq_hz / f_s - k as f64 / fft_len as f64;

                let w_k: Complex64 = weights
                    .iter()
                    .enumerate()
                    .map(|(n, w_n)| Complex64::from_polar(*w_n, 2.0 * PI * offset * n as f64))
                    .sum();

                let s_q = (component.amplitude / 2.0).powi(2) * w_k.norm_sqr();
                r_ref[k] += a_ah.scale(s_q);
            }
        }
    }

    r_ref
}

#[cfg(test)]
mod test {
    use nalgebra::SMatrix;

    use crate::{
        Complex64,
        covariance::Covariance,
        mic_array::MicArray,
        reference::reference_covariance,
        simulator::{Simulator, scene::Scene},
        stft::{Stft, StftStream},
        test_util::*,
    };

    /// Simulator -> StftStream -> Covariance over the whole scene; returns (R_est, L).
    fn stream_covariance(
        scene: &Scene,
        mic_array: &MicArray<16>,
    ) -> (Vec<SMatrix<Complex64, 16, 16>>, usize) {
        let mut sim = Simulator::new(scene.clone(), mic_array.clone());
        let mut stream = StftStream::<16>::new(Stft::new(FFT_LEN));
        let mut cov = Covariance::<16>::new(FFT_LEN / 2 + 1);

        let samples = read_all(&mut sim, 4096);
        stream.push(&samples, |snapshots, _l| cov.accumulate(snapshots));

        (cov.estimate(), cov.count())
    }

    #[test]
    fn one_sine_no_noise() {
        let scene = test_scene(-1000.0, 0.5);
        let mic_array = MicArray::uma16();

        let r_ref = reference_covariance(&scene, &mic_array, &Stft::new(FFT_LEN));
        let (r_est, l) = stream_covariance(&scene, &mic_array);
        assert!(l > 0, "no STFT frames");

        let err_rel = relative_error(&r_est, &r_ref);
        assert!(
            err_rel < 1e-9,
            "r_est should not deviate from r_ref, err_rel {err_rel}"
        );
    }

    fn relative_error(
        r_est: &[SMatrix<Complex64, 16, 16>],
        r_ref: &[SMatrix<Complex64, 16, 16>],
    ) -> f64 {
        (r_est
            .iter()
            .zip(r_ref)
            .map(|(r_est, r_ref)| (r_est - r_ref).norm_squared())
            .sum::<f64>()
            / r_ref.iter().map(|r_ref| r_ref.norm_squared()).sum::<f64>())
        .sqrt()
    }

    #[test]
    fn one_sine_with_mic_noise() {
        const DURATION_S: f64 = 0.5;

        let scene = test_scene(110.0, DURATION_S);
        let scene_long = test_scene(110.0, DURATION_S * 4.0);

        let mic_array = MicArray::uma16();

        let r_ref = reference_covariance(&scene, &mic_array, &Stft::new(FFT_LEN));
        let (r_est_short, l_short) = stream_covariance(&scene, &mic_array);
        assert!(l_short > 0, "no STFT frames");
        let (r_est_long, l_long) = stream_covariance(&scene_long, &mic_array);
        assert!(l_long > 0, "no STFT frames");

        let err_short = relative_error(&r_est_short, &r_ref);
        let err_long = relative_error(&r_est_long, &r_ref);

        // error should decay with 1/sqrt(l)
        let err_ratio_expected = (l_long as f64 / l_short as f64).sqrt();

        assert!(
            (err_ratio_expected - 0.5..err_ratio_expected + 0.5).contains(&(err_short / err_long)),
            "expected error ratio {err_ratio_expected}, got err_short {err_short}, err_long {err_long}"
        );
    }
}
