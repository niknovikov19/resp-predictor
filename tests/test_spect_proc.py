import unittest

import numpy as np

from experiments.phase_lock.spect_proc import (
    itc,
    morlet_phase_local,
    morlet_power,
    morlet_power_freqs,
    morlet_power_local,
)


def _angle_diff(a, b):
    return np.angle(np.exp(1j * (a - b)))


class TestPhaseProc(unittest.TestCase):

    def test_clean_sinusoid_recovers_phase(self):
        f = 8.0
        phase0 = 0.37
        n_cycles = 6
        dt = 1e-3
        t = np.arange(0.0, 3.0, dt)
        t0 = 1.25
        x = np.cos(2.0 * np.pi * f * t + phase0)

        phase = morlet_phase_local(x, t, t0, f, n_cycles)
        expected = 2.0 * np.pi * f * t0 + phase0

        self.assertLess(abs(_angle_diff(phase, expected)), 1e-2)

    def test_nans_outside_support_do_not_change_result(self):
        f = 10.0
        phase0 = -0.2
        n_cycles = 5
        dt = 5e-4
        t = np.arange(0.0, 2.0, dt)
        t0 = 1.0
        x = np.cos(2.0 * np.pi * f * t + phase0)

        phase_ref = morlet_phase_local(x, t, t0, f, n_cycles)

        sigma_t = n_cycles / (2.0 * np.pi * f)
        half_width = 4.0 * sigma_t
        x_nan = x.copy()
        x_nan[np.abs(t - t0) > half_width + dt] = np.nan

        phase_nan = morlet_phase_local(x_nan, t, t0, f, n_cycles)
        self.assertLess(abs(_angle_diff(phase_nan, phase_ref)), 1e-12)

    def test_partial_nans_inside_support_stay_close(self):
        f = 12.0
        phase0 = 0.9
        n_cycles = 7
        dt = 5e-4
        t = np.arange(0.0, 2.5, dt)
        t0 = 1.2
        x = np.cos(2.0 * np.pi * f * t + phase0)

        phase_ref = morlet_phase_local(x, t, t0, f, n_cycles)

        sigma_t = n_cycles / (2.0 * np.pi * f)
        support = np.abs(t - t0) <= 4.0 * sigma_t
        idx = np.flatnonzero(support)
        x_nan = x.copy()
        x_nan[idx[::5]] = np.nan

        phase_nan = morlet_phase_local(x_nan, t, t0, f, n_cycles)
        self.assertLess(abs(_angle_diff(phase_nan, phase_ref)), 6e-2)

    def test_all_nans_inside_support_returns_nan(self):
        f = 9.0
        n_cycles = 5
        dt = 1e-3
        t = np.arange(0.0, 2.0, dt)
        t0 = 0.8
        x = np.cos(2.0 * np.pi * f * t)

        sigma_t = n_cycles / (2.0 * np.pi * f)
        x[np.abs(t - t0) <= 4.0 * sigma_t] = np.nan

        phase = morlet_phase_local(x, t, t0, f, n_cycles)
        self.assertTrue(np.isnan(phase))

    def test_edge_t0_returns_finite_phase(self):
        f = 6.0
        phase0 = 0.15
        n_cycles = 5
        dt = 1e-3
        t = np.arange(0.0, 2.0, dt)
        t0 = 0.03
        x = np.cos(2.0 * np.pi * f * t + phase0)

        phase = morlet_phase_local(x, t, t0, f, n_cycles)
        self.assertTrue(np.isfinite(phase))

    def test_irregular_t_raises(self):
        f = 8.0
        n_cycles = 5
        t = np.arange(10, dtype=float) * 1e-3
        t[5] += 1e-4
        x = np.cos(2.0 * np.pi * f * t)

        with self.assertRaisesRegex(ValueError, "fixed time step"):
            morlet_phase_local(x, t, 0.003, f, n_cycles)

    def test_zero_local_signal_returns_nan(self):
        f = 8.0
        n_cycles = 5
        t = np.arange(0.0, 1.0, 1e-3)
        x = np.zeros_like(t)

        phase = morlet_phase_local(x, t, 0.5, f, n_cycles)
        self.assertTrue(np.isnan(phase))

    def test_power_is_positive_for_matching_sinusoid(self):
        f = 8.0
        n_cycles = 6
        dt = 1e-3
        t = np.arange(0.0, 3.0, dt)
        x = np.cos(2.0 * np.pi * f * t + 0.37)

        power = morlet_power_local(x, t, 1.25, f, n_cycles)
        self.assertGreater(power, 0.0)

    def test_power_returns_nan_when_local_window_is_missing(self):
        f = 9.0
        n_cycles = 5
        dt = 1e-3
        t = np.arange(0.0, 2.0, dt)
        t0 = 0.8
        x = np.cos(2.0 * np.pi * f * t)

        sigma_t = n_cycles / (2.0 * np.pi * f)
        x[np.abs(t - t0) <= 4.0 * sigma_t] = np.nan

        power = morlet_power_local(x, t, t0, f, n_cycles)
        self.assertTrue(np.isnan(power))

    def test_sliding_power_matches_local_power_at_sample(self):
        f = 8.0
        n_cycles = 6
        dt = 1e-3
        t = np.arange(0.0, 3.0, dt)
        x = np.cos(2.0 * np.pi * f * t + 0.37)

        t_power, power_trace = morlet_power(x, t, f, n_cycles, overlap=1.0)
        idx = 1250
        power_local = morlet_power_local(x, t, t[idx], f, n_cycles)

        self.assertAlmostEqual(t_power[idx], t[idx], places=12)
        self.assertAlmostEqual(power_trace[idx], power_local, places=12)

    def test_sliding_power_uses_sparser_grid_when_overlap_is_smaller(self):
        f = 8.0
        n_cycles = 5
        t = np.arange(0.0, 1.0, 1e-3)
        x = np.cos(2.0 * np.pi * f * t)

        t_power_dense, power_dense = morlet_power(x, t, f, n_cycles, overlap=1.0)
        t_power_sparse, power_sparse = morlet_power(x, t, f, n_cycles, overlap=0.5)

        self.assertEqual(t_power_dense.shape, power_dense.shape)
        self.assertEqual(t_power_sparse.shape, power_sparse.shape)
        self.assertEqual(t_power_dense[0], t[0])
        self.assertEqual(t_power_sparse[-1], t[-1])
        self.assertLess(t_power_sparse.size, t_power_dense.size)

    def test_sliding_power_returns_nan_over_fully_missing_region(self):
        f = 9.0
        n_cycles = 5
        dt = 1e-3
        t = np.arange(0.0, 2.0, dt)
        x = np.cos(2.0 * np.pi * f * t)
        t0 = 0.8

        sigma_t = n_cycles / (2.0 * np.pi * f)
        x[np.abs(t - t0) <= 4.0 * sigma_t] = np.nan

        t_power, power_trace = morlet_power(x, t, f, n_cycles, overlap=1.0)
        idx = np.argmin(np.abs(t_power - t0))
        self.assertTrue(np.isnan(power_trace[idx]))

    def test_range_power_matches_single_freq_when_fband_is_zero(self):
        f = 8.0
        n_cycles = 6
        dt = 1e-3
        t = np.arange(0.0, 3.0, dt)
        x = np.cos(2.0 * np.pi * f * t + 0.37)

        t_ref, power_ref = morlet_power(x, t, f, n_cycles, overlap=0.5)
        t_power, ff, power = morlet_power_freqs(
            x, t, f, 0.0, 0.5, n_cycles, overlap=0.5
        )

        np.testing.assert_allclose(t_power, t_ref)
        np.testing.assert_allclose(ff, np.array([f]))
        np.testing.assert_allclose(power[0], power_ref, equal_nan=True)

    def test_range_power_returns_frequency_by_time_array(self):
        f0 = 10.0
        fband = 3.0
        df = 1.0
        n_cycles = 5
        t = np.arange(0.0, 1.0, 1e-3)
        x = np.cos(2.0 * np.pi * 9.0 * t)

        t_power, ff, power = morlet_power_freqs(x, t, f0, fband, df, n_cycles)

        self.assertEqual(t_power.ndim, 1)
        self.assertEqual(ff.ndim, 1)
        self.assertEqual(power.shape, (ff.size, t_power.size))
        np.testing.assert_allclose(ff, np.arange(f0 - fband, f0 + fband + df, df))

    def test_range_power_is_larger_near_injected_frequencies(self):
        n_cycles = 6
        dt = 1e-3
        t = np.arange(0.0, 4.0, dt)
        x = (
            np.cos(2.0 * np.pi * 7.0 * t + 0.2)
            + 0.8 * np.cos(2.0 * np.pi * 15.0 * t - 0.4)
        )

        _, ff, power = morlet_power_freqs(
            x, t, 11.0, 5.0, 1.0, n_cycles, overlap=0.5, power_mode='amp2'
        )
        power_mean = np.nanmean(power, axis=1)
        idx_7 = np.argmin(np.abs(ff - 7.0))
        idx_11 = np.argmin(np.abs(ff - 11.0))
        idx_15 = np.argmin(np.abs(ff - 15.0))

        self.assertGreater(power_mean[idx_7], power_mean[idx_11])
        self.assertGreater(power_mean[idx_15], power_mean[idx_11])

    def test_range_power_returns_nan_over_fully_missing_region(self):
        f0 = 10.0
        fband = 2.0
        df = 1.0
        n_cycles = 5
        dt = 1e-3
        t = np.arange(0.0, 2.0, dt)
        t0 = 0.8
        x = np.cos(2.0 * np.pi * f0 * t)

        sigma_t = n_cycles / (2.0 * np.pi * (f0 - fband))
        x[np.abs(t - t0) <= 4.0 * sigma_t] = np.nan

        t_power, _, power = morlet_power_freqs(
            x, t, f0, fband, df, n_cycles, overlap=1.0
        )
        idx = np.argmin(np.abs(t_power - t0))
        self.assertTrue(np.all(np.isnan(power[:, idx])))

    def test_power_renormalization_keeps_partial_nan_power_close(self):
        f = 12.0
        phase0 = 0.9
        n_cycles = 7
        dt = 5e-4
        t = np.arange(0.0, 2.5, dt)
        t0 = 1.2
        x = np.cos(2.0 * np.pi * f * t + phase0)

        power_ref = morlet_power_local(x, t, t0, f, n_cycles)

        sigma_t = n_cycles / (2.0 * np.pi * f)
        support = np.abs(t - t0) <= 4.0 * sigma_t
        idx = np.flatnonzero(support)
        x_nan = x.copy()
        x_nan[idx[::5]] = np.nan

        power_nan = morlet_power_local(x_nan, t, t0, f, n_cycles)
        rel_err = abs(power_nan - power_ref) / power_ref
        self.assertLess(rel_err, 0.15)

    def test_itc_drop_window_returns_finite_result(self):
        f = 10.0
        phase0 = -0.4
        n_cycles = 5
        dt = 5e-4
        t = np.arange(0.0, 2.5, dt)
        x = np.cos(2.0 * np.pi * f * t + phase0)
        tt_stim = np.array([0.8, 1.0, 1.2, 1.4])

        itc_val, phi, phases = itc(
            x, t, tt_stim, f, n_cycles, drop_win=(-10e-3, 10e-3)
        )

        self.assertTrue(np.isfinite(itc_val))
        self.assertTrue(np.isfinite(phi))
        self.assertEqual(phases.shape, tt_stim.shape)
        self.assertTrue(np.all(np.isfinite(phases)))


if __name__ == "__main__":
    unittest.main()
