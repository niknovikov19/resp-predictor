import unittest

import numpy as np

import spect_proc as FIT


def _angle_diff(a, b):
    return np.angle(np.exp(1j * (a - b)))


class TestSpectProcFit(unittest.TestCase):

    def test_power_local_matches_fit_result_weighted_power(self):
        f = 8.0
        n_cycles = 6
        dt = 1e-3
        t = np.arange(0.0, 3.0, dt)
        t0 = 1.25
        x = np.cos(2.0 * np.pi * f * t + 0.37) + 0.6

        fit = FIT.sinusoid_fit_local(x, t, t0, f, n_cycles)
        power = FIT.sinusoid_fit_power_local(x, t, t0, f, n_cycles)

        self.assertAlmostEqual(power, fit['weighted_fit_power'], places=12)

    def test_sliding_power_matches_local_power_at_sample(self):
        f = 8.0
        n_cycles = 6
        dt = 1e-3
        t = np.arange(0.0, 3.0, dt)
        x = np.cos(2.0 * np.pi * f * t + 0.37)

        t_power, power_trace = FIT.sinusoid_fit_power(x, t, f, n_cycles, overlap=1.0)
        idx = 1250
        power_local = FIT.sinusoid_fit_power_local(x, t, t[idx], f, n_cycles)

        self.assertAlmostEqual(t_power[idx], t[idx], places=12)
        self.assertAlmostEqual(power_trace[idx], power_local, places=12)

    def test_sliding_power_uses_sparser_grid_when_overlap_is_smaller(self):
        f = 8.0
        n_cycles = 5
        t = np.arange(0.0, 1.0, 1e-3)
        x = np.cos(2.0 * np.pi * f * t)

        t_dense, p_dense = FIT.sinusoid_fit_power(x, t, f, n_cycles, overlap=1.0)
        t_sparse, p_sparse = FIT.sinusoid_fit_power(x, t, f, n_cycles, overlap=0.5)

        self.assertEqual(t_dense.shape, p_dense.shape)
        self.assertEqual(t_sparse.shape, p_sparse.shape)
        self.assertEqual(t_dense[0], t[0])
        self.assertEqual(t_sparse[-1], t[-1])
        self.assertLess(t_sparse.size, t_dense.size)

    def test_nans_inside_window_are_ignored(self):
        f = 12.0
        phase0 = 0.9
        n_cycles = 7
        dt = 5e-4
        t = np.arange(0.0, 2.5, dt)
        t0 = 1.2
        x = np.cos(2.0 * np.pi * f * t + phase0) + 0.8

        phase_ref = FIT.sinusoid_fit_phase_local(x, t, t0, f, n_cycles)

        sigma_t = n_cycles / (2.0 * np.pi * f)
        support = np.abs(t - t0) <= 4.0 * sigma_t
        idx = np.flatnonzero(support)
        x_nan = x.copy()
        x_nan[idx[::5]] = np.nan

        phase_nan = FIT.sinusoid_fit_phase_local(x_nan, t, t0, f, n_cycles)
        self.assertLess(abs(_angle_diff(phase_nan, phase_ref)), 6e-2)

    def test_edge_rejection_returns_failed_fit(self):
        f = 6.0
        n_cycles = 5
        dt = 1e-3
        t = np.arange(0.0, 2.0, dt)
        t0 = 0.03
        x = np.cos(2.0 * np.pi * f * t + 0.15)

        fit = FIT.sinusoid_fit_local(x, t, t0, f, n_cycles, exclude_edges=True)

        self.assertTrue(np.isnan(fit['phase']))
        self.assertTrue(np.isnan(fit['weighted_fit_power']))
        self.assertEqual(fit['rank'], 0)

    def test_fit_intercept_preserves_phase_with_offset(self):
        f = 10.0
        phase0 = 0.6
        n_cycles = 6
        dt = 1e-3
        t = np.arange(0.0, 2.0, dt)
        t0 = 1.0
        x = np.cos(2.0 * np.pi * f * (t - t0) + phase0) + 1.5

        phase = FIT.sinusoid_fit_phase_local(
            x, t, t0, f, n_cycles, fit_intercept=True, exclude_edges=False
        )
        self.assertLess(abs(_angle_diff(phase, phase0)), 1e-2)


if __name__ == "__main__":
    unittest.main()
