import unittest

import numpy as np
from numpy.testing import assert_allclose

from rate_model import RateModelWC


def run_old_full(model, h, r0, dt, nsteps):
    npops = model.npops
    h = h.reshape((npops, -1))
    r0 = r0.reshape((npops, 1))
    is_h_const = (h.shape[1] == 1)

    R = np.zeros((npops, nsteps))
    R[:, [0]] = r0
    for n in range(1, nsteps):
        r = R[:, [n - 1]]
        h_ = h if is_h_const else h[:, [n]]
        r_hat = model.gain(model.W @ r + h_)
        R[:, [n]] = r + (r_hat - r) * dt / model.tau
    return R


def run_old_final(model, h, r0, dt, nsteps):
    npops = model.npops
    h = h.reshape((npops, -1))
    r = r0.reshape((npops, 1)).copy()
    is_h_const = (h.shape[1] == 1)

    for n in range(1, nsteps):
        h_ = h if is_h_const else h[:, [n]]
        r_hat = model.gain(model.W @ r + h_)
        r += (r_hat - r) * dt / model.tau
    return r


class TestRateModelSyn(unittest.TestCase):

    def setUp(self):
        self.W = np.array([[0.4, -0.2], [0.3, 0.1]])
        self.model_par = dict(
            tau=np.array([[2.0], [3.0]]),
            rmax=10.0,
            gain_slope=0.7,
            gain_center=0.1,
        )

    def test_zero_tau_syn_matches_old_full(self):
        model = RateModelWC(self.W, tau_syn=0.0, **self.model_par)
        h = np.array([
            [0.2, 0.2, 0.4, 0.4, 0.1, 0.1],
            [0.0, 0.3, 0.3, 0.2, 0.2, 0.1],
        ])
        r0 = np.array([[0.1], [0.4]])
        dt = 0.5
        nsteps = h.shape[1]

        rr = model.run(h, r0, dt=dt, nsteps=nsteps)
        rr_old = run_old_full(model, h, r0, dt, nsteps)

        assert_allclose(rr.values, rr_old, atol=1e-12)

    def test_zero_tau_syn_matches_old_final(self):
        model = RateModelWC(self.W, tau_syn=0.0, **self.model_par)
        model.sim_res_type = 'final'
        h = np.array([[0.25], [0.15]])
        r0 = np.array([[0.2], [0.5]])

        r = model.run(h, r0, dt=0.25, nsteps=30)
        r_old = run_old_final(model, h, r0, dt=0.25, nsteps=30)

        assert_allclose(r, r_old, atol=1e-12)

    def test_filtered_synapse_keeps_same_steady_state(self):
        h = np.array([[0.2], [0.3]])
        r0 = np.zeros((2, 1))
        dt = 0.1
        nsteps = 1500

        model_inst = RateModelWC(self.W, tau_syn=0.0, **self.model_par)
        model_inst.sim_res_type = 'final'
        r_inst = model_inst.run(h, r0, dt=dt, nsteps=nsteps)

        tau_syn = np.array([[0.0, 6.0], [4.0, 0.0]])
        model_syn = RateModelWC(self.W, tau_syn=tau_syn, **self.model_par)
        model_syn.sim_res_type = 'final'
        r_syn = model_syn.run(h, r0, dt=dt, nsteps=nsteps)

        assert_allclose(r_syn, r_inst, atol=5e-3)

    def test_filtered_synapse_slows_transient(self):
        W = np.array([[0.0, 3.5], [0.0, 0.0]])
        tau = np.array([[2.0], [2.0]])
        h = np.zeros((2, 80))
        h[1, 10:] = 1.2
        r0 = np.zeros((2, 1))

        model_inst = RateModelWC(
            W, tau=tau, rmax=12.0, gain_slope=1.0, gain_center=0.0
        )
        rr_inst = model_inst.run(h, r0, dt=0.5, nsteps=h.shape[1])

        tau_syn = np.array([[0.0, 8.0], [0.0, 0.0]])
        model_syn = RateModelWC(
            W, tau=tau, tau_syn=tau_syn,
            rmax=12.0, gain_slope=1.0, gain_center=0.0
        )
        rr_syn = model_syn.run(h, r0, dt=0.5, nsteps=h.shape[1])

        self.assertLess(rr_syn[0, 20].item(), rr_inst[0, 20].item())
        assert_allclose(
            rr_syn[:, -1].values,
            rr_inst[:, -1].values,
            atol=1e-2,
        )

    def test_zero_tau_population_updates_instantly(self):
        model = RateModelWC(
            np.zeros((2, 2)),
            tau=np.array([[2.0], [0.0]]),
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        h = np.array([[0.0], [1.0]])
        r0 = np.zeros((2, 1))

        rr = model.run(h, r0, dt=0.5, nsteps=3)

        self.assertAlmostEqual(rr[1, 1].item(), model.gain(1.0, 1), places=12)

    def test_zero_tau_with_dynamic_synapse(self):
        W = np.array([[0.0, 0.0], [3.0, 0.0]])
        h = np.zeros((2, 40))
        h[0, 5:] = 2.0
        r0 = np.zeros((2, 1))

        model_inst = RateModelWC(
            W,
            tau=np.array([[1.0], [0.0]]),
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        rr_inst = model_inst.run(h, r0, dt=0.5, nsteps=h.shape[1])

        model_dyn = RateModelWC(
            W,
            tau=np.array([[1.0], [0.0]]),
            tau_syn=np.array([[0.0, 0.0], [4.0, 0.0]]),
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        rr_dyn = model_dyn.run(h, r0, dt=0.5, nsteps=h.shape[1])

        self.assertLess(rr_dyn[1, 6].item(), rr_inst[1, 6].item())
        assert_allclose(rr_dyn[1, -1].item(), rr_inst[1, -1].item(), atol=1e-2)

    def test_dynamic_synapse_starts_at_steady_state_in_full_run(self):
        W = np.array([[0.0, 1.6], [0.0, 0.0]])
        r0 = np.array([[4.0], [3.0]])
        tau = np.array([[2.0], [1.5]])
        tau_syn = np.array([[0.0, 5.0], [0.0, 0.0]])

        model = RateModelWC(
            W,
            tau=tau,
            tau_syn=tau_syn,
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        h = np.array([
            [float(model.gain_inv(r0[0, 0], 0) - W[0, 1] * r0[1, 0])],
            [float(model.gain_inv(r0[1, 0], 1))],
        ])

        rr = model.run(h, r0, dt=0.5, nsteps=3)

        assert_allclose(rr.values[:, [0]], r0, atol=1e-12)
        assert_allclose(rr.values[:, [1]], r0, atol=1e-12)

    def test_dynamic_synapse_starts_at_steady_state_in_run_1pop(self):
        W = np.array([[0.0, 1.6], [0.0, 0.0]])
        r0 = np.array([[4.0], [3.0]])

        model = RateModelWC(
            W,
            tau=np.array([[2.0], [1.5]]),
            tau_syn=np.array([[0.0, 5.0], [0.0, 0.0]]),
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        h = float(model.gain_inv(r0[0, 0], 0) - W[0, 1] * r0[1, 0])

        rr = model.run_1pop(0, h, r0, dt=0.5, nsteps=3)

        assert_allclose(rr.values[0, :2], [r0[0, 0], r0[0, 0]], atol=1e-12)

    def test_explicit_s0_changes_full_run_dynamic_synapse(self):
        W = np.array([[0.0, 1.6], [0.0, 0.0]])
        r0 = np.array([[4.0], [3.0]])
        tau_syn = np.array([[0.0, 5.0], [0.0, 0.0]])
        model = RateModelWC(
            W,
            tau=np.array([[2.0], [1.5]]),
            tau_syn=tau_syn,
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        h = np.array([
            [float(model.gain_inv(r0[0, 0], 0) - W[0, 1] * r0[1, 0])],
            [float(model.gain_inv(r0[1, 0], 1))],
        ])
        s0 = np.zeros((2, 2))

        rr_default = model.run(h, r0, dt=0.5, nsteps=3)
        rr_shifted = model.run(h, r0, dt=0.5, nsteps=3, s0=s0)

        self.assertLess(rr_shifted[0, 1].item(), rr_default[0, 1].item())

    def test_explicit_s0_changes_final_run_dynamic_synapse(self):
        W = np.array([[0.0, 1.6], [0.0, 0.0]])
        r0 = np.array([[4.0], [3.0]])
        tau_syn = np.array([[0.0, 5.0], [0.0, 0.0]])
        model = RateModelWC(
            W,
            tau=np.array([[2.0], [1.5]]),
            tau_syn=tau_syn,
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        model.sim_res_type = 'final'
        h = np.array([
            [float(model.gain_inv(r0[0, 0], 0) - W[0, 1] * r0[1, 0])],
            [float(model.gain_inv(r0[1, 0], 1))],
        ])
        s0 = np.zeros((2, 2))

        r_default = model.run(h, r0, dt=0.5, nsteps=3)
        r_shifted = model.run(h, r0, dt=0.5, nsteps=3, s0=s0)

        self.assertLess(r_shifted[0, 0], r_default[0, 0])

    def test_explicit_s0_changes_run_1pop_full_dynamic_synapse(self):
        W = np.array([[0.0, 1.6], [0.0, 0.0]])
        r0 = np.array([[4.0], [3.0]])
        model = RateModelWC(
            W,
            tau=np.array([[2.0], [1.5]]),
            tau_syn=np.array([[0.0, 5.0], [0.0, 0.0]]),
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        h = float(model.gain_inv(r0[0, 0], 0) - W[0, 1] * r0[1, 0])
        s0 = np.zeros((2, 2))

        rr_default = model.run_1pop(0, h, r0, dt=0.5, nsteps=3)
        rr_shifted = model.run_1pop(0, h, r0, dt=0.5, nsteps=3, s0=s0)

        self.assertLess(rr_shifted.values[0, 1], rr_default.values[0, 1])

    def test_explicit_s0_changes_run_1pop_final_dynamic_synapse(self):
        W = np.array([[0.0, 1.6], [0.0, 0.0]])
        r0 = np.array([[4.0], [3.0]])
        model = RateModelWC(
            W,
            tau=np.array([[2.0], [1.5]]),
            tau_syn=np.array([[0.0, 5.0], [0.0, 0.0]]),
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        model.sim_res_type = 'final'
        h = float(model.gain_inv(r0[0, 0], 0) - W[0, 1] * r0[1, 0])
        s0 = np.zeros((2, 2))

        r_default = model.run_1pop(0, h, r0, dt=0.5, nsteps=3)
        r_shifted = model.run_1pop(0, h, r0, dt=0.5, nsteps=3, s0=s0)

        self.assertLess(r_shifted, r_default)

    def test_instantaneous_s0_entries_are_ignored_in_full_run(self):
        W = np.array([[0.0, 1.6], [0.0, 0.0]])
        r0 = np.array([[4.0], [3.0]])
        tau_syn = np.array([[0.0, 5.0], [0.0, 0.0]])
        model = RateModelWC(
            W,
            tau=np.array([[2.0], [1.5]]),
            tau_syn=tau_syn,
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        h = np.array([
            [float(model.gain_inv(r0[0, 0], 0) - W[0, 1] * r0[1, 0])],
            [float(model.gain_inv(r0[1, 0], 1))],
        ])
        s0 = np.zeros((2, 2))
        s0[0, 1] = W[0, 1] * r0[1, 0]
        s0_alt = s0.copy()
        s0_alt[0, 0] = 123.0
        s0_alt[1, 1] = -456.0

        rr = model.run(h, r0, dt=0.5, nsteps=3, s0=s0)
        rr_alt = model.run(h, r0, dt=0.5, nsteps=3, s0=s0_alt)

        assert_allclose(rr_alt.values, rr.values, atol=1e-12)

    def test_instantaneous_s0_entries_are_ignored_in_run_1pop(self):
        W = np.array([[0.0, 1.6], [0.0, 0.0]])
        r0 = np.array([[4.0], [3.0]])
        model = RateModelWC(
            W,
            tau=np.array([[2.0], [1.5]]),
            tau_syn=np.array([[0.0, 5.0], [0.0, 0.0]]),
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        h = float(model.gain_inv(r0[0, 0], 0) - W[0, 1] * r0[1, 0])
        s0 = np.zeros((2, 2))
        s0[0, 1] = W[0, 1] * r0[1, 0]
        s0_alt = s0.copy()
        s0_alt[0, 0] = 123.0

        rr = model.run_1pop(0, h, r0, dt=0.5, nsteps=3, s0=s0)
        rr_alt = model.run_1pop(0, h, r0, dt=0.5, nsteps=3, s0=s0_alt)

        assert_allclose(rr_alt.values, rr.values, atol=1e-12)

    def test_zero_tau_dynamic_synapse_matches_reduced_update(self):
        W = np.array([[1.5]])
        h = np.array([[0.2]])
        r0 = np.array([[0.4]])
        dt = 0.1
        nsteps = 8

        model = RateModelWC(
            W,
            tau=0.0,
            tau_syn=2.0,
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        rr = model.run(h, r0, dt=dt, nsteps=nsteps)

        s = W[0, 0] * r0[0, 0]
        r_expected = [r0[0, 0]]
        for _ in range(1, nsteps):
            r_cur = float(model.gain(s + h[0, 0], 0))
            ds = (W[0, 0] * r_cur - s) / model.tau_syn[0, 0]
            s_mid = s + 0.5 * dt * ds
            r_mid = float(model.gain(s_mid + h[0, 0], 0))
            ds_mid = (W[0, 0] * r_mid - s_mid) / model.tau_syn[0, 0]
            s += dt * ds_mid
            r_expected.append(float(model.gain(s + h[0, 0], 0)))

        assert_allclose(rr.values[0], r_expected, atol=1e-12)

    def test_run_1pop_zero_tau(self):
        model = RateModelWC(
            np.array([[0.0, 0.0], [2.0, 0.0]]),
            tau=np.array([[1.0], [0.0]]),
            tau_syn=np.array([[0.0, 0.0], [3.0, 0.0]]),
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        model.sim_res_type = 'final'
        r0 = np.array([[1.0], [0.0]])

        r = model.run_1pop(1, 0.5, r0, dt=0.5, nsteps=10)

        self.assertAlmostEqual(r, model.gain(2.5, 1), places=12)

    def test_run_1pop_fast_path_ignores_s0(self):
        model = RateModelWC(
            np.array([[0.0, 0.0], [2.0, 0.0]]),
            tau=np.array([[1.0], [0.0]]),
            tau_syn=np.array([[0.0, 0.0], [0.0, 0.0]]),
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        model.sim_res_type = 'final'
        r0 = np.array([[1.0], [0.0]])
        s0 = np.full((2, 2), 999.0)

        r = model.run_1pop(1, 0.5, r0, dt=0.5, nsteps=10)
        r_alt = model.run_1pop(1, 0.5, r0, dt=0.5, nsteps=10, s0=s0)

        self.assertAlmostEqual(r_alt, r, places=12)

    def test_run_1pop_accepts_time_varying_h(self):
        model = RateModelWC(
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            tau=np.array([[1.0], [1.0]]),
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        h = np.array([0.1, 0.2, 0.4, 0.3, 0.5])
        r0 = np.array([[0.0], [0.0]])

        rr = model.run_1pop(0, h, r0, dt=0.5, nsteps=h.size)

        self.assertEqual(rr.dims, ('pop', 'time'))
        self.assertEqual(rr.shape, (1, h.size))
        self.assertTrue(np.all(np.isfinite(rr.values)))

    def test_dynamic_r0_drives_open_loop_self_row(self):
        model = RateModelWC(
            np.array([[1.5]]),
            tau=0.0,
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        h = np.zeros(5)
        r0 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        rr = model.run_1pop(0, h, r0, dt=0.5, nsteps=h.size)
        expected = np.asarray(model.gain(1.5 * r0[:, 1:], 0), dtype=float).ravel()

        assert_allclose(rr.values[0, 1:], expected, atol=1e-12)

    def test_dynamic_r0_uses_first_column_for_default_s0(self):
        W = np.array([[0.0, 1.6], [0.0, 0.0]])
        model = RateModelWC(
            W,
            tau=np.array([[2.0], [1.5]]),
            tau_syn=np.array([[0.0, 5.0], [0.0, 0.0]]),
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        h = float(model.gain_inv(4.0, 0) - W[0, 1] * 3.0)
        r0 = np.array([
            [4.0, 4.0, 4.0, 4.0],
            [3.0, 3.0, 5.0, 5.0],
        ])
        s0 = np.zeros((2, 2))

        rr_default = model.run_1pop(0, h, r0, dt=0.5, nsteps=r0.shape[1])
        rr_shifted = model.run_1pop(0, h, r0, dt=0.5, nsteps=r0.shape[1], s0=s0)

        assert_allclose(rr_default.values[0, :2], [4.0, 4.0], atol=1e-12)
        self.assertLess(rr_shifted.values[0, 1], rr_default.values[0, 1])
        self.assertGreater(rr_default.values[0, -1], rr_default.values[0, 1])

    def test_recurrent_zero_tau_block_raises(self):
        model = RateModelWC(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            tau=np.array([[0.0], [0.0]]),
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        h = np.zeros((2, 1))
        r0 = np.zeros((2, 1))

        with self.assertRaisesRegex(ValueError, "tau=0"):
            model.run(h, r0, dt=0.5, nsteps=3)


if __name__ == '__main__':
    unittest.main()
