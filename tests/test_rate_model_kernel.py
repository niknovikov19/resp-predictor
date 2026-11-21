import unittest
import sys
import types

import numpy as np
from numpy.testing import assert_allclose

from rate_model import RateModelKernel

if 'tqdm' not in sys.modules:
    tqdm_mod = types.ModuleType('tqdm')
    tqdm_mod.tqdm = lambda iterable, total=None: iterable
    sys.modules['tqdm'] = tqdm_mod

from experiments.resp_static.predictors.resp_pred_lin_multi_eps import ResponsePredictor
from experiments.resp_static.predictors.resp_pred_ord2 import ResponsePredictor_2


class TestRateModelKernel(unittest.TestCase):

    def setUp(self):
        self.model = RateModelKernel(
            np.array([[0.2, -0.1], [0.15, 0.1]]),
            tau=np.array([[2.0], [3.0]]),
            tau_syn=0.0,
            rmax=10.0,
            gain_slope=0.8,
            gain_center=0.1,
        )

    def test_gain_inverse_round_trip(self):
        mu = np.array([
            [-0.7, 0.0, 0.9],
            [0.2, -0.4, 1.1],
        ])
        r = self.model.gain(mu)
        mu_rt = self.model.gain_inv(r)
        assert_allclose(mu_rt, mu, atol=1e-9)

    def test_gain_deriv_is_positive_in_dynamic_range(self):
        r = np.array([
            [1.0, 3.5, 7.0],
            [0.5, 4.0, 8.5],
        ])
        deriv = self.model.gain_deriv(r)
        self.assertTrue(np.all(np.isfinite(deriv)))
        self.assertTrue(np.all(deriv > 0.0))

    def test_run_full_returns_pop_by_time_array(self):
        h = np.array([[0.6, 0.3, 0.3, 0.5], [0.2, 0.1, 0.4, 0.4]])
        r0 = np.array([[0.1], [0.2]])

        rr = self.model.run(h, r0, dt=0.5, nsteps=h.shape[1])

        self.assertEqual(rr.dims, ('pop', 'time'))
        self.assertEqual(rr.shape, (2, h.shape[1]))

    def test_run_final_returns_column_vector_and_scalar(self):
        self.model.sim_res_type = 'final'
        h = np.array([[0.4], [0.2]])
        r0 = np.array([[0.1], [0.2]])

        r = self.model.run(h, r0, dt=0.5, nsteps=12)
        r_pop = self.model.run_1pop(1, 0.3, r0, dt=0.5, nsteps=12)

        self.assertEqual(r.shape, (2, 1))
        self.assertTrue(np.isscalar(r_pop))

    def test_constant_input_relaxes_smoothly(self):
        model = RateModelKernel(
            np.zeros((2, 2)),
            tau=np.array([[2.0], [3.0]]),
            tau_syn=0.0,
            rmax=10.0,
            gain_slope=0.8,
            gain_center=0.1,
        )
        h = np.array([[0.8], [0.2]])
        r0 = np.zeros((2, 1))

        rr = model.run(h, r0, dt=0.5, nsteps=60)
        target = model.gain(h).ravel()

        self.assertTrue(np.all(np.diff(rr.values[0]) >= -1e-12))
        self.assertTrue(np.all(np.diff(rr.values[1]) >= -1e-12))
        assert_allclose(rr.values[:, -1], target, atol=2e-3)

    def test_dynamic_synapse_slows_transient_but_keeps_final_state(self):
        W = np.array([[0.0, 3.0], [0.0, 0.0]])
        h = np.array([[0.0], [1.2]])
        r0 = np.zeros((2, 1))
        common = dict(tau=np.array([[2.0], [2.0]]), rmax=12.0, gain_slope=1.0, gain_center=0.0)

        model_inst = RateModelKernel(W, tau_syn=0.0, **common)
        rr_inst = model_inst.run(h, r0, dt=0.5, nsteps=120)

        model_syn = RateModelKernel(
            W,
            tau_syn=np.array([[0.0, 8.0], [0.0, 0.0]]),
            **common,
        )
        rr_syn = model_syn.run(h, r0, dt=0.5, nsteps=120)

        self.assertLess(rr_syn[0, 6].item(), rr_inst[0, 6].item())
        assert_allclose(rr_syn[:, -1].values, rr_inst[:, -1].values, atol=1e-3)

    def test_explicit_s0_changes_dynamic_transient_and_ignores_inst_entries(self):
        model = RateModelKernel(
            np.array([[0.0, 1.6], [0.0, 0.0]]),
            tau=np.array([[2.0], [1.5]]),
            tau_syn=np.array([[0.0, 5.0], [0.0, 0.0]]),
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        h = np.array([[0.2], [0.1]])
        r0 = np.array([[4.0], [3.0]])
        s0 = np.zeros((2, 2))
        s0_alt = s0.copy()
        s0_alt[0, 0] = 123.0
        s0_alt[1, 0] = -456.0
        s0_alt[1, 1] = 789.0

        rr_default = model.run(h, r0, dt=0.5, nsteps=12)
        rr_shifted = model.run(h, r0, dt=0.5, nsteps=12, s0=s0)
        rr_shifted_alt = model.run(h, r0, dt=0.5, nsteps=12, s0=s0_alt)

        self.assertLess(rr_shifted[0, 1].item(), rr_default[0, 1].item())
        assert_allclose(rr_shifted_alt.values, rr_shifted.values, atol=1e-12)

    def test_run_1pop_accepts_time_varying_h(self):
        h = np.array([0.1, 0.4, 0.2, 0.5, 0.3])
        r0 = np.array([[0.2], [0.1]])

        rr = self.model.run_1pop(0, h, r0, dt=0.5, nsteps=h.size)

        self.assertEqual(rr.dims, ('pop', 'time'))
        self.assertEqual(rr.shape, (1, h.size))
        self.assertTrue(np.all(np.isfinite(rr.values)))

    def test_dynamic_r0_drives_open_loop_self_row(self):
        model = RateModelKernel(
            np.array([[1.2]]),
            tau=0.0,
            tau_syn=0.0,
            rmax=10.0,
            gain_slope=0.8,
            gain_center=0.1,
        )
        h = np.zeros(5)
        r0 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        rr = model.run_1pop(0, h, r0, dt=0.5, nsteps=h.size)
        expected = np.asarray(model.gain(1.2 * r0[:, 1:], 0), dtype=float).ravel()

        assert_allclose(rr.values[0, 1:], expected, atol=1e-12)

    def test_dynamic_r0_uses_first_column_for_default_s0(self):
        model = RateModelKernel(
            np.array([[0.0, 1.6], [0.0, 0.0]]),
            tau=np.array([[2.0], [1.5]]),
            tau_syn=np.array([[0.0, 5.0], [0.0, 0.0]]),
            rmax=10.0,
            gain_slope=1.0,
            gain_center=0.0,
        )
        h = float(model.gain_inv(4.0, 0) - 1.6 * 3.0)
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

    def test_feedback_correction_pass_changes_output(self):
        h = np.array([[0.5], [0.2]])
        r0 = np.zeros((2, 1))

        rr_plain = self.model.run(h, r0, dt=0.5, nsteps=20)
        rr_fb = self.model.run(
            h,
            r0,
            dt=0.5,
            nsteps=20,
            fb_proc=lambda r: -0.05 * r,
        )

        self.assertLess(rr_fb[:, -1].values.sum(), rr_plain[:, -1].values.sum())

    def test_response_predictor_full_mode_works(self):
        model = RateModelKernel(
            np.array([[0.2, -0.05], [0.1, 0.15]]),
            tau=np.array([[1.5], [2.0]]),
            tau_syn=np.array([[0.0, 3.0], [0.0, 0.0]]),
            rmax=10.0,
            gain_slope=0.7,
            gain_center=0.0,
        )
        h0 = np.array([[0.4], [0.25]])
        pred = ResponsePredictor(model, h0, dt=0.5, nsteps=20)

        pred.train(np.array([0.01, 0.02]), dr_vals=np.array([0.01, 0.02]))
        r_hat = pred.predict_r(np.array([[0.01], [0.015]]))

        self.assertTrue(np.all(np.isfinite(pred.r0)))
        self.assertTrue(np.all(np.isfinite(r_hat)))

    def test_response_predictor_2_final_mode_works(self):
        model = RateModelKernel(
            np.array([[0.2, -0.05], [0.1, 0.15]]),
            tau=np.array([[1.5], [2.0]]),
            tau_syn=np.array([[0.0, 3.0], [0.0, 0.0]]),
            rmax=10.0,
            gain_slope=0.7,
            gain_center=0.0,
        )
        model.sim_res_type = 'final'
        h0 = np.array([[0.4], [0.25]])
        pred = ResponsePredictor_2(model, h0, dt=0.5, nsteps=20)

        pred.train(dh=0.01, dr=0.01)
        r_hat = pred.predict_r(np.array([[0.01], [0.015]]))

        self.assertTrue(np.all(np.isfinite(pred.r0)))
        self.assertTrue(np.all(np.isfinite(r_hat)))

    def test_from_steady_state_returns_requested_fixed_point(self):
        np.random.seed(123)
        model, h0, J = RateModelKernel.from_steady_state(
            r0=np.array([2.5, 4.0, 5.5]),
            tau=np.array([[0.5], [1.0], [1.5]]),
            rmax=12.0,
            gain_slope=0.7,
            gain_center=0.1,
            R=0.8,
            J0=-0.6,
            tau_syn=np.array([
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]),
        )
        r0_col = np.array([[2.5], [4.0], [5.5]])
        r_hat = model.gain(model.W @ r0_col + h0)

        assert_allclose(r_hat, r0_col, atol=1e-10)
        self.assertEqual(J.shape, (3, 3))


if __name__ == '__main__':
    unittest.main()
