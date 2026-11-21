import os
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
from numpy.testing import assert_allclose

from rate_model.model_utils import (
    _build_W_h0_ei_2d_syn_from_eig_qphi,
    main_ei,
    main_rand,
    make_eigmode_start_ei_2d_syn,
    make_W_h0,
    make_W_h0_ei_2d,
    make_W_h0_ei_2d_syn,
    rate_phi_bounds,
    rate_qphi_to_eig_qphi,
    validate_rate_phi_ie,
    wrap_rate_phi_to_zero_branch,
)
from rate_model import RateModelWC


class TestModelUtils(unittest.TestCase):

    def test_rate_qphi_to_eig_qphi_reference(self):
        tau_syn = np.array([0.7, 0.3])
        q_rate = 1.4
        phi_rate = 0.8
        f = 7.0
        tau_decay = 9.0

        q_eig, phi_eig = rate_qphi_to_eig_qphi(
            q_rate, phi_rate, tau_syn, f, tau_decay
        )

        tau_e, tau_i = tau_syn
        lam = -1.0 / tau_decay + 1j * 2.0 * np.pi * f
        Ae = 1.0 + lam * tau_e
        Ai = 1.0 + lam * tau_i

        self.assertAlmostEqual(q_eig, q_rate * np.abs(Ae) / np.abs(Ai), places=12)
        self.assertAlmostEqual(
            phi_eig,
            phi_rate - np.angle(Ae) + np.angle(Ai),
            places=12,
        )

    def test_rate_phi_bounds_non_wrap(self):
        tau_syn = np.array([20e-3, 5e-3])
        lower, upper = rate_phi_bounds(tau_syn, 10.0, 1.0)

        self.assertLess(lower, upper)
        self.assertLessEqual(upper, np.pi)

    def test_rate_phi_bounds_wrap(self):
        tau_syn = np.array([2.5e-3, 100e-3])
        lower, upper = rate_phi_bounds(tau_syn, 10.0, 10.0)

        self.assertLess(lower, upper)
        self.assertLess(lower, 0.0)
        self.assertLessEqual(upper, np.pi)

    def test_wrap_rate_phi_to_zero_branch(self):
        tau_syn = np.array([5e-3, 2e-3])
        lower0, upper0 = rate_phi_bounds(tau_syn, 10.0, 20.0)
        phi_zero = wrap_rate_phi_to_zero_branch(4.0, tau_syn, 10.0, 20.0)

        self.assertAlmostEqual(
            phi_zero,
            lower0 + np.mod(4.0 - lower0, np.pi),
            places=12,
        )
        self.assertLessEqual(lower0, phi_zero)
        self.assertLess(phi_zero, lower0 + np.pi)

    def test_validate_rate_phi_ie_raises_zero_branch_error(self):
        tau_syn = np.array([2e-3, 50e-3])
        lower, upper = rate_phi_bounds(tau_syn, 10.0, 3.0)
        with self.assertRaisesRegex(
            ValueError,
            rf"phi_ie violates Dale constraint: use \({lower:.3f}, {upper:.3f}\], got 1\.200",
        ):
            validate_rate_phi_ie(1.2, tau_syn, 10.0, 3.0)

    def test_make_W_h0_matches_from_steady_state(self):
        model = RateModelWC(
            np.zeros((4, 4)),
            tau=np.linspace(0.3, 1.2, 4),
            rmax=10.0,
            gain_slope=0.6,
            gain_center=0.2,
            tau_syn=np.array([
                [0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]),
        )
        r0 = np.array([2.0, 3.0, 4.0, 5.0])

        np.random.seed(123)
        W, h0, J = make_W_h0(model, r0, R=0.8, J0=-0.7)

        np.random.seed(123)
        model_ref, h0_ref, J_ref = RateModelWC.from_steady_state(
            r0=r0,
            tau=model.tau,
            rmax=model.rmax,
            gain_slope=model.gain_slope,
            gain_center=model.gain_center,
            R=0.8,
            J0=-0.7,
            tau_syn=model.tau_syn,
        )

        assert_allclose(W, model_ref.W, atol=1e-12)
        assert_allclose(h0, h0_ref, atol=1e-12)
        assert_allclose(J, J_ref, atol=1e-12)

    def test_make_W_h0_returns_fixed_point(self):
        model = RateModelWC(
            np.zeros((3, 3)),
            tau=np.array([[0.5], [1.0], [1.5]]),
            rmax=12.0,
            gain_slope=0.7,
            gain_center=0.1,
            tau_syn=np.array([
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]),
        )
        r0 = np.array([3.0, 5.0, 7.0])
        W, h0, J = make_W_h0(model, r0, R=0.7)

        model.W = W
        r0_col = np.asarray(r0).reshape(-1, 1)
        r_hat = model.gain(W @ r0_col + h0)

        assert_allclose(r_hat, r0_col, atol=1e-10)
        self.assertEqual(J.shape, (3, 3))

    def test_make_W_h0_ei_2d_properties(self):
        tau = np.array([1.0, 0.5])
        f = 5.0
        tau_decay = 10.0
        r0 = np.array([5.0, 3.0])
        model = RateModelWC(
            np.zeros((2, 2)),
            tau=tau,
            rmax=10.0,
            gain_slope=0.5,
            gain_center=0.0,
        )

        phi_ie = -np.pi * 1.6
        W, h0, J = make_W_h0_ei_2d(
            model, r0, tau, f, tau_decay, phi_ie=phi_ie
        )

        self.assertTrue(np.all(W[:, 0] >= 0))
        self.assertTrue(np.all(W[:, 1] <= 0))

        eigvals = np.linalg.eigvals(J)
        freq = np.max(np.abs(eigvals.imag)) / (2 * np.pi)
        decay = np.mean(eigvals.real)
        self.assertAlmostEqual(freq, f, places=6)
        self.assertAlmostEqual(decay, -1.0 / tau_decay, places=6)

        model.W = W
        r0_col = np.asarray(r0).reshape(-1, 1)
        r_hat = model.gain(W @ r0_col + h0)
        assert_allclose(r_hat, r0_col, atol=1e-10)

    def test_make_W_h0_ei_2d_syn_properties(self):
        tau = 0.0
        tau_syn = np.array([1.0, 0.5])
        f = 5.0
        tau_decay = 10.0
        r0 = np.array([5.0, 3.0])
        model = RateModelWC(
            np.zeros((2, 2)),
            tau=tau,
            rmax=10.0,
            gain_slope=0.5,
            gain_center=0.0,
        )

        q_eig = 1.3
        phi_eig = 0.5
        lam = -1.0 / tau_decay + 1j * 2.0 * np.pi * f
        Ae = 1.0 + lam * tau_syn[0]
        Ai = 1.0 + lam * tau_syn[1]
        q_ie = q_eig * np.abs(Ai) / np.abs(Ae)
        phi_ie = phi_eig + np.angle(Ae) - np.angle(Ai)

        W, h0, J, tau_syn_mat = make_W_h0_ei_2d_syn(
            model, r0, tau_syn, f, tau_decay, q_ie=q_ie, phi_ie=phi_ie
        )
        W_ref, h0_ref, J_ref, tau_syn_ref = _build_W_h0_ei_2d_syn_from_eig_qphi(
            model, r0, tau_syn, f, tau_decay, q_ie=q_eig, phi_ie=phi_eig
        )

        self.assertTrue(np.all(W[:, 0] >= 0))
        self.assertTrue(np.all(W[:, 1] <= 0))
        assert_allclose(W, W_ref, atol=1e-12)
        assert_allclose(h0, h0_ref, atol=1e-12)
        assert_allclose(J, J_ref, atol=1e-12)
        assert_allclose(
            tau_syn_mat,
            np.array([[tau_syn[0], tau_syn[1]], [tau_syn[0], tau_syn[1]]]),
            atol=1e-12,
        )
        assert_allclose(tau_syn_mat, tau_syn_ref, atol=1e-12)

        eigvals = np.linalg.eigvals(J)
        freq = np.max(np.abs(eigvals.imag)) / (2 * np.pi)
        decay = np.mean(eigvals.real)
        self.assertAlmostEqual(freq, f, places=6)
        self.assertAlmostEqual(decay, -1.0 / tau_decay, places=6)

        gp = np.asarray(model.gain_deriv(r0), dtype=float).ravel()
        J_check = np.diag(1.0 / tau_syn) @ (
            np.diag(gp) @ W - np.eye(2)
        )
        assert_allclose(J, J_check, atol=1e-12)

        model.W = W
        model.tau_syn = tau_syn_mat
        r0_col = np.asarray(r0).reshape(-1, 1)
        r_hat = model.gain(W @ r0_col + h0)
        assert_allclose(r_hat, r0_col, atol=1e-10)

    def test_make_eigmode_start_ei_2d_syn_matches_rate_and_syn_state(self):
        tau_syn = np.array([25e-3, 50e-3])
        f = 10.0
        tau_decay = 0.5
        q_ie = 0.9
        phi_ie = 0.8
        dre_start = 0.12
        phase_start = 0.4
        r0 = np.array([5.0, 5.0])
        model = RateModelWC(
            np.zeros((2, 2)),
            tau=0.0,
            rmax=10.0,
            gain_slope=0.5,
            gain_center=0.0,
        )

        W, h0, J, tau_syn_mat = make_W_h0_ei_2d_syn(
            model, r0, tau_syn, f, tau_decay, q_ie=q_ie, phi_ie=phi_ie
        )
        phi_zero = validate_rate_phi_ie(phi_ie, tau_syn, f, tau_decay)
        r_start, s_start = make_eigmode_start_ei_2d_syn(
            W,
            r0,
            tau_syn,
            f,
            tau_decay,
            q_ie=q_ie,
            phi_ie=phi_ie,
            dre_start=dre_start,
            phase_start=phase_start,
        )

        lam = -1.0 / tau_decay + 1j * 2.0 * np.pi * f
        Ae = 1.0 + lam * tau_syn[0]
        Ai = 1.0 + lam * tau_syn[1]
        r_expected = np.array([
            [r0[0] + dre_start * np.cos(phase_start)],
            [r0[1] + dre_start * q_ie * np.cos(phase_start - phi_zero)],
        ])
        x0 = np.array([
            r0[0] + np.real(dre_start * np.exp(1j * phase_start) / Ae),
            r0[1] + np.real(dre_start * q_ie * np.exp(1j * (phase_start - phi_zero)) / Ai),
        ])
        s_expected = W * x0.reshape(1, 2)

        assert_allclose(r_start, r_expected, atol=1e-12)
        assert_allclose(s_start, s_expected, atol=1e-12)
        assert_allclose(tau_syn_mat, np.array([[tau_syn[0], tau_syn[1]], [tau_syn[0], tau_syn[1]]]), atol=1e-12)

    def test_eigmode_start_reduces_initial_transient(self):
        tau_syn = np.array([25e-3, 50e-3])
        f = 10.0
        tau_decay = 0.5
        q_ie = 1.0
        phi_ie = 1.0
        dre_start = 0.05
        phase_start = 0.3
        dt = 5e-4
        r0 = np.array([5.0, 5.0])
        model = RateModelWC(
            np.zeros((2, 2)),
            tau=0.0,
            rmax=10.0,
            gain_slope=0.5,
            gain_center=0.0,
        )

        W, h0, J, tau_syn_mat = make_W_h0_ei_2d_syn(
            model, r0, tau_syn, f, tau_decay, q_ie=q_ie, phi_ie=phi_ie
        )
        model.W = W
        model.tau_syn = tau_syn_mat
        r_start, s_start = make_eigmode_start_ei_2d_syn(
            W,
            r0,
            tau_syn,
            f,
            tau_decay,
            q_ie=q_ie,
            phi_ie=phi_ie,
            dre_start=dre_start,
            phase_start=phase_start,
        )

        phi_zero = validate_rate_phi_ie(phi_ie, tau_syn, f, tau_decay)
        lam = -1.0 / tau_decay + 1j * 2.0 * np.pi * f
        r_amp = np.array([
            dre_start * np.exp(1j * phase_start),
            dre_start * q_ie * np.exp(1j * (phase_start - phi_zero)),
        ])
        r_expected = r0.reshape(2, 1) + np.real(r_amp * np.exp(lam * dt)).reshape(2, 1)
        H = np.repeat(h0, 2, axis=1)

        rr_match = model.run(H, r_start, dt=dt, nsteps=2, s0=s_start)
        rr_plain = model.run(H, r_start, dt=dt, nsteps=2)

        err_match = np.max(np.abs(rr_match.values[:, [1]] - r_expected))
        err_plain = np.max(np.abs(rr_plain.values[:, [1]] - r_expected))

        self.assertLess(err_match, 1e-5)
        self.assertLess(err_match, err_plain)

    @patch('matplotlib.pyplot.show')
    def test_main_rand_runs(self, _show):
        np.random.seed(111)
        main_rand()

    @patch('matplotlib.pyplot.show')
    def test_main_ei_runs(self, _show):
        np.random.seed(111)
        main_ei()


if __name__ == '__main__':
    unittest.main()
