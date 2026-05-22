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


if __name__ == '__main__':
    unittest.main()
