import unittest

import numpy as np
from numpy.testing import assert_allclose

from rate_model import RateModelLin


class TestRateModelLin(unittest.TestCase):

    def test_run_1pop_opens_static_self_coupling_in_full_mode(self):
        model = RateModelLin(np.array([[1.5]]), tau=1.0)
        r0 = np.array([[1.0]])
        dt = 0.1
        nsteps = 5

        rr = model.run_1pop(0, 0.0, r0, dt=dt, nsteps=nsteps)

        target = 1.5
        expected = target + (1.0 - target) * np.exp(-dt * np.arange(nsteps))
        assert_allclose(rr.values[0], expected, atol=1e-12)

    def test_run_1pop_opens_static_self_coupling_in_final_mode(self):
        model = RateModelLin(np.array([[1.5]]), tau=1.0)
        model.sim_res_type = 'final'
        r0 = np.array([[1.0]])
        dt = 0.1
        nsteps = 6

        r = model.run_1pop(0, 0.0, r0, dt=dt, nsteps=nsteps)

        target = 1.5
        expected = target + (1.0 - target) * np.exp(-dt * (nsteps - 1))
        self.assertAlmostEqual(r, expected, places=12)

    def test_run_1pop_uses_time_varying_self_surrogate(self):
        model = RateModelLin(np.array([[1.5]]), tau=1.0)
        r0 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        dt = 0.2

        rr = model.run_1pop(0, 0.0, r0, dt=dt, nsteps=r0.shape[1])

        alpha = np.exp(-dt)
        expected = np.zeros(r0.shape[1], dtype=float)
        expected[0] = r0[0, 0]
        for n in range(1, r0.shape[1]):
            target = 1.5 * r0[0, n]
            expected[n] = alpha * expected[n - 1] + (1.0 - alpha) * target
        assert_allclose(rr.values[0], expected, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
