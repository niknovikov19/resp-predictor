import numpy as np
#import xarray as xr

from eig_utils import random_W_real
from .rate_model_base import RateModel, _to_array


class RateModelWC(RateModel):

    rmax: np.ndarray   # (npops, 1)
    gain_slope: np.ndarray   # (npops, 1)
    gain_center: np.ndarray   # (npops, 1)
    
    def __init__(
            self,
            W: np.ndarray,
            tau: np.ndarray | float,
            rmax: np.ndarray | float,
            gain_slope: np.ndarray | float,
            gain_center: np.ndarray | float,
            tau_syn: np.ndarray | float = 0.0
            ):
        super().__init__(W, tau, tau_syn)
        self.rmax = _to_array(rmax, (self.npops, 1))
        self.gain_slope = _to_array(gain_slope, (self.npops, 1))
        self.gain_center = _to_array(gain_center, (self.npops, 1))

    def gain(self,
             mu: np.ndarray | float,   # (npops x nsamples) or (1 x nsamples)
             pop_num: int | None = None
             ) -> np.ndarray | float:
        rmax, k, c  = self.rmax, self.gain_slope, self.gain_center
        if pop_num is None:
            mu = mu.reshape((self.npops, -1))
            return rmax / (1 + np.exp(-k * (mu - c)))
        else:
            if not np.isscalar(mu):
                mu = mu.reshape((1, -1))
            rmax, k, c = rmax[pop_num, 0], k[pop_num, 0], c[pop_num, 0]
            return rmax / (1 + np.exp(-k * (mu - c)))
    
    def gain_inv(self,
             r: np.ndarray | float,
             pop_num: int | None = None,
             eps: float = 1e-9
             ) -> np.ndarray | float:        
        rmax, k, c = self.rmax, self.gain_slope, self.gain_center
        if pop_num is None:
            r = np.array(r).reshape((self.npops, -1))
            r = np.clip(r, eps, rmax - eps)
            return c + (1.0 / k) * np.log(r / (rmax - r))
        else:
            if not np.isscalar(r):
                r = np.array(r).reshape((1, -1))
            rmax, k, c = rmax[pop_num, 0], k[pop_num, 0], c[pop_num, 0]
            r = np.clip(r, eps, rmax - eps)
            return c + (1.0 / k) * np.log(r / (rmax - r))

    def gain_deriv(self,
             r: np.ndarray | float,
             pop_num: int | None = None
             ) -> np.ndarray | float:
        rmax, k = self.rmax, self.gain_slope
        if pop_num is None:
            r = np.array(r).reshape((self.npops, -1))
            return k * r * (1 - r / rmax)
        else:
            if not np.isscalar(r):
                r = np.array(r).reshape((1, -1))
            rmax, k = rmax[pop_num, 0], k[pop_num, 0]
            return k * r * (1 - r / rmax)

    @classmethod
    def from_steady_state(cls,
            r0: np.ndarray | float,
            tau: np.ndarray | float,
            rmax: np.ndarray | float,
            gain_slope: np.ndarray | float,
            gain_center: np.ndarray | float,
            R: float,
            J0: float = -1.0,
            tau_syn: np.ndarray | float = 0.0
            ) -> tuple['RateModelWC', np.ndarray, np.ndarray]:
        """Build a Wilson-Cowan model with a prescribed steady state.

        The Jacobian is sampled as in ``ctrl_test.ipynb``:
        ``J = random_W_real(N, R) + J0 * I``.
        The returned tuple contains ``(model, h0, J)``, where ``h0`` is the
        constant input that makes ``r0`` a fixed point of the returned model.
        """
        r0 = np.asarray(r0).reshape((-1, 1))
        npops = r0.shape[0]

        model = cls(
            np.zeros((npops, npops)),
            tau=tau,
            tau_syn=tau_syn,
            rmax=rmax,
            gain_slope=gain_slope,
            gain_center=gain_center
        )

        gp = model.gain_deriv(r0)
        if np.any(np.isclose(gp, 0.0)):
            raise ValueError(
                "Target steady-state rates must give nonzero gain derivatives."
            )

        Ginv = np.diag(1.0 / gp.ravel())
        J = random_W_real(npops, R=R) + J0 * np.eye(npops)
        W = Ginv @ (np.eye(npops) + model.tau * J)
        h0 = model.gain_inv(r0) - W @ r0

        model.W = W
        return model, h0, J
