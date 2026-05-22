from abc import ABC, abstractmethod
from typing import Callable, Literal

import numpy as np
import xarray as xr

from eig_utils import random_W_real


def _to_array(x, shape):
    if np.isscalar(x):
        return np.full(shape, x)
    else:
        return np.array(x).reshape(shape)


class RateModel(ABC):

    npops: int
    W: np.ndarray   # (npops, npops)
    tau: np.ndarray   # (npops, 1)
    tau_syn: np.ndarray   # (npops, npops)

    sim_res_type: Literal['full', 'final']

    def __init__(
            self,
            W: np.ndarray,   # (npops, npops)
            tau: np.ndarray | float,   # (npops, 1)
            tau_syn: np.ndarray | float = 0.0   # (npops, npops)
            ):
        # TODO: checks
        self.W = W
        self.npops = W.shape[0]
        self.tau = _to_array(tau, (self.npops, 1))
        self.tau_syn = _to_array(tau_syn, (self.npops, self.npops))
        self.sim_res_type = 'full'

    def _init_syn(self, r: np.ndarray) -> np.ndarray:
        s = np.zeros((self.npops, self.npops))
        is_dyn = self.tau_syn > 0
        if np.any(is_dyn):
            s_hat = self.W * r.T
            s[is_dyn] = s_hat[is_dyn]
        return s
    
    def _calc_mu(
            self,
            r: np.ndarray,
            h: np.ndarray,
            s: np.ndarray
            ) -> np.ndarray:
        mu = h + s.sum(axis=1, keepdims=True)
        is_inst = (self.tau_syn == 0)
        if np.any(is_inst):
            mu = mu + (self.W * is_inst) @ r
        return mu
    
    def _step_syn(
            self,
            s: np.ndarray,
            r: np.ndarray,
            dt: float
            ) -> np.ndarray:
        is_dyn = self.tau_syn > 0
        if np.any(is_dyn):
            s_hat = self.W * r.T
            s[is_dyn] += (
                (s_hat[is_dyn] - s[is_dyn]) * dt / self.tau_syn[is_dyn]
            )
        return s
    
    @abstractmethod
    def gain(self,
             mu: np.ndarray | float,
             pop_num: int | None = None
             ) -> np.ndarray | float:
        pass

    @abstractmethod
    def gain_inv(self,
             r: np.ndarray | float,
             pop_num: int | None = None,
             eps: float = 1e-9
             ) -> np.ndarray | float:
        pass

    @abstractmethod
    def gain_deriv(self,
             mu: np.ndarray | float,
             pop_num: int | None = None
             ) -> np.ndarray | float:
        """Derivative of the gain function with respect to mu."""
        pass

    def _run_full(self,
            h: np.ndarray,   # input: (npops, 1 | nsteps)
            r0: np.ndarray,   # initial state: (npops, 1)
            dt: float = 1,
            nsteps: int = 10,
            fb_proc: Callable | None = None   # feedback function
            ) -> xr.DataArray:   # (npops, nsteps)

        # The result will be stored here
        R = np.zeros((self.npops, nsteps))

        # Time bins
        tvec = np.arange(0, nsteps * dt, dt)

        is_h_const = (h.shape[1] == 1)

        # Initial state
        R[:, [0]] = r0
        s = self._init_syn(r0)

        # Iterate to find the steady state
        for n in range(1, nsteps):
            r = R[:, [n - 1]]
            h_ = h if is_h_const else h[:, [n]]
            if fb_proc is not None:
                h_ = h_ + fb_proc(r)
            r_hat = self.gain(self._calc_mu(r, h_, s))
            R[:, [n]] = r + (r_hat - r) * dt / self.tau
            s = self._step_syn(s, r, dt)
        
        # Convert the result to xarray
        R = xr.DataArray(
            R,
            dims=['pop', 'time'],
            coords={'pop': np.arange(self.npops), 'time': tvec},
        )
        return R

    def _run(self,
            h: np.ndarray,   # input: (npops, 1 | nsteps)
            r0: np.ndarray,   # initial state: (npops, 1)
            dt: float = 1,
            nsteps: int = 10,
            fb_proc: Callable | None = None   # feedback function
            ) -> np.ndarray:   # (npops, 1)
        
        is_h_const = (h.shape[1] == 1)

        # Initial state
        r = r0.copy()
        s = self._init_syn(r0)

        # Iterate to find the steady state
        for n in range(1, nsteps):
            h_ = h if is_h_const else h[:, [n]]
            if fb_proc is not None:
                h_ = h_ + fb_proc(r)
            r_hat = self.gain(self._calc_mu(r, h_, s))
            s = self._step_syn(s, r, dt)
            r += (r_hat - r) * dt / self.tau
        
        return r
    
    def run(self,
            h: np.ndarray,   # input: (npops, 1 | nsteps)
            r0: np.ndarray,   # initial state: (npops, 1)
            dt: float = 1,
            nsteps: int = 10,
            fb_proc: Callable | None = None   # feedback function
            ) -> xr.DataArray | np.ndarray:   # (npops | 1, nsteps)
        # To column vectors
        r0 = r0.reshape((self.npops, 1))
        h = h.reshape((self.npops, -1))
        # Run (return full sim or the last bin only)
        if self.sim_res_type == 'full':
            return self._run_full(h, r0, dt, nsteps, fb_proc)
        elif self.sim_res_type == 'final':
            return self._run(h, r0, dt, nsteps, fb_proc)
        else:
            raise ValueError(f"Unknown simulation result type: {self.sim_res_type}")
    
    def _run_1pop_full(self,
            pop_num: int,
            h: float,
            r0: np.ndarray,   # surrogate rates (npops, 1)
            dt: float = 1,
            nsteps: int = 10
            ) -> xr.DataArray:   # (1, nsteps)
        
        # The result will be stored here
        R = np.zeros(nsteps)

        # Time bins
        tvec = np.arange(0, nsteps * dt, dt)

        # To column vector
        r0 = r0.reshape((self.npops, 1))

        # Initial state
        R[0] = r0[pop_num]

        # Synaptic state
        s = np.zeros(self.npops)
        is_dyn = self.tau_syn[pop_num, :] > 0
        if np.any(is_dyn):
            s_hat = self.W[pop_num, :] * r0[:, 0]
            s[is_dyn] = s_hat[is_dyn]

        # Steady-state
        # Iterate to find the steady state
        for n in range(1, nsteps):
            r = R[n - 1]
            mu = h + s.sum()
            if np.any(~is_dyn):
                mu += self.W[pop_num, ~is_dyn] @ r0[~is_dyn, 0]
            r_hat = self.gain(mu, pop_num)
            R[n] = r + (r_hat - r) * dt / self.tau[pop_num]
            if np.any(is_dyn):
                s_hat = self.W[pop_num, :] * r0[:, 0]
                s[is_dyn] += (
                    (s_hat[is_dyn] - s[is_dyn])
                    * dt / self.tau_syn[pop_num, is_dyn]
                )
        
        # Convert the result to xarray
        R = xr.DataArray(
            R.reshape((1, -1)),
            dims=['pop', 'time'],
            coords={'pop': [pop_num], 'time': tvec},
        )
        return R
    
    def _run_1pop(
            self,
            pop_num: int,
            h: float,
            r0: np.ndarray,   # surrogate rates (npops, 1)
            dt: float = 1,
            nsteps: int = 10
            ) -> float:
        # Return steady-state
        r0 = r0.reshape((self.npops, 1))
        if np.all(self.tau_syn[pop_num, :] == 0):
            r_hat = self.gain(
                float(self.W[pop_num, :] @ r0 + h), pop_num)
            return r_hat

        r = float(r0[pop_num, 0])
        s = np.zeros(self.npops)
        is_dyn = self.tau_syn[pop_num, :] > 0
        s_hat = self.W[pop_num, :] * r0[:, 0]
        s[is_dyn] = s_hat[is_dyn]

        for n in range(1, nsteps):
            mu = h + s.sum()
            if np.any(~is_dyn):
                mu += self.W[pop_num, ~is_dyn] @ r0[~is_dyn, 0]
            r_hat = self.gain(mu, pop_num)
            r += (r_hat - r) * dt / self.tau[pop_num]
            s[is_dyn] += (
                (s_hat[is_dyn] - s[is_dyn])
                * dt / self.tau_syn[pop_num, is_dyn]
            )
        return r
    
    def run_1pop(
            self,
            pop_num: int,
            h: float,
            r0: np.ndarray,   # surrogate rates (npops, 1)
            dt: float = 1,
            nsteps: int = 10
            ) -> xr.DataArray | float:
        # Run (return full sim or the last bin only)
        if self.sim_res_type == 'full':
            return self._run_1pop_full(pop_num, h, r0, dt, nsteps)
        elif self.sim_res_type == 'final':
            return self._run_1pop(pop_num, h, r0, dt, nsteps)
        else:
            raise ValueError(f"Unknown simulation result type: {self.sim_res_type}")
        

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
