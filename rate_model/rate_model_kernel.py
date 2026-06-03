from __future__ import annotations

import numpy as np
import xarray as xr

from eig_utils import random_W_real
from .rate_model_base import RateModel, _to_array


class RateModelKernel(RateModel):
    """Fast surrogate model for testing code that targets ``RateModel``."""

    rmax: np.ndarray
    gain_slope: np.ndarray
    gain_center: np.ndarray

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

    def gain(
            self,
            mu: np.ndarray | float,
            pop_num: int | None = None
            ) -> np.ndarray | float:
        rmax, k, c = self.rmax, self.gain_slope, self.gain_center
        if pop_num is None:
            mu = np.asarray(mu, dtype=float).reshape((self.npops, -1))
            return rmax / (1.0 + np.exp(-k * (mu - c)))
        if not np.isscalar(mu):
            mu = np.asarray(mu, dtype=float).reshape((1, -1))
        rmax, k, c = rmax[pop_num, 0], k[pop_num, 0], c[pop_num, 0]
        return rmax / (1.0 + np.exp(-k * (mu - c)))

    def gain_inv(
            self,
            r: np.ndarray | float,
            pop_num: int | None = None,
            eps: float = 1e-9
            ) -> np.ndarray | float:
        rmax, k, c = self.rmax, self.gain_slope, self.gain_center
        if pop_num is None:
            r = np.asarray(r, dtype=float).reshape((self.npops, -1))
            r = np.clip(r, eps, rmax - eps)
            return c + (1.0 / k) * np.log(r / (rmax - r))
        if not np.isscalar(r):
            r = np.asarray(r, dtype=float).reshape((1, -1))
        rmax, k, c = rmax[pop_num, 0], k[pop_num, 0], c[pop_num, 0]
        r = np.clip(r, eps, rmax - eps)
        return c + (1.0 / k) * np.log(r / (rmax - r))

    def gain_deriv(
            self,
            r: np.ndarray | float,
            pop_num: int | None = None
            ) -> np.ndarray | float:
        rmax, k = self.rmax, self.gain_slope
        if pop_num is None:
            r = np.asarray(r, dtype=float).reshape((self.npops, -1))
            return k * r * (1.0 - r / rmax)
        if not np.isscalar(r):
            r = np.asarray(r, dtype=float).reshape((1, -1))
        rmax, k = rmax[pop_num, 0], k[pop_num, 0]
        return k * r * (1.0 - r / rmax)

    @classmethod
    def from_steady_state(
            cls,
            r0: np.ndarray | float,
            tau: np.ndarray | float,
            rmax: np.ndarray | float,
            gain_slope: np.ndarray | float,
            gain_center: np.ndarray | float,
            R: float,
            J0: float = -1.0,
            tau_syn: np.ndarray | float = 0.0
            ) -> tuple['RateModelKernel', np.ndarray, np.ndarray]:
        """Build a kernel model with a prescribed steady state."""
        r0 = np.asarray(r0, dtype=float).reshape((-1, 1))
        npops = r0.shape[0]

        model = cls(
            np.zeros((npops, npops)),
            tau=tau,
            tau_syn=tau_syn,
            rmax=rmax,
            gain_slope=gain_slope,
            gain_center=gain_center,
        )

        gp = np.asarray(model.gain_deriv(r0), dtype=float).reshape((npops, 1))
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

    def _time_vector(self, dt: float, nsteps: int) -> np.ndarray:
        return np.arange(0.0, nsteps * dt, dt)

    def _expand_input(self, h: np.ndarray, nsteps: int) -> np.ndarray:
        h = np.asarray(h, dtype=float).reshape((self.npops, -1))
        if h.shape[1] == 1:
            return np.repeat(h, nsteps, axis=1)
        if h.shape[1] != nsteps:
            raise ValueError(
                "Time-varying input must have one column or exactly nsteps columns."
            )
        return h

    def _to_full_result(self, r_trace: np.ndarray, dt: float) -> xr.DataArray:
        return xr.DataArray(
            r_trace,
            dims=['pop', 'time'],
            coords={
                'pop': np.arange(self.npops),
                'time': self._time_vector(dt, r_trace.shape[1]),
            },
        )

    def _to_full_result_1pop(
            self,
            pop_num: int,
            r_trace: np.ndarray,
            dt: float
            ) -> xr.DataArray:
        return xr.DataArray(
            r_trace.reshape((1, -1)),
            dims=['pop', 'time'],
            coords={
                'pop': [pop_num],
                'time': self._time_vector(dt, r_trace.size),
            },
        )

    def _filter_synapses(
            self,
            source_trace: np.ndarray,
            r0: np.ndarray,
            dt: float,
            s0: np.ndarray | None = None
            ) -> np.ndarray:
        nsteps = source_trace.shape[1]
        drive = np.zeros((self.npops, nsteps))
        is_dyn = self.tau_syn > 0
        is_inst = ~is_dyn
        inst_w = self.W * is_inst
        s = self._init_syn(r0, s0)

        if not np.any(is_dyn):
            return inst_w @ source_trace

        alpha = np.ones_like(self.tau_syn, dtype=float)
        alpha[is_dyn] = np.exp(-dt / self.tau_syn[is_dyn])
        for n in range(nsteps):
            src_t = source_trace[:, [n]]
            drive[:, [n]] = s.sum(axis=1, keepdims=True) + inst_w @ src_t
            if n == nsteps - 1:
                continue
            s_hat = self._syn_target(src_t)
            s[is_dyn] = alpha[is_dyn] * s[is_dyn] + (1.0 - alpha[is_dyn]) * s_hat[is_dyn]
        return drive

    def _filter_synapses_row(
            self,
            pop_num: int,
            source_trace: np.ndarray,
            source_init: np.ndarray,
            dt: float,
            s0: np.ndarray | None = None
            ) -> np.ndarray:
        nsteps = source_trace.shape[1]
        drive = np.zeros(nsteps)
        is_dyn = self.tau_syn[pop_num, :] > 0
        is_inst = ~is_dyn
        s = self._init_syn_row(pop_num, source_init, s0)

        if not np.any(is_dyn):
            return (self.W[[pop_num], :] @ source_trace).ravel()

        alpha = np.ones(self.npops, dtype=float)
        alpha[is_dyn] = np.exp(-dt / self.tau_syn[pop_num, is_dyn])
        inst_w = self.W[pop_num, is_inst]
        for n in range(nsteps):
            src_t = source_trace[:, [n]]
            drive[n] = float(s.sum() + inst_w @ src_t[is_inst, 0])
            if n == nsteps - 1:
                continue
            s_hat = self._syn_target(src_t)[pop_num, :]
            s[is_dyn] = alpha[is_dyn] * s[is_dyn] + (1.0 - alpha[is_dyn]) * s_hat[is_dyn]
        return drive

    def _low_pass_rates(
            self,
            r_target: np.ndarray,
            r0: np.ndarray,
            dt: float
            ) -> np.ndarray:
        nsteps = r_target.shape[1]
        r_trace = np.zeros((self.npops, nsteps))
        r_trace[:, [0]] = r0

        zero_mask = (self.tau[:, 0] == 0)
        nonzero_mask = ~zero_mask
        alpha = np.ones(self.npops, dtype=float)
        alpha[nonzero_mask] = np.exp(-dt / self.tau[nonzero_mask, 0])

        for n in range(1, nsteps):
            r_trace[zero_mask, n] = r_target[zero_mask, n]
            r_trace[nonzero_mask, n] = (
                alpha[nonzero_mask] * r_trace[nonzero_mask, n - 1]
                + (1.0 - alpha[nonzero_mask]) * r_target[nonzero_mask, n]
            )
        return r_trace

    def _low_pass_rate_row(
            self,
            pop_num: int,
            r_target: np.ndarray,
            r0_pop: float,
            dt: float
            ) -> np.ndarray:
        nsteps = r_target.size
        r_trace = np.zeros(nsteps)
        r_trace[0] = r0_pop

        tau_pop = self.tau[pop_num, 0]
        if tau_pop == 0:
            r_trace[1:] = r_target[1:]
            return r_trace

        alpha = float(np.exp(-dt / tau_pop))
        for n in range(1, nsteps):
            r_trace[n] = alpha * r_trace[n - 1] + (1.0 - alpha) * r_target[n]
        return r_trace

    def _build_1pop_source_trace(
            self,
            pop_num: int,
            h_trace: np.ndarray,
            r0: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
        if r0.shape[1] == 1:
            source_trace = np.repeat(r0, h_trace.size, axis=1)
            source_trace[pop_num, :] = np.asarray(
                self.gain(h_trace.reshape((1, -1)), pop_num),
                dtype=float,
            ).ravel()
            return source_trace, r0
        return r0.copy(), r0[:, [0]]

    def _run_trace(
            self,
            h_trace: np.ndarray,
            r0: np.ndarray,
            dt: float,
            s0: np.ndarray | None = None
            ) -> np.ndarray:
        source_trace = np.asarray(self.gain(h_trace), dtype=float).reshape((self.npops, -1))
        drive = self._filter_synapses(source_trace, r0, dt, s0)
        r_target = np.asarray(
            self.gain(h_trace + drive),
            dtype=float,
        ).reshape((self.npops, -1))
        return self._low_pass_rates(r_target, r0, dt)

    def _apply_feedback(
            self,
            h_trace: np.ndarray,
            r0: np.ndarray,
            dt: float,
            fb_proc,
            s0: np.ndarray | None = None
            ) -> np.ndarray:
        r_prov = self._run_trace(h_trace, r0, dt, s0)
        h_fb = h_trace.copy()
        for n in range(h_trace.shape[1]):
            h_fb[:, [n]] += np.asarray(
                fb_proc(r_prov[:, [n]]),
                dtype=float,
            ).reshape((self.npops, 1))
        return self._run_trace(h_fb, r0, dt, s0)

    def run(
            self,
            h: np.ndarray,
            r0: np.ndarray,
            dt: float = 1,
            nsteps: int = 10,
            fb_proc=None,
            s0: np.ndarray | None = None
            ) -> xr.DataArray | np.ndarray:
        r0 = np.asarray(r0, dtype=float).reshape((self.npops, 1))
        h_trace = self._expand_input(h, nsteps)
        if fb_proc is None:
            r_trace = self._run_trace(h_trace, r0, dt, s0)
        else:
            r_trace = self._apply_feedback(h_trace, r0, dt, fb_proc, s0)

        if self.sim_res_type == 'full':
            return self._to_full_result(r_trace, dt)
        if self.sim_res_type == 'final':
            return r_trace[:, [-1]]
        raise ValueError(f"Unknown simulation result type: {self.sim_res_type}")

    def run_1pop(
            self,
            pop_num: int,
            h: float,
            r0: np.ndarray,
            dt: float = 1,
            nsteps: int = 10,
            s0: np.ndarray | None = None
            ) -> xr.DataArray | float:
        h_trace, r0 = self._prepare_1pop_inputs(h, r0, nsteps)
        source_trace, source_init = self._build_1pop_source_trace(pop_num, h_trace, r0)
        drive = self._filter_synapses_row(pop_num, source_trace, source_init, dt, s0)
        r_target = np.asarray(
            self.gain(h_trace.reshape((1, -1)) + drive.reshape((1, -1)), pop_num),
            dtype=float,
        ).ravel()
        r_trace = self._low_pass_rate_row(pop_num, r_target, float(r0[pop_num, 0]), dt)

        if self.sim_res_type == 'full':
            return self._to_full_result_1pop(pop_num, r_trace, dt)
        if self.sim_res_type == 'final':
            return float(r_trace[-1])
        raise ValueError(f"Unknown simulation result type: {self.sim_res_type}")
