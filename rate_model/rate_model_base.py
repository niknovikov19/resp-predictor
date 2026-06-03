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

    def _init_syn(self, r: np.ndarray, s0: np.ndarray | None = None) -> np.ndarray:
        s = np.zeros((self.npops, self.npops))
        is_dyn = self.tau_syn > 0
        if not np.any(is_dyn):
            return s
        if s0 is None:
            s_hat = self._syn_target(r)
            s[is_dyn] = s_hat[is_dyn]
            return s
        s0 = np.asarray(s0).reshape((self.npops, self.npops))
        s[is_dyn] = s0[is_dyn]
        return s

    def _init_syn_row(
            self,
            pop_num: int,
            r: np.ndarray,
            s0: np.ndarray | None = None
            ) -> np.ndarray:
        s = np.zeros(self.npops)
        is_dyn = self.tau_syn[pop_num, :] > 0
        if not np.any(is_dyn):
            return s
        if s0 is None:
            s_hat = self._syn_target(r)[pop_num, :]
            s[is_dyn] = s_hat[is_dyn]
            return s
        s0 = np.asarray(s0).reshape((self.npops, self.npops))
        s[is_dyn] = s0[pop_num, is_dyn]
        return s

    def _syn_target(self, r: np.ndarray) -> np.ndarray:
        return self.W * np.asarray(r).reshape((1, self.npops))
    
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
            s_hat = self._syn_target(r)
            s[is_dyn] += (
                (s_hat[is_dyn] - s[is_dyn]) * dt / self.tau_syn[is_dyn]
            )
        return s

    def _dsdt(
            self,
            s: np.ndarray,
            r: np.ndarray
            ) -> np.ndarray:
        ds = np.zeros_like(s)
        is_dyn = self.tau_syn > 0
        if np.any(is_dyn):
            s_hat = self._syn_target(r)
            ds[is_dyn] = (
                (s_hat[is_dyn] - s[is_dyn]) / self.tau_syn[is_dyn]
            )
        return ds

    def _expand_1pop_h(
            self,
            h: np.ndarray | float,
            nsteps: int
            ) -> np.ndarray:
        h = np.asarray(h, dtype=float).reshape(-1)
        if h.size == 1:
            return np.full(nsteps, float(h[0]))
        if h.size != nsteps:
            raise ValueError(
                "Time-varying input must have one sample or exactly nsteps samples."
            )
        return h.copy()

    def _expand_1pop_r0(
            self,
            r0: np.ndarray,
            nsteps: int
            ) -> np.ndarray:
        r0 = np.asarray(r0, dtype=float).reshape((self.npops, -1))
        if r0.shape[1] not in (1, nsteps):
            raise ValueError(
                "Surrogate rates must have one column or exactly nsteps columns."
            )
        return r0

    def _prepare_1pop_inputs(
            self,
            h: np.ndarray | float,
            r0: np.ndarray,
            nsteps: int
            ) -> tuple[np.ndarray, np.ndarray]:
        return self._expand_1pop_h(h, nsteps), self._expand_1pop_r0(r0, nsteps)

    def _surrogate_r_1pop(
            self,
            pop_num: int,
            r_pop: float,
            r0: np.ndarray,
            step_num: int | None = None
            ) -> np.ndarray:
        r0 = np.asarray(r0, dtype=float).reshape((self.npops, -1))
        if r0.shape[1] > 1:
            if step_num is None:
                raise ValueError("Time-varying surrogate rates require step_num.")
            return r0[:, [step_num]]
        r = r0[:, [0]].copy()
        r[pop_num, 0] = float(r_pop)
        return r

    def _calc_mu_1pop(
            self,
            pop_num: int,
            r_pop: float,
            r0: np.ndarray,
            h: float,
            s_row: np.ndarray,
            step_num: int | None = None
            ) -> float:
        r = self._surrogate_r_1pop(pop_num, r_pop, r0, step_num)
        mu = float(h + s_row.sum())
        is_inst = (self.tau_syn[pop_num, :] == 0)
        if np.any(is_inst):
            mu += float(self.W[pop_num, is_inst] @ r[is_inst, 0])
        return mu

    def _dsdt_row(
            self,
            pop_num: int,
            s_row: np.ndarray,
            r_pop: float,
            r0: np.ndarray,
            step_num: int | None = None
            ) -> np.ndarray:
        ds = np.zeros_like(s_row)
        is_dyn = self.tau_syn[pop_num, :] > 0
        if np.any(is_dyn):
            r = self._surrogate_r_1pop(pop_num, r_pop, r0, step_num)
            s_hat = self._syn_target(r)[pop_num, :]
            ds[is_dyn] = (
                (s_hat[is_dyn] - s_row[is_dyn])
                / self.tau_syn[pop_num, is_dyn]
            )
        return ds

    def _step_1pop_dynamic(
            self,
            pop_num: int,
            r: float,
            r0: np.ndarray,
            h: float,
            s: np.ndarray,
            dt: float,
            step_num: int | None = None
            ) -> tuple[float, np.ndarray]:
        zero_tau = (self.tau[pop_num, 0] == 0)

        # Match the full runner for rows with stored synapses
        if zero_tau:
            r_inst = float(self.gain(
                self._calc_mu_1pop(pop_num, r, r0, h, s, step_num), pop_num
            ))
        else:
            r_inst = float(r)
        r_hat = float(self.gain(
            self._calc_mu_1pop(pop_num, r_inst, r0, h, s, step_num), pop_num
        ))
        dr = 0.0 if zero_tau else (r_hat - r) / self.tau[pop_num, 0]
        ds = self._dsdt_row(pop_num, s, r_inst, r0, step_num)

        r_mid = float(r if zero_tau else r + 0.5 * dt * dr)
        s_mid = s + 0.5 * dt * ds
        if zero_tau:
            r_mid = float(self.gain(
                self._calc_mu_1pop(pop_num, r_mid, r0, h, s_mid, step_num), pop_num
            ))
        r_hat_mid = float(self.gain(
            self._calc_mu_1pop(pop_num, r_mid, r0, h, s_mid, step_num), pop_num
        ))
        dr_mid = 0.0 if zero_tau else (r_hat_mid - r_mid) / self.tau[pop_num, 0]
        ds_mid = self._dsdt_row(pop_num, s_mid, r_mid, r0, step_num)

        r_new = float(r if zero_tau else r + dt * dr_mid)
        s_new = s + dt * ds_mid
        if zero_tau:
            r_new = float(self.gain(
                self._calc_mu_1pop(pop_num, r_new, r0, h, s_new, step_num), pop_num
            ))
        return r_new, s_new

    def _zero_tau_order(self) -> np.ndarray:
        zero_mask = (self.tau[:, 0] == 0)
        zero_idx = np.flatnonzero(zero_mask)
        if zero_idx.size == 0:
            return zero_idx

        W_inst = self.W * (self.tau_syn == 0)
        A = (np.abs(W_inst[np.ix_(zero_idx, zero_idx)]) > 0)
        if np.any(np.diag(A)):
            raise ValueError(
                "Instantaneous self-coupling with tau=0 is not supported."
            )

        indeg = A.sum(axis=1).astype(int)
        order = []
        ready = [k for k, d in enumerate(indeg) if d == 0]
        while ready:
            k = ready.pop(0)
            order.append(k)
            outs = np.flatnonzero(A[:, k])
            for m in outs:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)

        if len(order) != len(zero_idx):
            raise ValueError(
                "Instantaneously recurrent tau=0 populations are not supported."
            )
        return zero_idx[np.array(order, dtype=int)]

    def _resolve_zero_tau(
            self,
            r: np.ndarray,
            h: np.ndarray,
            s: np.ndarray
            ) -> np.ndarray:
        order = self._zero_tau_order()
        if order.size == 0:
            return r

        r = r.copy()
        for pop_num in order:
            mu = float(self._calc_mu(r, h, s)[pop_num, 0])
            r[pop_num, 0] = self.gain(mu, pop_num)
        return r
    
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

    def h0_by_r0(self, r0):
        return self.gain_inv(r0) - self.W @ r0

    def _run_full(self,
            h: np.ndarray,   # input: (npops, 1 | nsteps)
            r0: np.ndarray,   # initial state: (npops, 1)
            dt: float = 1,
            nsteps: int = 10,
            fb_proc: Callable | None = None,   # feedback function
            s0: np.ndarray | None = None
            ) -> xr.DataArray:   # (npops, nsteps)

        # The result will be stored here
        R = np.zeros((self.npops, nsteps))

        # Time bins
        tvec = np.arange(0, nsteps * dt, dt)

        is_h_const = (h.shape[1] == 1)

        # Initial state
        R[:, [0]] = r0
        s = self._init_syn(r0, s0)

        # Iterate to find the steady state
        for n in range(1, nsteps):
            r_prev = R[:, [n - 1]]
            h_ = h if is_h_const else h[:, [n]]
            if fb_proc is not None:
                h_ = h_ + fb_proc(r_prev)
            zero_mask = (self.tau[:, 0] == 0)
            if np.any(self.tau_syn > 0):
                r_inst = self._resolve_zero_tau(r_prev, h_, s)
                r_hat = self.gain(self._calc_mu(r_inst, h_, s))
                dr = np.zeros_like(r_prev)
                dr[~zero_mask] = (
                    (r_hat[~zero_mask] - r_prev[~zero_mask]) / self.tau[~zero_mask]
                )
                ds = self._dsdt(s, r_inst)

                r_mid = r_prev.copy()
                r_mid[~zero_mask] += 0.5 * dt * dr[~zero_mask]
                s_mid = s + 0.5 * dt * ds
                r_mid = self._resolve_zero_tau(r_mid, h_, s_mid)
                r_hat_mid = self.gain(self._calc_mu(r_mid, h_, s_mid))
                dr_mid = np.zeros_like(r_prev)
                dr_mid[~zero_mask] = (
                    (r_hat_mid[~zero_mask] - r_mid[~zero_mask])
                    / self.tau[~zero_mask]
                )
                ds_mid = self._dsdt(s_mid, r_mid)

                r_new = r_prev.copy()
                r_new[~zero_mask] += dt * dr_mid[~zero_mask]
                s = s + dt * ds_mid
                r_new = self._resolve_zero_tau(r_new, h_, s)
            else:
                r_inst = self._resolve_zero_tau(r_prev, h_, s)
                r_hat = self.gain(self._calc_mu(r_inst, h_, s))
                r_new = r_prev.copy()
                r_new[~zero_mask] += (
                    (r_hat[~zero_mask] - r_prev[~zero_mask]) * dt / self.tau[~zero_mask]
                )
                r_new = self._resolve_zero_tau(r_new, h_, s)
            R[:, [n]] = r_new
        
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
            fb_proc: Callable | None = None,   # feedback function
            s0: np.ndarray | None = None
            ) -> np.ndarray:   # (npops, 1)
        
        is_h_const = (h.shape[1] == 1)

        # Initial state
        r = r0.copy()
        s = self._init_syn(r0, s0)

        # Iterate to find the steady state
        for n in range(1, nsteps):
            h_ = h if is_h_const else h[:, [n]]
            if fb_proc is not None:
                h_ = h_ + fb_proc(r)
            zero_mask = (self.tau[:, 0] == 0)
            if np.any(self.tau_syn > 0):
                r_inst = self._resolve_zero_tau(r, h_, s)
                r_hat = self.gain(self._calc_mu(r_inst, h_, s))
                dr = np.zeros_like(r)
                dr[~zero_mask] = (
                    (r_hat[~zero_mask] - r[~zero_mask]) / self.tau[~zero_mask]
                )
                ds = self._dsdt(s, r_inst)

                r_mid = r.copy()
                r_mid[~zero_mask] += 0.5 * dt * dr[~zero_mask]
                s_mid = s + 0.5 * dt * ds
                r_mid = self._resolve_zero_tau(r_mid, h_, s_mid)
                r_hat_mid = self.gain(self._calc_mu(r_mid, h_, s_mid))
                dr_mid = np.zeros_like(r)
                dr_mid[~zero_mask] = (
                    (r_hat_mid[~zero_mask] - r_mid[~zero_mask])
                    / self.tau[~zero_mask]
                )
                ds_mid = self._dsdt(s_mid, r_mid)

                r[~zero_mask] += dt * dr_mid[~zero_mask]
                s = s + dt * ds_mid
                r = self._resolve_zero_tau(r, h_, s)
            else:
                r_inst = self._resolve_zero_tau(r, h_, s)
                r_hat = self.gain(self._calc_mu(r_inst, h_, s))
                r[~zero_mask] += (
                    (r_hat[~zero_mask] - r[~zero_mask]) * dt / self.tau[~zero_mask]
                )
                r = self._resolve_zero_tau(r, h_, s)
        
        return r
    
    def run(self,
            h: np.ndarray,   # input: (npops, 1 | nsteps)
            r0: np.ndarray,   # initial state: (npops, 1)
            dt: float = 1,
            nsteps: int = 10,
            fb_proc: Callable | None = None,   # feedback function
            s0: np.ndarray | None = None
            ) -> xr.DataArray | np.ndarray:   # (npops | 1, nsteps)
        # To column vectors
        r0 = r0.reshape((self.npops, 1))
        h = h.reshape((self.npops, -1))
        # Run (return full sim or the last bin only)
        if self.sim_res_type == 'full':
            return self._run_full(h, r0, dt, nsteps, fb_proc, s0)
        elif self.sim_res_type == 'final':
            return self._run(h, r0, dt, nsteps, fb_proc, s0)
        else:
            raise ValueError(f"Unknown simulation result type: {self.sim_res_type}")
    
    def _run_1pop_full(self,
            pop_num: int,
            h: float,
            r0: np.ndarray,   # surrogate rates (npops, 1)
            dt: float = 1,
            nsteps: int = 10,
            s0: np.ndarray | None = None
            ) -> xr.DataArray:   # (1, nsteps)
        h_trace, r0 = self._prepare_1pop_inputs(h, r0, nsteps)
        is_r0_static = (r0.shape[1] == 1)
        is_dyn = self.tau_syn[pop_num, :] > 0

        # Keep the old path for static surrogate inputs
        if is_r0_static:

            # The result will be stored here
            R = np.zeros(nsteps)

            # Time bins
            tvec = np.arange(0, nsteps * dt, dt)

            # Initial state
            R[0] = float(r0[pop_num, 0])
            h0 = float(h_trace[0])

            if self.tau[pop_num, 0] == 0 and self.tau_syn[pop_num, pop_num] == 0 and self.W[pop_num, pop_num] != 0:
                raise ValueError(
                    "Instantaneous self-coupling with tau=0 is not supported."
                )

            # Keep the old path when this row has no stored synapses
            if not np.any(is_dyn):
                for n in range(1, nsteps):
                    r = R[n - 1]
                    h_n = float(h_trace[n])
                    mu = h_n + self.W[pop_num, :] @ r0[:, 0]
                    r_hat = self.gain(mu, pop_num)
                    if self.tau[pop_num, 0] == 0:
                        R[n] = r_hat
                    else:
                        R[n] = r + (r_hat - r) * dt / self.tau[pop_num, 0]
                return xr.DataArray(
                    R.reshape((1, -1)),
                    dims=['pop', 'time'],
                    coords={'pop': [pop_num], 'time': tvec},
                )

            # Initialize the stored synapses for the dynamic row
            s = self._init_syn_row(pop_num, r0, s0)
            r = float(r0[pop_num, 0])
            for n in range(1, nsteps):
                h_n = float(h_trace[n])
                r, s = self._step_1pop_dynamic(pop_num, r, r0, h_n, s, dt)
                R[n] = r

            return xr.DataArray(
                R.reshape((1, -1)),
                dims=['pop', 'time'],
                coords={'pop': [pop_num], 'time': tvec},
            )

        # The result will be stored here
        R = np.zeros(nsteps)

        # Time bins
        tvec = np.arange(0, nsteps * dt, dt)

        # Initial state
        R[0] = float(r0[pop_num, 0])
        s = self._init_syn_row(pop_num, r0[:, [0]], s0)
        r = float(R[0])

        for n in range(1, nsteps):
            h_n = float(h_trace[n])
            if np.any(is_dyn):
                r, s = self._step_1pop_dynamic(
                    pop_num, r, r0, h_n, s, dt, step_num=n
                )
            else:
                mu = self._calc_mu_1pop(pop_num, r, r0, h_n, s, step_num=n)
                r_hat = float(self.gain(mu, pop_num))
                if self.tau[pop_num, 0] == 0:
                    r = r_hat
                else:
                    r = r + (r_hat - r) * dt / self.tau[pop_num, 0]
            R[n] = r

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
            nsteps: int = 10,
            s0: np.ndarray | None = None
            ) -> float:
        h_trace, r0 = self._prepare_1pop_inputs(h, r0, nsteps)
        is_r0_static = (r0.shape[1] == 1)

        # Return steady-state
        if is_r0_static:
            if self.tau[pop_num, 0] == 0 and self.tau_syn[pop_num, pop_num] == 0 and self.W[pop_num, pop_num] != 0:
                raise ValueError(
                    "Instantaneous self-coupling with tau=0 is not supported."
                )
            if np.all(self.tau_syn[pop_num, :] == 0) and np.all(h_trace == h_trace[0]):
                r_hat = self.gain(
                    float(self.W[pop_num, :] @ r0[:, 0] + h_trace[0]), pop_num)
                return r_hat

        r = float(r0[pop_num, 0])
        s = self._init_syn_row(pop_num, r0[:, [0]], s0)
        is_dyn = self.tau_syn[pop_num, :] > 0

        for n in range(1, nsteps):
            h_n = float(h_trace[n])
            if np.any(is_dyn):
                step_num = None if is_r0_static else n
                r, s = self._step_1pop_dynamic(
                    pop_num, r, r0, h_n, s, dt, step_num=step_num
                )
            else:
                step_num = None if is_r0_static else n
                mu = self._calc_mu_1pop(pop_num, r, r0, h_n, s, step_num=step_num)
                r_hat = float(self.gain(mu, pop_num))
                if self.tau[pop_num, 0] == 0:
                    r = r_hat
                else:
                    r = r + (r_hat - r) * dt / self.tau[pop_num, 0]
        return r
    
    def run_1pop(
            self,
            pop_num: int,
            h: float,
            r0: np.ndarray,   # surrogate rates (npops, 1)
            dt: float = 1,
            nsteps: int = 10,
            s0: np.ndarray | None = None
            ) -> xr.DataArray | float:
        # Run (return full sim or the last bin only)
        if self.sim_res_type == 'full':
            return self._run_1pop_full(pop_num, h, r0, dt, nsteps, s0)
        elif self.sim_res_type == 'final':
            return self._run_1pop(pop_num, h, r0, dt, nsteps, s0)
        else:
            raise ValueError(f"Unknown simulation result type: {self.sim_res_type}")
        