from typing import Callable

import numpy as np
import xarray as xr

from eig_utils import random_W_real
from .rate_model_base import RateModel, _to_array


class RateModelLin(RateModel):
    """Linear rate model with the same public interface as RateModel.

    Dynamics for tau_syn == 0 are solved exactly under zero-order-hold input:

        tau_i dr_i/dt = -r_i + sum_j W_ij r_j + h_i(t)

    For unsupported cases, e.g. dynamic pairwise synapses or fb_proc, the class
    falls back to the parent RateModel time-stepper with identity gain.
    """

    def __init__(
            self,
            W: np.ndarray,
            tau: np.ndarray | float,
            tau_syn: np.ndarray | float = 0.0
            ):
        super().__init__(np.asarray(W, dtype=float), tau, tau_syn)
        self._disc_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def clear_cache(self) -> None:
        """Clear cached exact-discretization matrices after changing W/tau."""
        self._disc_cache.clear()

    # ---------------------------------------------------------------------
    # Linear identity gain
    # ---------------------------------------------------------------------

    def gain(
            self,
            mu: np.ndarray | float,
            pop_num: int | None = None
            ) -> np.ndarray | float:
        if np.isscalar(mu):
            return float(mu)

        mu = np.asarray(mu, dtype=float)

        if pop_num is None:
            return mu.reshape((self.npops, -1))
        else:
            return mu.reshape((1, -1))

    def gain_inv(
            self,
            r: np.ndarray | float,
            pop_num: int | None = None,
            eps: float = 1e-9
            ) -> np.ndarray | float:
        if np.isscalar(r):
            return float(r)

        r = np.asarray(r, dtype=float)

        if pop_num is None:
            return r.reshape((self.npops, -1))
        else:
            return r.reshape((1, -1))

    def gain_deriv(
            self,
            mu: np.ndarray | float,
            pop_num: int | None = None
            ) -> np.ndarray | float:
        if np.isscalar(mu):
            return 1.0

        mu = np.asarray(mu, dtype=float)

        if pop_num is None:
            return np.ones_like(mu.reshape((self.npops, -1)))
        else:
            return np.ones_like(mu.reshape((1, -1)))

    # ---------------------------------------------------------------------
    # Exact full-system solver for tau_syn == 0
    # ---------------------------------------------------------------------

    def _can_use_exact_full(
            self,
            fb_proc: Callable | None,
            s0: np.ndarray | None
            ) -> bool:
        return (
            fb_proc is None
            and s0 is None
            and np.all(self.tau[:, 0] > 0)
            and np.all(self.tau_syn == 0)
        )

    def _continuous_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return A, B for dr/dt = A r + B h."""
        tau = self.tau[:, 0]
        Tinv = np.diag(1.0 / tau)

        A = Tinv @ (self.W - np.eye(self.npops))
        B = Tinv

        return A, B

    def _discrete_matrices(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Exact zero-order-hold discretization.

        r[k+1] = Ad r[k] + Bd h[k]
        """
        dt = float(dt)

        if dt in self._disc_cache:
            return self._disc_cache[dt]

        from scipy.linalg import expm

        A, B = self._continuous_matrices()
        n = self.npops

        M = np.zeros((2 * n, 2 * n), dtype=float)
        M[:n, :n] = A
        M[:n, n:] = B

        EM = expm(M * dt)
        Ad = EM[:n, :n]
        Bd = EM[:n, n:]

        self._disc_cache[dt] = (Ad, Bd)
        return Ad, Bd

    def _prepare_h(
            self,
            h: np.ndarray,
            nsteps: int
            ) -> tuple[np.ndarray, bool]:
        h = np.asarray(h, dtype=float).reshape((self.npops, -1))

        if h.shape[1] not in (1, nsteps):
            raise ValueError(
                "Input h must have one column or exactly nsteps columns."
            )

        return h, h.shape[1] == 1

    def _run_full_exact(
            self,
            h: np.ndarray,
            r0: np.ndarray,
            dt: float,
            nsteps: int
            ) -> xr.DataArray:

        h, is_h_const = self._prepare_h(h, nsteps)
        r = np.asarray(r0, dtype=float).reshape((self.npops, 1)).copy()

        Ad, Bd = self._discrete_matrices(dt)

        R = np.zeros((self.npops, nsteps), dtype=float)
        R[:, [0]] = r

        for k in range(1, nsteps):
            h_k = h if is_h_const else h[:, [k]]
            r = Ad @ r + Bd @ h_k
            R[:, [k]] = r

        tvec = np.arange(0, nsteps * dt, dt)

        return xr.DataArray(
            R,
            dims=["pop", "time"],
            coords={"pop": np.arange(self.npops), "time": tvec},
        )

    def _run_final_exact(
            self,
            h: np.ndarray,
            r0: np.ndarray,
            dt: float,
            nsteps: int
            ) -> np.ndarray:

        h, is_h_const = self._prepare_h(h, nsteps)
        r = np.asarray(r0, dtype=float).reshape((self.npops, 1)).copy()

        Ad, Bd = self._discrete_matrices(dt)

        for k in range(1, nsteps):
            h_k = h if is_h_const else h[:, [k]]
            r = Ad @ r + Bd @ h_k

        return r

    def run(
            self,
            h: np.ndarray,
            r0: np.ndarray,
            dt: float = 1,
            nsteps: int = 10,
            fb_proc: Callable | None = None,
            s0: np.ndarray | None = None
            ) -> xr.DataArray | np.ndarray:

        r0 = np.asarray(r0, dtype=float).reshape((self.npops, 1))
        h = np.asarray(h, dtype=float).reshape((self.npops, -1))

        if not self._can_use_exact_full(fb_proc, s0):
            raise ValueError("Exact solver cannot be used with given fb_proc/s0/tau/tau_syn")
            #return super().run(h, r0, dt, nsteps, fb_proc, s0)

        if self.sim_res_type == "full":
            return self._run_full_exact(h, r0, dt, nsteps)
        elif self.sim_res_type == "final":
            return self._run_final_exact(h, r0, dt, nsteps)
        else:
            raise ValueError(f"Unknown simulation result type: {self.sim_res_type}")

    # ---------------------------------------------------------------------
    # Exact true-open-loop 1-pop solver for tau_syn row == 0
    # ---------------------------------------------------------------------

    def _can_use_exact_1pop(
            self,
            pop_num: int,
            s0: np.ndarray | None
            ) -> bool:
        return (
            s0 is None
            and self.tau[pop_num, 0] > 0
            and np.all(self.tau_syn[pop_num, :] == 0)
        )

    def _step_1pop_exact(
            self,
            pop_num: int,
            r: float,
            h_n: float,
            r_locked: np.ndarray,
            dt: float
            ) -> float:
        """Exact scalar update for one open-loop population."""
        tau = float(self.tau[pop_num, 0])
        forcing = float(h_n + self.W[pop_num, :] @ r_locked)
        a = -1.0 / tau
        b = forcing / tau

        if np.isclose(a, 0.0):
            return float(r + dt * b)

        ad = np.exp(a * dt)
        return float(ad * r + ((ad - 1.0) / a) * b)

    def _run_1pop_full_exact(
            self,
            pop_num: int,
            h: np.ndarray | float,
            r0: np.ndarray,
            dt: float,
            nsteps: int
            ) -> xr.DataArray:

        h_trace, r0 = self._prepare_1pop_inputs(h, r0, nsteps)
        is_r0_static = r0.shape[1] == 1

        R = np.zeros(nsteps, dtype=float)

        r = float(r0[pop_num, 0])
        R[0] = r

        for k in range(1, nsteps):
            r_locked = r0[:, 0] if is_r0_static else r0[:, k]

            r = self._step_1pop_exact(
                pop_num=pop_num,
                r=r,
                h_n=float(h_trace[k]),
                r_locked=r_locked,
                dt=dt,
            )

            R[k] = r

        tvec = np.arange(0, nsteps * dt, dt)

        return xr.DataArray(
            R.reshape((1, -1)),
            dims=["pop", "time"],
            coords={"pop": [pop_num], "time": tvec},
        )

    def _run_1pop_final_exact(
            self,
            pop_num: int,
            h: np.ndarray | float,
            r0: np.ndarray,
            dt: float,
            nsteps: int
            ) -> float:

        h_trace, r0 = self._prepare_1pop_inputs(h, r0, nsteps)
        is_r0_static = r0.shape[1] == 1

        r = float(r0[pop_num, 0])

        for k in range(1, nsteps):
            r_locked = r0[:, 0] if is_r0_static else r0[:, k]

            r = self._step_1pop_exact(
                pop_num=pop_num,
                r=r,
                h_n=float(h_trace[k]),
                r_locked=r_locked,
                dt=dt,
            )

        return float(r)

    def run_1pop(
            self,
            pop_num: int,
            h: np.ndarray | float,
            r0: np.ndarray,
            dt: float = 1,
            nsteps: int = 10,
            s0: np.ndarray | None = None
            ) -> xr.DataArray | float:

        if not self._can_use_exact_1pop(pop_num, s0):
            raise ValueError("Exact solver cannot be used with given fb_proc/s0/tau/tau_syn")
            #return super().run_1pop(pop_num, h, r0, dt, nsteps, s0)

        if self.sim_res_type == "full":
            return self._run_1pop_full_exact(pop_num, h, r0, dt, nsteps)
        elif self.sim_res_type == "final":
            return self._run_1pop_final_exact(pop_num, h, r0, dt, nsteps)
        else:
            raise ValueError(f"Unknown simulation result type: {self.sim_res_type}")
       

    @classmethod
    def from_steady_state(
            cls,
            r0: np.ndarray | float,
            tau: np.ndarray | float,
            R: float,
            J0: float = -1.0,
            tau_syn: np.ndarray | float = 0.0
            ) -> tuple["RateModelLin", np.ndarray, np.ndarray]:
        """Build a linear model with a prescribed steady state.

        For tau_syn == 0 and h = h0, the local dynamics are:

            d(r - r0)/dt = J (r - r0)

        where:

            J = random_W_real(npops, R=R) + J0 * I
            W = I + tau * J
            h0 = r0 - W @ r0
        """
        r0 = np.asarray(r0, dtype=float).reshape((-1, 1))
        npops = r0.shape[0]

        tau_arr = _to_array(tau, (npops, 1))

        J = random_W_real(npops, R=R) + J0 * np.eye(npops)
        W = np.eye(npops) + tau_arr * J
        h0 = r0 - W @ r0

        model = cls(W=W, tau=tau_arr, tau_syn=tau_syn)

        return model, h0, J
