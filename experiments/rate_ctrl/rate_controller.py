
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import xarray as xr


def _to_array(x, shape):
    if np.isscalar(x):
        return np.full(shape, x, dtype=float)
    return np.array(x, dtype=float).reshape(shape)


class RateController(ABC):

    par: dict[str, float | np.ndarray]
    dt: float
    nsteps: int
    sim_res_type: Literal['full', 'final']

    nvars: int
    tvec: np.ndarray
    state_xr: xr.Dataset

    state: dict[str, np.ndarray]
    step_num: int | None
    activate_t: float
    lock_t: float | None

    def __init__(
            self,
            r0: np.ndarray | float,      # target rate
            taum: np.ndarray | float,    # slow error integration time constant      
            ks: np.ndarray | float,      # slow correction sensitivity
            tauu: np.ndarray | float,    # dynamic highpass correction time const.
            ku: np.ndarray | float       # dynamic highpass correction sensitivity
            ):
        self.par = {'r0': r0, 'taum': taum, 'ks': ks, 'tauu': tauu, 'ku': ku}
        self.nvars = 1 if np.isscalar(r0) else len(r0)
        self.par = {k: _to_array(v, (self.nvars, 1))
                    for k, v in self.par.items()}
        self.sim_res_type = 'full'
        self.step_num = None
        self.activate_t = 0.0
        self.lock_t = None

    @abstractmethod
    def phi(self, m: np.ndarray, ) -> np.ndarray:
        pass

    def begin(self,
              dt: float = 1,
              nsteps: int = 10,
              m0: np.ndarray | float = 0.0,
              s0: np.ndarray | float = 0.0,
              u0: np.ndarray | float = 0.0,
              activate_t: float = 0.0,
              lock_t: float | None = None,
              sim_res_type: Literal['full', 'final'] = 'full'
              ):
        # Params
        self.dt = dt
        self.nsteps = nsteps
        self.activate_t = activate_t
        self.lock_t = lock_t
        self.sim_res_type = sim_res_type

        # State
        self.state = {'m': m0, 's': s0, 'u': u0, 'e': 0, 'q': 0, 'z': 0}
        self.state = {k: _to_array(v, (self.nvars, 1)) 
                      for k, v in self.state.items()}
        self.state['z'] = self.state['s'] + self.state['u']

        # Time bins
        self.tvec = np.arange(0, nsteps * dt, dt)

        # The result will be stored here
        self.state_xr = {}
        if sim_res_type:
            for k, v in self.state.items():
                X = xr.DataArray(
                    np.zeros((self.nvars, nsteps)),
                    dims=['var', 'time'],
                    coords={'var': np.arange(self.nvars), 'time': self.tvec},
                )
                #X[:, 0] = v
                self.state_xr[k] = X
            self.state_xr = xr.Dataset(self.state_xr)

        self.step_num = 0

    def step(self, r: np.ndarray) -> np.ndarray:
        if self.step_num is None:
            raise RuntimeError('Simulation not started. Call begin() first.')
        if self.step_num >= self.nsteps:
            raise RuntimeError('Simulation finished. No more steps allowed.')

        r = np.array(r, dtype=float).reshape((self.nvars, 1))

        e, m, q, s, u, z = (
            self.state[k] for k in ('e', 'm', 'q', 's', 'u', 'z'))
        t = self.tvec[self.step_num]
        r0, taum, ks, ku, tauu = (
            self.par[k] for k in ('r0', 'taum', 'ks', 'ku', 'tauu'))
        is_active = t >= self.activate_t
        is_locked = (
            self.lock_t is not None and t >= self.lock_t
        )
        
        # Update the state
        if is_active:
            e = r - r0                              # online error
            m += (e - m) * self.dt / taum           # slow mean error
            q = e - m                               # fast error residual
            if not is_locked:
                s += -ks * self.phi(m) * self.dt    # slow correction until locking
                u += (-u - ku * q) * self.dt / tauu # dynamic high-pass correction until locking
            else:
                u[:] = 0.0
            z = s + u                               # total correction
        else:
            e[:] = 0.0
            q[:] = 0.0
            m[:] = 0.0
            s[:] = 0.0
            u[:] = 0.0
            z[:] = 0.0

        # Store the state
        self.state = {'m': m, 's': s, 'u': u, 'e': e, 'q': q, 'z': z}
        if self.sim_res_type == 'full':
            for k, v in self.state.items():
                self.state_xr[k][:, self.step_num] = v[:, 0]
        self.step_num += 1

        return self.state


class RateControllerLin(RateController):

    def phi(
        self,
        m: np.ndarray,
    ) -> np.ndarray:
        return m


class DeadZoneRateController(RateController):

    eps_m: np.ndarray

    def __init__(
        self,
        npops: int,
        r0: np.ndarray | float,
        taum: np.ndarray | float,
        ks: np.ndarray | float,
        tauu: np.ndarray | float,
        ku: np.ndarray | float,
        eps_m: np.ndarray | float,
    ):
        super().__init__(npops, r0, taum, ks, tauu, ku)
        self.eps_m = _to_array(eps_m, (self.npops, 1))

    def phi(
        self,
        m: np.ndarray,
    ) -> np.ndarray:
        return np.sign(m) * np.maximum(np.abs(m) - self.eps_m, 0.0)
