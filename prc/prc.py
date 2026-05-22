"""
prc.py — Competing modules P1 and P2 with generic latent dynamics
===================================================================

Architecture
------------
Each module P_i is defined by a *latent state* z_i ∈ R^d that evolves as

    τ_z · dz_i/dt = A_z @ z_i  +  B_u @ u_i

with input vector  u_i = [c_i(t),  m(t),  x_j(t)]^T

    c_i(t) : module-specific excitatory signal  (c1 drives P1, c2 drives P2)
    m(t)   : common periodic signal shared by both modules
    x_j(t) : observable of the *competing* module  (x2 for P1, x1 for P2)

The scalar *observable* of each module is a nonlinear readout of the latent state:

    x_i(t) = C_out · activation(z_i(t))

Parameters:
    A_z   (d × d)  intrinsic latent dynamics (decay, oscillations, memory)
    B_u   (d × 3)  input coupling matrix — columns correspond to [c_i, m, x_j]
                   A negative entry in the x_j column produces competition;
                   a positive entry produces cooperation.
    C_out (1 × d)  linear readout from the activated latent state
    tau_z float | (d,)  latent time constant(s)

Response-kernel equivalence
----------------------------
Ignoring the nonlinearity and treating z_i as linear, the impulse response of
x_i to a delta-pulse in input channel k (at lag τ ≥ 0) is

    K_k(τ) = C_out · expm(A_z · τ) · B_u[:, k]

So (A_z, B_u, C_out) together fully specify the *response kernels* from each
input channel to the observable.  Parameterising the module via these matrices
is therefore equivalent to parameterising it via its response kernels — the
latter could in principle be estimated from data and then factored back into
(A_z, B_u, C_out) via a rank-d approximation.

Classes
-------
Module                   Abstract base class
LinearNonlinearModule    Concrete module with linear latent ODE + nonlinear readout
CompetitiveSystem        Couples two Module instances; handles Euler integration

Utilities
---------
square_wave(t, ...)      Returns a square-wave value (scalar or array)
make_signal(t_arr, ...)  Wraps square_wave into a time-array evaluation
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import xarray as xr


# ---------------------------------------------------------------------------
# Abstract module interface
# ---------------------------------------------------------------------------

class Module(ABC):
    """Abstract base class for a single dynamical module.

    A module maintains a latent state z ∈ R^d and exposes:
      - dz_dt(z, u): the time-derivative of z given input vector u
      - observe(z):  the scalar observable x derived from z
    """

    @property
    @abstractmethod
    def d(self) -> int:
        """Dimensionality of the latent state."""
        ...

    @abstractmethod
    def dz_dt(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Time derivative of latent state.

        Parameters
        ----------
        z : (d,) latent state vector
        u : (n_inputs,) input vector, e.g. [c_i, m, x_j]

        Returns
        -------
        dz : (d,) derivative
        """
        ...

    @abstractmethod
    def observe(self, z: np.ndarray) -> float:
        """Scalar observable derived from latent state.

        Parameters
        ----------
        z : (d,) latent state vector

        Returns
        -------
        x : scalar observable
        """
        ...


# ---------------------------------------------------------------------------
# Concrete module: linear latent ODE + nonlinear readout
# ---------------------------------------------------------------------------

class LinearNonlinearModule(Module):
    """Module with linear latent ODE and nonlinear readout.

    Latent dynamics (Euler-integrated externally):
        τ_z · dz/dt = A_z @ z  +  B_u @ u

    Observable:
        x = C_out @ activation(z)

    Parameters
    ----------
    d : int
        Latent state dimensionality.
    A_z : array-like (d, d)
        Intrinsic dynamics matrix.  Eigenvalues control decay rate and
        natural frequency of the latent dynamics.
    B_u : array-like (d, n_inputs)
        Input coupling matrix.  Each column maps one input channel to
        the latent dimensions.  Negative entry → inhibitory coupling.
    C_out : array-like (1, d) or (d,)
        Readout vector / matrix from the activated latent state.
    tau_z : float or array-like (d,), default 1.0
        Latent time constant(s).  Scalar applies the same constant to all
        dimensions; a vector allows per-dimension time scales.
    activation : callable, default np.tanh
        Element-wise nonlinearity applied to z before the readout.
        Use ``lambda z: z`` for a purely linear module.
    """

    def __init__(
        self,
        d: int,
        A_z: np.ndarray,
        B_u: np.ndarray,
        C_out: np.ndarray,
        tau_z: float | np.ndarray = 1.0,
        activation=np.tanh,
    ) -> None:
        self._d = d
        self.A_z = np.asarray(A_z, dtype=float)
        self.B_u = np.asarray(B_u, dtype=float)
        self.C_out = np.atleast_2d(np.asarray(C_out, dtype=float))
        self.tau_z = np.asarray(tau_z, dtype=float)
        self.activation = activation

        # validate shapes
        assert self.A_z.shape == (d, d), f"A_z must be ({d},{d}), got {self.A_z.shape}"
        assert self.B_u.shape[0] == d,   f"B_u must have {d} rows, got {self.B_u.shape}"
        assert self.C_out.shape[1] == d, f"C_out must have {d} columns, got {self.C_out.shape}"

    @property
    def d(self) -> int:
        return self._d

    def dz_dt(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Linear ODE: (A_z @ z + B_u @ u) / tau_z."""
        return (self.A_z @ z + self.B_u @ u) / self.tau_z

    def observe(self, z: np.ndarray) -> float:
        """Nonlinear readout: C_out @ activation(z)."""
        return float(self.C_out @ self.activation(z))


# ---------------------------------------------------------------------------
# Competitive system: couples two modules
# ---------------------------------------------------------------------------

class CompetitiveSystem:
    """Couples two Module instances so that their observables
    mutually inhibit (or excite) each other through the latent dynamics.

    The input vector to each module is built at every time step as::

        u1 = [c1(t),  m(t),  x2(t)]   # x2 feeds into P1's latent state
        u2 = [c2(t),  m(t),  x1(t)]   # x1 feeds into P2's latent state

    The sign and magnitude of the x_j channel in B_u controls
    whether the competition is inhibitory or excitatory.

    Parameters
    ----------
    mod1 : Module   Module P1
    mod2 : Module   Module P2
    """

    def __init__(self, mod1: Module, mod2: Module) -> None:
        self.mod1 = mod1
        self.mod2 = mod2

    # ------------------------------------------------------------------
    def run(
        self,
        c1: xr.DataArray | np.ndarray,
        c2: xr.DataArray | np.ndarray,
        m:  xr.DataArray | np.ndarray,
        z1_init: np.ndarray | None = None,
        z2_init: np.ndarray | None = None,
    ) -> xr.Dataset:
        """Simulate the system using forward-Euler integration.

        Parameters
        ----------
        c1 : xr.DataArray or array (n_steps,)
            Module-specific excitatory signal for P1.
        c2 : xr.DataArray or array (n_steps,)
            Module-specific excitatory signal for P2.
        m : xr.DataArray or array (n_steps,)
            Common periodic signal for both modules.
            If DataArray with a "time" coordinate, the time axis is used
            for output coordinates and dt is inferred as t[1]-t[0].
        z1_init : (d1,) array or None
            Initial latent state for P1.  Defaults to zeros.
        z2_init : (d2,) array or None
            Initial latent state for P2.  Defaults to zeros.

        Returns
        -------
        ds : xr.Dataset
            Variables:
              x1, x2       (time,)              — scalar observables
              z1           (latent_dim_1, time)  — P1 latent trajectory
              z2           (latent_dim_2, time)  — P2 latent trajectory
              c1, c2, m    (time,)              — input signals
        """
        # extract numpy arrays and time coordinate
        def _vals(sig):
            return np.asarray(sig.values if isinstance(sig, xr.DataArray) else sig,
                              dtype=float)

        C1 = _vals(c1)
        C2 = _vals(c2)
        M  = _vals(m)
        n_steps = len(C1)

        # infer time axis from first DataArray that carries a "time" coord
        t_arr = None
        for sig in (c1, c2, m):
            if isinstance(sig, xr.DataArray) and "time" in sig.coords:
                t_arr = np.asarray(sig.coords["time"].values, dtype=float)
                break
        if t_arr is None:
            t_arr = np.arange(n_steps, dtype=float)
        dt = float(t_arr[1] - t_arr[0]) if n_steps > 1 else 1.0

        d1, d2 = self.mod1.d, self.mod2.d

        # storage
        X1 = np.empty(n_steps)
        X2 = np.empty(n_steps)
        Z1 = np.empty((d1, n_steps))
        Z2 = np.empty((d2, n_steps))

        # initial conditions
        z1 = np.zeros(d1) if z1_init is None else np.asarray(z1_init, dtype=float).copy()
        z2 = np.zeros(d2) if z2_init is None else np.asarray(z2_init, dtype=float).copy()

        x1 = self.mod1.observe(z1)
        x2 = self.mod2.observe(z2)

        for i in range(n_steps):
            c1v = C1[i]
            c2v = C2[i]
            mv  = M[i]

            # record current state
            X1[i], X2[i] = x1, x2
            Z1[:, i] = z1
            Z2[:, i] = z2

            # input vectors: [module signal, common signal, competitor observable]
            u1 = np.array([c1v, mv, x2])
            u2 = np.array([c2v, mv, x1])

            # Euler step
            z1 = z1 + dt * self.mod1.dz_dt(z1, u1)
            z2 = z2 + dt * self.mod2.dz_dt(z2, u2)

            # update observables
            x1 = self.mod1.observe(z1)
            x2 = self.mod2.observe(z2)

        coords_time = {"time": t_arr}
        ds = xr.Dataset(
            {
                "x1": xr.DataArray(X1, dims=["time"], coords=coords_time),
                "x2": xr.DataArray(X2, dims=["time"], coords=coords_time),
                "z1": xr.DataArray(
                    Z1,
                    dims=["latent_dim_1", "time"],
                    coords={"latent_dim_1": np.arange(d1), "time": t_arr},
                ),
                "z2": xr.DataArray(
                    Z2,
                    dims=["latent_dim_2", "time"],
                    coords={"latent_dim_2": np.arange(d2), "time": t_arr},
                ),
                "c1": xr.DataArray(C1, dims=["time"], coords=coords_time),
                "c2": xr.DataArray(C2, dims=["time"], coords=coords_time),
                "m":  xr.DataArray(M,  dims=["time"], coords=coords_time),
            }
        )
        return ds

    # kept for backward-compatibility
    def run_from_callables(
        self,
        T: float,
        dt: float,
        c1_fn,
        c2_fn,
        m_fn,
        z1_init: np.ndarray | None = None,
        z2_init: np.ndarray | None = None,
    ) -> xr.Dataset:
        """Build signal arrays from callables then call run()."""
        t_arr = np.arange(int(round(T / dt))) * dt
        c1 = xr.DataArray(
            np.array([float(c1_fn(t)) for t in t_arr], dtype=float),
            dims=["time"], coords={"time": t_arr},
        )
        c2 = xr.DataArray(
            np.array([float(c2_fn(t)) for t in t_arr], dtype=float),
            dims=["time"], coords={"time": t_arr},
        )
        m = xr.DataArray(
            np.array([float(m_fn(t)) for t in t_arr], dtype=float),
            dims=["time"], coords={"time": t_arr},
        )
        return self.run(c1, c2, m, z1_init=z1_init, z2_init=z2_init)

    # ------------------------------------------------------------------
    @staticmethod
    def plot(ds: xr.Dataset, title: str = "Competitive Modules") -> plt.Figure:
        """Three-panel plot of simulation results.

        Panel 1 — Observables:  x1(t), x2(t)
        Panel 2 — Latent states: all dims of z1 and z2
        Panel 3 — Input signals: c1(t), c2(t), m(t)
        """
        t = ds.coords["time"].values

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(title, fontsize=13)

        # ---- Panel 1: observables ----
        ax = axes[0]
        ax.plot(t, ds["x1"].values, label="x1 (P1)", color="steelblue", lw=1.5)
        ax.plot(t, ds["x2"].values, label="x2 (P2)", color="tomato",    lw=1.5)
        ax.set_ylabel("Observable x")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title("Observables")

        # ---- Panel 2: latent states ----
        ax = axes[1]
        colors_1 = plt.cm.Blues(np.linspace(0.4, 0.9, ds.sizes["latent_dim_1"]))
        colors_2 = plt.cm.Reds( np.linspace(0.4, 0.9, ds.sizes["latent_dim_2"]))
        for k in range(ds.sizes["latent_dim_1"]):
            ax.plot(t, ds["z1"].values[k], color=colors_1[k],
                    lw=1.2, label=f"z1[{k}]")
        for k in range(ds.sizes["latent_dim_2"]):
            ax.plot(t, ds["z2"].values[k], color=colors_2[k],
                    lw=1.2, linestyle="--", label=f"z2[{k}]")
        ax.set_ylabel("Latent state z")
        ax.legend(loc="upper right", fontsize=9, ncol=2)
        ax.set_title("Latent States")

        # ---- Panel 3: input signals ----
        ax = axes[2]
        ax.plot(t, ds["c1"].values, label="c1 (P1 input)",  color="steelblue", lw=1.2)
        ax.plot(t, ds["c2"].values, label="c2 (P2 input)",  color="tomato",    lw=1.2)
        ax.plot(t, ds["m"].values,  label="m (common)",     color="forestgreen",lw=1.2)
        ax.set_ylabel("Signal amplitude")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title("Input Signals")

        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Signal utilities
# ---------------------------------------------------------------------------

def square_wave(
    t: float | np.ndarray,
    amplitude: float = 1.0,
    frequency: float = 1.0,
    phase: float = 0.0,
    duty: float = 0.5,
    offset: float = 0.0,
) -> float | np.ndarray:
    """Square wave signal.

    Parameters
    ----------
    t         : time (scalar or array, seconds)
    amplitude : half-amplitude (peak value = offset + amplitude)
    frequency : oscillation frequency (Hz)
    phase     : phase offset (radians)
    duty      : duty cycle in [0, 1] (fraction of period that is high)
    offset    : DC offset

    Returns
    -------
    signal : same shape as t; values in [offset - amplitude, offset + amplitude]
    """
    angle = 2.0 * np.pi * frequency * np.asarray(t) + phase
    return offset + amplitude * scipy.signal.square(angle, duty=duty)


def make_signal(
    t_arr: np.ndarray,
    amplitude: float = 1.0,
    frequency: float = 1.0,
    phase: float = 0.0,
    duty: float = 0.5,
    offset: float = 0.0,
) -> xr.DataArray:
    """Return a square-wave signal as an xr.DataArray with a 'time' coordinate.

    Parameters
    ----------
    t_arr     : (n,) time array (seconds)
    amplitude : half-amplitude
    frequency : Hz
    phase     : phase offset (radians)
    duty      : duty cycle in [0, 1]
    offset    : DC offset

    Returns
    -------
    xr.DataArray with dim 'time'
    """
    values = square_wave(t_arr, amplitude=amplitude, frequency=frequency,
                         phase=phase, duty=duty, offset=offset)
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims=["time"],
        coords={"time": np.asarray(t_arr, dtype=float)},
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # Module parameters (d=2, damped-oscillator latent)                   #
    # ------------------------------------------------------------------ #
    #
    # A_z has eigenvalues -0.5 ± 2i  →  each module has an intrinsic     #
    # damped oscillation with angular frequency ~2 rad/s.                 #
    #
    # B_u columns:  [c_i , m  , x_j ]                                    #
    #   dim-0:       1.0    0.5  -1.5   (excitatory signal + competition)  #
    #   dim-1:       0.0    0.2   0.0   (only weakly driven by m)          #
    #
    # Negative entry in B_u[:,2] = x_j column → x_j inhibits z_i         #
    # → reduces x_i  (competition / mutual inhibition)                    #
    # ------------------------------------------------------------------ #

    d = 2

    A_z = np.array([[-0.5, -2.0],
                    [ 2.0, -0.5]])

    B_u = np.array([[ 1.0,  0.5, -1.5],
                    [ 0.0,  0.2,  0.0]])

    C_out = np.array([[1.0, 0.0]])   # observe first latent dimension

    tau_z = 1.0

    mod1 = LinearNonlinearModule(d, A_z, B_u, C_out, tau_z=tau_z)
    mod2 = LinearNonlinearModule(d, A_z, B_u, C_out, tau_z=tau_z)

    system = CompetitiveSystem(mod1, mod2)

    T  = 30.0
    dt = 0.01
    t_arr = np.arange(int(round(T / dt))) * dt

    c1 = make_signal(t_arr, amplitude=2.0, frequency=0.2, phase=0.0)
    c2 = make_signal(t_arr, amplitude=2.0, frequency=0.2, phase=np.pi)
    m  = make_signal(t_arr, amplitude=0.5, frequency=0.5, phase=0.0)

    ds = system.run(c1, c2, m)

    fig = CompetitiveSystem.plot(ds, title="Competing Modules P1 & P2 — demo")
    plt.show()
