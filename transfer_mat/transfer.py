from abc import abstractmethod
import hashlib
from itertools import product
from pathlib import Path
import sys

DIRPATH_REPO = Path(__file__).parents[1]
sys.path.append(str(DIRPATH_REPO))

import numpy as np
import xarray as xr

from rate_model import RateModel


class TransferMatBuiderBase:
    """Explicit three-stage scaffold for probe traces, gains, and transfers.

    The base class owns the shared artifacts:
    - ``R_probe``: upstream open-loop probe traces
    - ``D``: direct input-to-rate gains derived from ``R_probe``
    - ``C``: transfer coefficients derived from ``R_probe``

    Subclasses define how to produce ``R_probe`` and how to compute one entry
    of ``D`` and ``C`` from the shared probe dataset.
    """

    def __init__(self):
        self.pops = ['P1', 'P2']
        self.freqs = np.array([10, 15, 20])
        self.amps = [1, 2]

        self._prepare()

    def _prepare(self):
        """Prepare array sizes and result containers for the explicit pipeline."""
        self.npops = len(self.pops)
        self.nfreqs = len(self.freqs)
        self.namps = len(self.amps)
        self.R_probe = None

        self.C = xr.DataArray(
            np.full((self.npops, self.npops, self.nfreqs, self.namps), np.nan, dtype=complex),
            dims=['pop_pre', 'pop_post', 'freq', 'amp'],
            coords={
                'pop_pre': np.arange(self.npops),
                'pop_post': np.arange(self.npops),
                'freq': self.freqs,
                'amp': self.amps,
            },
        )
        self.D = xr.DataArray(
            np.full((self.npops, self.nfreqs, self.namps), np.nan, dtype=complex),
            dims=['pop', 'freq', 'amp'],
            coords={
                'pop': np.arange(self.npops),
                'freq': self.freqs,
                'amp': self.amps,
            },
        )

    @abstractmethod
    def compute_input_traces(self):
        """Run the shared open-loop probe experiment and store ``R_probe``."""

    @abstractmethod
    def _probe_gain(self, pop_num, f, amp):
        """Return one direct input-to-rate gain from ``R_probe``."""

    @abstractmethod
    def _probe_coef(self, pop_pre, pop_post, f, amp):
        """Return one transfer coefficient from ``R_probe``."""

    def _iter_transfer_product(self):
        """Iterate over all output-input-frequency-amplitude transfer entries."""
        return product(range(self.npops), range(self.npops), self.freqs, self.amps)

    def _iter_gain_product(self):
        """Iterate over all population-frequency-amplitude gain entries."""
        return product(range(self.npops), self.freqs, self.amps)

    def compute_input_gains(self):
        """Fill the cached direct gains from the shared probe dataset."""
        if self.R_probe is None:
            raise RuntimeError(
                "compute_input_traces() must be called before compute_input_gains()."
            )

        self.D[:] = np.nan
        for pop_num, f, amp in self._iter_gain_product():
            cc = dict(pop=pop_num, freq=f, amp=amp)
            self.D.loc[cc] = self._probe_gain(pop_num, f, amp)
        return self.D

    def compute_transfer_mat(self):
        """Fill the cached transfer matrix from the shared probe dataset."""
        if self.R_probe is None:
            raise RuntimeError(
                "compute_input_traces() must be called before compute_transfer_mat()."
            )

        self.C[:] = np.nan
        for pop_pre, pop_post, f, amp in self._iter_transfer_product():
            cc = dict(pop_pre=pop_pre, pop_post=pop_post, freq=f, amp=amp)
            self.C.loc[cc] = self._probe_coef(pop_pre, pop_post, f, amp)
        return self.C


class TransferMatBuider(TransferMatBuiderBase):
    """Estimate transfer coefficients from one-pop probes."""

    def __init__(
            self,
            model: RateModel,
            freqs,
            amps,
            h0,
            dt: float,
            sigma_h: np.ndarray | float = 0.0,
            seed: int = 0,
            warmup_time: float = 8.0,
            measure_time: float = 24.0,
            welch_time: float = 4.0,
            welch_overlap: float = 0.5,
            steady_time: float = 20.0,
            ):
        self.model = model
        self.pops = np.arange(model.npops)
        self.freqs = np.asarray(freqs, dtype=float)
        self.amps = np.asarray(amps, dtype=float)
        self.h0 = np.asarray(h0, dtype=float).reshape((model.npops, 1))
        self.dt = float(dt)
        self.sigma_h = np.asarray(sigma_h, dtype=float)
        self.seed = int(seed)
        self.warmup_time = float(warmup_time)
        self.measure_time = float(measure_time)
        self.welch_time = float(welch_time)
        self.welch_overlap = float(welch_overlap)
        self.steady_time = float(steady_time)

        self.r0 = None
        self.steady_steps = None
        self.warmup_steps = None
        self.measure_steps = None
        self.keep_steps = None
        self.total_probe_steps = None
        self.welch_target_steps = None
        self.time = None

        self._prepare()

    def _prepare(self):
        """Prepare cached step counts and cached traces."""
        super()._prepare()
        self.r0 = None
        self.steady_steps = self._time_to_steps(self.steady_time)
        self.warmup_steps = self._time_to_steps(self.warmup_time)
        self.measure_steps = self._time_to_steps(self.measure_time)
        self.keep_steps = self.warmup_steps + self.measure_steps
        self.total_probe_steps = self.steady_steps + self.keep_steps
        self.welch_target_steps = self._time_to_steps(self.welch_time)
        self.time = np.arange(self.keep_steps, dtype=float) * self.dt
        self.D.attrs = {
            'dt': self.dt,
            'steady_time': self.steady_time,
            'warmup_time': self.warmup_time,
            'measure_time': self.measure_time,
            'definition': 'direct input-to-rate complex gain from true open-loop probes',
        }
        self._validate_time_config()

    def _time_to_steps(self, time_value: float) -> int:
        """Convert a duration in seconds to a floored step count."""
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if time_value < 0:
            raise ValueError("Time windows must be non-negative.")
        raw_steps = float(time_value) / self.dt
        return int(np.floor(raw_steps + 1e-12))

    def _period_steps(self, f: float) -> int:
        """Return the integer number of simulation steps per probe period."""
        raw = 1.0 / (float(f) * self.dt)
        period_steps = int(round(raw))
        if period_steps <= 0 or not np.isclose(raw, period_steps, rtol=1e-9, atol=1e-12):
            raise ValueError(
                f"dt={self.dt} does not give an integer period for f={float(f)}."
            )
        return period_steps

    def _welch_config(self, f: float) -> tuple[int, int, int]:
        """Return the Welch window length, FFT bin, and hop size for a probe frequency."""
        period_steps = self._period_steps(f)
        window_cycles = self.welch_target_steps // period_steps
        if window_cycles <= 0:
            raise ValueError(
                f"welch_time={self.welch_time} is too short for one full cycle at f={float(f)}."
            )
        window_steps = int(window_cycles * period_steps)
        hop_steps = max(1, int(round(window_steps * (1.0 - self.welch_overlap))))
        return window_steps, int(window_cycles), hop_steps

    def _validate_time_config(self):
        """Validate the configured time windows against the simulation grid."""
        if self.nfreqs == 0:
            raise ValueError("At least one probe frequency is required.")
        if self.steady_steps <= 0:
            raise ValueError("steady_time must contain at least one time step.")
        if self.measure_steps <= 0:
            raise ValueError("measure_time must contain at least one time step.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_time must be non-negative.")
        if self.keep_steps <= 0:
            raise ValueError("warmup_time + measure_time must contain at least one time step.")
        if self.welch_target_steps <= 0:
            raise ValueError("welch_time must contain at least one time step.")
        if not 0.0 <= self.welch_overlap < 1.0:
            raise ValueError("welch_overlap must be in the interval [0, 1).")

        for f in self.freqs:
            window_steps, _, _ = self._welch_config(float(f))
            if window_steps > self.measure_steps:
                raise ValueError(
                    f"welch_time={self.welch_time} produces a window longer than measure_time "
                    f"for f={float(f)}."
                )

    def _noise_sigma(self, pop_num: int) -> float:
        """Return the configured noise scale for one population."""
        if self.sigma_h.ndim == 0:
            return float(self.sigma_h)
        sigma = self.sigma_h.reshape(-1)
        if sigma.size != self.npops:
            raise ValueError("sigma_h must be scalar or have one value per population.")
        return float(sigma[pop_num])

    def _seed_from_key(self, *parts) -> int:
        """Build a deterministic RNG seed from the global seed and a probe key."""
        key = '|'.join(str(part) for part in parts)
        digest = hashlib.sha256(key.encode('utf-8')).digest()
        return (self.seed + int.from_bytes(digest[:8], 'little')) % (2**32)

    def _noise_trace(self, pop_num: int, nsteps: int, *key_parts) -> np.ndarray:
        """Generate a deterministic white-noise trace for one population."""
        sigma = self._noise_sigma(pop_num)
        if sigma == 0:
            return np.zeros(nsteps)
        rng = np.random.default_rng(self._seed_from_key(*key_parts))
        return sigma * rng.standard_normal(nsteps) / np.sqrt(self.dt)

    def _run_model_final(self, h: np.ndarray, r0: np.ndarray, nsteps: int) -> np.ndarray:
        """Run the full model and return only the final state."""
        sim_res_type = self.model.sim_res_type
        self.model.sim_res_type = 'final'
        try:
            return np.asarray(
                self.model.run(h, r0, dt=self.dt, nsteps=nsteps),
                dtype=float,
            ).reshape((self.npops, 1))
        finally:
            self.model.sim_res_type = sim_res_type

    def _run_1pop_full(
            self,
            pop_num: int,
            h: np.ndarray | float,
            r0: np.ndarray,
            nsteps: int,
            ) -> np.ndarray:
        """Run one population and return its full time trace."""
        sim_res_type = self.model.sim_res_type
        self.model.sim_res_type = 'full'
        try:
            rr = self.model.run_1pop(pop_num, h, r0, dt=self.dt, nsteps=nsteps)
            return np.asarray(rr.values[0], dtype=float)
        finally:
            self.model.sim_res_type = sim_res_type

    def _ensure_r0(self) -> np.ndarray:
        """Compute and cache the steady-state baseline rates."""
        if self.r0 is not None:
            return self.r0
        self.r0 = self._run_model_final(
            self.h0,
            np.zeros((self.npops, 1)),
            self.steady_steps,
        )
        return self.r0

    def _build_probe_input(self, pop_num: int, f: float, amp: float) -> np.ndarray:
        """Build the noisy cosine drive used to probe one input population."""
        tt = np.arange(self.total_probe_steps, dtype=float) * self.dt
        h = np.full(self.total_probe_steps, float(self.h0[pop_num, 0]))
        h += float(amp) * np.cos(2 * np.pi * float(f) * tt)
        h += self._noise_trace(pop_num, self.total_probe_steps, 'input', pop_num, f, amp)
        return h

    def _estimate_complex_amplitude(
            self,
            x: np.ndarray,
            t: np.ndarray,
            f: float,
            ) -> complex:
        """Fit one sinusoid and return its complex amplitude."""
        x = np.asarray(x, dtype=float).reshape(-1)
        t = np.asarray(t, dtype=float).reshape(-1)
        omega_t = 2 * np.pi * float(f) * t
        design = np.column_stack([
            np.cos(omega_t),
            np.sin(omega_t),
            np.ones_like(omega_t),
        ])
        coef, *_ = np.linalg.lstsq(design, x, rcond=None)
        cos_coef, sin_coef, _ = coef
        return complex(cos_coef - 1j * sin_coef)

    def _welch_transfer(self, x: np.ndarray, y: np.ndarray, f: float) -> complex:
        """Estimate the transfer coefficient with Welch-averaged FFT bins."""
        window_steps, bin_num, hop_steps = self._welch_config(f)
        if window_steps > x.size:
            raise ValueError("Welch window is longer than the analysis trace.")

        # Slide the Welch window over the measurement-only trace.
        starts = np.arange(0, x.size - window_steps + 1, hop_steps, dtype=int)
        if starts.size == 0:
            raise ValueError("No Welch windows fit inside the analysis trace.")

        # Average the cross-spectrum and autospectrum at the probe-frequency bin.
        window = np.hanning(window_steps)
        sxy = np.zeros(starts.size, dtype=complex)
        sxx = np.zeros(starts.size, dtype=float)
        for n, start in enumerate(starts):
            xw = x[start:start + window_steps]
            yw = y[start:start + window_steps]
            xw = (xw - xw.mean()) * window
            yw = (yw - yw.mean()) * window
            X = np.fft.rfft(xw)[bin_num]
            Y = np.fft.rfft(yw)[bin_num]
            sxy[n] = np.conj(X) * Y
            sxx[n] = np.abs(X) ** 2

        return sxy.mean() / (sxx.mean() + 1e-30)

    def compute_input_traces(self) -> xr.DataArray:
        """Run and store the shared open-loop input probes."""
        # Rebuilding the shared upstream probe dataset invalidates downstream
        # derived artifacts until they are recomputed explicitly.
        self.D[:] = np.nan
        self.C[:] = np.nan

        # Start each probe from the shared steady-state baseline.
        r0 = self._ensure_r0()
        rr = np.empty((self.npops, self.nfreqs, self.namps, self.keep_steps), dtype=float)

        # Run each one-pop probe and keep only the post-steady segment.
        for pop_pre, ifreq, iamp in product(range(self.npops), range(self.nfreqs), range(self.namps)):
            f = float(self.freqs[ifreq])
            amp = float(self.amps[iamp])
            h = self._build_probe_input(pop_pre, f, amp)
            trace = self._run_1pop_full(pop_pre, h, r0, self.total_probe_steps)
            rr[pop_pre, ifreq, iamp, :] = trace[self.steady_steps:]

        # Store the full probe-response tensor on a physical time axis.
        self.R_probe = xr.DataArray(
            rr,
            dims=['pop_pre', 'freq', 'amp', 'time'],
            coords={
                'pop_pre': np.arange(self.npops),
                'freq': self.freqs,
                'amp': self.amps,
                'time': self.time,
            },
            attrs={
                'dt': self.dt,
                'steady_time': self.steady_time,
                'warmup_time': self.warmup_time,
                'measure_time': self.measure_time,
                'welch_time': self.welch_time,
            },
        )
        return self.R_probe

    def _probe_gain(self, pop_num, f, amp):
        """Estimate one direct gain entry from the shared open-loop probe trace."""
        x = np.asarray(
            self.R_probe.loc[dict(pop_pre=pop_num, freq=f, amp=amp)].values,
            dtype=float,
        )
        t_measure = self.time[self.warmup_steps:] + self.steady_time
        return (
            self._estimate_complex_amplitude(
                x[self.warmup_steps:],
                t_measure,
                float(f),
            )
            / float(amp)
        )

    def _probe_coef(self, pop_pre, pop_post, f, amp):
        """Compute one transfer coefficient from the shared probe trace."""
        # Use the shared input-pop probe trace to build the surrogate state.
        x = np.asarray(
            self.R_probe.loc[dict(pop_pre=pop_pre, freq=f, amp=amp)].values,
            dtype=float,
        )
        r0_trace = np.repeat(self._ensure_r0(), x.size, axis=1)
        r0_trace[pop_pre, :] = x

        # Simulate the output population driven by the surrogate pre-population trace.
        h = np.full(x.size, float(self.h0[pop_post, 0]))
        h += self._noise_trace(pop_post, x.size, 'output', pop_pre, pop_post, f, amp)
        y = self._run_1pop_full(pop_post, h, r0_trace, x.size)

        # Estimate the transfer on the measurement-only part of the trace.
        x_measure = x[self.warmup_steps:]
        y_measure = y[self.warmup_steps:]
        return self._welch_transfer(x_measure, y_measure, float(f))
