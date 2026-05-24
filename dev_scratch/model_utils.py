import os
from pathlib import Path
import sys

DIRPATH_REPO = Path(__file__).parents[1]
sys.path.append(str(DIRPATH_REPO))

import matplotlib.pyplot as plt
import numpy as np
from numpy import real, imag, abs, pi, sin, cos
from scipy.signal import butter, sosfiltfilt

from rate_model import RateModelWC



def sample_disk_points_conjugate_symmetric(N: int, R: float):
    vals = []
    k = 0
    while k < N:
        if k == N - 1:
            x = np.random.uniform(-R, R)
            vals.append(complex(x, 0.0))
            k += 1
        else:
            u = np.random.rand()
            theta = 2 * np.pi * np.random.rand()
            lam = R * np.sqrt(u) * np.exp(1j * theta)

            if abs(lam.imag) < 1e-12 or np.random.rand() < 0.3:
                vals.append(complex(lam.real, 0.0))
                k += 1
            else:
                vals.append(lam)
                vals.append(np.conj(lam))
                k += 2
    return vals[:N]


def block_diag(blocks):
    n = sum(b.shape[0] for b in blocks)
    out = np.zeros((n, n), dtype=float)
    i = 0
    for b in blocks:
        m = b.shape[0]
        out[i:i+m, i:i+m] = b
        i += m
    return out


def random_W_real(N: int, R: float) -> np.ndarray:
    eigvals = sample_disk_points_conjugate_symmetric(N, R)

    blocks = []
    i = 0
    while i < N:
        lam = eigvals[i]
        if abs(lam.imag) < 1e-12:
            blocks.append(np.array([[lam.real]]))
            i += 1
        else:
            a, b = lam.real, lam.imag
            blocks.append(np.array([[a, -b],
                                    [b,  a]]))
            i += 2

    J = block_diag(blocks)

    Q = np.random.randn(N, N)
    while np.linalg.matrix_rank(Q) < N:
        Q = np.random.randn(N, N)

    W = Q @ J @ np.linalg.inv(Q)
    return W


def make_W_h0(model, r0, R, J0=-1.0):
    model_, h0, J = RateModelWC.from_steady_state(
        r0=r0,
        tau=model.tau,
        rmax=model.rmax,
        gain_slope=model.gain_slope,
        gain_center=model.gain_center,
        R=R,
        J0=J0,
        tau_syn=model.tau_syn,
    )
    return model_.W, h0, J


def make_W_h0_ei_2d(model, r0, tau, f, tau_decay, q_ie=1, phi_ie=0, tol=1e-12):
    """
    Assumes:
        dr/dt = T^{-1}[-r + gain(W @ r + h0)]

    tau and tau_decay must be in seconds if f is in Hz.
    """

    r0 = np.asarray(r0, dtype=float).reshape(-1, 1)
    if r0.shape != (2, 1):
        raise ValueError("r0 must have shape (2,) or (2,1)")

    tau = np.asarray(tau, dtype=float).ravel()
    if tau.size == 1:
        tau = np.repeat(tau.item(), 2)
    if tau.size != 2:
        raise ValueError("tau must be scalar or length 2")

    tau_e, tau_i = tau
    if tau_e <= 0 or tau_i <= 0:
        raise ValueError("tau values must be positive")
    if tau_decay <= 0:
        raise ValueError("tau_decay must be positive")
    if f < 0:
        raise ValueError("f must be nonnegative")

    mu0 = np.asarray(model.gain_inv(r0), dtype=float).reshape(2, 1)
    gp = np.asarray(model.gain_deriv(r0), dtype=float).ravel()

    if gp.size != 2:
        raise ValueError("gain_deriv must return length 2")
    if np.any(gp <= 0):
        raise ValueError(f"gain derivatives must be positive; got {gp}")

    g_e, g_i = gp

    a = -1.0 / tau_decay
    b = 2.0 * np.pi * f

    alpha_min = max(
        -1.0 / tau_e,
        2.0 * a + 1.0 / tau_i,
    )

    alpha = a + b / np.tan(phi_ie)

    if alpha < alpha_min - tol:
        if b == 0:
            raise ValueError("phi_ie violates Dale constraint: no valid phi_ie")
        phi_max = np.arctan2(b, alpha_min - a)
        n = np.floor(phi_ie / np.pi)
        phi_lo = n * np.pi
        phi_hi = phi_lo + phi_max
        _raise_phi_error(phi_ie, phi_lo, phi_hi)

    K = (alpha - a) ** 2 + b ** 2
    sqrtK = np.sqrt(K)

    beta = sqrtK / q_ie
    gamma = q_ie * sqrtK

    J_target = np.array([
        [alpha, -beta],
        [gamma, 2.0 * a - alpha],
    ])

    T = np.diag([tau_e, tau_i])
    Ginv = np.diag([1.0 / g_e, 1.0 / g_i])

    W = Ginv @ (np.eye(2) + T @ J_target)
    h0 = mu0 - W @ r0

    G = np.diag([g_e, g_i])
    J = np.diag(1.0 / tau) @ (G @ W - np.eye(2))

    if not (
        W[0, 0] >= -tol and
        W[0, 1] <=  tol and
        W[1, 0] >= -tol and
        W[1, 1] <=  tol
    ):
        raise RuntimeError(f"Dale sign violation:\nW=\n{W}")

    return W, h0, J


def _syn_eig_factors(tau_syn, f, tau_decay):
    tau_e, tau_i = np.asarray(tau_syn, dtype=float).ravel()
    a = -1.0 / tau_decay
    b = 2.0 * np.pi * f
    lam = a + 1j * b
    Ae = 1.0 + lam * tau_e
    Ai = 1.0 + lam * tau_i
    return tau_e, tau_i, a, b, Ae, Ai


def rate_qphi_to_eig_qphi(q_rate, phi_rate, tau_syn, f, tau_decay):
    tau_e, tau_i, a, b, Ae, Ai = _syn_eig_factors(tau_syn, f, tau_decay)
    q_eig = q_rate * np.abs(Ae) / np.abs(Ai)
    phi_eig = phi_rate - np.angle(Ae) + np.angle(Ai)
    return q_eig, phi_eig


def _eig_phi_shift(tau_syn, f, tau_decay):
    tau_e, tau_i, a, b, Ae, Ai = _syn_eig_factors(tau_syn, f, tau_decay)
    return np.angle(Ae) - np.angle(Ai)


def _eig_phi_upper(tau_syn, f, tau_decay, tol=1e-12):
    tau_e, tau_i, a, b, Ae, Ai = _syn_eig_factors(tau_syn, f, tau_decay)
    alpha_min = max(-1.0 / tau_e, 2.0 * a + 1.0 / tau_i)
    if b == 0:
        if a >= alpha_min - tol:
            return np.pi
        return np.nan
    phi_upper = np.arctan2(b, alpha_min - a)
    if phi_upper > 0:
        return float(phi_upper)
    return np.nan


def rate_phi_bounds(tau_syn, f, tau_decay):
    # Keep the displayed branch around zero
    phi_upper = _eig_phi_upper(tau_syn, f, tau_decay)
    if not np.isfinite(phi_upper):
        return np.nan, np.nan
    lower = np.mod(_eig_phi_shift(tau_syn, f, tau_decay), np.pi)
    upper = lower + phi_upper
    if upper > np.pi:
        lower -= np.pi
        upper -= np.pi
    return lower, upper


def wrap_rate_phi_to_zero_branch(phi_ie, tau_syn, f, tau_decay):
    # Shift phi_ie onto the branch that contains zero
    lower, upper = rate_phi_bounds(tau_syn, f, tau_decay)
    if not (np.isfinite(lower) and np.isfinite(upper)):
        return np.nan
    return lower + np.mod(phi_ie - lower, np.pi)


def _raise_phi_error(phi_ie, lower, upper):
    raise ValueError(
        f"phi_ie violates Dale constraint: use ({lower:.3f}, {upper:.3f}], got {phi_ie:.3f}"
    )


def validate_rate_phi_ie(phi_ie, tau_syn, f, tau_decay, tol=1e-12):
    # Check phi_ie in rate coordinates before any eig conversion
    lower, upper = rate_phi_bounds(tau_syn, f, tau_decay)
    if not (np.isfinite(lower) and np.isfinite(upper)):
        raise ValueError("phi_ie violates Dale constraint: no valid phi_ie")
    phi_zero = wrap_rate_phi_to_zero_branch(phi_ie, tau_syn, f, tau_decay)
    if not (lower < phi_zero <= upper + tol):
        _raise_phi_error(phi_ie, lower, upper)
    return phi_zero


def _build_W_h0_ei_2d_syn_from_eig_qphi(model, r0, tau_syn, f, tau_decay,
                                        q_ie=1, phi_ie=0, tol=1e-12):
    """
    Assumes almost instantaneous rate dynamics:
        x' = T_syn^{-1}[-x + r]
        r ~= gain(W @ x + h0)

    tau_syn and tau_decay must be in seconds if f is in Hz.
    Negative tau_decay is allowed and gives a positive real part
    in the target eigenvalues.
    tau_syn = [tau_e, tau_i] means synapses from E have tau_e and
    synapses from I have tau_i.

    Returns:
        W, h0, J, tau_syn_mat
    where tau_syn_mat is the 2x2 matrix to assign to model.tau_syn.
    """

    r0 = np.asarray(r0, dtype=float).reshape(-1, 1)
    if r0.shape != (2, 1):
        raise ValueError("r0 must have shape (2,) or (2,1)")

    tau_syn = np.asarray(tau_syn, dtype=float).ravel()
    if tau_syn.size == 1:
        tau_syn = np.repeat(tau_syn.item(), 2)
    if tau_syn.size != 2:
        raise ValueError("tau_syn must be scalar or length 2")

    tau_e, tau_i = tau_syn
    if tau_e <= 0 or tau_i <= 0:
        raise ValueError("tau_syn values must be positive")
    if tau_decay == 0:
        raise ValueError("tau_decay must be nonzero")
    if f < 0:
        raise ValueError("f must be nonnegative")

    mu0 = np.asarray(model.gain_inv(r0), dtype=float).reshape(2, 1)
    gp = np.asarray(model.gain_deriv(r0), dtype=float).ravel()

    if gp.size != 2:
        raise ValueError("gain_deriv must return length 2")
    if np.any(gp <= 0):
        raise ValueError(f"gain derivatives must be positive; got {gp}")

    g_e, g_i = gp

    # The public wrapper already checked the rate-space Dale constraint
    tau_e, tau_i, a, b, Ae, Ai = _syn_eig_factors(tau_syn, f, tau_decay)
    alpha = a + b / np.tan(phi_ie)

    K = (alpha - a) ** 2 + b ** 2
    sqrtK = np.sqrt(K)

    beta = sqrtK / q_ie
    gamma = q_ie * sqrtK

    J_target = np.array([
        [alpha, -beta],
        [gamma, 2.0 * a - alpha],
    ])

    T = np.diag([tau_e, tau_i])
    Ginv = np.diag([1.0 / g_e, 1.0 / g_i])

    W = Ginv @ (np.eye(2) + T @ J_target)
    h0 = mu0 - W @ r0

    J = np.diag(1.0 / tau_syn) @ (np.diag([g_e, g_i]) @ W - np.eye(2))

    if not (
        W[0, 0] >= -tol and
        W[0, 1] <=  tol and
        W[1, 0] >= -tol and
        W[1, 1] <=  tol
    ):
        raise RuntimeError(f"Dale sign violation:\nW=\n{W}")

    tau_syn_mat = np.array([
        [tau_e, tau_i],
        [tau_e, tau_i],
    ])

    return W, h0, J, tau_syn_mat


def make_W_h0_ei_2d_syn(model, r0, tau_syn, f, tau_decay,
                        q_ie=1, phi_ie=0, tol=1e-12):
    # Validate phi_ie in rate space before converting to eig coordinates
    phi_zero = validate_rate_phi_ie(phi_ie, tau_syn, f, tau_decay, tol=tol)

    # Build the EI system from the equivalent eig-space q/phi
    q_eig, phi_eig = rate_qphi_to_eig_qphi(q_ie, phi_zero, tau_syn, f, tau_decay)
    return _build_W_h0_ei_2d_syn_from_eig_qphi(
        model, r0, tau_syn, f, tau_decay, q_ie=q_eig, phi_ie=phi_eig, tol=tol
    )


def make_eigmode_start_ei_2d_syn(
        W,
        r0,
        tau_syn,
        f,
        tau_decay,
        q_ie=1,
        phi_ie=0,
        dre_start=0.1,
        phase_start=0.0,
        tol=1e-12
        ):
    r0 = np.asarray(r0, dtype=float).reshape(2, 1)
    W = np.asarray(W, dtype=float).reshape(2, 2)
    tau_e, tau_i, a, b, Ae, Ai = _syn_eig_factors(tau_syn, f, tau_decay)
    phi_zero = validate_rate_phi_ie(phi_ie, np.array([tau_e, tau_i]), f, tau_decay, tol=tol)

    # Build the rate amplitudes at t=0
    re_amp = dre_start * np.exp(1j * phase_start)
    ri_amp = dre_start * q_ie * np.exp(1j * (phase_start - phi_zero))
    r_amp = np.array([re_amp, ri_amp], dtype=complex)
    r_start = r0 + np.real(r_amp).reshape(2, 1)

    # Convert the rate amplitudes to the filtered synaptic state
    x_amp = np.array([re_amp / Ae, ri_amp / Ai], dtype=complex)
    x0 = r0[:, 0] + np.real(x_amp)
    tau_syn_mat = np.array([
        [tau_e, tau_i],
        [tau_e, tau_i],
    ])
    s_start = np.zeros((2, 2), dtype=float)
    is_dyn = tau_syn_mat > 0
    s_hat = W * x0.reshape(1, 2)
    s_start[is_dyn] = s_hat[is_dyn]

    return r_start, s_start


def main_rand():
    #np.random.seed(114)

    # Num. populations
    N = 5

    tau = np.linspace(0.2, 2, N)
    rho = 0.8   # spectral radius   

    # Baseline rate vector
    r0 = np.linspace(2, 8, N).reshape(-1, 1)

    # Model
    dummy_W = np.zeros((N, N))
    model = RateModelWC(
        dummy_W, tau=tau, rmax=10, gain_slope=0.5, gain_center=0)
    W, h0, J = make_W_h0(model, r0, rho)
    model.W = W

    # Simulation params
    T = 20
    dt = 0.05
    nsteps = int(T / dt)

    # Add zero-mean noise around that operating point
    sigma_h = 0.05
    H = h0 + sigma_h * np.random.randn(N, nsteps) / np.sqrt(dt)

    # Initial state
    r_start = r0.copy()

    # Run simulation
    R = model.run(
        H, r_start, dt, nsteps
    )
    tt = R.coords['time']

    # Plot result
    plt.figure(figsize=(10, 5))
    for n in range(N):
        plt.plot(tt, R.isel(pop=n))
    plt.xlabel('Time')
    plt.ylabel('Rate')
    plt.show()


def main_ei():
    tau = 0

    # Synapses
    taue_syn = 25e-3
    taui_syn = 50e-3
    tau_syn = np.array([taue_syn, taui_syn])

    # Eigenmode
    f = 10
    tau_decay = 0.5
    q_ie = 1
    phi_ie = 1

    # Steady-state
    r0 = np.array([5, 5])

    # Simulation
    T = 2
    dt = 5e-4
    nsteps = int(T / dt)
    tt = np.arange(nsteps) * dt

    # Initial point
    dre_start = 1
    phase_start = -pi / 2

    # Noise
    sigma_h = 0.00

    # Pulses
    pulse_t0 = 0
    print(f't0={pulse_t0:.04f}')
    pulse_f = 10
    pulse_w = 5e-3
    pulse_amp_e = 0
    pulse_amp_i = -0.0

    # Pulse-locked epochs
    epoch_t_win = (-0.05, 0.05)

    # Bandpass filter
    bp_f = (5.0, 20.0)
    bp_order = 4

    # Convert rate q/phi to eig q/phi
    q_eig, phi_eig = rate_qphi_to_eig_qphi(q_ie, phi_ie, tau_syn, f, tau_decay)

    # Build the model
    model = RateModelWC(
        np.zeros((2, 2)), tau=tau, rmax=10, gain_slope=0.5, gain_center=0)
    W, h0, J, tau_syn_mat = make_W_h0_ei_2d_syn(
        model, r0, tau_syn, f, tau_decay, q_ie=q_ie, phi_ie=phi_ie
    )
    model.W = W
    model.tau_syn = tau_syn_mat
    r_start, s_start = make_eigmode_start_ei_2d_syn(
        W,
        r0,
        tau_syn,
        f,
        tau_decay,
        q_ie=q_ie,
        phi_ie=phi_ie,
        dre_start=dre_start,
        phase_start=phase_start,
    )

    print(f"W:\n{W}")
    print(f"rate q/phi: {q_ie}, {phi_ie}")
    print(f"eig q/phi: {q_eig}, {phi_eig}")
    L = np.linalg.eigvals(J)
    print(f"\nJacobian eigenvalues: {L}")
    print(f'f: {abs(imag(L[0])) / 2 / pi}')
    print(f'tau_decay: {-1.0 / real(L[0])}')

    # Build the input signal
    H = np.repeat(h0, nsteps, axis=1)
    H += sigma_h * np.random.randn(2, nsteps) / np.sqrt(dt)
    pulse_t_mask = np.zeros(nsteps, dtype=bool)
    npulses = int(np.floor((T - pulse_t0) * pulse_f)) + 1
    pulse_kw = max(1, int(np.round(pulse_w / dt)))
    for n in range(npulses):
        pulse_t = pulse_t0 + n / pulse_f
        k0 = int(np.round(pulse_t / dt))
        k1 = min(k0 + pulse_kw, nsteps)
        if 0 <= k0 < nsteps:
            pulse_t_mask[k0:k1] = True
    H[0, pulse_t_mask] += pulse_amp_e
    H[1, pulse_t_mask] += pulse_amp_i
    
    # Start from a rate/synapse state on the eigenmode
    R = model.run(H, r_start, dt, nsteps, s0=s_start)
    tt = R.coords['time']

    # Average over pulse-triggered epochs
    pulse_idx = np.flatnonzero(pulse_t_mask)
    epoch_k0 = int(np.round(epoch_t_win[0] / dt))
    epoch_k1 = int(np.round(epoch_t_win[1] / dt))
    epoch_len = epoch_k1 - epoch_k0
    epoch_start_idx = pulse_idx[
        (pulse_idx + epoch_k0 >= 0) & (pulse_idx + epoch_k1 <= nsteps)
    ]
    tt_epoch = np.arange(epoch_k0, epoch_k1) * dt

    sos = butter(bp_order, bp_f, btype='bandpass', fs=1.0 / dt, output='sos')
    S = {'raw': R.values, 'filt': sosfiltfilt(sos, R.values, axis=1)}
    E = {}
    for k, X in S.items():
        epochs = np.stack(
            [X[:, n + epoch_k0:n + epoch_k1] for n in epoch_start_idx],
            axis=0,
        )
        E[k] = epochs.mean(axis=0)
    
    # Plot the signals
    fig, ax = plt.subplots(
        2, 2, figsize=(12, 7), gridspec_kw={'width_ratios': [3, 1]}
    )
    row_titles = {
        'raw': f'E-I oscillations (f={f} Hz, τ_decay={tau_decay})',
        'filt': f'Bandpass {bp_f[0]}-{bp_f[1]} Hz',
    }
    row_ylabel = {'raw': 'Rate', 'filt': 'Filtered rate'}
    row_label_suffix = {'raw': '', 'filt': ', filtered'}

    for i, key in enumerate(['raw', 'filt']):
        ax[i, 0].plot(tt, S[key][0], label=f'Excitatory{row_label_suffix[key]}')
        ax[i, 0].plot(tt, S[key][1], label=f'Inhibitory{row_label_suffix[key]}')
        #for t_pulse in tt[pulse_t_mask]:
        #    ax[i, 0].axvline(t_pulse, color='k', linestyle='--', alpha=0.01)
        if key == 'raw':
            ax[i, 0].axhline(r0[0], color='C0', linestyle='--', alpha=0.3)
            ax[i, 0].axhline(r0[1], color='C1', linestyle='--', alpha=0.3)
        ax[i, 0].set_xlabel('Time')
        ax[i, 0].set_ylabel(row_ylabel[key])
        ax[i, 0].legend()
        ax[i, 0].set_title(row_titles[key])

        ax[i, 1].plot(tt_epoch, E[key][0], label=f'Excitatory{row_label_suffix[key]}')
        ax[i, 1].plot(tt_epoch, E[key][1], label=f'Inhibitory{row_label_suffix[key]}')
        if key == 'raw':
            ax[i, 1].axhline(r0[0], color='C0', linestyle='--', alpha=0.3)
            ax[i, 1].axhline(r0[1], color='C1', linestyle='--', alpha=0.3)
        ax[i, 1].axvline(0, color='k', linestyle='--', alpha=0.5)
        ax[i, 1].set_xlabel('Time from pulse')
        ax[i, 1].set_ylabel(row_ylabel[key])
        ax[i, 1].set_title(
            f'Epoch average over {len(epoch_start_idx)} pulses'
            if key == 'raw' else 'Filtered epoch average'
        )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main_ei()
    #main_rand()
