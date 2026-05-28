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
from rate_model_eig import (
    make_W_h0_ei_2d,
    rate_qphi_to_eig_qphi,
    rate_phi_bounds,
    wrap_rate_phi_to_zero_branch,
    validate_rate_phi_ie,
    _build_W_h0_ei_2d_syn_from_eig_qphi,
    make_W_h0_ei_2d_syn,
    make_eigmode_start_ei_2d_syn,
)



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

    # Set the synaptic time constants
    # Synapses
    taue_syn = 25e-3
    taui_syn = 50e-3
    tau_syn = np.array([taue_syn, taui_syn])

    # Define the target EI eigenmode
    # Eigenmode
    f = 10
    tau_decay = -0.7
    q_ie = 1
    phi_ie = 0.8

    # Steady-state
    r0 = np.array([5, 5])

    # Set the simulation timeline
    # Simulation
    T = 2
    dt = 5e-4
    nsteps = int(T / dt)
    tt = np.arange(nsteps) * dt

    # Initial point
    dre_start = 1
    phase_start = -pi / 2

    # Set the external perturbations
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

    # Convert the rate coordinates to eig coordinates
    # Convert rate q/phi to eig q/phi
    q_eig, phi_eig = rate_qphi_to_eig_qphi(q_ie, phi_ie, tau_syn, f, tau_decay)

    # Build the EI model and aligned initial state
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

    # Build the pulse-driven input trace
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
    
    # Simulate from the eigenmode-matched initial state
    # Start from a rate/synapse state on the eigenmode
    R = model.run(H, r_start, dt, nsteps, s0=s_start)
    tt = R.coords['time']

    # Average pulse-triggered raw and filtered epochs
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
    
    # Plot the full traces and epoch averages
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
