import sys
from pathlib import Path

DIRPATH_REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(DIRPATH_REPO))
print(DIRPATH_REPO)

import matplotlib.pyplot as plt
import numpy as np

from rate_model import RateModelLin, RateModelWC
from transfer_mat import TransferMatBuider


def estimate_complex_amplitude(x, t, freq):
    """Fit one sinusoid and return its complex amplitude."""
    x = np.asarray(x, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    omega_t = 2 * np.pi * float(freq) * t
    design = np.column_stack([
        np.cos(omega_t),
        np.sin(omega_t),
        np.ones_like(omega_t),
    ])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    cos_coef, sin_coef, _ = coef
    return cos_coef - 1j * sin_coef


def build_closed_loop_drive(h0, drive_complex, freq, dt, nsteps):
    """Construct a full-pop sinusoidal drive around the baseline input."""
    tt = np.arange(nsteps, dtype=float) * dt
    phase = np.exp(1j * 2 * np.pi * float(freq) * tt)
    return np.repeat(h0, nsteps, axis=1) + np.real(drive_complex[:, None] * phase[None, :])


def predict_closed_loop_response(C, direct_gain, drive_complex):
    """Predict the closed-loop rate response from the measured open-loop transfer matrix."""
    loop_mat = C.transpose('pop_post', 'pop_pre').values
    return np.linalg.solve(
        np.eye(loop_mat.shape[0], dtype=complex) - loop_mat,
        np.asarray(direct_gain, dtype=complex) * drive_complex,
    )


np.random.seed(0)

# Target operating point
#r0 = np.array([1, 3, 5, 7, 9]).reshape(-1, 1)
r0 = np.array([3, 4, 5, 6, 7]).reshape(-1, 1)
n_pop = r0.shape[0]

# Model parameters
tau = np.linspace(0.4, 1.8, n_pop).reshape(-1, 1)

# Model type
model_type = 'lin'
#model_type = 'wc'

# Create model
if model_type == 'lin':
    model, h0, J = RateModelLin.from_steady_state(
        r0=r0, tau=tau, R=0.5
    )
elif model_type == 'wc':
    model, h0, J = RateModelWC.from_steady_state(
        r0=r0, tau=tau, R=0.5,
        rmax=10, gain_slope=0.5, gain_center=0
    )
else:
    raise ValueError(f"Invalid model type: {model_type}")

# Simulation params
T = 10
dt = 0.01
nsteps = int(T / dt)

# Probe params
probe_freq = 5
probe_amp = [1, 5, 10]

# Add input noise
sigma_h = 0.1
H = h0 + sigma_h * np.random.randn(n_pop, nsteps) / np.sqrt(dt)

""" # Test: run model and plot traces
rr_noise = model.run(H, r0, dt=dt, nsteps=nsteps)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for pop in range(n_pop):
    ax.plot(rr_noise.time, rr_noise.isel(pop=pop), label=f'pop {pop} (r0={r0[pop, 0]:.0f})')
    ax.axhline(r0[pop, 0], color=f'C{pop}', linestyle=':', linewidth=0.9)
ax.set_xlabel('Time')
ax.set_ylabel('Rate')
ax.legend(loc='upper right', ncol=2)
 """

# Build transfer matrix
tmb = TransferMatBuider(
    model,
    freqs=np.atleast_1d(probe_freq),
    amps=np.atleast_1d(probe_amp),
    h0=h0,
    dt=dt,
    sigma_h=sigma_h
)
R1 = tmb.compute_input_traces()
D = tmb.compute_input_gains()
C = tmb.compute_transfer_mat()

#print((D.squeeze(drop=True)).round(3))
#print((C.squeeze(drop=True)).round(2))

""" # Plot responses to external drive for unconn. system
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for pop in range(n_pop):
    tt, rr = R1.time, R1.isel(pop_pre=pop).squeeze(drop=True)
    ax.plot(tt, rr, label=f'pop {pop} (r0={r0[pop, 0]:.0f})')
    ax.axhline(r0[pop, 0], color=f'C{pop}', linestyle=':', linewidth=0.9)
ax.set_xlabel('Time')
ax.set_ylabel('Rate')
ax.legend(loc='upper right', ncol=2) """


# External drive to each pop at freq w 
drive_complex = np.array([
    1.0 * np.exp(1j * 0.0),
    0.8 * np.exp(1j * 0.7),
    0.6 * np.exp(-1j * 0.5),
    0.5 * np.exp(1j * 1.4),
    0.9 * np.exp(-1j * 1.1),
], dtype=complex)

# Prediction of closed-loop response to the drive
closed_loop_pred = predict_closed_loop_response(
    C.squeeze(drop=True),
    D.squeeze(drop=True).values,
    drive_complex,
)

H_closed = build_closed_loop_drive(
    h0=h0,
    drive_complex=drive_complex,
    freq=probe_freq,
    dt=dt,
    nsteps=tmb.total_probe_steps,
)
rr_closed = model.run(
    H_closed,
    np.zeros((n_pop, 1)),
    dt=dt,
    nsteps=tmb.total_probe_steps,
)
rr_closed_keep = rr_closed.isel(time=slice(tmb.steady_steps, None))
closed_loop_obs = np.array([
    estimate_complex_amplitude(
        rr_closed.isel(pop=pop).values[tmb.steady_steps + tmb.warmup_steps:],
        np.arange(tmb.steady_steps + tmb.warmup_steps, tmb.total_probe_steps) * dt,
        probe_freq,
    )
    for pop in range(n_pop)
])
rel_err = np.abs(closed_loop_obs - closed_loop_pred) / np.maximum(np.abs(closed_loop_pred), 1e-12)

print("\nClosed-loop complex-amplitude test")
for pop in range(n_pop):
    print(
        f"pop {pop}: "
        f"drive={drive_complex[pop]: .3f}, "
        f"pred={closed_loop_pred[pop]: .3f}, "
        f"obs={closed_loop_obs[pop]: .3f}, "
        f"rel_err={rel_err[pop]:.3%}"
    )
print(f"max relative error: {rel_err.max():.3%}")

fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for pop in range(n_pop):
    axs[0].plot(
        rr_closed_keep.time,
        rr_closed_keep.isel(pop=pop),
        label=f'pop {pop}',
    )
axs[0].set_ylabel('Closed-loop rate')
axs[0].legend(loc='upper right', ncol=2)

pops = np.arange(n_pop)
axs[1].plot(pops, np.abs(closed_loop_pred), 'o-', label='predicted |R|')
axs[1].plot(pops, np.abs(closed_loop_obs), 's--', label='simulated |R|')
axs[1].set_xlabel('Population')
axs[1].set_ylabel('Complex amplitude magnitude')
axs[1].legend(loc='best')

fig_phase, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.plot(pops, np.angle(closed_loop_pred), 'o-', label='predicted phase')
ax.plot(pops, np.angle(closed_loop_obs), 's--', label='simulated phase')
ax.set_xlabel('Population')
ax.set_ylabel('Phase (rad)')
ax.legend(loc='best')


plt.show()
