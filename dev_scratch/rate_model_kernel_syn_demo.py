import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '.matplotlib')

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from rate_model_kernel import RateModelKernel

ARTIFACT_DIR = Path(__file__).resolve().parent / 'artifacts' / 'rate_model_ker'
ARTIFACT_PATH = ARTIFACT_DIR / 'rate_model_kernel_syn_demo.png'


def solve_h0_for_kernel(model, r0, tol=1e-12, max_iter=2000):
    """Find h0 so the kernel model keeps r0 as its constant-input fixed point."""
    r0 = np.asarray(r0, dtype=float).reshape((-1, 1))
    mu0 = model.gain_inv(r0)
    q = r0.copy()

    for _ in range(max_iter):
        q_new = np.asarray(model.gain(mu0 - model.W @ q), dtype=float).reshape(q.shape)
        if np.max(np.abs(q_new - q)) < tol:
            q = q_new
            break
        q = q_new
    else:
        raise RuntimeError('Failed to derive h0 for the requested kernel fixed point.')

    h0 = mu0 - model.W @ q
    return h0, q


def main():
    np.random.seed(123)

    # Target operating point
    r0 = np.array([2.0, 4.0, 5.0, 6.0, 8.0]).reshape(-1, 1)
    n_pop = r0.shape[0]

    # Kernel model parameters
    tau = np.linspace(0.4, 1.8, n_pop).reshape(-1, 1)
    tau_syn = np.array([
        [0.0, 2.0, 0.0, 0.0, 1.5],
        [1.0, 0.0, 2.5, 0.0, 0.0],
        [0.0, 1.5, 0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 2.5],
        [1.5, 0.0, 0.0, 1.0, 0.0],
    ])

    model, _, J = RateModelKernel.from_steady_state(
        r0=r0,
        tau=tau,
        rmax=10.0,
        gain_slope=0.5,
        gain_center=0.0,
        R=0.8,
        J0=-0.8,
        tau_syn=tau_syn,
    )
    h0, q0 = solve_h0_for_kernel(model, r0)

    # Simulate noisy fluctuations around the operating point
    T = 20.0
    dt = 0.05
    nsteps = int(T / dt)
    sigma_h = 0.01
    H = h0 + sigma_h * np.random.randn(n_pop, nsteps) / np.sqrt(dt)

    rr = model.run(H, r0, dt=dt, nsteps=nsteps)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for pop in range(n_pop):
        ax.plot(rr.time, rr.isel(pop=pop), label=f'pop {pop} (r0={r0[pop, 0]:.0f})')
        ax.axhline(r0[pop, 0], color=f'C{pop}', linestyle=':', linewidth=0.9)

    ax.set_xlabel('time')
    ax.set_ylabel('rate')
    ax.set_title('5-population kernel model with exponential synapses and noisy input around h0')
    ax.legend(loc='upper right', ncol=2)

    fig.tight_layout()
    fig.savefig(ARTIFACT_PATH, dpi=150)

    print(f'Saved {ARTIFACT_PATH}')
    print('Target rates :', r0.ravel())
    print('Mean rates   :', rr.mean(dim='time').values)
    print('Final rates  :', rr.isel(time=-1).values)
    print('h0           :', h0.ravel())
    print('gain(h0)     :', q0.ravel())
    print('Jacobian eig :', np.linalg.eigvals(J))


if __name__ == '__main__':
    main()
