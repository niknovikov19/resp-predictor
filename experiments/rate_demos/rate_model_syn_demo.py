import os
import sys
from pathlib import Path

# Add the repo root for direct script runs
if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

os.environ.setdefault('MPLCONFIGDIR', '.matplotlib')

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from rate_model import RateModelWC


def main():
    W = np.array([[0.0, 3.5], [0.0, 0.0]])
    tau = np.array([[2.0], [2.0]])
    tau_syn = np.array([[0.0, 8.0], [0.0, 0.0]])

    h = np.zeros((2, 80))
    h[1, 10:] = 1.2
    r0 = np.zeros((2, 1))
    dt = 0.5

    model_inst = RateModelWC(
        W, tau=tau,
        rmax=12.0, gain_slope=1.0, gain_center=0.0
    )
    model_syn = RateModelWC(
        W, tau=tau, tau_syn=tau_syn,
        rmax=12.0, gain_slope=1.0, gain_center=0.0
    )

    rr_inst = model_inst.run(h, r0, dt=dt, nsteps=h.shape[1])
    rr_syn = model_syn.run(h, r0, dt=dt, nsteps=h.shape[1])

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax[0].plot(rr_inst.time, rr_inst.isel(pop=0), label='instant')
    ax[0].plot(rr_syn.time, rr_syn.isel(pop=0), label='filtered')
    ax[0].set_ylabel('pop 0 rate')
    ax[0].set_title('Filtered synapse slows the transient but keeps the same fixed point')
    ax[0].legend()

    ax[1].plot(rr_inst.time, rr_inst.isel(pop=1), label='driver')
    ax[1].plot(rr_syn.time, rr_syn.isel(pop=1), '--', label='driver (same)')
    ax[1].plot(rr_inst.time, h[1], ':', label='external input to pop 1')
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('pop 1 / input')
    ax[1].legend()

    fig.tight_layout()
    fig.savefig('rate_model_syn_demo.png', dpi=150)

    print('Saved rate_model_syn_demo.png')
    print('Final rate, instant :', rr_inst.isel(time=-1).values)
    print('Final rate, filtered:', rr_syn.isel(time=-1).values)


if __name__ == '__main__':
    main()
