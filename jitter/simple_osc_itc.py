import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


def osc_kernel(fs, freq_hz, tau_ms, amp=1.0, n_tau=6):
    tau_s = tau_ms / 1000
    kt = np.arange(0, n_tau * tau_s + 1 / fs, 1 / fs)
    h = np.exp(-kt / tau_s) * np.cos(2 * np.pi * freq_hz * kt)
    h *= amp / np.max(np.abs(h))
    return h


def apply_kernel(stim_times, fs, dur, freq_hz, tau_ms, amp=1.0):
    t = np.arange(0, dur, 1 / fs)
    stim_idx = np.round(np.asarray(stim_times) * fs).astype(int)
    stim_idx = stim_idx[(stim_idx >= 0) & (stim_idx < len(t))]
    impulses = np.zeros_like(t)
    impulses[stim_idx] = 1.0
    h = osc_kernel(fs, freq_hz, tau_ms, amp=amp)
    x = fftconvolve(impulses, h, mode="full")[: len(t)]
    return t, x, stim_idx, h


def jittered_onsets(f0, dur, jitter_sd, n_trials, t0=1.0, t1=1.0, seed=1):
    rng = np.random.default_rng(seed)
    base = np.arange(t0, dur - t1, 1 / f0)
    trials = []
    for _ in range(n_trials):
        onset = base + rng.normal(0, jitter_sd, size=len(base))
        onset = onset[(onset >= t0) & (onset <= dur - t1)]
        trials.append(np.sort(onset))
    return trials


def morlet_phase(x, fs, f0, n_cycles=5):
    sigma_t = n_cycles / (2 * np.pi * f0)
    wt = np.arange(-4 * sigma_t, 4 * sigma_t + 1 / fs, 1 / fs)
    psi = np.exp(2j * np.pi * f0 * wt) * np.exp(-0.5 * (wt / sigma_t) ** 2)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
    return np.angle(fftconvolve(x, np.conj(psi[::-1]), mode="same"))


def itc_curve(fs, dur, f0, freq_hz, layer, jitter_sds, n_trials=50):
    rows = []
    for jitter_sd in jitter_sds:
        phases = []
        for onsets in jittered_onsets(f0, dur, jitter_sd, n_trials, seed=1):
            _, x, stim_idx, _ = apply_kernel(onsets, fs, dur, freq_hz, layer["tau_ms"], layer["amp"])
            phase = morlet_phase(x, fs, f0)
            edge = int(np.ceil(4 * (5 / (2 * np.pi * f0)) * fs))
            valid = stim_idx[(stim_idx >= edge) & (stim_idx < len(x) - edge)]
            if len(valid):
                phases.append(phase[valid])
        phases = np.concatenate(phases)
        itc = np.abs(np.mean(np.exp(1j * phases)))
        rows.append(itc)
    raw = np.asarray(rows)
    return raw, raw / raw[0]


fs = 500
dur = 8.0
f0 = 10.0
freq_hz = 1.0
jitter_sds = np.linspace(0, 0.15, 40)

layers = {
    "layer_1": {"amp": 1, "tau_ms": 10},
    "layer_2": {"amp": 1, "tau_ms": 50},
}


# 1) Kernel application check: one stimulus, no noise
single_stim = [2.0]
kernel_data = {}
for name, layer in layers.items():
    kernel_data[name] = apply_kernel(single_stim, fs, 4.0, freq_hz, layer["tau_ms"], layer["amp"])

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
for name, (t, x, stim_idx, _) in kernel_data.items():
    axes[0].plot(t, x, label=f"{name}: amp={layers[name]['amp']}, tau={layers[name]['tau_ms']} ms")
    axes[0].axvline(t[stim_idx[0]], color="k", linestyle=":", linewidth=1)
axes[0].set_title("One stimulus, no noise")
axes[0].set_ylabel("signal")
axes[0].legend()

for name, (_, _, _, h) in kernel_data.items():
    kt = np.arange(len(h)) / fs
    axes[1].plot(kt, h, label=name)
axes[1].set_title("Applied kernels")
axes[1].set_xlabel("time (s)")
axes[1].set_ylabel("kernel")
axes[1].legend()
plt.tight_layout()


# 2) ITC curves
itc = {}
for name, layer in layers.items():
    raw, norm = itc_curve(fs, dur, f0, freq_hz, layer, jitter_sds)
    itc[name] = {"raw": raw, "norm": norm}

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
for name in layers:
    axes[0].plot(jitter_sds * 1000, itc[name]["raw"], "o-", label=name)
    axes[1].plot(jitter_sds * 1000, itc[name]["norm"], "o-", label=name)
axes[0].set_title("Raw ITC")
axes[0].set_xlabel("jitter SD (ms)")
axes[0].set_ylabel("ITC")
axes[1].set_title("Normalized ITC")
axes[1].set_xlabel("jitter SD (ms)")
axes[1].set_ylabel("ITC / ITC(0)")
for ax in axes:
    ax.legend()
plt.tight_layout()


# 3) Signals with marked stimulus times
example_onsets = jittered_onsets(f0, dur, 0.06, n_trials=1, seed=3)[0]
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
for ax, name in zip(axes, layers):
    t, x, stim_idx, _ = apply_kernel(example_onsets, fs, dur, freq_hz, layers[name]["tau_ms"], layers[name]["amp"])
    ax.plot(t, x, color="black")
    ax.vlines(t[stim_idx], ymin=np.min(x), ymax=np.max(x), color="red", alpha=0.35)
    ax.set_title(name)
    ax.set_ylabel("signal")
axes[-1].set_xlabel("time (s)")
plt.tight_layout()
plt.show()