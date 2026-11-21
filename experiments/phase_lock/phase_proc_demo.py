import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.phase_lock.spect_proc import (
    itc, morlet_phase_local, morlet_power, morlet_power_local,
)


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / Path(__file__).stem
LOCAL_ARTIFACT_PATH = ARTIFACT_DIR / "phase_proc_local_demo.png"
ITC_ARTIFACT_PATH = ARTIFACT_DIR / "phase_proc_itc_demo.png"
POWER_ARTIFACT_PATH = ARTIFACT_DIR / "phase_proc_power_demo.png"


def _angle_diff(a, b):
    return float(np.angle(np.exp(1j * (a - b))))


def main():
    # Set the shared phase-analysis parameters
    fs = 1000.0
    dt = 1.0 / fs
    dur = 2.0
    f = 10.0
    n_cycles = 6
    phase0 = 0.6
    t0 = 1.0
    power_overlap = 0.9

    # Build a synthetic oscillatory trace
    t = np.arange(0.0, dur, dt)
    x_clean = (
        np.cos(2.0 * np.pi * f * t + phase0)
        + 0.20 * np.cos(2.0 * np.pi * 3.0 * f * t - 0.4)
    )

    # Mark the local support window around t0
    sigma_t = n_cycles / (2.0 * np.pi * f)
    support = np.abs(t - t0) <= 4.0 * sigma_t
    support_idx = np.flatnonzero(support)

    # Drop samples both inside and outside the support
    x_nan = x_clean.copy()
    x_nan[:120] = np.nan
    x_nan[-150:] = np.nan
    x_nan[support_idx[::6]] = np.nan

    # Compare phase estimates against the clean reference
    phase_clean = morlet_phase_local(x_clean, t, t0, f, n_cycles)
    phase_nan = morlet_phase_local(x_nan, t, t0, f, n_cycles)
    phase_expected = 2.0 * np.pi * f * t0 + phase0

    diff_clean = _angle_diff(phase_clean, phase_expected)
    diff_nan = _angle_diff(phase_nan, phase_expected)
    diff_between = _angle_diff(phase_nan, phase_clean)
    power_clean = morlet_power_local(x_clean, t, t0, f, n_cycles)
    power_nan = morlet_power_local(x_nan, t, t0, f, n_cycles)
    t_power_clean, power_trace_clean = morlet_power(
        x_clean, t, f, n_cycles, overlap=power_overlap
    )
    t_power_nan, power_trace_nan = morlet_power(
        x_nan, t, f, n_cycles, overlap=power_overlap
    )

    # Build a stimulus-locked trace with strong local artifacts
    rng = np.random.default_rng(7)
    tt_stim = 0.25 + np.arange(12) / f
    x_itc = np.cos(2.0 * np.pi * f * t + phase0)
    for t_stim in tt_stim:
        amp = 20.0 * rng.normal()
        width = 6e-3
        x_itc += amp * np.exp(-0.5 * ((t - (t_stim - 3e-3)) / width) ** 2)

    # Compare ITC with and without dropping the stimulus window
    drop_win = (-6e-3, 6e-3)
    itc_raw, phi_raw, phases_raw = itc(x_itc, t, tt_stim, f, n_cycles)
    itc_drop, phi_drop, phases_drop = itc(
        x_itc, t, tt_stim, f, n_cycles, drop_win=drop_win
    )
    phase_expected_wrapped = float(np.angle(np.exp(1j * phase_expected)))
    phase_expected_stim = np.angle(np.exp(1j * (2.0 * np.pi * f * tt_stim + phase0)))
    phase_err_raw = np.angle(np.exp(1j * (phases_raw - phase_expected_stim)))
    phase_err_drop = np.angle(np.exp(1j * (phases_drop - phase_expected_stim)))

    # Prepare the output location
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Plot the local phase demo
    fig_local, ax_local = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax_local[0].plot(t, x_clean, label="clean signal", color="C0", linewidth=1.2)
    ax_local[0].axvspan(
        t0 - 4.0 * sigma_t, t0 + 4.0 * sigma_t, color="C2", alpha=0.12
    )
    ax_local[0].axvline(t0, color="C3", linestyle="--", linewidth=1.0, label="t0")
    ax_local[0].set_ylabel("signal")
    ax_local[0].set_title("Local Morlet phase on a clean oscillation")
    ax_local[0].legend(loc="upper right")

    ax_local[1].plot(t, x_clean, color="0.75", linewidth=1.0, label="clean reference")
    ax_local[1].plot(t, x_nan, color="C1", linewidth=1.2, label="signal with NaNs")
    ax_local[1].scatter(
        t[np.isnan(x_nan)],
        np.zeros(np.isnan(x_nan).sum()),
        s=10,
        color="C3",
        label="NaN samples",
        zorder=3,
    )
    ax_local[1].axvspan(
        t0 - 4.0 * sigma_t, t0 + 4.0 * sigma_t, color="C2", alpha=0.12
    )
    ax_local[1].axvline(t0, color="C3", linestyle="--", linewidth=1.0)
    ax_local[1].set_xlabel("time (s)")
    ax_local[1].set_ylabel("signal")
    ax_local[1].set_title(
        "Phase stays stable with missing samples inside and outside support"
    )
    ax_local[1].legend(loc="upper right")

    # Summarize the local phase results on the figure
    text_local = (
        f"expected phase unwrapped: {phase_expected:.3f} rad\n"
        f"expected phase wrapped  : {phase_expected_wrapped:.3f} rad\n"
        f"clean phase             : {phase_clean:.3f} rad (err {diff_clean:+.3e})\n"
        f"NaN phase               : {phase_nan:.3f} rad (err {diff_nan:+.3e})\n"
        f"NaN-clean diff          : {diff_between:+.3e} rad\n"
        f"clean power             : {power_clean:.3e}\n"
        f"NaN power               : {power_nan:.3e}"
    )
    fig_local.text(0.12, 0.02, text_local, family="monospace")

    # Save the local phase figure
    fig_local.tight_layout(rect=(0, 0.10, 1, 1))
    fig_local.savefig(LOCAL_ARTIFACT_PATH, dpi=150)

    # Plot the sliding power demo
    fig_power, ax_power = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax_power[0].plot(t, x_clean, color="0.7", linewidth=1.0, label="clean signal")
    ax_power[0].plot(t, x_nan, color="C1", linewidth=1.2, label="signal with NaNs")
    ax_power[0].axvline(t0, color="C3", linestyle="--", linewidth=1.0, label="t0")
    ax_power[0].set_ylabel("signal")
    ax_power[0].set_title("Signal used for sliding Morlet power")
    ax_power[0].legend(loc="upper right")

    ax_power[1].plot(
        t_power_clean,
        power_trace_clean,
        color="C0",
        linewidth=1.2,
        label="clean power",
    )
    ax_power[1].plot(
        t_power_nan,
        power_trace_nan,
        color="C2",
        linewidth=1.2,
        label="NaN-robust power",
    )
    ax_power[1].axvline(t0, color="C3", linestyle="--", linewidth=1.0)
    ax_power[1].scatter([t0], [power_clean], color="C0", zorder=3)
    ax_power[1].scatter([t0], [power_nan], color="C2", zorder=3)
    ax_power[1].set_xlabel("time (s)")
    ax_power[1].set_ylabel("power")
    ax_power[1].set_title("Sliding Morlet power compensates for missing-kernel energy")
    ax_power[1].legend(loc="upper right")

    # Summarize the power values on the figure
    text_power = (
        f"local clean power : {power_clean:.3e}\n"
        f"local NaN power   : {power_nan:.3e}\n"
        f"window overlap    : {power_overlap:.2f}\n"
        f"trace max clean   : {np.nanmax(power_trace_clean):.3e}\n"
        f"trace max NaN     : {np.nanmax(power_trace_nan):.3e}"
    )
    fig_power.text(0.12, 0.02, text_power, family="monospace")

    # Save the sliding power figure
    fig_power.tight_layout(rect=(0, 0.09, 1, 1))
    fig_power.savefig(POWER_ARTIFACT_PATH, dpi=150)

    # Plot the ITC demo
    fig_itc, ax_itc = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    ax_itc[0].plot(t, x_itc, color="C0", linewidth=1.0, label="stimulus-locked signal")
    for t_stim in tt_stim:
        ax_itc[0].axvline(t_stim, color="0.7", linestyle="--", linewidth=0.8)
        ax_itc[0].axvspan(
            t_stim + drop_win[0],
            t_stim + drop_win[1],
            color="C3",
            alpha=0.08,
        )
    ax_itc[0].set_xlabel("time (s)")
    ax_itc[0].set_ylabel("signal")
    ax_itc[0].set_title("Stimulus-locked signal with strong peri-stimulus artifacts")
    ax_itc[0].legend(loc="upper right")

    # Plot the per-stimulus phase errors
    stim_idx = np.arange(tt_stim.size)
    ax_itc[1].axhline(0.0, color="0.5", linestyle="--", linewidth=1.0)
    ax_itc[1].plot(stim_idx, phase_err_raw, "o-", color="C1", label="raw phase error")
    ax_itc[1].plot(stim_idx, phase_err_drop, "o-", color="C2", label="masked phase error")
    ax_itc[1].set_xlabel("stimulus index")
    ax_itc[1].set_ylabel("phase error (rad)")
    ax_itc[1].set_title("Masking tightens the stimulus-locked phase estimates")
    ax_itc[1].legend(loc="upper right")

    # Summarize the ITC results on the figure
    text_itc = (
        f"ITC raw/drop            : {itc_raw:.3f} / {itc_drop:.3f}\n"
        f"Avg phase raw/drop      : {phi_raw:.3f} / {phi_drop:.3f} rad"
    )
    fig_itc.text(0.12, 0.02, text_itc, family="monospace")

    # Save the ITC figure
    fig_itc.tight_layout(rect=(0, 0.09, 1, 1))
    fig_itc.savefig(ITC_ARTIFACT_PATH, dpi=150)

    # Print the saved artifact locations
    print(f"Saved {LOCAL_ARTIFACT_PATH}")
    print(f"Saved {POWER_ARTIFACT_PATH}")
    print(f"Saved {ITC_ARTIFACT_PATH}")

    # Print the local phase summary
    print(f"Expected phase unwrapped : {phase_expected:.6f} rad")
    print(f"Expected phase wrapped   : {phase_expected_wrapped:.6f} rad")
    print(f"Clean phase              : {phase_clean:.6f} rad")
    print(f"NaN phase                : {phase_nan:.6f} rad")
    print(f"Clean error              : {diff_clean:+.6e} rad")
    print(f"NaN error                : {diff_nan:+.6e} rad")
    print(f"NaN-clean diff           : {diff_between:+.6e} rad")
    print(f"Clean power              : {power_clean:.6e}")
    print(f"NaN power                : {power_nan:.6e}")

    # Print the ITC summary
    print(f"ITC raw                  : {itc_raw:.6f}")
    print(f"ITC drop                 : {itc_drop:.6f}")
    print(f"Avg phase raw            : {phi_raw:.6f} rad")
    print(f"Avg phase drop           : {phi_drop:.6f} rad")
    print(f"Phases raw               : {np.array2string(phases_raw, precision=3)}")
    print(f"Phases drop              : {np.array2string(phases_drop, precision=3)}")


if __name__ == "__main__":
    main()
