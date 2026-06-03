import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from rate_model_kernel import RateModelKernel


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "rate_model_ker"
SUMMARY_PATH = ARTIFACT_DIR / "summary.json"
TRACE_PATH = ARTIFACT_DIR / "traces.npz"
FULL_FINAL_PNG = ARTIFACT_DIR / "full_vs_final.png"
SYN_PNG = ARTIFACT_DIR / "dynamic_synapse.png"
ONEPOP_PNG = ARTIFACT_DIR / "run_1pop_consistency.png"


def _to_scalar(x) -> float:
    return float(np.asarray(x, dtype=float).reshape(-1)[0])


def _max_abs_diff(a, b) -> float:
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    if aa.size != bb.size:
        raise ValueError(f"Cannot compare arrays with different sizes: {aa.size} vs {bb.size}")
    return float(np.max(np.abs(aa - bb)))


def _build_demo_model(tau_syn=0.0) -> RateModelKernel:
    return RateModelKernel(
        np.array([
            [0.20, -0.12, 0.08],
            [0.10, 0.16, -0.05],
            [0.05, 0.10, 0.18],
        ]),
        tau=np.array([[1.5], [2.2], [0.0]]),
        tau_syn=tau_syn,
        rmax=np.array([[12.0], [10.0], [9.0]]),
        gain_slope=np.array([[0.8], [0.9], [1.1]]),
        gain_center=np.array([[0.1], [-0.1], [0.0]]),
    )


def _run_full_vs_final_case():
    model_full = _build_demo_model(tau_syn=0.0)
    model_final = _build_demo_model(tau_syn=0.0)
    model_final.sim_res_type = "final"

    dt = 0.25
    nsteps = 80
    tt = np.arange(nsteps) * dt
    h = np.vstack([
        0.25 + 0.30 * (tt >= 4.0) + 0.08 * np.sin(2.0 * np.pi * tt / 6.0),
        -0.10 + 0.20 * (tt >= 8.0) - 0.04 * np.cos(2.0 * np.pi * tt / 5.0),
        0.15 + 0.10 * np.sin(2.0 * np.pi * tt / 9.0 + 0.3),
    ])
    r0 = np.array([[0.2], [0.8], [0.1]])

    rr_full = model_full.run(h, r0, dt=dt, nsteps=nsteps)
    r_final = model_final.run(h, r0, dt=dt, nsteps=nsteps)

    rr_fb_full = model_full.run(
        h, r0, dt=dt, nsteps=nsteps, fb_proc=lambda r: -0.04 * r
    )
    r_fb_final = model_final.run(
        h, r0, dt=dt, nsteps=nsteps, fb_proc=lambda r: -0.04 * r
    )

    return {
        "dt": dt,
        "nsteps": nsteps,
        "time": tt,
        "h": h,
        "r0": r0,
        "rr_full": rr_full,
        "r_final": r_final,
        "rr_fb_full": rr_fb_full,
        "r_fb_final": r_fb_final,
        "max_diff_plain": _max_abs_diff(rr_full.isel(time=-1).values, r_final),
        "max_diff_feedback": _max_abs_diff(rr_fb_full.isel(time=-1).values, r_fb_final),
    }


def _run_dynamic_synapse_case():
    W = np.array([
        [0.0, 2.2, 0.0],
        [0.0, 0.0, 1.6],
        [0.0, 0.0, 0.0],
    ])
    tau = np.array([[1.8], [1.5], [0.0]])
    common = dict(
        tau=tau,
        rmax=np.array([[11.0], [10.0], [8.0]]),
        gain_slope=np.array([[0.9], [1.0], [1.2]]),
        gain_center=np.array([[0.0], [0.0], [0.0]]),
    )

    model_inst = RateModelKernel(W, tau_syn=0.0, **common)
    model_syn = RateModelKernel(
        W,
        tau_syn=np.array([
            [0.0, 5.5, 0.0],
            [0.0, 0.0, 3.5],
            [0.0, 0.0, 0.0],
        ]),
        **common,
    )
    model_syn_s0 = RateModelKernel(
        W,
        tau_syn=np.array([
            [0.0, 5.5, 0.0],
            [0.0, 0.0, 3.5],
            [0.0, 0.0, 0.0],
        ]),
        **common,
    )

    dt = 0.2
    nsteps = 120
    tt = np.arange(nsteps) * dt
    h = np.zeros((3, nsteps))
    h[2, tt >= 2.0] = 1.2
    h[1, tt >= 7.0] += 0.35
    r0 = np.zeros((3, 1))
    s0 = np.zeros((3, 3))
    s0[0, 1] = 1.3
    s0[1, 2] = 0.8

    rr_inst = model_inst.run(h, r0, dt=dt, nsteps=nsteps)
    rr_syn = model_syn.run(h, r0, dt=dt, nsteps=nsteps)
    rr_syn_s0 = model_syn_s0.run(h, r0, dt=dt, nsteps=nsteps, s0=s0)

    return {
        "dt": dt,
        "nsteps": nsteps,
        "time": tt,
        "h": h,
        "r0": r0,
        "rr_inst": rr_inst,
        "rr_syn": rr_syn,
        "rr_syn_s0": rr_syn_s0,
        "steady_state_diff": _max_abs_diff(
            rr_inst.isel(time=-1).values,
            rr_syn.isel(time=-1).values,
        ),
        "s0_transient_shift": float(
            rr_syn.isel(pop=0, time=6).item() - rr_syn_s0.isel(pop=0, time=6).item()
        ),
    }


def _run_onepop_case():
    model_full = RateModelKernel(
        np.array([
            [0.0, 1.5, -0.4],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]),
        tau=np.array([[2.0], [1.0], [0.0]]),
        tau_syn=np.array([
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]),
        rmax=np.array([[11.0], [9.0], [8.0]]),
        gain_slope=np.array([[0.9], [1.0], [1.1]]),
        gain_center=np.zeros((3, 1)),
    )
    model_final = RateModelKernel(
        model_full.W.copy(),
        tau=model_full.tau.copy(),
        tau_syn=model_full.tau_syn.copy(),
        rmax=model_full.rmax.copy(),
        gain_slope=model_full.gain_slope.copy(),
        gain_center=model_full.gain_center.copy(),
    )
    model_final.sim_res_type = "final"

    pop_num = 0
    dt = 0.25
    nsteps = 70
    h_pop = 0.55
    r0 = np.array([[0.2], [2.4], [1.3]])
    s0 = np.zeros((3, 3))
    s0[0, 1] = 1.2

    rr_1pop = model_full.run_1pop(pop_num, h_pop, r0, dt=dt, nsteps=nsteps)
    r_1pop_final = model_final.run_1pop(pop_num, h_pop, r0, dt=dt, nsteps=nsteps)
    rr_1pop_s0 = model_full.run_1pop(pop_num, h_pop, r0, dt=dt, nsteps=nsteps, s0=s0)

    h_equiv = np.array([
        [h_pop],
        [model_full.gain_inv(r0[1, 0], 1)],
        [model_full.gain_inv(r0[2, 0], 2)],
    ])
    rr_full = model_full.run(h_equiv, r0, dt=dt, nsteps=nsteps, s0=s0)

    return {
        "dt": dt,
        "nsteps": nsteps,
        "pop_num": pop_num,
        "h_pop": h_pop,
        "r0": r0,
        "s0": s0,
        "rr_1pop": rr_1pop,
        "r_1pop_final": r_1pop_final,
        "rr_1pop_s0": rr_1pop_s0,
        "rr_full": rr_full,
        "full_vs_1pop_max_diff": _max_abs_diff(
            rr_full.isel(pop=pop_num).values,
            rr_1pop_s0.isel(pop=0).values,
        ),
        "onepop_full_vs_final_diff": abs(
            rr_1pop.isel(time=-1).item() - r_1pop_final
        ),
    }


def _plot_full_vs_final(case):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    tt = case["time"]
    rr_full = case["rr_full"]
    rr_fb_full = case["rr_fb_full"]

    for pop in range(rr_full.sizes["pop"]):
        ax[0].plot(tt, rr_full.isel(pop=pop), label=f"pop {pop}")
        ax[0].axhline(case["r_final"][pop, 0], linestyle="--", color=f"C{pop}")
    ax[0].set_ylabel("rate")
    ax[0].set_title("`run` full trace and `run(..., sim_res_type='final')` agree at the last bin")
    ax[0].legend(loc="upper right")

    for pop in range(rr_fb_full.sizes["pop"]):
        ax[1].plot(tt, rr_fb_full.isel(pop=pop), label=f"pop {pop}")
        ax[1].axhline(case["r_fb_final"][pop, 0], linestyle="--", color=f"C{pop}")
    ax[1].set_ylabel("rate")
    ax[1].set_title("The same full/final consistency holds with one-pass feedback")
    ax[1].legend(loc="upper right")

    for pop in range(case["h"].shape[0]):
        ax[2].plot(tt, case["h"][pop], label=f"h[{pop}]")
    ax[2].set_xlabel("time")
    ax[2].set_ylabel("input")
    ax[2].set_title(
        f"max final diff plain={case['max_diff_plain']:.2e}, feedback={case['max_diff_feedback']:.2e}"
    )
    ax[2].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(FULL_FINAL_PNG, dpi=150)
    plt.close(fig)


def _plot_dynamic_synapse(case):
    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    tt = case["time"]

    for pop in range(3):
        ax[pop].plot(tt, case["rr_inst"].isel(pop=pop), label="instant")
        ax[pop].plot(tt, case["rr_syn"].isel(pop=pop), label="dynamic")
        ax[pop].plot(tt, case["rr_syn_s0"].isel(pop=pop), "--", label="dynamic + explicit s0")
        ax[pop].set_ylabel(f"pop {pop}")
        ax[pop].legend(loc="upper right")

    ax[0].set_title("Dynamic synapses slow transients but preserve the fixed point")

    for pop in range(3):
        ax[3].plot(tt, case["h"][pop], label=f"h[{pop}]")
    ax[3].set_xlabel("time")
    ax[3].set_ylabel("input")
    ax[3].set_title(
        f"steady-state diff={case['steady_state_diff']:.2e}, transient s0 shift={case['s0_transient_shift']:.3f}"
    )
    ax[3].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(SYN_PNG, dpi=150)
    plt.close(fig)


def _plot_onepop(case):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    tt = np.arange(case["nsteps"]) * case["dt"]

    ax[0].plot(tt, case["rr_1pop"].isel(pop=0), label="run_1pop full")
    ax[0].axhline(case["r_1pop_final"], color="C1", linestyle="--", label="run_1pop final")
    ax[0].set_ylabel("rate")
    ax[0].set_title("`run_1pop` full and final outputs match at the last bin")
    ax[0].legend(loc="upper right")

    ax[1].plot(tt, case["rr_1pop"].isel(pop=0), label="run_1pop default s0")
    ax[1].plot(tt, case["rr_1pop_s0"].isel(pop=0), "--", label="run_1pop explicit s0")
    ax[1].set_ylabel("rate")
    ax[1].set_title("Explicit synaptic state changes the row-level transient")
    ax[1].legend(loc="upper right")

    ax[2].plot(tt, case["rr_full"].isel(pop=case["pop_num"]), label="full run pop 0")
    ax[2].plot(tt, case["rr_1pop_s0"].isel(pop=0), "--", label="run_1pop pop 0")
    ax[2].set_xlabel("time")
    ax[2].set_ylabel("rate")
    ax[2].set_title(
        f"equivalent full-vs-1pop diff={case['full_vs_1pop_max_diff']:.2e}, "
        f"1pop full-vs-final diff={case['onepop_full_vs_final_diff']:.2e}"
    )
    ax[2].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(ONEPOP_PNG, dpi=150)
    plt.close(fig)


def _save_summary(full_final_case, syn_case, onepop_case):
    payload = {
        "artifact_dir": str(ARTIFACT_DIR),
        "figures": {
            "full_vs_final": str(FULL_FINAL_PNG),
            "dynamic_synapse": str(SYN_PNG),
            "run_1pop_consistency": str(ONEPOP_PNG),
        },
        "metrics": {
            "full_vs_final": {
                "max_diff_plain": full_final_case["max_diff_plain"],
                "max_diff_feedback": full_final_case["max_diff_feedback"],
            },
            "dynamic_synapse": {
                "steady_state_diff": syn_case["steady_state_diff"],
                "s0_transient_shift": syn_case["s0_transient_shift"],
            },
            "run_1pop": {
                "onepop_full_vs_final_diff": onepop_case["onepop_full_vs_final_diff"],
                "equivalent_full_vs_1pop_diff": onepop_case["full_vs_1pop_max_diff"],
            },
        },
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as fobj:
        json.dump(payload, fobj, indent=2, sort_keys=True)


def _save_traces(full_final_case, syn_case, onepop_case):
    np.savez_compressed(
        TRACE_PATH,
        time_full_final=full_final_case["time"],
        h_full_final=full_final_case["h"],
        rr_full=full_final_case["rr_full"].values,
        r_final=full_final_case["r_final"],
        rr_fb_full=full_final_case["rr_fb_full"].values,
        r_fb_final=full_final_case["r_fb_final"],
        time_syn=syn_case["time"],
        h_syn=syn_case["h"],
        rr_inst=syn_case["rr_inst"].values,
        rr_syn=syn_case["rr_syn"].values,
        rr_syn_s0=syn_case["rr_syn_s0"].values,
        time_onepop=np.arange(onepop_case["nsteps"]) * onepop_case["dt"],
        rr_1pop=onepop_case["rr_1pop"].values,
        rr_1pop_s0=onepop_case["rr_1pop_s0"].values,
        r_1pop_final=np.array([onepop_case["r_1pop_final"]]),
        rr_full_equiv=onepop_case["rr_full"].values,
    )


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    full_final_case = _run_full_vs_final_case()
    syn_case = _run_dynamic_synapse_case()
    onepop_case = _run_onepop_case()

    _plot_full_vs_final(full_final_case)
    _plot_dynamic_synapse(syn_case)
    _plot_onepop(onepop_case)
    _save_summary(full_final_case, syn_case, onepop_case)
    _save_traces(full_final_case, syn_case, onepop_case)

    print(f"Saved {FULL_FINAL_PNG}")
    print(f"Saved {SYN_PNG}")
    print(f"Saved {ONEPOP_PNG}")
    print(f"Saved {SUMMARY_PATH}")
    print(f"Saved {TRACE_PATH}")
    print(f"full/final max diff          : {full_final_case['max_diff_plain']:.6e}")
    print(f"full/final feedback max diff : {full_final_case['max_diff_feedback']:.6e}")
    print(f"dynamic steady-state diff    : {syn_case['steady_state_diff']:.6e}")
    print(f"dynamic explicit-s0 shift    : {syn_case['s0_transient_shift']:.6f}")
    print(f"run_1pop full/final diff     : {onepop_case['onepop_full_vs_final_diff']:.6e}")
    print(f"equivalent full-vs-1pop diff : {onepop_case['full_vs_1pop_max_diff']:.6e}")


if __name__ == "__main__":
    main()
