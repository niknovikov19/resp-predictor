from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "diagrams"


def add_box(ax, x, y, w, h, title, lines, facecolor, edgecolor="#1f2937", title_size=14, body_size=10):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.02 * w,
        y + h - 0.08 * h,
        title,
        fontsize=title_size,
        weight="bold",
        va="top",
        ha="left",
        color="#111827",
    )
    ax.text(
        x + 0.02 * w,
        y + h - 0.23 * h,
        "\n".join(lines),
        fontsize=body_size,
        va="top",
        ha="left",
        color="#1f2937",
        family="monospace",
    )


def add_arrow(ax, start, end, color="#475569", lw=2.0, connectionstyle="arc3,rad=0.0"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def style_canvas(ax, title, subtitle):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.975, title, fontsize=22, weight="bold", va="top", ha="left", color="#111827")
    ax.text(0.02, 0.94, subtitle, fontsize=11, va="top", ha="left", color="#475569")


def render_call_chain():
    fig, ax = plt.subplots(figsize=(16, 10))
    style_canvas(
        ax,
        "multisine_demo.py Call Chain",
        "The common path is: demo preset -> _run_cases -> generate_multisine -> system model -> analysis -> summary/plot",
    )

    add_box(
        ax,
        0.03,
        0.69,
        0.18,
        0.2,
        "Entry Points",
        [
            "run_demo()",
            "run_strong_harmonic_guard_demo()",
            "run_overlap_sensitive_harmonic_guard_demo()",
            "",
            "Each builds:",
            "- cases",
            "- system_kwargs",
            "- truth_func",
            "- plot/summary flavor",
        ],
        "#dbeafe",
    )
    add_box(
        ax,
        0.28,
        0.69,
        0.18,
        0.2,
        "_run_cases()",
        [
            "for each case:",
            "1. generate_multisine()",
            "2. system_fn(...)",
            "3. analyze_periodic_response()",
            "4. summary_fn(...)",
            "5. plot_fn(...)",
            "",
            "Finally:",
            "- summary_printer()",
            "- plt.show()",
        ],
        "#e0f2fe",
    )
    add_box(
        ax,
        0.53,
        0.69,
        0.18,
        0.2,
        "Analysis Core",
        [
            "analyze_periodic_response()",
            "",
            "Computes:",
            "- H on driven bins",
            "- coherence",
            "- line SNR",
            "- distortion metrics",
            "- driven/unexcited spectra",
        ],
        "#dcfce7",
    )
    add_box(
        ax,
        0.78,
        0.69,
        0.18,
        0.2,
        "Outputs",
        [
            "summarize_case()",
            "summarize_overlap_sensitive_case()",
            "print_*_table()",
            "plot_case()",
            "plot_overlap_sensitive_case()",
            "_print_overlap_examples()",
        ],
        "#fef3c7",
    )

    add_arrow(ax, (0.21, 0.79), (0.28, 0.79))
    add_arrow(ax, (0.46, 0.79), (0.53, 0.79))
    add_arrow(ax, (0.71, 0.79), (0.78, 0.79))

    add_box(
        ax,
        0.25,
        0.39,
        0.25,
        0.2,
        "Probe Construction",
        [
            "generate_multisine()",
            "",
            "returns:",
            "- t",
            "- x",
            "- info dict",
            "",
            "info carries freqs, bins, amps,",
            "phases, x_period, params",
        ],
        "#ede9fe",
    )
    add_box(
        ax,
        0.57,
        0.39,
        0.25,
        0.2,
        "System Models",
        [
            "simulate_demo_system()",
            "simulate_harmonic_guard_demo_system()",
            "",
            "Both consume:",
            "- x",
            "- fs",
            "- probe_info",
            "- transfer_func",
            "- noise/distortion kwargs",
        ],
        "#fee2e2",
    )

    add_arrow(ax, (0.37, 0.69), (0.37, 0.59))
    add_arrow(ax, (0.50, 0.49), (0.57, 0.49))
    add_arrow(ax, (0.695, 0.59), (0.695, 0.69))

    add_box(
        ax,
        0.04,
        0.15,
        0.2,
        0.17,
        "Bin Selection Helpers",
        [
            "_candidate_bins()",
            "_valid_bin()",
            "_nearest_valid_bin()",
            "_coverage_fill_bin()",
            "",
            "select_harmonic_bins()",
        ],
        "#eff6ff",
        body_size=9,
    )
    add_box(
        ax,
        0.28,
        0.15,
        0.2,
        0.17,
        "Waveform Helpers",
        [
            "crest_factor()",
            "_synthesize_period()",
            "choose_phases()",
            "",
            "used inside",
            "generate_multisine()",
        ],
        "#f5f3ff",
        body_size=9,
    )
    add_box(
        ax,
        0.52,
        0.15,
        0.2,
        0.17,
        "System Ingredients",
        [
            "transfer_demo()",
            "transfer_overlap_sensitive_demo()",
            "colored_noise()",
            "",
            "plugged into",
            "simulate_*()",
        ],
        "#fff1f2",
        body_size=9,
    )
    add_box(
        ax,
        0.76,
        0.15,
        0.2,
        0.17,
        "Analysis Helpers",
        [
            "harmonic_overlap_report()",
            "harmonic_overlap_target_bins()",
            "reconstruct_response_on_grid()",
            "_band_coverage()",
        ],
        "#fefce8",
        body_size=9,
    )

    add_arrow(ax, (0.24, 0.24), (0.30, 0.39), connectionstyle="arc3,rad=0.15")
    add_arrow(ax, (0.38, 0.32), (0.38, 0.39))
    add_arrow(ax, (0.62, 0.32), (0.69, 0.39), connectionstyle="arc3,rad=-0.12")
    add_arrow(ax, (0.82, 0.32), (0.66, 0.39), connectionstyle="arc3,rad=0.12")

    ax.text(
        0.03,
        0.05,
        "Key idea: the same info dict produced by generate_multisine() is reused later for overlap checks, plotting, and reconstruction.",
        fontsize=11,
        color="#334155",
        ha="left",
        va="bottom",
    )

    out = OUT_DIR / "multisine_demo_call_chain.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def render_parameters():
    fig, ax = plt.subplots(figsize=(16, 10))
    style_canvas(
        ax,
        "multisine_demo.py Main Parameters",
        "Grouped by where they matter most and which functions consume them",
    )

    add_box(
        ax,
        0.03,
        0.55,
        0.29,
        0.33,
        "1. Probe / Excitation Parameters",
        [
            "Consumed by generate_multisine()",
            "",
            "Timing:",
            "  fs, T, n_cycles",
            "Band coverage:",
            "  fmin, fmax, n_lines",
            "Line selection:",
            "  odd_only, min_gap_bins, harmonic_guard, dense",
            "Amplitude shaping:",
            "  gamma, f_ref, amp_ratio_cap, rms",
            "Phase search:",
            "  phase_mode, n_phase_trials, seed",
        ],
        "#dbeafe",
        body_size=10,
    )

    add_box(
        ax,
        0.36,
        0.55,
        0.29,
        0.33,
        "2. System / Distortion Parameters",
        [
            "Consumed by simulate_demo_system() or",
            "simulate_harmonic_guard_demo_system()",
            "",
            "Linear system:",
            "  transfer_func",
            "Background variability:",
            "  noise_beta, noise_std, drift_std",
            "Generic cubic demo:",
            "  cubic",
            "Harmonic-guard demos:",
            "  harmonic_order, harmonic_gain",
            "",
            "These change y, not x.",
        ],
        "#fee2e2",
        body_size=10,
    )

    add_box(
        ax,
        0.69,
        0.55,
        0.28,
        0.33,
        "3. Analysis / Visualization Parameters",
        [
            "Consumed after simulate_*()",
            "",
            "analyze_periodic_response():",
            "  band",
            "summarize_*() / plot_*():",
            "  truth_func",
            "reconstruct_response_on_grid():",
            "  band, n_points",
            "overlap-sensitive plots:",
            "  reconstruction_band",
            "",
            "These change interpretation, not data generation.",
        ],
        "#dcfce7",
        body_size=10,
    )

    add_box(
        ax,
        0.05,
        0.15,
        0.24,
        0.25,
        "run_demo()",
        [
            "Purpose:",
            "  broad overview of good/bad probe designs",
            "",
            "Uses:",
            "  transfer_demo",
            "  simulate_demo_system",
            "  plot_case",
            "  print_summary_table",
        ],
        "#ede9fe",
    )
    add_box(
        ax,
        0.38,
        0.15,
        0.24,
        0.25,
        "run_strong_harmonic_guard_demo()",
        [
            "Purpose:",
            "  isolate line corruption from explicit harmonics",
            "",
            "Uses:",
            "  transfer_demo",
            "  simulate_harmonic_guard_demo_system",
            "  plot_case",
            "  print_harmonic_guard_table",
        ],
        "#fef3c7",
    )
    add_box(
        ax,
        0.71,
        0.15,
        0.24,
        0.25,
        "run_overlap_sensitive_harmonic_guard_demo()",
        [
            "Purpose:",
            "  show when harmonic guard changes inferred H(f)",
            "",
            "Uses:",
            "  transfer_overlap_sensitive_demo",
            "  simulate_harmonic_guard_demo_system",
            "  plot_overlap_sensitive_case",
            "  print_overlap_sensitive_guard_table",
        ],
        "#fde68a",
    )

    add_arrow(ax, (0.17, 0.55), (0.17, 0.40))
    add_arrow(ax, (0.50, 0.55), (0.50, 0.40))
    add_arrow(ax, (0.83, 0.55), (0.83, 0.40))

    ax.text(
        0.03,
        0.05,
        "Rule of thumb: if you want different x(t), edit probe parameters; if you want different y(t), edit system parameters; if you want different diagnostics, edit analysis parameters.",
        fontsize=11,
        color="#334155",
        ha="left",
        va="bottom",
    )

    out = OUT_DIR / "multisine_demo_parameters.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [render_call_chain(), render_parameters()]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
