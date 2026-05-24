import matplotlib.pyplot as plt
import numpy as np


# Probe signal generation and line selection

def crest_factor(x):
    """Return the crest factor of a waveform. """
    x = np.asarray(x, float)
    return np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-15)


def _candidate_bins(T, fmin, fmax, odd_only):
    """Return allowed FFT bins inside the requested band. """
    nmin = int(np.ceil(fmin * T))
    nmax = int(np.floor(fmax * T))
    bins = np.arange(nmin, nmax + 1)
    if odd_only:
        bins = bins[bins % 2 == 1]
    return bins


def _valid_bin(c, chosen, min_gap_bins=1, harmonic_guard=1):
    """Check whether a candidate bin respects spacing and harmonic rules. """
    for b in chosen:
        if abs(c - b) <= min_gap_bins:
            return False
        if harmonic_guard > 0:
            for m in (2, 3):
                if abs(c - m * b) <= harmonic_guard:
                    return False
                if abs(m * c - b) <= harmonic_guard:
                    return False
    return True


def _nearest_valid_bin(target, allowed, chosen, min_gap_bins=1, harmonic_guard=1):
    """Pick the valid allowed bin closest to a target bin. """
    # Visit candidates in order of proximity to the target bin.
    order = np.lexsort((allowed, np.abs(allowed - target)))
    for c in allowed[order]:
        if _valid_bin(c, chosen, min_gap_bins=min_gap_bins, harmonic_guard=harmonic_guard):
            return int(c)
    return None


def _coverage_fill_bin(allowed, chosen, min_gap_bins=1, harmonic_guard=1):
    """Fill the widest remaining gap with a valid bin. """
    if not chosen:
        return None
    # Keep only candidates that still satisfy the spacing and harmonic rules.
    chosen_sorted = np.array(sorted(chosen), dtype=int)
    remaining = [int(c) for c in allowed if c not in chosen]
    candidates = [
        c for c in remaining
        if _valid_bin(c, chosen, min_gap_bins=min_gap_bins, harmonic_guard=harmonic_guard)
    ]
    if not candidates:
        return None
    # Score each candidate by how well it fills the widest gap in the current set.
    lo = int(allowed[0])
    hi = int(allowed[-1])
    scored = []
    for c in candidates:
        idx = np.searchsorted(chosen_sorted, c)
        left = lo if idx == 0 else int(chosen_sorted[idx - 1])
        right = hi if idx == len(chosen_sorted) else int(chosen_sorted[idx])
        midpoint = 0.5 * (left + right)
        gap_width = right - left
        scored.append((gap_width, -abs(c - midpoint), -c, c))
    scored.sort(reverse=True)
    return scored[0][-1]


def select_harmonic_bins(
    T,
    fmin=2.0,
    fmax=100.0,
    n_lines=40,
    odd_only=True,
    min_gap_bins=1,
    harmonic_guard=1,
    dense=False,
):
    """Select driven bins that span the requested band. """
    # Build the set of candidate bins in the requested frequency band.
    allowed = _candidate_bins(T, fmin, fmax, odd_only=odd_only)
    if len(allowed) == 0:
        raise ValueError(
            f"No allowed bins for T={T}, fmin={fmin}, fmax={fmax}, odd_only={odd_only}."
        )
    if dense:
        return allowed.copy()
    # Place logarithmically spaced targets first to get broad low-to-high coverage.
    target_bins = np.geomspace(allowed[0], allowed[-1], n_lines)
    chosen = []
    for target in target_bins:
        c = _nearest_valid_bin(
            target,
            allowed,
            chosen,
            min_gap_bins=min_gap_bins,
            harmonic_guard=harmonic_guard,
        )
        if c is not None and c not in chosen:
            chosen.append(c)
    # Fill any leftover slots by covering the widest remaining gaps.
    while len(chosen) < n_lines:
        c = _coverage_fill_bin(
            allowed,
            chosen,
            min_gap_bins=min_gap_bins,
            harmonic_guard=harmonic_guard,
        )
        if c is None or c in chosen:
            break
        chosen.append(c)
    chosen = np.array(sorted(set(chosen)), dtype=int)
    if len(chosen) < n_lines:
        raise ValueError(
            "Could not select the requested number of driven lines. "
            f"Requested n_lines={n_lines}, got {len(chosen)} with "
            f"T={T}, fmin={fmin}, fmax={fmax}, odd_only={odd_only}, "
            f"min_gap_bins={min_gap_bins}, harmonic_guard={harmonic_guard}."
        )
    return chosen


def _synthesize_period(freqs, amps, phases, fs, T):
    """Synthesize one multisine period from frequencies, amplitudes, and phases. """
    n_per = int(round(fs * T))
    t = np.arange(n_per) / fs
    x = np.sum(
        amps[:, None] * np.cos(2 * np.pi * freqs[:, None] * t[None, :] + phases[:, None]),
        axis=0,
    )
    return x


def choose_phases(freqs, amps, fs, T, n_trials=100, seed=0, mode="best-random"):
    """Choose probe phases with optional crest-factor search. """
    rng = np.random.default_rng(seed)
    # Zero phase is mainly here as a deliberately bad baseline.
    if mode == "zero":
        return np.zeros(len(freqs))
    # Search random phase assignments and keep the one with the lowest crest factor.
    best_phi = None
    best_cf = np.inf
    for _ in range(max(1, n_trials)):
        phi = rng.uniform(0, 2 * np.pi, size=len(freqs))
        x = _synthesize_period(freqs, amps, phi, fs, T)
        cf = crest_factor(x)
        if cf < best_cf:
            best_cf = cf
            best_phi = phi
    return best_phi


def generate_multisine(
    fs=500.0,
    T=10.0,
    n_cycles=20,
    fmin=2.0,
    fmax=100.0,
    n_lines=40,
    gamma=0.5,
    f_ref=10.0,
    amp_ratio_cap=8.0,
    odd_only=True,
    min_gap_bins=1,
    harmonic_guard=1,
    dense=False,
    phase_mode="best-random",
    n_phase_trials=200,
    rms=1.0,
    seed=0,
):
    """Generate a repeated multisine probe and its metadata. """
    # Select the driven FFT bins that define the probe support.
    bins = select_harmonic_bins(
        T=T,
        fmin=fmin,
        fmax=fmax,
        n_lines=n_lines,
        odd_only=odd_only,
        min_gap_bins=min_gap_bins,
        harmonic_guard=harmonic_guard,
        dense=dense,
    )
    freqs = bins / T
    # Shape amplitudes across frequency and choose a phase assignment.
    amps = (freqs / f_ref) ** gamma
    amps /= amps.min()
    amps = np.minimum(amps, amp_ratio_cap)
    phases = choose_phases(
        freqs=freqs,
        amps=amps,
        fs=fs,
        T=T,
        n_trials=n_phase_trials,
        seed=seed,
        mode=phase_mode,
    )
    # Normalize one period to the requested RMS and tile it across cycles.
    x_per = _synthesize_period(freqs, amps, phases, fs, T)
    x_per -= x_per.mean()
    x_per *= rms / (np.sqrt(np.mean(x_per**2)) + 1e-15)
    x = np.tile(x_per, n_cycles)
    t = np.arange(len(x)) / fs
    # Package metadata reused later by the simulators, summaries, and plots.
    info = {
        "fs": fs,
        "T": T,
        "n_cycles": n_cycles,
        "freqs": freqs,
        "bins": bins,
        "amps": amps,
        "phases": phases,
        "x_period": x_per,
        "crest_factor": crest_factor(x_per),
        "params": {
            "fmin": fmin,
            "fmax": fmax,
            "n_lines": n_lines,
            "gamma": gamma,
            "f_ref": f_ref,
            "amp_ratio_cap": amp_ratio_cap,
            "odd_only": odd_only,
            "min_gap_bins": min_gap_bins,
            "harmonic_guard": harmonic_guard,
            "dense": dense,
            "phase_mode": phase_mode,
            "n_phase_trials": n_phase_trials,
            "rms": rms,
        },
    }
    return t, x, info


# Simulation ingredients and demo system models

def transfer_demo(freqs):
    """Return the default smooth demo transfer function. """
    freqs = np.asarray(freqs)
    # Combine a low-pass envelope with two broad resonant peaks.
    lowpass = 1.0 / np.sqrt(1.0 + (freqs / 45.0) ** 2)
    peak1 = 1.0 + 2.0 * np.exp(-0.5 * ((freqs - 6.0) / 1.3) ** 2)
    peak2 = 1.0 + 0.5 * np.exp(-0.5 * ((freqs - 18.0) / 3.0) ** 2)
    mag = 0.8 * lowpass * peak1 * peak2
    phase = -2 * np.pi * freqs * 0.012 - 0.35 * np.arctan(freqs / 20.0)
    return mag * np.exp(1j * phase)


def transfer_overlap_sensitive_demo(freqs):
    """Return a sharper transfer function for overlap-sensitive demos. """
    freqs = np.asarray(freqs)
    # Build a response with narrow features so a few bad lines matter more.
    base = 0.18 + 0.9 / (1.0 + (freqs / 14.0) ** 1.4)
    peak1 = 1.7 * np.exp(-0.5 * ((freqs - 4.6) / 0.22) ** 2)
    peak2 = 1.1 * np.exp(-0.5 * ((freqs - 12.2) / 0.45) ** 2)
    notch = 0.35 * np.exp(-0.5 * ((freqs - 7.9) / 0.18) ** 2)
    mag = np.clip(base + peak1 + peak2 - notch, 0.03, None)
    phase = -2 * np.pi * freqs * 0.01 - 0.15 * np.arctan(freqs / 8.0)
    return mag * np.exp(1j * phase)


def colored_noise(n, fs, beta=1.0, seed=0):
    """Generate unit-variance colored noise with spectral slope beta. """
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    z = rng.normal(size=len(freqs)) + 1j * rng.normal(size=len(freqs))
    z[0] = 0.0
    if n % 2 == 0:
        z[-1] = z[-1].real + 0j
    # Shape the spectrum in the frequency domain and transform back to time.
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.maximum(freqs[1:], 1e-6) ** (beta / 2.0)
    x = np.fft.irfft(z * scale, n=n)
    x -= x.mean()
    x /= np.std(x) + 1e-15
    return x


def simulate_user_system(x, fs):
    """Placeholder for a user-supplied system simulator. """
    raise NotImplementedError("Replace this with your simulator F(x) -> y.")


def simulate_demo_system(
    x,
    fs,
    probe_info=None,
    seed=1,
    transfer_func=transfer_demo,
    noise_beta=1.0,
    noise_std=0.8,
    drift_std=0.15,
    cubic=0.08,
):
    """Simulate the default nonlinear demo system. """
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    # Apply the linear transfer function in the frequency domain.
    H = transfer_func(freqs)
    y_lin = np.fft.irfft(np.fft.rfft(x) * H, n=n)
    # Add slow multiplicative drift to mimic trial-to-trial gain changes.
    slow = colored_noise(n, fs, beta=3.0, seed=seed + 10)
    slow = slow / (np.max(np.abs(slow)) + 1e-15)
    gain = 1.0 + drift_std * slow
    y = gain * y_lin
    # Add a broad cubic nonlinearity that creates intermodulation products.
    y = y + cubic * (y**3 - np.mean(y**3))
    # Add colored background noise on top of the periodic response.
    bg = noise_std * colored_noise(n, fs, beta=noise_beta, seed=seed + 20)
    return y + bg


def simulate_harmonic_guard_demo_system(
    x,
    fs,
    probe_info,
    seed=1,
    transfer_func=transfer_demo,
    harmonic_order=3,
    harmonic_gain=0.35,
    noise_beta=1.0,
    noise_std=0.03,
    drift_std=0.0,
):
    """Simulate a system with explicit harmonic contamination lines. """
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    # Apply the linear transfer function first to get the baseline response.
    H = transfer_func(freqs)
    y_lin = np.fft.irfft(np.fft.rfft(x) * H, n=n)
    # Synthesize harmonics directly from the driven probe lines.
    driven_freqs = np.asarray(probe_info["freqs"])
    driven_amps = np.asarray(probe_info["amps"])
    driven_phases = np.asarray(probe_info["phases"])
    keep = harmonic_order * driven_freqs <= 0.5 * fs
    harm_freqs = harmonic_order * driven_freqs[keep]
    harm_amps = harmonic_gain * driven_amps[keep] ** harmonic_order
    if len(harm_amps) > 0:
        harm_amps /= np.max(harm_amps)
        harm_amps *= harmonic_gain * np.max(np.abs(y_lin))
        harm_phases = harmonic_order * driven_phases[keep]
        y_harm = np.tile(
            _synthesize_period(harm_freqs, harm_amps, harm_phases, fs, probe_info["T"]),
            probe_info["n_cycles"],
        )
    else:
        y_harm = np.zeros_like(y_lin)
    # Optionally apply slow gain drift to the linear part of the response.
    if drift_std > 0:
        slow = colored_noise(n, fs, beta=3.0, seed=seed + 10)
        slow = slow / (np.max(np.abs(slow)) + 1e-15)
        y_lin = (1.0 + drift_std * slow) * y_lin
    # Add background colored noise on top of the linear and harmonic terms.
    bg = noise_std * colored_noise(n, fs, beta=noise_beta, seed=seed + 20)
    return y_lin + y_harm + bg


# Response analysis and overlap diagnostics

def analyze_periodic_response(x, y, fs, T, driven_freqs, band=(2.0, 100.0)):
    """Estimate periodic transfer and distortion metrics from repeated cycles. """
    n_per = int(round(fs * T))
    n_cycles = min(len(x), len(y)) // n_per
    # Keep only full periods so cycle-averaged FFTs stay perfectly aligned.
    x = np.asarray(x[: n_cycles * n_per])
    y = np.asarray(y[: n_cycles * n_per])
    Xc = np.fft.rfft(x.reshape(n_cycles, n_per), axis=1)
    Yc = np.fft.rfft(y.reshape(n_cycles, n_per), axis=1)
    freqs = np.fft.rfftfreq(n_per, d=1 / fs)
    # Split the analysis band into driven and unexcited bins.
    driven_bins = np.unique(np.round(np.asarray(driven_freqs) * T).astype(int))
    band_bins = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    det_bins = np.setdiff1d(band_bins, driven_bins)
    # Average the cycle FFTs to estimate the locked periodic response.
    Xbar = Xc.mean(axis=0)
    Ybar = Yc.mean(axis=0)
    H = Ybar[driven_bins] / (Xbar[driven_bins] + 1e-15)
    Sxx = np.mean(np.abs(Xc) ** 2, axis=0)
    Syy = np.mean(np.abs(Yc) ** 2, axis=0)
    Sxy = np.mean(np.conj(Xc) * Yc, axis=0)
    coherence = np.abs(Sxy) ** 2 / (Sxx * Syy + 1e-30)
    driven_lock = np.abs(Ybar[driven_bins])
    det_lock = np.abs(Ybar[det_bins])
    even_det_bins = det_bins[det_bins % 2 == 0]
    odd_det_bins = det_bins[det_bins % 2 == 1]
    # Summarize how much periodic energy leaks onto unexcited lines.
    has_det_bins = len(det_bins) > 0
    if not has_det_bins:
        distortion_ratio = np.nan
        even_distortion = np.nan
        odd_unexcited_distortion = np.nan
    else:
        distortion_ratio = np.sqrt(
            np.sum(np.abs(Ybar[det_bins]) ** 2) / (np.sum(np.abs(Ybar[driven_bins]) ** 2) + 1e-30)
        )
        even_distortion = np.sqrt(
            np.sum(np.abs(Ybar[even_det_bins]) ** 2) / (np.sum(np.abs(Ybar[driven_bins]) ** 2) + 1e-30)
        )
        odd_unexcited_distortion = np.sqrt(
            np.sum(np.abs(Ybar[odd_det_bins]) ** 2) / (np.sum(np.abs(Ybar[driven_bins]) ** 2) + 1e-30)
        )
    # Estimate how strongly each driven line rises above cycle-to-cycle variation.
    locked_fraction = np.abs(Ybar) ** 2 / (np.mean(np.abs(Yc) ** 2, axis=0) + 1e-30)
    cycle_noise = np.std(Yc - Ybar[None, :], axis=0)
    line_snr = np.abs(Ybar[driven_bins]) / (cycle_noise[driven_bins] + 1e-15)
    return {
        "freqs": freqs,
        "driven_freqs": freqs[driven_bins],
        "driven_bins": driven_bins,
        "det_bins": det_bins,
        "Xbar": Xbar,
        "Ybar": Ybar,
        "H": H,
        "coherence": coherence,
        "line_snr": line_snr,
        "locked_fraction": locked_fraction,
        "distortion_ratio": distortion_ratio,
        "even_distortion": even_distortion,
        "odd_unexcited_distortion": odd_unexcited_distortion,
        "has_detection_bins": has_det_bins,
        "driven_lock": driven_lock,
        "det_lock": det_lock,
        "n_cycles": n_cycles,
    }


def harmonic_overlap_report(bins, T, harmonics=(2, 3)):
    """Report driven bins that land on selected harmonic multiples. """
    bins = np.unique(np.round(np.asarray(bins)).astype(int))
    bin_set = set(int(b) for b in bins)
    hits = []
    # Enumerate driven-to-driven harmonic overlaps in bin units.
    for base in bins:
        for harmonic in harmonics:
            target = int(harmonic * base)
            if target in bin_set:
                hits.append(
                    {
                        "base_bin": int(base),
                        "target_bin": target,
                        "harmonic": harmonic,
                        "base_freq": float(base / T),
                        "target_freq": float(target / T),
                    }
                )
    return hits


def harmonic_overlap_target_bins(bins, T, harmonics=(2, 3)):
    """Return the unique target bins that participate in harmonic overlaps. """
    hits = harmonic_overlap_report(bins, T, harmonics=harmonics)
    return np.array(sorted({hit["target_bin"] for hit in hits}), dtype=int)


def _band_coverage(probe_info, analysis):
    """Summarize the analyzed band coverage of the driven lines. """
    f = analysis["driven_freqs"]
    if len(f) == 0:
        return np.nan, np.nan, False
    # Compare the actual driven span with the allowed probe band.
    params = probe_info["params"]
    allowed = _candidate_bins(
        probe_info["T"],
        params["fmin"],
        params["fmax"],
        odd_only=params["odd_only"],
    )
    fmin = allowed[0] / probe_info["T"]
    fmax = allowed[-1] / probe_info["T"]
    tol = 1e-12
    return float(f.min()), float(f.max()), bool(f.min() <= fmin + tol and f.max() >= fmax - tol)


def reconstruct_response_on_grid(analysis, band=None, n_points=2000):
    """Interpolate the estimated transfer function onto a dense frequency grid. """
    driven_freqs = np.asarray(analysis["driven_freqs"])
    H = np.asarray(analysis["H"])
    if band is None:
        fmin = float(driven_freqs.min())
        fmax = float(driven_freqs.max())
    else:
        fmin, fmax = band
    # Interpolate real and imaginary parts separately to get a dense response curve.
    fg = np.linspace(fmin, fmax, n_points)
    Hr = np.interp(fg, driven_freqs, H.real)
    Hi = np.interp(fg, driven_freqs, H.imag)
    return fg, Hr + 1j * Hi


# Summaries and table formatting

def summarize_case(name, probe_info, analysis, truth_func=None):
    """Summarize one case with coverage, noise, and distortion metrics. """
    f = analysis["driven_freqs"]
    low = (f >= 4) & (f <= 8)
    high = (f >= 40) & (f <= 80)
    min_freq, max_freq, full_band_covered = _band_coverage(probe_info, analysis)
    harmonic_hits = harmonic_overlap_report(probe_info["bins"], probe_info["T"])
    # Collect the main probe and response quality metrics into one row.
    msg = {
        "name": name,
        "crest": probe_info["crest_factor"],
        "n_lines": len(probe_info["freqs"]),
        "harmonic_hits": len(harmonic_hits),
        "min_freq": min_freq,
        "max_freq": max_freq,
        "full_band": full_band_covered,
        "median_coh": float(np.median(analysis["coherence"][analysis["driven_bins"]])),
        "median_snr_4_8Hz": float(np.median(analysis["line_snr"][low])) if np.any(low) else np.nan,
        "median_snr_40_80Hz": float(np.median(analysis["line_snr"][high])) if np.any(high) else np.nan,
        "distortion": float(analysis["distortion_ratio"]),
        "even_dist": float(analysis["even_distortion"]),
        "odd_unexcited_dist": float(analysis["odd_unexcited_distortion"]),
    }
    # Compare estimated transfer values against the truth when a reference is available.
    if truth_func is not None:
        Htrue = truth_func(f)
        rel_err = np.abs(analysis["H"] - Htrue) / np.maximum(np.abs(Htrue), 1e-12)
        msg["median_rel_err"] = float(np.median(rel_err))
        msg["max_rel_err"] = float(np.max(rel_err))
        overlap_bins = harmonic_overlap_target_bins(probe_info["bins"], probe_info["T"])
        overlap_mask = np.isin(analysis["driven_bins"], overlap_bins)
        if np.any(overlap_mask):
            msg["overlap_rel_err"] = float(np.median(rel_err[overlap_mask]))
            msg["max_overlap_rel_err"] = float(np.max(rel_err[overlap_mask]))
    return msg


def summarize_overlap_sensitive_case(
    name,
    probe_info,
    analysis,
    truth_func=None,
    reconstruction_band=(2.0, 40.0),
):
    """Summarize overlap-sensitive cases using reconstructed H(f). """
    msg = summarize_case(name, probe_info, analysis, truth_func=truth_func)
    if truth_func is None:
        return msg
    # Compare a dense reconstruction of H(f) so narrow features matter explicitly.
    fg, Hrec = reconstruct_response_on_grid(analysis, band=reconstruction_band)
    Htrue = truth_func(fg)
    rel_rec = np.abs(Hrec - Htrue) / np.maximum(np.abs(Htrue), 1e-12)
    msg["dense_median_rel_err"] = float(np.median(rel_rec))
    msg["dense_max_rel_err"] = float(np.max(rel_rec))
    est_peak_idx = int(np.argmax(np.abs(Hrec)))
    true_peak_idx = int(np.argmax(np.abs(Htrue)))
    msg["peak_freq_err"] = float(abs(fg[est_peak_idx] - fg[true_peak_idx]))
    msg["peak_mag_rel_err"] = float(
        abs(np.abs(Hrec[est_peak_idx]) - np.abs(Htrue[true_peak_idx]))
        / np.maximum(np.abs(Htrue[true_peak_idx]), 1e-12)
    )
    return msg


def _format_table_value(v):
    """Format a scalar table value for aligned console output. """
    if isinstance(v, str):
        return f"{v:>20s}"
    if isinstance(v, (bool, np.bool_)):
        return f"{'yes' if v else 'no':>20s}"
    if v is None:
        return f"{'n/a':>20s}"
    try:
        if np.isnan(v):
            return f"{'n/a':>20s}"
    except TypeError:
        pass
    return f"{v:20.4g}"


def print_summary_table(rows):
    """Print the default summary table for multiple cases. """
    headers = [
        "name",
        "crest",
        "n_lines",
        "harmonic_hits",
        "min_freq",
        "max_freq",
        "full_band",
        "median_coh",
        "median_snr_4_8Hz",
        "median_snr_40_80Hz",
        "distortion",
        "even_dist",
        "odd_unexcited_dist",
        "median_rel_err",
    ]
    # Print a fixed-width table so demo outputs stay easy to compare by eye.
    print(" | ".join(f"{h:>20s}" for h in headers))
    print("-" * (24 * len(headers)))
    for row in rows:
        vals = []
        for h in headers:
            v = row.get(h, np.nan)
            vals.append(_format_table_value(v))
        print(" | ".join(vals))


def print_harmonic_guard_table(rows):
    """Print the focused summary table for harmonic-guard demos. """
    headers = [
        "name",
        "n_lines",
        "harmonic_hits",
        "median_coh",
        "distortion",
        "odd_unexcited_dist",
        "overlap_rel_err",
        "max_overlap_rel_err",
        "median_rel_err",
    ]
    # Print only the fields that are most relevant to harmonic contamination.
    print(" | ".join(f"{h:>20s}" for h in headers))
    print("-" * (24 * len(headers)))
    for row in rows:
        vals = []
        for h in headers:
            vals.append(_format_table_value(row.get(h, np.nan)))
        print(" | ".join(vals))


def print_overlap_sensitive_guard_table(rows):
    """Print the summary table for overlap-sensitive H(f) demos. """
    headers = [
        "name",
        "n_lines",
        "harmonic_hits",
        "overlap_rel_err",
        "max_overlap_rel_err",
        "dense_median_rel_err",
        "dense_max_rel_err",
        "peak_freq_err",
        "peak_mag_rel_err",
    ]
    # Print the dense-reconstruction errors that matter for inferred transfer shape.
    print(" | ".join(f"{h:>20s}" for h in headers))
    print("-" * (24 * len(headers)))
    for row in rows:
        vals = []
        for h in headers:
            vals.append(_format_table_value(row.get(h, np.nan)))
        print(" | ".join(vals))


# Plotting helpers

def plot_case(axs, name, probe_info, analysis, truth_func=None):
    """Plot the standard six-panel overview for one case. """
    fs = probe_info["fs"]
    n_per = len(probe_info["x_period"])
    tp = np.arange(n_per) / fs
    freqs = analysis["freqs"]
    driven = analysis["driven_bins"]
    det = analysis["det_bins"]
    overlap_bins = harmonic_overlap_target_bins(probe_info["bins"], probe_info["T"])
    overlap_mask = np.isin(driven, overlap_bins)
    show_overlap_markers = np.any(overlap_mask) and len(overlap_bins) <= 20
    # Plot one probe period in the time domain.
    axs[0].plot(tp, probe_info["x_period"], lw=1.2)
    axs[0].set_title(f"{name}: one period of x(t)")
    axs[0].set_xlabel("Time (s)")
    # Plot the selected probe line amplitudes across frequency.
    axs[1].stem(probe_info["freqs"], probe_info["amps"], basefmt=" ", markerfmt=" ")
    axs[1].set_title(f"{name}: input line amplitudes")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_xlim(0, 105)
    # Plot the sampled transfer magnitude and highlight harmonic-overlap bins.
    axs[2].plot(freqs[driven], np.abs(analysis["H"]), "o-", ms=3, label="estimated")
    if truth_func is not None:
        fg = np.linspace(2, 100, 1000)
        axs[2].plot(fg, np.abs(truth_func(fg)), "-", lw=1.5, label="truth")
    if show_overlap_markers:
        axs[2].plot(
            freqs[driven][overlap_mask],
            np.abs(analysis["H"])[overlap_mask],
            "o",
            ms=8,
            mfc="none",
            mec="crimson",
            mew=1.5,
            label="harmonic-overlap lines",
        )
    axs[2].set_title(f"{name}: |H(f)|")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_xlim(0, 105)
    axs[2].legend()
    # Plot either transfer error against truth or coherence when no truth exists.
    if truth_func is not None:
        rel_err = np.abs(analysis["H"] - truth_func(freqs[driven])) / np.maximum(
            np.abs(truth_func(freqs[driven])),
            1e-12,
        )
        axs[3].semilogy(freqs[driven], rel_err, "o-", ms=3, label="relative error")
        if show_overlap_markers:
            axs[3].semilogy(
                freqs[driven][overlap_mask],
                rel_err[overlap_mask],
                "o",
                ms=8,
                mfc="none",
                mec="crimson",
                mew=1.5,
                label="harmonic-overlap lines",
            )
        axs[3].set_title(f"{name}: transfer relative error")
        axs[3].set_ylabel("|H_est - H_true| / |H_true|")
        axs[3].legend()
    else:
        axs[3].plot(freqs[driven], analysis["coherence"][driven], "o-", ms=3)
        axs[3].set_title(f"{name}: coherence at driven lines")
        axs[3].set_ylim(0, 1.05)
    axs[3].set_xlabel("Frequency (Hz)")
    axs[3].set_xlim(0, 105)
    # Plot periodic output energy on driven and unexcited lines.
    axs[4].semilogy(freqs[driven], np.abs(analysis["Ybar"][driven]), "o", ms=3, label="driven lines")
    axs[4].semilogy(freqs[det], np.abs(analysis["Ybar"][det]), ".", ms=2, label="unexcited lines")
    if show_overlap_markers:
        axs[4].semilogy(
            freqs[driven][overlap_mask],
            np.abs(analysis["Ybar"][driven])[overlap_mask],
            "o",
            ms=8,
            mfc="none",
            mec="crimson",
            mew=1.5,
            label="harmonic-overlap lines",
        )
    if analysis["has_detection_bins"]:
        distortion_label = (
            f"dist={analysis['distortion_ratio']:.3g}, "
            f"even={analysis['even_distortion']:.3g}, "
            f"odd-unexc={analysis['odd_unexcited_distortion']:.3g}"
        )
    else:
        distortion_label = "dist=n/a, even=n/a, odd-unexc=n/a"
    axs[4].set_title(f"{name}: periodic output spectrum\n{distortion_label}")
    axs[4].set_xlabel("Frequency (Hz)")
    axs[4].set_xlim(0, 105)
    axs[4].legend()
    # Plot line-wise SNR across the driven frequencies.
    axs[5].plot(freqs[driven], analysis["line_snr"], "o-", ms=3)
    axs[5].set_title(f"{name}: line SNR")
    axs[5].set_xlabel("Frequency (Hz)")
    axs[5].set_xlim(0, 105)


def plot_overlap_sensitive_case(
    axs,
    name,
    probe_info,
    analysis,
    truth_func=None,
    reconstruction_band=(2.0, 40.0),
):
    """Plot the overlap-sensitive H(f) comparison panels. """
    freqs = analysis["freqs"]
    driven = analysis["driven_bins"]
    overlap_bins = harmonic_overlap_target_bins(probe_info["bins"], probe_info["T"])
    overlap_mask = np.isin(driven, overlap_bins)
    show_overlap_markers = np.any(overlap_mask) and len(overlap_bins) <= 30
    # Plot the probe line amplitudes and mark bins that can be harmonically contaminated.
    axs[0].stem(probe_info["freqs"], probe_info["amps"], basefmt=" ", markerfmt=" ")
    if show_overlap_markers:
        axs[0].plot(
            freqs[driven][overlap_mask],
            probe_info["amps"][overlap_mask],
            "o",
            ms=7,
            mfc="none",
            mec="crimson",
            mew=1.5,
            label="harmonic-overlap lines",
        )
        axs[0].legend()
    axs[0].set_title(f"{name}: input line amplitudes")
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_xlim(reconstruction_band[0], reconstruction_band[1])
    # Plot the sampled transfer magnitude on the driven lines only.
    axs[1].plot(freqs[driven], np.abs(analysis["H"]), "o-", ms=3, label="estimated")
    if truth_func is not None:
        fg = np.linspace(reconstruction_band[0], reconstruction_band[1], 2000)
        axs[1].plot(fg, np.abs(truth_func(fg)), "-", lw=1.5, label="truth")
    if show_overlap_markers:
        axs[1].plot(
            freqs[driven][overlap_mask],
            np.abs(analysis["H"])[overlap_mask],
            "o",
            ms=8,
            mfc="none",
            mec="crimson",
            mew=1.5,
            label="harmonic-overlap lines",
        )
    axs[1].set_title(f"{name}: sampled |H(f)|")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_xlim(reconstruction_band[0], reconstruction_band[1])
    axs[1].legend()
    # Plot transfer error on the sampled lines to expose overlap-driven failures.
    if truth_func is not None:
        Htrue_driven = truth_func(freqs[driven])
        rel_err = np.abs(analysis["H"] - Htrue_driven) / np.maximum(np.abs(Htrue_driven), 1e-12)
        axs[2].semilogy(freqs[driven], rel_err, "o-", ms=3, label="sampled relative error")
        if show_overlap_markers:
            axs[2].semilogy(
                freqs[driven][overlap_mask],
                rel_err[overlap_mask],
                "o",
                ms=8,
                mfc="none",
                mec="crimson",
                mew=1.5,
                label="harmonic-overlap lines",
            )
        axs[2].set_title(f"{name}: sampled relative error")
        axs[2].set_ylabel("|H_est - H_true| / |H_true|")
        axs[2].legend()
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_xlim(reconstruction_band[0], reconstruction_band[1])
    # Plot the dense reconstruction to show how bad lines distort the inferred curve.
    if truth_func is not None:
        fg, Hrec = reconstruct_response_on_grid(analysis, band=reconstruction_band)
        Htrue = truth_func(fg)
        axs[3].plot(fg, np.abs(Hrec), "-", lw=1.5, label="reconstructed from driven lines")
        axs[3].plot(fg, np.abs(Htrue), "--", lw=1.5, label="truth")
        axs[3].set_title(f"{name}: reconstructed |H(f)|")
        axs[3].legend()
    axs[3].set_xlabel("Frequency (Hz)")
    axs[3].set_xlim(reconstruction_band[0], reconstruction_band[1])
    # Plot dense reconstruction error to show where the inferred transfer curve goes wrong.
    if truth_func is not None:
        rel_rec = np.abs(Hrec - Htrue) / np.maximum(np.abs(Htrue), 1e-12)
        axs[4].semilogy(fg, rel_rec, "-", lw=1.5, label="reconstruction relative error")
        axs[4].set_title(f"{name}: reconstruction relative error")
        axs[4].set_ylabel("|H_rec - H_true| / |H_true|")
        axs[4].legend()
    axs[4].set_xlabel("Frequency (Hz)")
    axs[4].set_xlim(reconstruction_band[0], reconstruction_band[1])


# Demo orchestration and runnable presets

def _run_cases(
    cases,
    system_fn=simulate_demo_system,
    system_kwargs=None,
    truth_func=transfer_demo,
    summary_fn=summarize_case,
    summary_printer=print_summary_table,
    fig_title=None,
    plot_fn=plot_case,
    n_cols=6,
):
    """Run a set of cases through generation, simulation, analysis, and plotting. """
    system_kwargs = {} if system_kwargs is None else dict(system_kwargs)
    results = []
    # Create one row of panels per case using the requested plotting function.
    fig, axs = plt.subplots(
        len(cases),
        n_cols,
        figsize=(4.4 * n_cols, 4.8 * len(cases)),
        constrained_layout=True,
    )
    if len(cases) == 1:
        axs = axs[None, :]
    # Run the full pipeline for each named preset in the case dictionary.
    for row, (name, cfg) in enumerate(cases.items()):
        _, x, info = generate_multisine(**cfg)
        case_system_kwargs = {"seed": 10 + row}
        case_system_kwargs.update(system_kwargs)
        y = system_fn(x, fs=cfg["fs"], probe_info=info, **case_system_kwargs)
        analysis = analyze_periodic_response(x, y, fs=cfg["fs"], T=cfg["T"], driven_freqs=info["freqs"])
        summary = summary_fn(name, info, analysis, truth_func=truth_func)
        results.append(
            {
                "name": name,
                "summary": summary,
                "probe_info": info,
                "analysis": analysis,
            }
        )
        plot_fn(axs[row], name, info, analysis, truth_func=truth_func)
    # Finish the figure and print the tabular summary.
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=16)
    summary_printer([item["summary"] for item in results])
    plt.show()
    return results


def _print_overlap_examples(results, max_examples=4):
    """Print short examples of harmonic line overlaps for each case. """
    for item in results:
        hits = harmonic_overlap_report(item["probe_info"]["bins"], item["probe_info"]["T"])
        if not hits:
            print(f"{item['name']}: no 2x/3x overlaps among driven lines.")
            continue
        # Show a few example overlaps so the summaries stay readable.
        examples = ", ".join(
            f"{hit['harmonic']}x{hit['base_freq']:.1f}->{hit['target_freq']:.1f} Hz"
            for hit in hits[:max_examples]
        )
        print(f"{item['name']}: {len(hits)} overlaps ({examples})")


def run_demo():
    """Run the original overview demo with several probe design variants. """
    fs = 500.0
    T = 10.0
    n_cycles = 20
    # Compare a well-designed probe against a few intentionally bad design choices.
    cases = {
        "good": dict(
            fs=fs,
            T=T,
            n_cycles=n_cycles,
            fmin=2,
            fmax=100,
            n_lines=42,
            gamma=0.55,
            odd_only=True,
            min_gap_bins=1,
            harmonic_guard=1,
            dense=False,
            phase_mode="best-random",
            n_phase_trials=250,
            rms=1.0,
            seed=1,
        ),
        "bad_zero_phase": dict(
            fs=fs,
            T=T,
            n_cycles=n_cycles,
            fmin=2,
            fmax=100,
            n_lines=42,
            gamma=0.0,
            odd_only=False,
            min_gap_bins=0,
            harmonic_guard=0,
            dense=False,
            phase_mode="zero",
            n_phase_trials=1,
            rms=1.0,
            seed=2,
        ),
        "bad_no_harmonic_guard": dict(
            fs=fs,
            T=T,
            n_cycles=n_cycles,
            fmin=2,
            fmax=100,
            n_lines=42,
            gamma=0.55,
            odd_only=False,
            min_gap_bins=1,
            harmonic_guard=0,
            dense=False,
            phase_mode="best-random",
            n_phase_trials=250,
            rms=1.0,
            seed=4,
        ),
        "bad_dense_all_bins": dict(
            fs=fs,
            T=T,
            n_cycles=n_cycles,
            fmin=2,
            fmax=100,
            n_lines=42,
            gamma=0.0,
            odd_only=False,
            min_gap_bins=0,
            harmonic_guard=0,
            dense=True,
            phase_mode="zero",
            n_phase_trials=1,
            rms=1.0,
            seed=3,
        ),
    }
    return _run_cases(cases)


def run_strong_harmonic_guard_demo():
    """Run the explicit harmonic-contamination demo. """
    fs = 500.0
    T = 10.0
    n_cycles = 20
    # Keep the probe definition fixed and compare guard on versus off.
    base_cfg = dict(
        fs=fs,
        T=T,
        n_cycles=n_cycles,
        fmin=2,
        fmax=100,
        n_lines=60,
        gamma=0.55,
        odd_only=True,
        min_gap_bins=1,
        dense=False,
        phase_mode="best-random",
        n_phase_trials=250,
        rms=1.0,
    )
    cases = {
        "guarded_strong_nl": dict(base_cfg, harmonic_guard=1, seed=21),
        "unguarded_strong_nl": dict(base_cfg, harmonic_guard=0, seed=21),
    }
    # Use explicit harmonic injection so overlap failures are easy to localize.
    system_kwargs = dict(seed=77, noise_beta=1.0, noise_std=0.02, drift_std=0.0, harmonic_gain=1.2)
    results = _run_cases(
        cases,
        system_fn=simulate_harmonic_guard_demo_system,
        system_kwargs=system_kwargs,
        summary_printer=print_harmonic_guard_table,
        fig_title="Strong 3rd-harmonic distortion: harmonic guard comparison",
    )
    _print_overlap_examples(results)
    return results


def run_overlap_sensitive_harmonic_guard_demo():
    """Run the sharper H(f) demo where guard changes the inferred transfer curve. """
    fs = 500.0
    T = 20.0
    n_cycles = 20
    # Use denser line coverage so the guard does not win by simply leaving a large hole.
    base_cfg = dict(
        fs=fs,
        T=T,
        n_cycles=n_cycles,
        fmin=2,
        fmax=40,
        n_lines=100,
        gamma=0.2,
        odd_only=False,
        min_gap_bins=1,
        dense=False,
        phase_mode="best-random",
        n_phase_trials=250,
        rms=1.0,
        seed=5,
    )
    cases = {
        "guarded_overlap_sensitive": dict(base_cfg, harmonic_guard=1),
        "unguarded_overlap_sensitive": dict(base_cfg, harmonic_guard=0),
    }
    # Inject second-harmonic overlap into a transfer curve with sharper features.
    system_kwargs = dict(
        seed=41,
        transfer_func=transfer_overlap_sensitive_demo,
        harmonic_order=2,
        harmonic_gain=0.8,
        noise_beta=1.0,
        noise_std=0.005,
        drift_std=0.0,
    )
    results = _run_cases(
        cases,
        system_fn=simulate_harmonic_guard_demo_system,
        system_kwargs=system_kwargs,
        truth_func=transfer_overlap_sensitive_demo,
        summary_fn=summarize_overlap_sensitive_case,
        summary_printer=print_overlap_sensitive_guard_table,
        fig_title="Overlap-sensitive H(f): when harmonic guard helps the transfer estimate",
        plot_fn=plot_overlap_sensitive_case,
        n_cols=5,
    )
    _print_overlap_examples(results)
    return results


if __name__ == "__main__":
    run_demo()
