import os
import hashlib
import json
from pathlib import Path
import sys

DIRPATH_REPO = Path(__file__).parents[1]
sys.path.append(str(DIRPATH_REPO))
os.environ.setdefault('MPLCONFIGDIR', '.matplotlib')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy import real, imag, abs, pi, sin, cos
from scipy.signal import butter, sosfiltfilt

from spect_proc import itc, sinusoid_fit_itc
from rate_model import RateModelWC
from rate_model_eig import (
    rate_qphi_to_eig_qphi,
    make_W_h0_ei_2d_syn,
    make_eigmode_start_ei_2d_syn
)


ARTIFACT_ROOT = Path(__file__).resolve().parent / 'artifacts' / Path(__file__).stem


def create_params():
    # Instantaneous rates
    tau = 0

    # Synapses
    taue_syn = 25e-3
    taui_syn = 50e-3

    # Eigenmode
    f = 10
    tau_decay = 0.7
    #tau_decay = -0.7
    q_ie = 1
    phi_ie = 0.8

    # Steady-state
    r0 = np.array([5, 5])

    # Simulation
    T = 20
    dt = 5e-4
    seed = 1111

    # Initial point
    dre_start = 1
    phase_start = -pi / 2

    # Noise
    sigma_h = 0.01

    # Pulses
    pulse_t0 = 0.5
    pulse_f = 10
    pulse_w = 5e-3
    pulse_amp_e = 0
    pulse_amp_i = 0.5

    # Phase and ITC analysis
    phase_signal_type = 'raw'
    phase_pop_id = 0
    phase_f = f
    phase_n_cycles = 5
    phase_method = 'fit'
    phase_progression_plot = 'wrapped_strip'
    phase_subtract_local_mean = 1
    phase_fit_intercept = 1
    phase_rcond = None
    phase_exclude_edges = 1
    phase_min_valid_mass_frac = 0
    phase_drop_win = 10e-3
    phase_drop_win_sweep = np.array(
        [0.0, 2e-3, 4e-3, 6e-3, 8e-3, 10e-3, 15e-3, 20e-3, 50e-3]
    )

    # Pulse-locked epochs
    epoch_t_win = (-0.05, 0.05)

    # Bandpass filter
    bp_f = (5.0, 20.0)
    bp_order = 4

    # Collect_params
    def collect_params():
        model_par = {
            'tau': tau,
            'taue_syn': taue_syn,
            'taui_syn': taui_syn,
            'r0': r0,
            'sigma_h': sigma_h
        }
        eigmode_par = {
            'f': f,
            'tau_decay': tau_decay,
            'q_ie': q_ie,
            'phi_ie': phi_ie,
        }
        sim_par = {
            'T': T,
            'dt': dt,
            'seed': seed,
            'dre_start': dre_start,
            'phase_start': phase_start,
        }
        pulse_par = {
            'pulse_t0': pulse_t0,
            'pulse_f': pulse_f,
            'pulse_w': pulse_w,
            'pulse_amp_e': pulse_amp_e,
            'pulse_amp_i': pulse_amp_i,
        }
        sigproc_par = {
            'epoch_t_win': epoch_t_win,
            'bp_f': bp_f,
            'bp_order': bp_order,
        }
        phaseproc_par = {
            'method': phase_method,
            'phase_progression_plot': phase_progression_plot,
            'signal': phase_signal_type,
            'pop': phase_pop_id,
            'f': phase_f,
            'n_cycles': phase_n_cycles,
            'subtract_local_mean': phase_subtract_local_mean,
            'fit_intercept': phase_fit_intercept,
            'rcond': phase_rcond,
            'exclude_edges': phase_exclude_edges,
            'min_valid_mass_frac': phase_min_valid_mass_frac,
            'drop_win': phase_drop_win,
            'drop_win_sweep': phase_drop_win_sweep,
        }
        return {
            'model': model_par, 'eigmode': eigmode_par, 
            'sim': sim_par, 'pulse': pulse_par,
            'sigproc': sigproc_par,
            'phaseproc': phaseproc_par,
        }
    return collect_params()


def _json_ready(value):
    # Convert arrays and numpy scalars into JSON-friendly values
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _format_drop_win(drop_win):
    # Format the phase drop window for labels and file names
    if np.isscalar(drop_win):
        value = float(drop_win)
        return f"{-value:g}_{value:g}"
    drop_win = np.asarray(drop_win, dtype=float).ravel()
    return f"{drop_win[0]:g}_{drop_win[1]:g}"


def _hash_payload(payload):
    # Compute a short stable hash for a JSON-ready payload
    payload_json = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(payload_json.encode('utf-8')).hexdigest()[:8]


def _get_active_phaseproc(phaseproc):
    # Keep only the phase-analysis params that matter for the selected method
    method = phaseproc['method']
    if method not in {'morlet', 'fit'}:
        raise ValueError("phaseproc['method'] must be 'morlet' or 'fit'")

    active = {
        'method': method,
        'phase_progression_plot': phaseproc['phase_progression_plot'],
        'signal': phaseproc['signal'],
        'pop': phaseproc['pop'],
        'f': phaseproc['f'],
        'n_cycles': phaseproc['n_cycles'],
        'exclude_edges': bool(phaseproc['exclude_edges']),
        'min_valid_mass_frac': float(phaseproc['min_valid_mass_frac']),
        'drop_win': phaseproc['drop_win'],
        'drop_win_sweep': phaseproc['drop_win_sweep'],
    }
    if method == 'morlet':
        active['subtract_local_mean'] = bool(phaseproc['subtract_local_mean'])
    else:
        active['fit_intercept'] = bool(phaseproc['fit_intercept'])
        active['rcond'] = phaseproc['rcond']
    return active


def _normalize_drop_win(drop_win):
    # Expand scalar drop windows to symmetric pairs
    if np.isscalar(drop_win):
        value = float(drop_win)
        return (-value, value)
    drop_win = np.asarray(drop_win, dtype=float).ravel()
    return float(drop_win[0]), float(drop_win[1])


def _phase_progression_strip_coords(tt_valid, phases_valid, ymin, ymax):
    # Replicate wrapped phases across the visible 2π strip
    tt_valid = np.asarray(tt_valid, dtype=float)
    phases_valid = np.asarray(phases_valid, dtype=float)
    k_min = int(np.floor((ymin - np.max(phases_valid)) / (2.0 * np.pi)))
    k_max = int(np.ceil((ymax - np.min(phases_valid)) / (2.0 * np.pi)))

    x_plot = []
    y_plot = []
    for k in range(k_min, k_max + 1):
        phase_k = phases_valid + 2.0 * np.pi * k
        keep = (phase_k >= ymin) & (phase_k <= ymax)
        if np.any(keep):
            x_plot.append(tt_valid[keep])
            y_plot.append(phase_k[keep])

    if not x_plot:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.concatenate(x_plot), np.concatenate(y_plot)


def split_params_for_hashing(params):
    """Split params into simulation and analysis subsets."""

    # Separate simulation-driving params from analysis-only params
    sim_payload = _json_ready({
        'model': params['model'],
        'eigmode': params['eigmode'],
        'sim': params['sim'],
        'pulse': params['pulse'],
    })
    analysis_payload = _json_ready({
        'sigproc': params['sigproc'],
        'phaseproc': _get_active_phaseproc(params['phaseproc']),
    })
    return sim_payload, analysis_payload


def build_artifact_paths(params):
    """Build the output folder and file paths for a parameter set."""

    # Split the params into simulation and analysis subsets
    sim_payload, analysis_payload = split_params_for_hashing(params)
    sim_hash = _hash_payload(sim_payload)
    analysis_hash = _hash_payload(analysis_payload)

    # Build the simulation-specific artifact folder
    T = params['sim']['T']
    pulse_amp_e = params['pulse']['pulse_amp_e']
    pulse_amp_i = params['pulse']['pulse_amp_i']
    seed = params['sim']['seed']
    phase_method = params['phaseproc']['method']
    phase_drop_win = _format_drop_win(params['phaseproc']['drop_win'])
    run_dir = ARTIFACT_ROOT / (
        f"res_T_{T:g}_amp_{pulse_amp_e:g}_{pulse_amp_i:g}_seed_{seed}_{sim_hash}"
    )

    # Build the cache and analysis-specific output file paths
    return {
        'dir': run_dir,
        'sim_cache': run_dir / 'sim_cache.npz',
        'signals_png': run_dir / (
            f'phase_lock_signals_{phase_method}_drop_{phase_drop_win}_{analysis_hash}.png'
        ),
        'phase_png': run_dir / (
            f'phase_lock_phase_itc_{phase_method}_drop_{phase_drop_win}_{analysis_hash}.png'
        ),
        'phase_sweep_png': run_dir / (
            f'phase_lock_phase_itc_sweep_{phase_method}_{analysis_hash}.png'
        ),
        'sim_params_json': run_dir / 'sim_params.json',
        'analysis_params_json': run_dir / f'analysis_params_{analysis_hash}.json',
        'sim_payload': sim_payload,
        'analysis_payload': analysis_payload,
        'sim_hash': sim_hash,
        'analysis_hash': analysis_hash,
    }


def save_json(path, payload):
    # Save a JSON payload with stable formatting
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fobj:
        json.dump(payload, fobj, indent=2, sort_keys=True)


def save_sim_cache(path, R, tt_pulse, J):
    # Save the raw simulation outputs needed for later analyses
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        R_values=R.values,
        R_time=R.coords['time'].values,
        R_pop=R.coords['pop'].values,
        tt_pulse=np.asarray(tt_pulse, dtype=float),
        J=np.asarray(J, dtype=float),
    )


def load_sim_cache(path):
    # Load a cached raw simulation and reconstruct the xarray signal
    with np.load(path) as data:
        R = xr.DataArray(
            data['R_values'],
            dims=['pop', 'time'],
            coords={'pop': data['R_pop'], 'time': data['R_time']},
        )
        tt_pulse = data['tt_pulse']
        J = data['J']
    return R, tt_pulse, J


def compute_pulse_t_mask(tt_pulse, params):
    """Compute the pulse mask from pulse times. """
    # Unpack the pulse timing parameters
    T, dt = (params['sim'][p] for p in ['T', 'dt'])
    pulse_w = params['pulse']['pulse_w']
    nsteps = int(T / dt)
    pulse_kw = max(1, int(np.round(pulse_w / dt)))

    # Mark the pulse samples on the simulation grid
    pulse_t_mask = np.zeros(nsteps, dtype=bool)
    for pulse_t in np.asarray(tt_pulse, dtype=float):
        k0 = int(np.round(pulse_t / dt))
        k1 = min(k0 + pulse_kw, nsteps)
        if 0 <= k0 < nsteps:
            pulse_t_mask[k0:k1] = True
    return pulse_t_mask


def run_model(params):
    """Returns: R, tt_pulse, J """

    # Unpack params
    tau, taue_syn, taui_syn, r0, sigma_h = (
        params['model'][p] for p in ['tau', 'taue_syn', 'taui_syn', 'r0', 'sigma_h'])
    f, tau_decay, q_ie, phi_ie = (
        params['eigmode'][p] for p in ['f', 'tau_decay', 'q_ie', 'phi_ie'])
    T, dt, seed, dre_start, phase_start = (
        params['sim'][p] for p in ['T', 'dt', 'seed', 'dre_start', 'phase_start'])
    pulse_t0, pulse_f, pulse_w, pulse_amp_e, pulse_amp_i = (
        params['pulse'][p] for p in ['pulse_t0', 'pulse_f', 'pulse_w',
                                     'pulse_amp_e', 'pulse_amp_i'])

    # Derived vars
    tau_syn = np.array([taue_syn, taui_syn])
    nsteps = int(T / dt)
    rng = np.random.default_rng(seed)

    # Initialize the model
    model = RateModelWC(
        np.zeros((2, 2)), tau=tau, rmax=10, gain_slope=0.5, gain_center=0)

    # Set model params that yield given eigenmode
    W, h0, J, tau_syn_mat = make_W_h0_ei_2d_syn(
        model, r0, tau_syn, f, tau_decay, q_ie=q_ie, phi_ie=phi_ie
    )
    model.W = W
    model.tau_syn = tau_syn_mat

    # Initial point
    r_start, s_start = make_eigmode_start_ei_2d_syn(
        W, r0, tau_syn, f, tau_decay, q_ie=q_ie, phi_ie=phi_ie,
        dre_start=dre_start, phase_start=phase_start
    )

    """ print(f"W:\n{W}")
    print(f"rate q/phi: {q_ie}, {phi_ie}")
    print(f"eig q/phi: {q_eig}, {phi_eig}")
    L = np.linalg.eigvals(J)
    print(f"\nJacobian eigenvalues: {L}")
    print(f'f: {abs(imag(L[0])) / 2 / pi}')
    print(f'tau_decay: {-1.0 / real(L[0])}') """

    # Build the input signal
    H = np.repeat(h0, nsteps, axis=1)   # const level
    H += sigma_h * rng.standard_normal((2, nsteps)) / np.sqrt(dt)   # noise

    # Build the pulse times
    npulses = int(np.floor((T - pulse_t0) * pulse_f)) + 1
    tt_pulse = []
    for n in range(npulses):
        pulse_t = pulse_t0 + n / pulse_f
        if 0 <= pulse_t < T:
            tt_pulse.append(pulse_t)
    tt_pulse = np.array(tt_pulse, dtype=float)

    # Add pulses to the signal
    pulse_t_mask = compute_pulse_t_mask(tt_pulse, params)
    H[0, pulse_t_mask] += pulse_amp_e
    H[1, pulse_t_mask] += pulse_amp_i

    # Run starting from a rate/synapse state on the eigenmode
    R = model.run(H, r_start, dt, nsteps, s0=s_start)

    return R, tt_pulse, J


def get_simulation(params, artifact_paths):
    """Load cached simulation outputs or run and cache them."""

    # Reuse the raw simulation if the sim cache already exists
    if artifact_paths['sim_cache'].exists():
        R, tt_pulse, J = load_sim_cache(artifact_paths['sim_cache'])
        return R, tt_pulse, J, True

    # Otherwise run the simulation and persist the raw outputs
    R, tt_pulse, J = run_model(params)
    save_sim_cache(artifact_paths['sim_cache'], R, tt_pulse, J)
    return R, tt_pulse, J, False


def filt_signal(R, params):
    """Filter a signal. """
    bp_f, bp_order = (params['sigproc'][p] for p in ['bp_f', 'bp_order'])
    dt = params['sim']['dt']
    sos = butter(bp_order, bp_f, btype='bandpass', fs=1.0 / dt, output='sos')

    # Filter the xarray data and preserve its coordinates
    R_filt_values = sosfiltfilt(sos, R.values, axis=1)
    return R.copy(data=R_filt_values)


def epoch_signals(signals, tt_pulse, params):
    """Epoch signals by pulse times and avg. over epochs. """

    # Unpack the epoching parameters
    T, dt = (params['sim'][p] for p in ['T', 'dt'])
    epoch_t_win = params['sigproc']['epoch_t_win']
    nsteps = int(T / dt)

    # Recover the pulse onset indices from the pulse mask
    pulse_t_mask = compute_pulse_t_mask(tt_pulse, params)
    pulse_start_mask = pulse_t_mask & ~np.r_[False, pulse_t_mask[:-1]]
    pulse_idx = np.flatnonzero(pulse_start_mask)

    # Build the epoch index window
    epoch_k0 = int(np.round(epoch_t_win[0] / dt))
    epoch_k1 = int(np.round(epoch_t_win[1] / dt))
    epoch_start_idx = pulse_idx[
        (pulse_idx + epoch_k0 >= 0) & (pulse_idx + epoch_k1 <= nsteps)
    ]
    tt_epoch = np.arange(epoch_k0, epoch_k1) * dt

    # Average over pulse-triggered epochs
    epochs_mean = {}
    for key, X in signals.items():
        epochs = np.stack(
            [X.values[:, n + epoch_k0:n + epoch_k1] for n in epoch_start_idx],
            axis=0,
        )
        epoch_mean_values = epochs.mean(axis=0)
        epoch_template = X.isel(time=slice(0, epoch_k1 - epoch_k0)).copy(
            data=epoch_mean_values
        )
        epoch_template = epoch_template.assign_coords(time=tt_epoch)
        epochs_mean[key] = epoch_template

    return tt_epoch, epochs_mean, len(epoch_start_idx)


def analyze_phase_itc(R_signals, tt_pulse, params):
    """Compute stimulus-locked phases and ITC."""

    # Unpack the phase-analysis parameters
    phaseproc = _get_active_phaseproc(params['phaseproc'])
    signal_key, pop_idx = (phaseproc[p] for p in ['signal', 'pop'])
    signal = R_signals[signal_key].isel(pop=pop_idx)
    drop_win = _normalize_drop_win(phaseproc['drop_win'])

    # Compute the stimulus-locked phases
    if phaseproc['method'] == 'morlet':
        itc_value, phi, phases = itc(
            signal.values,
            signal.coords['time'].values,
            tt_pulse,
            phaseproc['f'],
            phaseproc['n_cycles'],
            drop_win=drop_win,
            subtract_local_mean=phaseproc['subtract_local_mean'],
            exclude_edges=phaseproc['exclude_edges'],
            min_valid_mass_frac=phaseproc['min_valid_mass_frac'],
        )
    else:
        itc_value, phi, phases = sinusoid_fit_itc(
            signal.values,
            signal.coords['time'].values,
            tt_pulse,
            phaseproc['f'],
            phaseproc['n_cycles'],
            drop_win=drop_win,
            fit_intercept=phaseproc['fit_intercept'],
            exclude_edges=phaseproc['exclude_edges'],
            min_valid_mass_frac=phaseproc['min_valid_mass_frac'],
            rcond=phaseproc['rcond'],
        )

    # Package the phase-analysis outputs
    result = {
        'method': phaseproc['method'],
        'phase_progression_plot': phaseproc['phase_progression_plot'],
        'signal_key': signal_key,
        'pop': pop_idx,
        'f': phaseproc['f'],
        'n_cycles': phaseproc['n_cycles'],
        'exclude_edges': phaseproc['exclude_edges'],
        'min_valid_mass_frac': phaseproc['min_valid_mass_frac'],
        'drop_win': drop_win,
        'tt_stim': np.asarray(tt_pulse, dtype=float),
        'phases': phases,
        'itc': itc_value,
        'phi': phi,
    }
    if phaseproc['method'] == 'morlet':
        result['subtract_local_mean'] = phaseproc['subtract_local_mean']
    else:
        result['fit_intercept'] = phaseproc['fit_intercept']
        result['rcond'] = phaseproc['rcond']
    return result


def analyze_phase_itc_drop_sweep(R_signals, tt_pulse, params):
    """Sweep symmetric drop windows and summarize ITC and mean phase."""

    # Unpack the phase-analysis parameters
    phaseproc = _get_active_phaseproc(params['phaseproc'])
    signal_key, pop_idx = (phaseproc[p] for p in ['signal', 'pop'])
    signal = R_signals[signal_key].isel(pop=pop_idx)
    widths = np.asarray(phaseproc['drop_win_sweep'], dtype=float).ravel()

    # Evaluate ITC and mean phase for each symmetric drop width
    itc_vals = np.full(widths.shape, np.nan, dtype=float)
    phi_vals = np.full(widths.shape, np.nan, dtype=float)
    for i, width in enumerate(widths):
        drop_win = (-width, width)
        if phaseproc['method'] == 'morlet':
            itc_val, phi_val, phases = itc(
                signal.values,
                signal.coords['time'].values,
                tt_pulse,
                phaseproc['f'],
                phaseproc['n_cycles'],
                drop_win=drop_win,
                subtract_local_mean=phaseproc['subtract_local_mean'],
                exclude_edges=phaseproc['exclude_edges'],
                min_valid_mass_frac=phaseproc['min_valid_mass_frac'],
            )
        else:
            itc_val, phi_val, phases = sinusoid_fit_itc(
                signal.values,
                signal.coords['time'].values,
                tt_pulse,
                phaseproc['f'],
                phaseproc['n_cycles'],
                drop_win=drop_win,
                fit_intercept=phaseproc['fit_intercept'],
                exclude_edges=phaseproc['exclude_edges'],
                min_valid_mass_frac=phaseproc['min_valid_mass_frac'],
                rcond=phaseproc['rcond'],
            )
        itc_vals[i] = itc_val
        phi_vals[i] = phi_val

    # Package the sweep outputs
    result = {
        'method': phaseproc['method'],
        'signal_key': signal_key,
        'pop': pop_idx,
        'f': phaseproc['f'],
        'n_cycles': phaseproc['n_cycles'],
        'exclude_edges': phaseproc['exclude_edges'],
        'min_valid_mass_frac': phaseproc['min_valid_mass_frac'],
        'drop_widths': widths,
        'itc': itc_vals,
        'phi': phi_vals,
    }
    if phaseproc['method'] == 'morlet':
        result['subtract_local_mean'] = phaseproc['subtract_local_mean']
    else:
        result['fit_intercept'] = phaseproc['fit_intercept']
        result['rcond'] = phaseproc['rcond']
    return result


def plot_results(R_signals, tt_epoch, epochs_mean, n_epochs, params, artifact_paths):
    """Plot the full time series and pulse-triggered averages."""

    # Unpack the plot labels and references
    r0 = params['model']['r0']
    f, tau_decay = (params['eigmode'][p] for p in ['f', 'tau_decay'])
    bp_f = params['sigproc']['bp_f']
    drop_win = _normalize_drop_win(params['phaseproc']['drop_win'])

    # Build the plot layout
    fig, ax = plt.subplots(
        2, 2, figsize=(12, 7), gridspec_kw={'width_ratios': [3, 1]}
    )
    row_titles = {
        'raw': f'E-I oscillations (f={f} Hz, τ_decay={tau_decay})',
        'filt': f'Bandpass {bp_f[0]}-{bp_f[1]} Hz',
    }
    row_ylabel = {'raw': 'Rate', 'filt': 'Filtered rate'}
    row_label_suffix = {'raw': '', 'filt': ', filtered'}
    tt = R_signals['raw'].coords['time']

    # Plot the timecourses and epoch averages
    for i, key in enumerate(['raw', 'filt']):
        ax[i, 0].plot(
            tt, R_signals[key].isel(pop=0), label=f'Excitatory{row_label_suffix[key]}'
        )
        ax[i, 0].plot(
            tt, R_signals[key].isel(pop=1), label=f'Inhibitory{row_label_suffix[key]}'
        )
        if key == 'raw':
            ax[i, 0].axhline(r0[0], color='C0', linestyle='--', alpha=0.3)
            ax[i, 0].axhline(r0[1], color='C1', linestyle='--', alpha=0.3)
        ax[i, 0].set_xlabel('Time')
        ax[i, 0].set_ylabel(row_ylabel[key])
        ax[i, 0].legend()
        ax[i, 0].set_title(row_titles[key])

        # Plot the average pulse-triggered responses
        ax[i, 1].plot(
            tt_epoch, epochs_mean[key].isel(pop=0), label=f'Excitatory{row_label_suffix[key]}'
        )
        ax[i, 1].plot(
            tt_epoch, epochs_mean[key].isel(pop=1), label=f'Inhibitory{row_label_suffix[key]}'
        )
        ax[i, 1].axvspan(drop_win[0], drop_win[1], color='0.5', alpha=0.15)
        if key == 'raw':
            ax[i, 1].axhline(r0[0], color='C0', linestyle='--', alpha=0.3)
            ax[i, 1].axhline(r0[1], color='C1', linestyle='--', alpha=0.3)
        ax[i, 1].axvline(0, color='k', linestyle='--', alpha=0.5)
        ax[i, 1].set_xlabel('Time from pulse')
        ax[i, 1].set_ylabel(row_ylabel[key])
        ax[i, 1].set_title(
            f'Epoch average over {n_epochs} pulses'
            if key == 'raw' else 'Filtered epoch average'
        )

    # Save the signals and epochs figure
    artifact_paths['dir'].mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(artifact_paths['signals_png'], dpi=150)
    plt.close(fig)


def plot_phase_itc_results(phase_results, artifact_paths):
    """Plot the stimulus phases and complex-plane ITC summary."""

    # Unpack the phase-analysis outputs
    tt_stim = phase_results['tt_stim']
    phases = np.asarray(phase_results['phases'], dtype=float)
    valid = np.isfinite(phases)
    phases_valid = phases[valid]
    tt_valid = tt_stim[valid]
    itc_value = phase_results['itc']
    phi = phase_results['phi']
    signal_key = phase_results['signal_key']
    pop_idx = phase_results['pop']
    method = phase_results['method']
    phase_progression_plot = phase_results['phase_progression_plot']
    drop_win = phase_results['drop_win']

    # Unwrap the phase progression and anchor it to the first absolute phase
    phase_prog = np.array([], dtype=float)
    if phases_valid.size:
        phase_prog = np.unwrap(phases_valid)
        phase_prog += phases_valid[0] - phase_prog[0]

    # Build the plot layout
    fig, ax = plt.subplots(
        1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [3, 1.3]}
    )

    # Set the left-panel y-range
    if phase_progression_plot == 'wrapped_strip':
        y_min = -2.0 * np.pi
        y_max = 2.0 * np.pi
    elif phase_prog.size:
        y_min = float(np.min(phase_prog))
        y_max = float(np.max(phase_prog))
    else:
        y_min = -2.0 * np.pi
        y_max = 2.0 * np.pi

    # Add reference lines at nearby multiples of pi
    pi_k_min = int(np.floor(y_min / np.pi))
    pi_k_max = int(np.ceil(y_max / np.pi))
    for k in range(pi_k_min, pi_k_max + 1):
        ax[0].axhline(
            k * np.pi,
            color='k' if abs(k) % 2 == 0 else 'r',
            linestyle='--',
            linewidth=1.0,
            alpha=0.5,
        )

    # Plot the phase progression at stimulus times
    if phase_progression_plot == 'wrapped_strip':
        xx, yy = _phase_progression_strip_coords(tt_valid, phases_valid, y_min, y_max)
        ax[0].plot(
            xx, yy, linestyle='None', marker='o', markersize=2.5,
            color='C0', label='Phase copies'
        )
        ax[0].set_ylim(y_min, y_max)
    else:
        ax[0].plot(tt_valid, phase_prog, '-', color='C0', label='Phase progression')

    ax[0].set_xlabel('Stimulus time')
    ax[0].set_ylabel('Phase progression')
    ax[0].set_title(
        f'Phase progression at stimuli ({signal_key}, pop {pop_idx}, {method}, {phase_progression_plot})'
    )
    ax[0].legend()

    # Plot the phases on the complex plane
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax[1].plot(np.cos(theta), np.sin(theta), color='0.75')
    ax[1].scatter(
        np.cos(phases_valid), np.sin(phases_valid), 
        color='C1', alpha=0.1, label='Stimulus phases'
    )
    ax[1].plot(
        [0.0, np.cos(phi)],
        [0.0, np.sin(phi)],
        color='0.6',
        linestyle='--',
        linewidth=1.2,
        label='Avg phase direction',
    )
    ax[1].plot(
        [0.0, itc_value * np.cos(phi)],
        [0.0, itc_value * np.sin(phi)],
        color='C0',
        linewidth=2.0,
        label='ITC vector',
    )
    ax[1].set_aspect('equal', 'box')
    ax[1].set_xlim(-1.1, 1.1)
    ax[1].set_ylim(-1.1, 1.1)
    ax[1].set_xlabel('Real')
    ax[1].set_ylabel('Imag')
    ax[1].set_title(f'Complex plane ITC={itc_value:.3f} ({method})')
    #ax[1].legend(loc='upper right', fontsize=8)

    # Add the phase-analysis summary to the figure
    summary = [
        f'method = {method}',
        f'phase_plot = {phase_progression_plot}',
        f'phase_drop_win = {drop_win}',
    ]
    if method == 'morlet':
        summary.append(
            f"subtract_local_mean = {bool(phase_results['subtract_local_mean'])}"
        )
    else:
        summary.append(f"fit_intercept = {bool(phase_results['fit_intercept'])}")
    fig.text(0.12, 0.02, '    '.join(summary), family='monospace')

    # Save the phase and ITC figure
    artifact_paths['dir'].mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(artifact_paths['phase_png'], dpi=150)
    plt.close(fig)


def plot_phase_itc_sweep_results(sweep_results, artifact_paths):
    """Plot ITC and mean phase versus symmetric drop width."""

    # Unpack the sweep outputs
    widths = np.asarray(sweep_results['drop_widths'], dtype=float)
    itc_vals = np.asarray(sweep_results['itc'], dtype=float)
    phi_vals = np.asarray(sweep_results['phi'], dtype=float)
    signal_key = sweep_results['signal_key']
    pop_idx = sweep_results['pop']
    method = sweep_results['method']

    # Build the plot layout
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    # Plot the ITC versus drop width
    ax[0].plot(widths, itc_vals, 'o-', color='C0')
    ax[0].set_xlabel('Symmetric drop half-width')
    ax[0].set_ylabel('ITC')
    ax[0].set_title(f'ITC vs drop width ({signal_key}, pop {pop_idx}, {method})')

    # Plot the circular-mean phase versus drop width
    ax[1].plot(widths, phi_vals, 'o-', color='C1')
    ax[1].set_xlabel('Symmetric drop half-width')
    ax[1].set_ylabel('Mean phase (rad)')
    ax[1].set_title(f'Mean phase vs drop width ({method})')

    # Save the sweep figure
    artifact_paths['dir'].mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(artifact_paths['phase_sweep_png'], dpi=150)
    plt.close(fig)


def main():
    # Define params
    params = create_params()
    artifact_paths = build_artifact_paths(params)

    # Save the simulation and analysis parameter subsets
    save_json(artifact_paths['sim_params_json'], artifact_paths['sim_payload'])
    save_json(artifact_paths['analysis_params_json'], artifact_paths['analysis_payload'])

    # Create or load the raw simulation
    R, tt_pulse, J, cache_hit = get_simulation(params, artifact_paths)

    # Filter the output signal
    R_filt = filt_signal(R, params)

    # Collect the raw and filtered xarray signals
    R_signals = {'raw': R, 'filt': R_filt}

    # Epoch the signals around the pulses
    tt_epoch, epochs_mean, n_epochs = epoch_signals(R_signals, tt_pulse, params)

    # Plot the signals and epochs
    plot_results(R_signals, tt_epoch, epochs_mean, n_epochs, params, artifact_paths)

    # Compute the phase and ITC summary
    phase_results = analyze_phase_itc(R_signals, tt_pulse, params)
    phase_sweep_results = analyze_phase_itc_drop_sweep(R_signals, tt_pulse, params)

    # Plot the phase and ITC results
    plot_phase_itc_results(phase_results, artifact_paths)
    plot_phase_itc_sweep_results(phase_sweep_results, artifact_paths)

    # Report the saved artifact paths
    cache_msg = 'Loaded' if cache_hit else 'Saved'
    print(f"{cache_msg} {artifact_paths['sim_cache']}")
    print(f"Saved {artifact_paths['signals_png']}")
    print(f"Saved {artifact_paths['phase_png']}")
    print(f"Saved {artifact_paths['phase_sweep_png']}")
    print(f"Saved {artifact_paths['sim_params_json']}")
    print(f"Saved {artifact_paths['analysis_params_json']}")


if __name__ == '__main__':
    main()
