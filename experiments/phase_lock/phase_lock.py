import os
import hashlib
import json
import copy
from pathlib import Path
import sys

DIRPATH_REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(DIRPATH_REPO))
os.environ.setdefault('MPLCONFIGDIR', '.matplotlib')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy import real, imag, abs, pi, sin, cos
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt

from experiments.phase_lock.spect_proc import (
    itc,
    morlet_power,
    morlet_power_freqs,
    sinusoid_fit_itc,
    sinusoid_fit_power,
    sinusoid_fit_power_freqs,
)
from rate_model import RateModelWC
from rate_model.rate_model_eig import (
    rate_qphi_to_eig_qphi,
    make_W_h0_ei_2d_syn,
    make_eigmode_start_ei_2d_syn
)


ARTIFACT_ROOT = Path(__file__).resolve().parent / 'artifacts' / Path(__file__).stem

SIM_VARIANT_SPECS = (
    {'key': 'noise_pulses', 'label': 'Noise + pulses', 'color': 'C0'},
    {'key': 'noise_only', 'label': 'Noise only', 'color': 'C1'},
    {'key': 'pulses_only', 'label': 'Pulses only', 'color': 'C2'},
)

DERIVED_SIGNAL_SPECS = {
    'np_minus_p': {
        'label': 'Noise+pulses - pulses only',
        'color': 'C3',
        'terms': (
            ('noise_pulses', 1.0),
            ('pulses_only', -1.0)
        )
    },
    'np_minus_n': {
        'label': 'Noise+pulses - noise only',
        'color': 'C4',
        'terms': (
            ('noise_pulses', 1.0),
            ('noise_only', -1.0)
        )
    },
    'np_minus_n_p': {
        'label': 'Noise+pulses - pulses only - noise only',
        'color': 'C5',
        'terms': (
            ('noise_pulses', 1.0),
            ('pulses_only', -1.0),
            ('noise_only', -1.0)
        )
    },
    'n_plus_p': {
        'label': 'Pulses only + noise only',
        'color': 'C6',
        'terms': (
            ('pulses_only', 1.0),
            ('noise_only', 1.0)
        )
    }
}


def create_params():
    # Instantaneous rates
    tau = 0

    # Synapses
    taue_syn = 25e-3
    taui_syn = 50e-3

    # Eigenmode
    f = 10
    #tau_decay = -0.7
    tau_decay = 0.7
    q_ie = 1
    phi_ie = 0.8

    # Steady-state
    r0 = np.array([5, 5])

    # Simulation
    T = 10
    dt = 5e-4
    seed = 1111

    # Initial point
    dre_start = 1
    #phase_start = -pi / 2
    phase_start = 0

    # Noise
    sigma_h = 0.01

    # Pulses
    pulse_t0 = 0.5
    pulse_f = 10
    pulse_w = 5e-3
    pulse_amp_e = 0
    pulse_amp_i = 0.0

    # Phase and ITC analysis
    phase_signal_type = 'raw'
    phase_pop_id = 0
    phase_f = f
    phase_n_cycles = 5
    phase_method = 'fit'
    phase_progression_plot = 'wrapped_strip'
    phase_power_mode = 'amp2'
    phase_power_overlap = 0.9
    phase_power_fband = 5
    phase_power_df = 0.5
    phase_subtract_local_mean = 1
    phase_fit_intercept = 1
    phase_rcond = None
    phase_exclude_edges = 1
    phase_min_valid_mass_frac = 0
    phase_drop_win = 10e-3
    phase_drop_win_sweep = np.array(
        [0.0, 2e-3, 4e-3, 6e-3, 8e-3, 10e-3, 15e-3, 20e-3]
    )
    phase_distr_smooth_sigma = 1

    # Pulse-locked epochs
    epoch_t_win = (-0.05, 0.05)
    epoch_plot_subtract_r0 = 1
    raw_zoom_t_win = (8, 10)
    variant_t_win = (5, 10)

    # Time window for pulse inclusion
    analysis_t_win = (5, T)

    # Bandpass filter
    bp_f = (5.0, 20.0)
    bp_order = 4

    # Simulation variants
    sim_variants_enabled = 1
    derived_signals_enabled = 1
    derived_signals_subtract_r0_before_combine = 1
    derived_signals_use_specs = ['n_plus_p']

    # Analysis actions
    action_signals = 1
    action_signal_vars = 1
    action_itc = 1
    action_itc_wvar = 0
    action_pow_wvar = 0
    action_pow_fband = 1
    action_phase_distr = 1

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
        sim_variants_par = {
            'enabled': sim_variants_enabled,
        }
        derived_signals_par = {
            'enabled': derived_signals_enabled,
            'subtract_r0_before_combine': (
                derived_signals_subtract_r0_before_combine
            ),
            'use_specs': derived_signals_use_specs,
        }
        action_flags_par = {
            'signals': action_signals,
            'signal_vars': action_signal_vars,
            'itc': action_itc,
            'itc_wvar': action_itc_wvar,
            'pow_wvar': action_pow_wvar,
            'pow_fband': action_pow_fband,
            'phase_distr': action_phase_distr,
        }
        sigproc_par = {
            'epoch_t_win': epoch_t_win,
            'epoch_plot_subtract_r0': epoch_plot_subtract_r0,
            'raw_zoom_t_win': raw_zoom_t_win,
            'variant_t_win': variant_t_win,
            'analysis_t_win': analysis_t_win,
            'bp_f': bp_f,
            'bp_order': bp_order,
        }
        phaseproc_par = {
            'method': phase_method,
            'phase_progression_plot': phase_progression_plot,
            'power_mode': phase_power_mode,
            'power_overlap': phase_power_overlap,
            'power_fband': phase_power_fband,
            'power_df': phase_power_df,
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
            'phase_distr_xlim': (-2.0 * np.pi, 2.0 * np.pi),
            'phase_distr_bins': 64,
            'phase_distr_smooth_sigma': phase_distr_smooth_sigma,
        }
        return {
            'model': model_par, 'eigmode': eigmode_par, 
            'sim': sim_par, 'pulse': pulse_par,
            'sim_variants': sim_variants_par,
            'derived_signals': derived_signals_par,
            'action_flags': action_flags_par,
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


def _format_drop_win_filename(drop_win):
    # Format the phase drop window compactly for file names
    t_pre, t_post = _normalize_drop_win(drop_win)
    if np.isclose(t_pre, -t_post):
        return f"w_{abs(t_post):g}"
    return f"w_{t_pre:g}_{t_post:g}"


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
        'power_overlap': float(phaseproc['power_overlap']),
        'power_fband': float(phaseproc['power_fband']),
        'power_df': float(phaseproc['power_df']),
        'exclude_edges': bool(phaseproc['exclude_edges']),
        'min_valid_mass_frac': float(phaseproc['min_valid_mass_frac']),
        'drop_win': phaseproc['drop_win'],
        'drop_win_sweep': phaseproc['drop_win_sweep'],
        'phase_distr_xlim': phaseproc['phase_distr_xlim'],
        'phase_distr_bins': phaseproc['phase_distr_bins'],
        'phase_distr_smooth_sigma': float(phaseproc['phase_distr_smooth_sigma']),
    }
    if method == 'morlet':
        active['subtract_local_mean'] = bool(phaseproc['subtract_local_mean'])
        active['power_mode'] = phaseproc['power_mode']
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


def _apply_drop_win_mask(x, t, tt_stim, drop_win):
    # Mask all peri-stimulus windows on a 1D signal
    x_masked = np.asarray(x, dtype=float).copy()
    t_pre, t_post = _normalize_drop_win(drop_win)
    for t0 in np.asarray(tt_stim, dtype=float):
        drop_mask = (t >= t0 + t_pre) & (t <= t0 + t_post)
        x_masked[drop_mask] = np.nan
    return x_masked


def _get_variant_specs(params):
    # Select the simulation variants for this run
    if params['sim_variants']['enabled']:
        return list(SIM_VARIANT_SPECS)
    return [SIM_VARIANT_SPECS[0]]


def _get_selected_derived_signal_specs(params):
    # Select the named derived-signal specs requested for this run
    derived_signals = params['derived_signals']
    if not derived_signals['enabled']:
        return []

    specs = []
    for name in derived_signals['use_specs']:
        if name not in DERIVED_SIGNAL_SPECS:
            raise ValueError(f'Unknown derived signal spec: {name}')
        spec = dict(DERIVED_SIGNAL_SPECS[name])
        spec['key'] = name
        specs.append(spec)
    return specs


def _get_primary_variant_key():
    # Keep one primary variant for shared reference plots
    return 'noise_pulses'


def _make_variant_params(params, variant_key):
    # Derive one simulation variant from the base params
    variant_params = copy.deepcopy(params)
    if variant_key == 'noise_only':
        variant_params['pulse']['pulse_amp_e'] = 0.0
        variant_params['pulse']['pulse_amp_i'] = 0.0
    elif variant_key == 'pulses_only':
        variant_params['model']['sigma_h'] = 0.0
    elif variant_key != 'noise_pulses':
        raise ValueError(f'Unknown simulation variant: {variant_key}')
    return variant_params


def _get_sim_cache_path(artifact_paths, params, variant_key):
    # Build the raw-cache path for one simulation variant
    if not params['sim_variants']['enabled'] and variant_key == _get_primary_variant_key():
        return artifact_paths['sim_cache']
    return artifact_paths['dir'] / f'sim_cache_{variant_key}.npz'


def _filter_analysis_pulse_times(tt_pulse, params):
    # Keep only pulse times that belong to the shared analysis window
    tt_pulse = np.asarray(tt_pulse, dtype=float)
    t0, t1 = map(float, params['sigproc']['analysis_t_win'])
    keep = (tt_pulse >= t0) & (tt_pulse <= t1)
    return tt_pulse[keep]


def split_params_for_hashing(params):
    """Split params into simulation and analysis subsets."""

    # Separate simulation-driving params from analysis-only params
    sim_payload = _json_ready({
        'model': params['model'],
        'eigmode': params['eigmode'],
        'sim': params['sim'],
        'pulse': params['pulse'],
        'sim_variants': params['sim_variants'],
    })
    analysis_payload = _json_ready({
        'action_flags': params['action_flags'],
        'derived_signals': params['derived_signals'],
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
    tau_decay = params['eigmode']['tau_decay']
    pulse_amp_e = params['pulse']['pulse_amp_e']
    pulse_amp_i = params['pulse']['pulse_amp_i']
    seed = params['sim']['seed']
    sigma_h = params['model']['sigma_h']
    phase_drop_win = _format_drop_win_filename(params['phaseproc']['drop_win'])
    run_dir = ARTIFACT_ROOT / (
        #f"res_T_{T:g}_amp_{pulse_amp_e:g}_{pulse_amp_i:g}_seed_{seed}_{sim_hash}"
        f"res_tau_{tau_decay}_amp_{pulse_amp_e:g}_{pulse_amp_i:g}_"
        f"noise_{sigma_h}_seed_{seed}_T_{T:g}_{sim_hash}"
    )

    # Build the cache and analysis-specific output file paths
    return {
        'dir': run_dir,
        'sim_cache': run_dir / 'sim_cache.npz',
        'signals_png': run_dir / (
            f'signals_{phase_drop_win}_{analysis_hash}.png'
        ),
        'signal_vars_png': run_dir / (
            f'signal_vars_{analysis_hash}.png'
        ),
        'phase_png': run_dir / (
            f'itc_{phase_drop_win}_{analysis_hash}.png'
        ),
        'phase_sweep_png': run_dir / (
            f'itc_wvar_{analysis_hash}.png'
        ),
        'power_sweep_png': run_dir / (
            f'pow_wvar_{analysis_hash}.png'
        ),
        'power_fband_png': run_dir / (
            f'pow_fband_{analysis_hash}.png'
        ),
        'phase_distr_png': run_dir / (
            f'phase_distr_{analysis_hash}.png'
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


def get_simulated_variant_sources(params, artifact_paths):
    """Load or run all simulated variant sources. """

    # Build each requested simulation variant
    tt_pulse_ref = None
    variant_sources = {}
    for spec in _get_variant_specs(params):
        variant_key = spec['key']
        variant_params = _make_variant_params(params, variant_key)
        sim_cache_path = _get_sim_cache_path(artifact_paths, params, variant_key)

        # Reuse the variant cache when it already exists
        if sim_cache_path.exists():
            R, tt_pulse, J = load_sim_cache(sim_cache_path)
            cache_hit = True
        else:
            R, tt_pulse, J = run_model(variant_params)
            save_sim_cache(sim_cache_path, R, tt_pulse, J)
            cache_hit = False

        # Verify that all variants share the same pulse times
        tt_pulse = np.asarray(tt_pulse, dtype=float)
        if tt_pulse_ref is None:
            tt_pulse_ref = tt_pulse
        elif not np.array_equal(tt_pulse, tt_pulse_ref):
            raise ValueError('Simulation variants must share identical pulse times')

        variant_sources[variant_key] = {
            'key': variant_key,
            'label': spec['label'],
            'color': spec['color'],
            'params': variant_params,
            'R': R,
            'tt_pulse': tt_pulse_ref,
            'epoch_plot_ref': np.asarray(variant_params['model']['r0'], dtype=float),
            'J': J,
            'cache_hit': cache_hit,
            'sim_cache': sim_cache_path,
        }
    return variant_sources


def add_derived_variant_sources(variant_sources, params):
    """Append derived variant sources before downstream analyses. """

    if not params['sim_variants']['enabled']:
        return variant_sources

    def centered_signal(variant_source):
        # Center one simulated source by its steady-state reference
        r0 = np.asarray(variant_source['params']['model']['r0'], dtype=float)[:, None]
        return variant_source['R'] - r0

    variant_sources = dict(variant_sources)
    for spec in _get_selected_derived_signal_specs(params):
        terms = [(variant_sources[key], coef) for key, coef in spec['terms']]
        tt_pulse_ref = np.asarray(terms[0][0]['tt_pulse'], dtype=float)
        for source, _ in terms[1:]:
            if not np.array_equal(source['tt_pulse'], tt_pulse_ref):
                raise ValueError('Derived variants must share identical pulse times')

        R_derived = None
        epoch_plot_ref = None
        for source, coef in terms:
            if params['derived_signals']['subtract_r0_before_combine']:
                term_signal = centered_signal(source) * coef
            else:
                term_signal = source['R'] * coef
                ref_term = np.asarray(source['epoch_plot_ref'], dtype=float) * coef
                epoch_plot_ref = ref_term if epoch_plot_ref is None else (epoch_plot_ref + ref_term)
            R_derived = term_signal if R_derived is None else (R_derived + term_signal)

        variant_sources[spec['key']] = {
            'key': spec['key'],
            'label': spec['label'],
            'color': spec['color'],
            'params': copy.deepcopy(params),
            'R': R_derived,
            'tt_pulse': tt_pulse_ref,
            'epoch_plot_ref': (
                np.zeros_like(params['model']['r0'], dtype=float)
                if params['derived_signals']['subtract_r0_before_combine']
                else epoch_plot_ref
            ),
            'derived_from': [key for key, _ in spec['terms']],
        }
    return variant_sources


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

    if epoch_start_idx.size == 0:
        epochs_mean = {}
        for key, X in signals.items():
            epoch_shape = (X.sizes['pop'], epoch_k1 - epoch_k0)
            epoch_mean_values = np.full(epoch_shape, np.nan, dtype=float)
            epoch_template = X.isel(time=slice(0, epoch_k1 - epoch_k0)).copy(
                data=epoch_mean_values
            )
            epoch_template = epoch_template.assign_coords(time=tt_epoch)
            epochs_mean[key] = epoch_template
        return tt_epoch, epochs_mean, 0

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


def _compute_phase_itc_for_signal(signal, tt_stim, phaseproc, drop_win):
    # Compute phases and ITC for one 1D signal with the selected method
    if phaseproc['method'] == 'morlet':
        return itc(
            signal.values,
            signal.coords['time'].values,
            tt_stim,
            phaseproc['f'],
            phaseproc['n_cycles'],
            drop_win=drop_win,
            subtract_local_mean=phaseproc['subtract_local_mean'],
            exclude_edges=phaseproc['exclude_edges'],
            min_valid_mass_frac=phaseproc['min_valid_mass_frac'],
        )
    return sinusoid_fit_itc(
        signal.values,
        signal.coords['time'].values,
        tt_stim,
        phaseproc['f'],
        phaseproc['n_cycles'],
        drop_win=drop_win,
        fit_intercept=phaseproc['fit_intercept'],
        exclude_edges=phaseproc['exclude_edges'],
        min_valid_mass_frac=phaseproc['min_valid_mass_frac'],
        rcond=phaseproc['rcond'],
    )


def analyze_phase_itc(R_signals, tt_pulse, params):
    """Compute stimulus-locked phases and ITC."""

    # Unpack the phase-analysis parameters
    phaseproc = _get_active_phaseproc(params['phaseproc'])
    signal_key, pop_idx = (phaseproc[p] for p in ['signal', 'pop'])
    signal = R_signals[signal_key].isel(pop=pop_idx)
    drop_win = _normalize_drop_win(phaseproc['drop_win'])

    # Compute the stimulus-locked phases
    itc_value, phi, phases = _compute_phase_itc_for_signal(
        signal, tt_pulse, phaseproc, drop_win
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
        itc_val, phi_val, phases = _compute_phase_itc_for_signal(
            signal, tt_pulse, phaseproc, drop_win
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


def analyze_phase_distr(R_signals, tt_pulse, params):
    """Compute pulse-time phase samples for both populations. """

    phaseproc = _get_active_phaseproc(params['phaseproc'])
    signal_key = phaseproc['signal']
    signal = R_signals[signal_key]
    drop_win = _normalize_drop_win(phaseproc['drop_win'])
    _, _, phases_e = _compute_phase_itc_for_signal(
        signal.isel(pop=0), tt_pulse, phaseproc, drop_win
    )
    _, _, phases_i = _compute_phase_itc_for_signal(
        signal.isel(pop=1), tt_pulse, phaseproc, drop_win
    )

    result = {
        'method': phaseproc['method'],
        'signal_key': signal_key,
        'f': phaseproc['f'],
        'n_cycles': phaseproc['n_cycles'],
        'exclude_edges': phaseproc['exclude_edges'],
        'min_valid_mass_frac': phaseproc['min_valid_mass_frac'],
        'drop_win': drop_win,
        'tt_stim': np.asarray(tt_pulse, dtype=float),
        'phases_e': np.asarray(phases_e, dtype=float),
        'phases_i': np.asarray(phases_i, dtype=float),
        'xlim': tuple(map(float, phaseproc['phase_distr_xlim'])),
        'bins': phaseproc['phase_distr_bins'],
        'smooth_sigma': float(phaseproc['phase_distr_smooth_sigma']),
    }
    if phaseproc['method'] == 'morlet':
        result['subtract_local_mean'] = phaseproc['subtract_local_mean']
    else:
        result['fit_intercept'] = phaseproc['fit_intercept']
        result['rcond'] = phaseproc['rcond']
    return result


def analyze_power_fband(R_signals, tt_pulse, params):
    """Summarize sliding power over a frequency band for E and I. """

    phaseproc = _get_active_phaseproc(params['phaseproc'])
    signal_key = phaseproc['signal']
    signal = R_signals[signal_key]
    tt = signal.coords['time'].values
    f0 = float(phaseproc['f'])
    fband = float(phaseproc['power_fband'])
    df = float(phaseproc['power_df'])
    drop_win = _normalize_drop_win(phaseproc['drop_win'])
    power_curves = []
    ff = None

    # Compute one mean power-vs-frequency curve per population
    for pop_idx in [0, 1]:
        x_masked = _apply_drop_win_mask(
            signal.isel(pop=pop_idx).values, tt, tt_pulse, drop_win
        )
        if phaseproc['method'] == 'morlet':
            _, ff_pop, power = morlet_power_freqs(
                x_masked,
                tt,
                f0,
                fband,
                df,
                phaseproc['n_cycles'],
                overlap=phaseproc['power_overlap'],
                power_mode=phaseproc['power_mode'],
                subtract_local_mean=phaseproc['subtract_local_mean'],
                exclude_edges=phaseproc['exclude_edges'],
                min_valid_mass_frac=phaseproc['min_valid_mass_frac'],
            )
        else:
            _, ff_pop, power = sinusoid_fit_power_freqs(
                x_masked,
                tt,
                f0,
                fband,
                df,
                phaseproc['n_cycles'],
                overlap=phaseproc['power_overlap'],
                fit_intercept=phaseproc['fit_intercept'],
                exclude_edges=phaseproc['exclude_edges'],
                min_valid_mass_frac=phaseproc['min_valid_mass_frac'],
                rcond=phaseproc['rcond'],
            )
        if ff is None:
            ff = ff_pop
        power_curves.append(np.nanmean(power, axis=1))

    result = {
        'method': phaseproc['method'],
        'signal_key': signal_key,
        'f0': f0,
        'fband': fband,
        'df': df,
        'ff': np.asarray(ff, dtype=float),
        'power_e': np.asarray(power_curves[0], dtype=float),
        'power_i': np.asarray(power_curves[1], dtype=float),
        'power_overlap': phaseproc['power_overlap'],
        'drop_win': drop_win,
        'exclude_edges': phaseproc['exclude_edges'],
        'min_valid_mass_frac': phaseproc['min_valid_mass_frac'],
    }
    if phaseproc['method'] == 'morlet':
        result['subtract_local_mean'] = phaseproc['subtract_local_mean']
        result['power_mode'] = phaseproc['power_mode']
    else:
        result['fit_intercept'] = phaseproc['fit_intercept']
        result['rcond'] = phaseproc['rcond']
    return result


def analyze_power_drop_sweep(R_signals, tt_pulse, params):
    """Sweep symmetric drop windows and summarize sliding-window power. """

    # Unpack the power-analysis parameters
    phaseproc = _get_active_phaseproc(params['phaseproc'])
    signal_key = phaseproc['signal']
    signal = R_signals[signal_key]
    tt = signal.coords['time'].values
    widths = np.asarray(phaseproc['drop_win_sweep'], dtype=float).ravel()
    power_e = np.full(widths.shape, np.nan, dtype=float)
    power_i = np.full(widths.shape, np.nan, dtype=float)

    # Evaluate the sliding-window power summary for each population and width
    for i, width in enumerate(widths):
        drop_win = (-width, width)
        for pop_idx, power_out in enumerate([power_e, power_i]):
            x_masked = _apply_drop_win_mask(
                signal.isel(pop=pop_idx).values, tt, tt_pulse, drop_win
            )
            if phaseproc['method'] == 'morlet':
                _, power_trace = morlet_power(
                    x_masked,
                    tt,
                    phaseproc['f'],
                    phaseproc['n_cycles'],
                    overlap=phaseproc['power_overlap'],
                    power_mode=phaseproc['power_mode'],
                    subtract_local_mean=phaseproc['subtract_local_mean'],
                    exclude_edges=phaseproc['exclude_edges'],
                    min_valid_mass_frac=phaseproc['min_valid_mass_frac'],
                )
            else:
                _, power_trace = sinusoid_fit_power(
                    x_masked,
                    tt,
                    phaseproc['f'],
                    phaseproc['n_cycles'],
                    overlap=phaseproc['power_overlap'],
                    fit_intercept=phaseproc['fit_intercept'],
                    exclude_edges=phaseproc['exclude_edges'],
                    min_valid_mass_frac=phaseproc['min_valid_mass_frac'],
                    rcond=phaseproc['rcond'],
                )
            power_out[i] = np.nanmean(power_trace)

    # Package the power sweep outputs
    result = {
        'method': phaseproc['method'],
        'signal_key': signal_key,
        'f': phaseproc['f'],
        'n_cycles': phaseproc['n_cycles'],
        'power_overlap': phaseproc['power_overlap'],
        'exclude_edges': phaseproc['exclude_edges'],
        'min_valid_mass_frac': phaseproc['min_valid_mass_frac'],
        'drop_widths': widths,
        'power_e': power_e,
        'power_i': power_i,
    }
    if phaseproc['method'] == 'morlet':
        result['subtract_local_mean'] = phaseproc['subtract_local_mean']
        result['power_mode'] = phaseproc['power_mode']
    else:
        result['fit_intercept'] = phaseproc['fit_intercept']
        result['rcond'] = phaseproc['rcond']
    return result


def process_variant_source(variant_source, params):
    """Run all downstream analyses for one variant source. """

    variant_params = variant_source.get('params', params)
    action_flags = params['action_flags']
    tt_pulse_analysis = _filter_analysis_pulse_times(variant_source['tt_pulse'], params)

    # Package the fully processed variant result
    variant_result = dict(variant_source)
    variant_result['tt_pulse_analysis'] = tt_pulse_analysis
    needs_signals = any(
        bool(action_flags[key])
        for key in [
            'signals', 'signal_vars', 'itc', 'itc_wvar',
            'pow_wvar', 'pow_fband', 'phase_distr'
        ]
    )

    if needs_signals:
        # Build the raw and filtered signal views once for all enabled analyses
        R_filt = filt_signal(variant_source['R'], variant_params)
        R_signals = {'raw': variant_source['R'], 'filt': R_filt}
        variant_result['R_signals'] = R_signals

        if action_flags['signals']:
            tt_epoch, epochs_mean, n_epochs = epoch_signals(
                R_signals, tt_pulse_analysis, variant_params
            )
            variant_result['tt_epoch'] = tt_epoch
            variant_result['epochs_mean'] = epochs_mean
            variant_result['n_epochs'] = n_epochs

        if action_flags['itc']:
            variant_result['phase_results'] = analyze_phase_itc(
                R_signals, tt_pulse_analysis, params
            )

        if action_flags['itc_wvar']:
            variant_result['phase_sweep_results'] = analyze_phase_itc_drop_sweep(
                R_signals, tt_pulse_analysis, params
            )

        if action_flags['pow_wvar']:
            variant_result['power_sweep_results'] = analyze_power_drop_sweep(
                R_signals, tt_pulse_analysis, params
            )

        if action_flags['pow_fband']:
            variant_result['power_fband_results'] = analyze_power_fband(
                R_signals, tt_pulse_analysis, params
            )

        if action_flags['phase_distr']:
            variant_result['phase_distr_results'] = analyze_phase_distr(
                R_signals, tt_pulse_analysis, params
            )
    return variant_result


def process_variant_sources(variant_sources, params):
    """Run the shared processing pipeline over all variant sources. """

    return {
        key: process_variant_source(variant_source, params)
        for key, variant_source in variant_sources.items()
    }


def plot_results(variant_results, params, artifact_paths):
    """Plot the full time series and pulse-triggered averages."""

    # Unpack the plot labels and references
    r0 = params['model']['r0']
    f, tau_decay = (params['eigmode'][p] for p in ['f', 'tau_decay'])
    raw_zoom_t_win = params['sigproc']['raw_zoom_t_win']
    epoch_plot_subtract_r0 = bool(params['sigproc']['epoch_plot_subtract_r0'])
    drop_win = _normalize_drop_win(params['phaseproc']['drop_win'])
    primary = variant_results[_get_primary_variant_key()]
    primary_signals = primary['R_signals']
    tt_epoch = primary['tt_epoch']
    raw_signal = primary_signals['raw']
    tt = raw_signal.coords['time'].values

    # Build the plot layout
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))

    # Plot the full raw timecourse
    ax[0, 0].plot(tt, raw_signal.isel(pop=0), label='Excitatory')
    ax[0, 0].plot(tt, raw_signal.isel(pop=1), label='Inhibitory')
    ax[0, 0].axhline(r0[0], color='C0', linestyle='--', alpha=0.3)
    ax[0, 0].axhline(r0[1], color='C1', linestyle='--', alpha=0.3)
    ax[0, 0].set_xlabel('Time')
    ax[0, 0].set_ylabel('Rate')
    ax[0, 0].set_title(f'E-I oscillations (f={f} Hz, τ_decay={tau_decay})')
    ax[0, 0].legend()

    # Plot a zoomed chunk of the raw timecourse
    zoom_mask = (tt >= raw_zoom_t_win[0]) & (tt <= raw_zoom_t_win[1])
    ax[1, 0].plot(tt[zoom_mask], raw_signal.isel(pop=0).values[zoom_mask], label='Excitatory')
    ax[1, 0].plot(tt[zoom_mask], raw_signal.isel(pop=1).values[zoom_mask], label='Inhibitory')
    ax[1, 0].axhline(r0[0], color='C0', linestyle='--', alpha=0.3)
    ax[1, 0].axhline(r0[1], color='C1', linestyle='--', alpha=0.3)
    ax[1, 0].set_xlabel('Time')
    ax[1, 0].set_ylabel('Rate')
    ax[1, 0].set_title(f'Raw zoom ({raw_zoom_t_win[0]:g}, {raw_zoom_t_win[1]:g})')
    ax[1, 0].legend()

    # Plot the raw pulse-triggered averages separately for E and I
    epoch_ylabel = 'Rate - r0' if epoch_plot_subtract_r0 else 'Rate'
    for pop_idx, axis in enumerate([ax[0, 1], ax[1, 1]]):
        for variant in variant_results.values():
            epoch_trace = variant['epochs_mean']['raw'].isel(pop=pop_idx).values
            if epoch_plot_subtract_r0:
                epoch_trace = epoch_trace - variant['epoch_plot_ref'][pop_idx]
            axis.plot(
                tt_epoch,
                epoch_trace,
                color=variant['color'],
                label=variant['label'],
            )
        axis.axvspan(drop_win[0], drop_win[1], color='0.5', alpha=0.15)
        axis.axvline(0, color='k', linestyle='--', alpha=0.5)
        if epoch_plot_subtract_r0:
            axis.axhline(0, color='k', linestyle='--', alpha=0.3)
        else:
            axis.axhline(r0[pop_idx], color=f'C{pop_idx}', linestyle='--', alpha=0.3)
        axis.set_xlabel('Time from pulse')
        axis.set_ylabel(epoch_ylabel)
        axis.set_title(
            f"{'E' if pop_idx == 0 else 'I'} epoch average over {primary['n_epochs']} pulses"
        )
        axis.legend(fontsize=8)

    # Save the signals and epochs figure
    artifact_paths['dir'].mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(artifact_paths['signals_png'], dpi=150)
    plt.close(fig)


def plot_variant_signals_results(variant_results, params, artifact_paths):
    """Plot one raw time-window panel per active variant. """

    variant_t_win = params['sigproc']['variant_t_win']
    t0, t1 = map(float, variant_t_win)
    variants = list(variant_results.values())
    n_variants = len(variants)

    fig, ax = plt.subplots(
        n_variants, 1, figsize=(14, max(2.6 * n_variants, 4.0)), sharex=True
    )
    if n_variants == 1:
        ax = [ax]

    # Plot one raw time window per variant with both populations
    for axis, variant in zip(ax, variants):
        signal = variant['R_signals']['raw']
        tt = signal.coords['time'].values
        keep = (tt >= t0) & (tt <= t1)
        axis.plot(
            tt[keep], signal.isel(pop=0).values[keep],
            color='C0', linestyle='-', label='E'
        )
        axis.plot(
            tt[keep], signal.isel(pop=1).values[keep],
            color='C1', linestyle='-', label='I'
        )
        axis.set_ylabel('Rate')
        axis.set_title(variant['label'])
        axis.legend(loc='upper right', fontsize=8)

    ax[-1].set_xlabel('Time')
    ax[-1].set_xlim(t0, t1)
    artifact_paths['dir'].mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(artifact_paths['signal_vars_png'], dpi=150)
    plt.close(fig)


def plot_phase_itc_results(variant_results, artifact_paths):
    """Plot the stimulus phases and complex-plane ITC summary."""

    # Unpack the phase-analysis outputs
    phase_results_ref = variant_results[_get_primary_variant_key()]['phase_results']
    signal_key = phase_results_ref['signal_key']
    pop_idx = phase_results_ref['pop']
    method = phase_results_ref['method']
    phase_progression_plot = phase_results_ref['phase_progression_plot']
    drop_win = phase_results_ref['drop_win']

    # Collect the y-limits needed across variants
    y_min = -2.0 * np.pi
    y_max = 2.0 * np.pi
    if phase_progression_plot == 'unwrapped':
        phase_prog_all = []
        for variant in variant_results.values():
            phases = np.asarray(variant['phase_results']['phases'], dtype=float)
            valid = np.isfinite(phases)
            phases_valid = phases[valid]
            if phases_valid.size:
                phase_prog = np.unwrap(phases_valid)
                phase_prog += phases_valid[0] - phase_prog[0]
                phase_prog_all.append(phase_prog)
        if phase_prog_all:
            y_min = min(float(np.min(phase_prog)) for phase_prog in phase_prog_all)
            y_max = max(float(np.max(phase_prog)) for phase_prog in phase_prog_all)

    # Build the plot layout
    fig, ax = plt.subplots(
        1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [3, 1.3]}
    )

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

    # Plot the phase progression at stimulus times for all variants
    for variant in variant_results.values():
        phase_results = variant['phase_results']
        tt_stim = phase_results['tt_stim']
        phases = np.asarray(phase_results['phases'], dtype=float)
        valid = np.isfinite(phases)
        tt_valid = tt_stim[valid]
        phases_valid = phases[valid]
        if phase_progression_plot == 'wrapped_strip':
            xx, yy = _phase_progression_strip_coords(
                tt_valid, phases_valid, y_min, y_max
            )
            ax[0].plot(
                xx, yy, linestyle='None', marker='o', markersize=2.5,
                color=variant['color'], label=variant['label'],
            )
        elif phases_valid.size:
            phase_prog = np.unwrap(phases_valid)
            phase_prog += phases_valid[0] - phase_prog[0]
            ax[0].plot(
                tt_valid, phase_prog, '-', color=variant['color'],
                label=variant['label'],
            )

    ax[0].set_xlabel('Stimulus time')
    ax[0].set_ylabel('Phase progression')
    ax[0].set_title(
        f'Phase progression at stimuli ({signal_key}, pop {pop_idx}, {method}, {phase_progression_plot})'
    )
    ax[0].set_ylim(y_min, y_max)
    ax[0].legend()

    # Plot the phases on the complex plane for all variants
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax[1].plot(np.cos(theta), np.sin(theta), color='0.75')
    for variant in variant_results.values():
        phase_results = variant['phase_results']
        phases = np.asarray(phase_results['phases'], dtype=float)
        valid = np.isfinite(phases)
        phases_valid = phases[valid]
        phi = phase_results['phi']
        itc_value = phase_results['itc']
        ax[1].scatter(
            np.cos(phases_valid), np.sin(phases_valid),
            color=variant['color'], alpha=0.1,
        )
        if np.isfinite(phi):
            ax[1].plot(
                [0.0, np.cos(phi)],
                [0.0, np.sin(phi)],
                color=variant['color'],
                linestyle='--',
                linewidth=1.2,
            )
        if np.isfinite(phi) and np.isfinite(itc_value):
            ax[1].plot(
                [0.0, itc_value * np.cos(phi)],
                [0.0, itc_value * np.sin(phi)],
                color=variant['color'],
                linewidth=2.0,
                label=variant['label'],
            )
    ax[1].set_aspect('equal', 'box')
    ax[1].set_xlim(-1.1, 1.1)
    ax[1].set_ylim(-1.1, 1.1)
    ax[1].set_xlabel('Real')
    ax[1].set_ylabel('Imag')
    ax[1].set_title(f'Complex plane ITC ({method})')
    ax[1].legend(loc='upper right', fontsize=8)

    # Add the phase-analysis summary to the figure
    summary = [
        f'method = {method}',
        f'phase_plot = {phase_progression_plot}',
        f'phase_drop_win = {drop_win}',
    ]
    if len(variant_results) > 1:
        summary.append('sim_variants = enabled')
    if method == 'morlet':
        summary.append(
            f"subtract_local_mean = {bool(phase_results_ref['subtract_local_mean'])}"
        )
    else:
        summary.append(f"fit_intercept = {bool(phase_results_ref['fit_intercept'])}")
    fig.text(0.12, 0.02, '    '.join(summary), family='monospace')

    # Save the phase and ITC figure
    artifact_paths['dir'].mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(artifact_paths['phase_png'], dpi=150)
    plt.close(fig)


def plot_phase_itc_sweep_results(variant_results, artifact_paths):
    """Plot ITC and mean phase versus symmetric drop width."""

    # Unpack the sweep outputs
    sweep_results_ref = variant_results[_get_primary_variant_key()]['phase_sweep_results']
    widths = np.asarray(sweep_results_ref['drop_widths'], dtype=float)
    signal_key = sweep_results_ref['signal_key']
    pop_idx = sweep_results_ref['pop']
    method = sweep_results_ref['method']

    # Build the plot layout
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    # Plot the ITC and mean phase sweeps for all variants
    for variant in variant_results.values():
        sweep_results = variant['phase_sweep_results']
        ax[0].plot(
            widths, np.asarray(sweep_results['itc'], dtype=float),
            'o-', color=variant['color'], label=variant['label'],
        )
        ax[1].plot(
            widths, np.asarray(sweep_results['phi'], dtype=float),
            'o-', color=variant['color'], label=variant['label'],
        )
    ax[0].set_xlabel('Symmetric drop half-width')
    ax[0].set_ylabel('ITC')
    ax[0].set_title(f'ITC vs drop width ({signal_key}, pop {pop_idx}, {method})')
    ax[0].legend()

    ax[1].set_xlabel('Symmetric drop half-width')
    ax[1].set_ylabel('Mean phase (rad)')
    ax[1].set_title(f'Mean phase vs drop width ({method})')
    ax[1].legend()

    # Save the sweep figure
    artifact_paths['dir'].mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(artifact_paths['phase_sweep_png'], dpi=150)
    plt.close(fig)


def plot_power_sweep_results(variant_results, artifact_paths):
    """Plot power versus symmetric drop width for E and I populations. """

    # Unpack the power-sweep metadata
    power_results_ref = variant_results[_get_primary_variant_key()]['power_sweep_results']
    widths = np.asarray(power_results_ref['drop_widths'], dtype=float)
    signal_key = power_results_ref['signal_key']
    method = power_results_ref['method']

    # Build the plot layout
    fig, ax = plt.subplots(2, 1, figsize=(7, 8.5))
    pop_specs = [
        ('power_e', 'Excitatory power vs drop width'),
        ('power_i', 'Inhibitory power vs drop width'),
    ]

    # Plot the power sweeps for all variants
    for axis, (power_key, title) in zip(ax, pop_specs):
        for variant in variant_results.values():
            power_results = variant['power_sweep_results']
            axis.plot(
                widths,
                np.asarray(power_results[power_key], dtype=float),
                'o-',
                color=variant['color'],
                label=variant['label'],
            )
        axis.set_xlabel('Symmetric drop half-width')
        axis.set_ylabel('Power')
        axis.set_title(f'{title} ({signal_key}, {method})')
        axis.legend()

    # Save the power sweep figure
    artifact_paths['dir'].mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(artifact_paths['power_sweep_png'], dpi=150)
    plt.close(fig)


def plot_power_fband_results(variant_results, artifact_paths):
    """Plot power versus frequency for E and I populations. """

    power_ref = variant_results[_get_primary_variant_key()]['power_fband_results']
    ff = np.asarray(power_ref['ff'], dtype=float)
    signal_key = power_ref['signal_key']
    method = power_ref['method']
    f0 = float(power_ref['f0'])
    fband = float(power_ref['fband'])
    df = float(power_ref['df'])
    drop_win = power_ref['drop_win']

    fig, ax = plt.subplots(2, 1, figsize=(7, 8.5))
    pop_specs = [
        ('power_e', 'Excitatory power vs frequency'),
        ('power_i', 'Inhibitory power vs frequency'),
    ]

    # Plot the mean power-vs-frequency curves for all variants
    for axis, (power_key, title) in zip(ax, pop_specs):
        for variant in variant_results.values():
            power_results = variant['power_fband_results']
            axis.plot(
                ff,
                np.asarray(power_results[power_key], dtype=float),
                'o-',
                color=variant['color'],
                label=variant['label'],
            )
        axis.set_xlabel('Frequency')
        axis.set_ylabel('Mean power')
        axis.set_title(f'{title} ({signal_key}, {method})')
        axis.legend()

    fig.text(
        0.12, 0.02,
        (
            f'f0 = {f0:g}    fband = {fband:g}    df = {df:g}'
            f'    drop_win = {drop_win}'
        ),
        family='monospace'
    )
    artifact_paths['dir'].mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(artifact_paths['power_fband_png'], dpi=150)
    plt.close(fig)


def _replicate_phases_to_xlim(phases, xlim):
    # Replicate wrapped phases across the visible x-range
    phases = np.asarray(phases, dtype=float)
    phases = phases[np.isfinite(phases)]
    if phases.size == 0:
        return phases

    xmin, xmax = map(float, xlim)
    k_min = int(np.floor((xmin - np.max(phases)) / (2.0 * np.pi)))
    k_max = int(np.ceil((xmax - np.min(phases)) / (2.0 * np.pi)))
    phases_rep = [phases + 2.0 * np.pi * k for k in range(k_min, k_max + 1)]
    phases_rep = np.concatenate(phases_rep)
    keep = (phases_rep >= xmin) & (phases_rep <= xmax)
    return phases_rep[keep]


def plot_phase_distr_results(variant_results, artifact_paths):
    """Plot phase distributions at pulse times for E and I. """

    phase_distr_ref = variant_results[_get_primary_variant_key()]['phase_distr_results']
    xlim = phase_distr_ref['xlim']
    nbins = int(phase_distr_ref['bins'])
    smooth_sigma = float(phase_distr_ref['smooth_sigma'])
    bins = np.linspace(xlim[0], xlim[1], nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    method = phase_distr_ref['method']
    signal_key = phase_distr_ref['signal_key']

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    pop_specs = [('phases_e', 'E phase distribution'), ('phases_i', 'I phase distribution')]

    for axis, (phase_key, title) in zip(ax, pop_specs):
        for variant in variant_results.values():
            phase_distr = variant['phase_distr_results']
            phases_rep = _replicate_phases_to_xlim(phase_distr[phase_key], xlim)
            if phases_rep.size == 0:
                continue
            counts, _ = np.histogram(phases_rep, bins=bins)
            probs = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
            if smooth_sigma > 0:
                probs = gaussian_filter1d(probs, smooth_sigma, mode='nearest')
                if probs.sum() > 0:
                    probs = probs / probs.sum()
            axis.plot(centers, probs, color=variant['color'], label=variant['label'])
        axis.set_xlim(xlim)
        axis.set_xlabel('Phase (rad)')
        axis.set_ylabel('Probability')
        axis.set_title(f'{title} ({signal_key}, {method}, sigma={smooth_sigma:g})')
        axis.legend(fontsize=8)

    artifact_paths['dir'].mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(artifact_paths['phase_distr_png'], dpi=150)
    plt.close(fig)


def main():
    # Define params
    params = create_params()
    action_flags = params['action_flags']
    artifact_paths = build_artifact_paths(params)

    # Save the simulation and analysis parameter subsets
    save_json(artifact_paths['sim_params_json'], artifact_paths['sim_payload'])
    save_json(artifact_paths['analysis_params_json'], artifact_paths['analysis_payload'])

    # Build the pre-analysis variant sources
    variant_sources = get_simulated_variant_sources(params, artifact_paths)
    variant_sources = add_derived_variant_sources(variant_sources, params)

    # Run the shared downstream processing pipeline
    variant_results = process_variant_sources(variant_sources, params)

    # Plot the signals, phases, and sweep results
    if action_flags['signals']:
        plot_results(variant_results, params, artifact_paths)
    if action_flags['signal_vars']:
        plot_variant_signals_results(variant_results, params, artifact_paths)
    if action_flags['itc']:
        plot_phase_itc_results(variant_results, artifact_paths)
    if action_flags['itc_wvar']:
        plot_phase_itc_sweep_results(variant_results, artifact_paths)
    if action_flags['pow_wvar']:
        plot_power_sweep_results(variant_results, artifact_paths)
    if action_flags['pow_fband']:
        plot_power_fband_results(variant_results, artifact_paths)
    if action_flags['phase_distr']:
        plot_phase_distr_results(variant_results, artifact_paths)

    # Report the saved artifact paths
    for variant in variant_results.values():
        if 'sim_cache' not in variant:
            continue
        cache_msg = 'Loaded' if variant['cache_hit'] else 'Saved'
        print(f"{cache_msg} {variant['sim_cache']}")
    if action_flags['signals']:
        print(f"Saved {artifact_paths['signals_png']}")
    if action_flags['signal_vars']:
        print(f"Saved {artifact_paths['signal_vars_png']}")
    if action_flags['itc']:
        print(f"Saved {artifact_paths['phase_png']}")
    if action_flags['itc_wvar']:
        print(f"Saved {artifact_paths['phase_sweep_png']}")
    if action_flags['pow_wvar']:
        print(f"Saved {artifact_paths['power_sweep_png']}")
    if action_flags['pow_fband']:
        print(f"Saved {artifact_paths['power_fband_png']}")
    if action_flags['phase_distr']:
        print(f"Saved {artifact_paths['phase_distr_png']}")
    print(f"Saved {artifact_paths['sim_params_json']}")
    print(f"Saved {artifact_paths['analysis_params_json']}")


if __name__ == '__main__':
    main()
