import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '.matplotlib')

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rate_model.model_utils import (
    rate_phi_bounds,
    make_W_h0_ei_2d_syn,
)
from rate_model import RateModelWC


BOUND_NAMES = ('lower', 'upper')
RANGE_KEYS = ('taue_syn', 'taui_syn', 'tau_decay')
DEFAULT_ARTIFACT_DIR = (
    Path(__file__).resolve().parent / 'artifacts' / Path(__file__).stem
)
DEFAULT_RANGES = {
    'taue_syn': np.linspace(1e-3, 100e-3, 15),
    'taui_syn': np.linspace(1e-3, 100e-3, 15),
    'tau_decay': np.array([0.1, 1, 10]),
}


def _as_1d_float_array(values):
    return np.asarray(values, dtype=float).ravel()


def _validate_model_inputs(model, r0, f):
    r0_arr = np.asarray(r0, dtype=float).reshape(-1, 1)
    if model.npops != 2:
        raise ValueError('model must have 2 populations')
    if not np.allclose(model.tau, 0.0):
        raise ValueError('model.tau must be zero')
    if f < 0:
        raise ValueError('f must be nonnegative')
    return r0_arr


def _validate_ranges(ranges):
    return {key: _as_1d_float_array(ranges[key]) for key in RANGE_KEYS}

def compute_phi_ie_bounds(model, r0, f, ranges):
    r0_arr = _validate_model_inputs(model, r0, f)
    ranges = _validate_ranges(ranges)
    data = np.full(
        (
            len(BOUND_NAMES),
            ranges['taue_syn'].size,
            ranges['taui_syn'].size,
            ranges['tau_decay'].size,
        ),
        np.nan,
        dtype=float,
    )

    for i_taue, taue_syn in enumerate(ranges['taue_syn']):
        for i_taui, taui_syn in enumerate(ranges['taui_syn']):
            for i_decay, tau_decay in enumerate(ranges['tau_decay']):
                lower, upper = rate_phi_bounds(
                    np.array([taue_syn, taui_syn], dtype=float), f, tau_decay
                )
                data[0, i_taue, i_taui, i_decay] = lower
                data[1, i_taue, i_taui, i_decay] = upper

    return xr.DataArray(
        data,
        dims=('bound', 'taue_syn', 'taui_syn', 'tau_decay'),
        coords={
            'bound': list(BOUND_NAMES),
            'taue_syn': ranges['taue_syn'],
            'taui_syn': ranges['taui_syn'],
            'tau_decay': ranges['tau_decay'],
        },
        attrs={
            'description': 'Allowed rate-space phi_ie bounds',
            'phi_ie_convention': 'lower may be below 0 when the interval wraps',
            'r0': r0_arr[:, 0].tolist(),
            'f': float(f),
        },
    )


def _format_token(value):
    return f'{float(value):g}'


def _serialize_model(model):
    return {
        'npops': int(model.npops),
        'W': np.asarray(model.W, dtype=float).tolist(),
        'tau': np.asarray(model.tau, dtype=float).tolist(),
        'tau_syn': np.asarray(model.tau_syn, dtype=float).tolist(),
        'rmax': np.asarray(model.rmax, dtype=float).tolist(),
        'gain_slope': np.asarray(model.gain_slope, dtype=float).tolist(),
        'gain_center': np.asarray(model.gain_center, dtype=float).tolist(),
    }


def _build_payload(model, r0, f, ranges, phi_bounds):
    return {
        'model': _serialize_model(model),
        'r0': np.asarray(r0, dtype=float).ravel().tolist(),
        'f': float(f),
        'ranges': {
            key: np.asarray(ranges[key], dtype=float).ravel().tolist()
            for key in RANGE_KEYS
        },
        'phi_bounds': {
            'dims': list(phi_bounds.dims),
            'coords': {
                key: np.asarray(phi_bounds.coords[key]).tolist()
                for key in phi_bounds.dims
            },
            'values': np.asarray(phi_bounds.values, dtype=float).tolist(),
            'attrs': dict(phi_bounds.attrs),
        },
    }


def _build_stem(r0, f, phi_bounds, payload):
    dims_token = '_'.join(str(phi_bounds.sizes[key]) for key in phi_bounds.dims)
    r0_token = '_'.join(_format_token(val) for val in np.asarray(r0).ravel())
    payload_hash = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode('utf-8')
    ).hexdigest()[:8]
    return (
        f'phi_maps_r0_{r0_token}_f_{_format_token(f)}'
        f'_dim_{dims_token}_{payload_hash}'
    )


def _plot_phi_bounds(phi_bounds):
    tau_decay_vals = phi_bounds.coords['tau_decay'].values
    taue_vals = phi_bounds.coords['taue_syn'].values
    taui_vals = phi_bounds.coords['taui_syn'].values

    fig, axes = plt.subplots(
        len(BOUND_NAMES),
        tau_decay_vals.size,
        figsize=(4.6 * tau_decay_vals.size, 4.2 * len(BOUND_NAMES)),
        squeeze=False,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='#d9d9d9')

    for i_bound, bound_name in enumerate(BOUND_NAMES):
        row_vals = np.asarray(phi_bounds.isel(bound=i_bound).values, dtype=float)
        row_vals = row_vals[np.isfinite(row_vals)]
        if row_vals.size:
            row_vmin = float(row_vals.min())
            row_vmax = float(row_vals.max())
            if np.isclose(row_vmin, row_vmax):
                row_vmax = row_vmin + 1e-9
        else:
            row_vmin = 0.0
            row_vmax = 1.0

        row_mesh = None
        for i_decay, tau_decay in enumerate(tau_decay_vals):
            ax = axes[i_bound, i_decay]
            panel = phi_bounds.isel(bound=i_bound, tau_decay=i_decay).transpose(
                'taue_syn', 'taui_syn'
            )
            row_mesh = ax.pcolormesh(
                taui_vals,
                taue_vals,
                panel.values,
                shading='auto',
                cmap=cmap,
                vmin=row_vmin,
                vmax=row_vmax,
            )
            ax.set_title(f'tau_decay={_format_token(tau_decay)}')
            ax.set_xlabel('taui_syn')
            ax.set_ylabel(f'{bound_name} phi_ie\ntaue_syn')

        fig.colorbar(
            row_mesh,
            ax=axes[i_bound, :].tolist(),
            label=f'{bound_name} phi_ie (rad)',
        )
    return fig


def save_demo_artifacts(output_dir=None, r0=None, f=10.0, ranges=None):
    r0_arr = np.array([5.0, 5.0] if r0 is None else r0, dtype=float)
    ranges = _validate_ranges(DEFAULT_RANGES if ranges is None else ranges)

    model = RateModelWC(
        np.zeros((2, 2)),
        tau=0.0,
        rmax=10.0,
        gain_slope=0.5,
        gain_center=0.0,
    )
    phi_bounds = compute_phi_ie_bounds(model, r0_arr, f, ranges)
    payload = _build_payload(model, r0_arr, f, ranges, phi_bounds)
    stem = _build_stem(r0_arr, f, phi_bounds, payload)

    output_path = DEFAULT_ARTIFACT_DIR if output_dir is None else Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig = _plot_phi_bounds(phi_bounds)
    png_path = output_path / f'{stem}.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    payload['artifact_dir'] = str(output_path)
    payload['artifact_stem'] = stem
    payload['png_path'] = str(png_path)
    payload['json_path'] = str(output_path / f'{stem}.json')

    json_path = output_path / f'{stem}.json'
    with json_path.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    return {
        'phi_bounds': phi_bounds,
        'output_dir': output_path,
        'png_path': png_path,
        'json_path': json_path,
        'payload': payload,
    }


def main():
    result = save_demo_artifacts()

    phi_bounds = result['phi_bounds']
    lower = float(phi_bounds.sel(bound='lower').isel(taue_syn=0, taui_syn=0, tau_decay=0))
    upper = float(phi_bounds.sel(bound='upper').isel(taue_syn=0, taui_syn=0, tau_decay=0))
    phi_ie = 0.5 * (lower + upper)

    if np.isfinite(phi_ie):
        make_W_h0_ei_2d_syn(
            RateModelWC(
                np.zeros((2, 2)),
                tau=0.0,
                rmax=10.0,
                gain_slope=0.5,
                gain_center=0.0,
            ),
            phi_bounds.attrs['r0'],
            np.array([
                float(phi_bounds.coords['taue_syn'][0]),
                float(phi_bounds.coords['taui_syn'][0]),
            ]),
            float(phi_bounds.attrs['f']),
            float(phi_bounds.coords['tau_decay'][0]),
            q_ie=1.0,
            phi_ie=phi_ie,
        )

    print(f"Saved {result['png_path']}")
    print(f"Saved {result['json_path']}")


if __name__ == '__main__':
    main()
