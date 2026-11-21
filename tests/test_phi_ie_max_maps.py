import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from rate_model.model_utils import (
    make_W_h0_ei_2d_syn,
    rate_phi_bounds,
)
from experiments.ei_modes.phi_ie_max_maps import (
    DEFAULT_RANGES,
    compute_phi_ie_bounds,
    main,
)
from rate_model import RateModelWC


class TestPhiIeMaxMaps(unittest.TestCase):

    def setUp(self):
        self.model = RateModelWC(
            np.zeros((2, 2)),
            tau=0.0,
            rmax=10.0,
            gain_slope=0.5,
            gain_center=0.0,
        )
        self.r0 = np.array([5.0, 5.0])

    def test_compute_phi_ie_bounds_shape_and_values(self):
        ranges = {
            'taue_syn': np.array([7e-3, 9e-3]),
            'taui_syn': np.array([2e-3, 4e-3]),
            'tau_decay': np.array([20.0]),
        }

        phi_bounds = compute_phi_ie_bounds(self.model, self.r0, 10.0, ranges)

        self.assertIsInstance(phi_bounds, xr.DataArray)
        self.assertEqual(
            phi_bounds.dims,
            ('bound', 'taue_syn', 'taui_syn', 'tau_decay'),
        )
        self.assertEqual(phi_bounds.shape, (2, 2, 2, 1))

        tau_syn = np.array([ranges['taue_syn'][0], ranges['taui_syn'][0]])
        lower_expected, upper_expected = rate_phi_bounds(tau_syn, 10.0, 20.0)

        self.assertAlmostEqual(
            phi_bounds.sel(bound='lower').values[0, 0, 0],
            lower_expected,
            places=12,
        )
        self.assertAlmostEqual(
            phi_bounds.sel(bound='upper').values[0, 0, 0],
            upper_expected,
            places=12,
        )

    def test_compute_phi_ie_bounds_has_non_wrap_case(self):
        ranges = {
            'taue_syn': np.array([20e-3]),
            'taui_syn': np.array([5e-3]),
            'tau_decay': np.array([1.0]),
        }

        phi_bounds = compute_phi_ie_bounds(self.model, self.r0, 10.0, ranges)
        lower = float(phi_bounds.sel(bound='lower').values[0, 0, 0])
        upper = float(phi_bounds.sel(bound='upper').values[0, 0, 0])
        lower_expected, upper_expected = rate_phi_bounds(
            np.array([20e-3, 5e-3]), 10.0, 1.0
        )

        self.assertLess(lower, upper)
        self.assertLessEqual(upper, np.pi)
        self.assertAlmostEqual(lower, lower_expected, places=12)
        self.assertAlmostEqual(upper, upper_expected, places=12)

    def test_compute_phi_ie_bounds_has_wrap_case(self):
        ranges = {
            'taue_syn': np.array([2.5e-3]),
            'taui_syn': np.array([100e-3]),
            'tau_decay': np.array([10.0]),
        }

        phi_bounds = compute_phi_ie_bounds(self.model, self.r0, 10.0, ranges)
        lower = float(phi_bounds.sel(bound='lower').values[0, 0, 0])
        upper = float(phi_bounds.sel(bound='upper').values[0, 0, 0])
        lower_expected, upper_expected = rate_phi_bounds(
            np.array([2.5e-3, 100e-3]), 10.0, 10.0
        )

        self.assertLess(lower, upper)
        self.assertLess(lower, 0.0)
        self.assertLessEqual(upper, np.pi)
        self.assertAlmostEqual(lower, lower_expected, places=12)
        self.assertAlmostEqual(upper, upper_expected, places=12)

    def test_phi_bounds_match_make_W_h0_boundary(self):
        ranges = {
            'taue_syn': np.array([5e-3]),
            'taui_syn': np.array([2e-3]),
            'tau_decay': np.array([20.0]),
        }
        tau_syn = np.array([ranges['taue_syn'][0], ranges['taui_syn'][0]])
        phi_bounds = compute_phi_ie_bounds(self.model, self.r0, 10.0, ranges)
        phi_lo, phi_hi = rate_phi_bounds(tau_syn, 10.0, 20.0)
        phi_ok = 0.5 * (phi_lo + phi_hi)

        make_W_h0_ei_2d_syn(
            self.model,
            self.r0,
            tau_syn,
            10.0,
            ranges['tau_decay'][0],
            q_ie=0.8,
            phi_ie=phi_ok,
        )

        with self.assertRaisesRegex(ValueError, 'phi_ie violates Dale constraint'):
            make_W_h0_ei_2d_syn(
                self.model,
                self.r0,
                tau_syn,
                10.0,
                ranges['tau_decay'][0],
                q_ie=0.8,
                phi_ie=phi_hi + 1e-6,
            )

    def test_main_writes_png_and_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'phi_ie_max_maps'
            with patch(
                'experiments.ei_modes.phi_ie_max_maps.DEFAULT_ARTIFACT_DIR',
                output_dir,
            ):
                main()

            png_files = sorted(output_dir.glob('*.png'))
            json_files = sorted(output_dir.glob('*.json'))

            self.assertEqual(len(png_files), 1)
            self.assertEqual(len(json_files), 1)
            self.assertTrue(png_files[0].name.startswith('phi_maps_r0_5_5_f_10'))

            payload = json.loads(json_files[0].read_text(encoding='utf-8'))
            self.assertEqual(payload['artifact_dir'], str(output_dir))
            self.assertEqual(
                payload['phi_bounds']['dims'],
                ['bound', 'taue_syn', 'taui_syn', 'tau_decay'],
            )
            self.assertEqual(
                len(payload['ranges']['taue_syn']),
                DEFAULT_RANGES['taue_syn'].size,
            )
            self.assertEqual(
                len(payload['ranges']['taui_syn']),
                DEFAULT_RANGES['taui_syn'].size,
            )
            self.assertEqual(
                len(payload['ranges']['tau_decay']),
                DEFAULT_RANGES['tau_decay'].size,
            )


if __name__ == '__main__':
    unittest.main()
