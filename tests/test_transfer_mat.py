import unittest

import numpy as np
from numpy.testing import assert_allclose

from rate_model.rate_model_kernel import RateModelKernel
from transfer_mat.transfer import TransferMatBuider


def _make_kernel_model(W, tau_syn=0.0):
    return RateModelKernel(
        np.asarray(W, dtype=float),
        tau=np.array([[2.0], [2.5]]),
        tau_syn=tau_syn,
        rmax=12.0,
        gain_slope=0.9,
        gain_center=0.0,
    )


class CountingTransferMatBuider(TransferMatBuider):
    def __init__(self, *args, **kwargs):
        self.trace_calls = 0
        super().__init__(*args, **kwargs)

    def compute_input_traces(self):
        self.trace_calls += 1
        return super().compute_input_traces()


class TestTransferMatBuider(unittest.TestCase):

    def setUp(self):
        self.freqs = np.array([0.5])
        self.amps = np.array([0.05])
        self.h0 = np.array([[-0.2], [-0.1]])
        self.common = dict(
            freqs=self.freqs,
            amps=self.amps,
            h0=self.h0,
            dt=0.1,
            warmup_time=8.0,
            measure_time=16.0,
            welch_time=4.0,
            welch_overlap=0.5,
            steady_time=12.0,
        )

    def test_zero_coupling_gives_near_zero_transfer(self):
        builder = TransferMatBuider(_make_kernel_model(np.zeros((2, 2))), **self.common)

        builder.compute_input_traces()
        C = builder.compute_transfer_mat()

        self.assertLess(np.max(np.abs(C.values)), 5e-3)

    def test_one_way_coupling_gives_forward_but_not_reverse_transfer(self):
        builder = TransferMatBuider(
            _make_kernel_model(np.array([[0.0, 0.3], [0.0, 0.0]])),
            **self.common,
        )

        builder.compute_input_traces()
        C = builder.compute_transfer_mat()
        forward = C.loc[dict(pop_pre=1, pop_post=0, freq=0.5, amp=0.05)].item()
        reverse = C.loc[dict(pop_pre=0, pop_post=1, freq=0.5, amp=0.05)].item()

        self.assertGreater(np.abs(forward), 1e-3)
        self.assertLess(np.abs(reverse), 5e-3)

    def test_dynamic_synapse_adds_phase_lag(self):
        inst = TransferMatBuider(
            _make_kernel_model(np.array([[0.0, 0.3], [0.0, 0.0]]), tau_syn=0.0),
            **self.common,
        )
        syn = TransferMatBuider(
            _make_kernel_model(
                np.array([[0.0, 0.3], [0.0, 0.0]]),
                tau_syn=np.array([[0.0, 4.0], [0.0, 0.0]]),
            ),
            **self.common,
        )

        inst.compute_input_traces()
        syn.compute_input_traces()
        C_inst = inst.compute_transfer_mat().loc[
            dict(pop_pre=1, pop_post=0, freq=0.5, amp=0.05)
        ].item()
        C_syn = syn.compute_transfer_mat().loc[
            dict(pop_pre=1, pop_post=0, freq=0.5, amp=0.05)
        ].item()
        phase_diff = np.angle(C_syn / C_inst)

        self.assertLess(phase_diff, -1e-3)

    def test_compute_input_traces_recomputes_and_metadata_matches(self):
        builder = CountingTransferMatBuider(
            _make_kernel_model(np.array([[0.0, 1.0], [0.0, 0.0]])),
            freqs=np.array([0.5, 1.0]),
            amps=np.array([0.05]),
            h0=self.h0,
            dt=0.1,
            warmup_time=6.0,
            measure_time=12.0,
            welch_time=4.0,
            welch_overlap=0.5,
            steady_time=8.0,
        )

        traces_a = builder.compute_input_traces()
        traces_b = builder.compute_input_traces()

        self.assertEqual(builder.trace_calls, 2)
        self.assertEqual(builder.R_probe.dims, ('pop_pre', 'freq', 'amp', 'time'))
        self.assertEqual(builder.R_probe.sizes['time'], 180)
        self.assertEqual(float(builder.R_probe.coords['time'][0]), 0.0)
        assert_allclose(np.diff(builder.R_probe.coords['time'].values), builder.dt)
        self.assertFalse(np.isnan(builder.R_probe.values).any())
        assert_allclose(traces_a.values, traces_b.values, atol=1e-12)
        self.assertEqual(builder.R_probe.attrs['steady_time'], 8.0)
        self.assertEqual(builder.R_probe.attrs['warmup_time'], 6.0)
        self.assertEqual(builder.R_probe.attrs['measure_time'], 12.0)
        self.assertEqual(builder.R_probe.attrs['welch_time'], 4.0)

    def test_downstream_steps_require_explicit_probe_traces(self):
        builder = TransferMatBuider(
            _make_kernel_model(np.array([[0.0, 0.3], [0.0, 0.0]])),
            **self.common,
        )

        with self.assertRaisesRegex(RuntimeError, 'compute_input_traces'):
            builder.compute_input_gains()
        with self.assertRaisesRegex(RuntimeError, 'compute_input_traces'):
            builder.compute_transfer_mat()

    def test_compute_input_gains_matches_probe_response_over_drive(self):
        builder = TransferMatBuider(
            _make_kernel_model(np.array([[0.0, 0.3], [0.0, 0.0]])),
            **self.common,
        )

        traces = builder.compute_input_traces()
        gains = builder.compute_input_gains()
        gains_again = builder.compute_input_gains()

        self.assertIs(gains_again, gains)
        self.assertEqual(gains.dims, ('pop', 'freq', 'amp'))
        self.assertEqual(gains.attrs['definition'],
                         'direct input-to-rate complex gain from true open-loop probes')

        t_measure = traces.time.values[builder.warmup_steps:] + builder.steady_time
        x = traces.loc[dict(pop_pre=0, freq=0.5, amp=0.05)].values[builder.warmup_steps:]
        X = np.column_stack([
            np.cos(2 * np.pi * 0.5 * t_measure),
            np.sin(2 * np.pi * 0.5 * t_measure),
            np.ones_like(t_measure),
        ])
        coef, *_ = np.linalg.lstsq(X, x, rcond=None)
        expected = (coef[0] - 1j * coef[1]) / 0.05

        assert_allclose(
            gains.loc[dict(pop=0, freq=0.5, amp=0.05)].item(),
            expected,
            atol=1e-12,
        )

    def test_explicit_pipeline_has_no_hidden_probe_runs(self):
        builder = CountingTransferMatBuider(
            _make_kernel_model(np.array([[0.0, 0.3], [0.0, 0.0]])),
            **self.common,
        )

        builder.compute_input_traces()
        gains_1 = builder.compute_input_gains()
        gains_2 = builder.compute_input_gains()
        transfer_1 = builder.compute_transfer_mat()
        transfer_2 = builder.compute_transfer_mat()

        self.assertEqual(builder.trace_calls, 1)
        self.assertIs(gains_1, gains_2)
        self.assertIs(transfer_1, transfer_2)

    def test_reproducibility_with_and_without_noise(self):
        model = _make_kernel_model(np.array([[0.0, 1.0], [0.0, 0.0]]))
        builder_0a = TransferMatBuider(model, sigma_h=0.0, seed=7, **self.common)
        builder_0b = TransferMatBuider(_make_kernel_model(np.array([[0.0, 1.0], [0.0, 0.0]])),
                                       sigma_h=0.0, seed=13, **self.common)
        builder_na = TransferMatBuider(_make_kernel_model(np.array([[0.0, 1.0], [0.0, 0.0]])),
                                       sigma_h=0.02, seed=11, **self.common)
        builder_nb = TransferMatBuider(_make_kernel_model(np.array([[0.0, 1.0], [0.0, 0.0]])),
                                       sigma_h=0.02, seed=11, **self.common)

        builder_0a.compute_input_traces()
        builder_0b.compute_input_traces()
        builder_na.compute_input_traces()
        builder_nb.compute_input_traces()

        D0a = builder_0a.compute_input_gains()
        D0b = builder_0b.compute_input_gains()
        Dna = builder_na.compute_input_gains()
        Dnb = builder_nb.compute_input_gains()
        C0a = builder_0a.compute_transfer_mat()
        C0b = builder_0b.compute_transfer_mat()
        Cna = builder_na.compute_transfer_mat()
        Cnb = builder_nb.compute_transfer_mat()

        assert_allclose(D0a.values, D0b.values, atol=1e-12)
        assert_allclose(Dna.values, Dnb.values, atol=1e-12)
        assert_allclose(C0a.values, C0b.values, atol=1e-12)
        assert_allclose(Cna.values, Cnb.values, atol=1e-12)

    def test_time_windows_floor_to_steps_and_whole_cycles(self):
        builder = TransferMatBuider(
            _make_kernel_model(np.array([[0.0, 0.3], [0.0, 0.0]])),
            freqs=np.array([2.0]),
            amps=np.array([0.05]),
            h0=self.h0,
            dt=0.1,
            warmup_time=0.36,
            measure_time=1.09,
            welch_time=0.95,
            welch_overlap=0.5,
            steady_time=1.26,
        )

        traces = builder.compute_input_traces()
        window_steps, bin_num, hop_steps = builder._welch_config(2.0)

        self.assertEqual(builder.steady_steps, 12)
        self.assertEqual(builder.warmup_steps, 3)
        self.assertEqual(builder.measure_steps, 10)
        self.assertEqual(traces.sizes['time'], 13)
        self.assertEqual(window_steps, 5)
        self.assertEqual(bin_num, 1)
        self.assertEqual(hop_steps, 2)


if __name__ == '__main__':
    unittest.main()
