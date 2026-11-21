# Repository move log

Paths are relative to the repository root. A directory entry covers every file below it, with the same relative path and filename at the destination.

## Group experiments by topic

| Previous path | Path after this pass |
| --- | --- |
| `jitter/` | `experiments/jitter/` |
| `prc/` | `experiments/prc/` |
| `rate_controller/` | `experiments/rate_ctrl/` |
| `demo/rate_model_demo.ipynb` | `experiments/rate_demos/rate_model_demo.ipynb` |
| `demo/transfer_mat.demo.py` | `experiments/osc_transfer/transfer_mat.demo.py` |
| `demo/transfer_mat_demo.ipynb` | `experiments/osc_transfer/transfer_mat_demo.ipynb` |
| `docs/diagrams/` | `experiments/osc_transfer/diagrams/` |
| `docs/render_multisine_demo_diagrams.py` | `experiments/osc_transfer/render_multisine_demo_diagrams.py` |
| `description.ipynb` | `experiments/resp_static/description.ipynb` |
| `description.md` | `experiments/resp_static/description.md` |
| `resp_pred_test_1.ipynb` | `experiments/resp_static/resp_pred_test_1.ipynb` |
| `transfer_estimate.ipynb` | `experiments/osc_transfer/transfer_estimate.ipynb` |
| `dev_scratch/model_utils.py` | `model_utils.py` |
| `dev_scratch/multisine_demo.py` | `experiments/osc_transfer/multisine_demo.py` |
| `dev_scratch/optimize_w.ipynb` | `experiments/ei_modes/optimize_w.ipynb` |
| `dev_scratch/optimize_w.py` | `experiments/ei_modes/optimize_w.py` |
| `dev_scratch/phi_ie_max_maps.py` | `experiments/ei_modes/phi_ie_max_maps.py` |
| `dev_scratch/artifacts/phi_ie_max_maps/` | `experiments/ei_modes/artifacts/phi_ie_max_maps/` |
| `dev_scratch/phase_lock.py` | `experiments/phase_lock/phase_lock.py` |
| `dev_scratch/phase_proc_demo.py` | `experiments/phase_lock/phase_proc_demo.py` |
| `dev_scratch/artifacts/phase_lock/` | `experiments/phase_lock/artifacts/phase_lock/` |
| `dev_scratch/rate_model_kernel_demo.py` | `experiments/rate_demos/rate_model_kernel_demo.py` |
| `dev_scratch/rate_model_kernel_syn_demo.py` | `experiments/rate_demos/rate_model_kernel_syn_demo.py` |
| `dev_scratch/rate_model_syn_demo.py` | `experiments/rate_demos/rate_model_syn_demo.py` |
| `dev_scratch/artifacts/rate_model_ker/` | `experiments/rate_demos/artifacts/rate_model_ker/` |

The controller was already grouped in `rate_controller/` in the working tree. In the prior Git revision, `rate_controller.py`, `ctrl_test.ipynb`, and `ctrl_test_multi.ipynb` were at the root. Their current paths are under `experiments/rate_ctrl/`.

`dev_scratch/tasks.md` and `dev_scratch/.matplotlib/` stayed in place.

## Organize the remaining root Python files

| Previous path | Current path |
| --- | --- |
| `resp_pred.py` | `experiments/resp_static/resp_pred.py` |
| `resp_predictor.py` | `experiments/resp_static/resp_predictor.py` |
| `resp_predictor_2.py` | `experiments/resp_static/resp_predictor_2.py` |
| `rank1_hess.py` | `experiments/resp_static/rank1_hess.py` |
| `eig_utils.py` | `rate_model/eig_utils.py` |
| `model_utils.py` | `rate_model/model_utils.py` |
| `rate_model_eig.py` | `rate_model/rate_model_eig.py` |
| `spect_proc.py` | `experiments/phase_lock/spect_proc.py` |
| `wc.py` | `_legacy/wc.py` |

`rate_model/`, `transfer_mat/`, `io_fit/`, and `tests/` stay at the root.
