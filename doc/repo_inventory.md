# Repository inventory

This repository explores how to estimate the input–output response of neural population models from simulations. [Project description](../experiments/resp_static/description.md) · [File move log](repo_moves.md)

## Root directories

| Directory | Contents |
| --- | --- |
| `_legacy/` | Standalone legacy Wilson–Cowan simulation. |
| `dev_scratch/` | Loose development notes and the existing Matplotlib cache. |
| `doc/` | Repository inventory and file move log. |
| `experiments/` | Notebooks, scripts, and artifacts grouped by research topic. |
| `io_fit/` | Task notes on approximating model input–output relationships. |
| `rate_model/` | Base, linear, kernel, and Wilson–Cowan rate models, plus shared model and eigenmode helpers. |
| `tests/` | Tests for rate models, spectral processing, transfer matrices, and experiment utilities. |
| `transfer_mat/` | Transfer-matrix estimation code. |

## Experiment topics

| Topic | Contents |
| --- | --- |
| [`ei_modes/`](../experiments/ei_modes/) | Weight optimization and maps of valid excitatory–inhibitory phase ranges. |
| [`jitter/`](../experiments/jitter/) | Oscillation jitter and inter-trial coherence. |
| [`osc_transfer/`](../experiments/osc_transfer/) | Oscillatory transfer estimation, multisine probes, and diagrams. |
| [`phase_lock/`](../experiments/phase_lock/) | Phase locking, spectral processing, and phase-processing demos. |
| [`prc/`](../experiments/prc/) | Phase-response curve code, notebook, and figures. |
| [`rate_ctrl/`](../experiments/rate_ctrl/) | Feedback controller implementation, single-run notebook, and parameter sweep. |
| [`rate_demos/`](../experiments/rate_demos/) | Rate-model, synaptic, and kernel demos. |
| [`resp_static/`](../experiments/resp_static/) | Static response predictors, experiments, and project descriptions. |

## Python modules moved from the root

| Current path | Purpose |
| --- | --- |
| [`rate_model/eig_utils.py`](../rate_model/eig_utils.py) | Eigenvalue sampling and real random matrix construction. |
| [`rate_model/model_utils.py`](../rate_model/model_utils.py) | Shared model construction and experiment helpers. |
| [`experiments/resp_static/rank1_hess.py`](../experiments/resp_static/rank1_hess.py) | Rank-one Hessian estimation. |
| [`rate_model/rate_model_eig.py`](../rate_model/rate_model_eig.py) | Two-population excitatory–inhibitory model construction from eigenmode parameters. |
| [`experiments/resp_static/resp_pred.py`](../experiments/resp_static/resp_pred.py) | Steady-state calculations, perturbation matrices, and response prediction experiments. |
| [`experiments/resp_static/resp_predictor.py`](../experiments/resp_static/resp_predictor.py) | `ResponsePredictor` and tools for evaluating perturbation predictions. |
| [`experiments/resp_static/resp_predictor_2.py`](../experiments/resp_static/resp_predictor_2.py) | `ResponsePredictor_2` implementation. |
| [`experiments/phase_lock/spect_proc.py`](../experiments/phase_lock/spect_proc.py) | Morlet and sinusoid-fit methods for phase, power, and inter-trial coherence. |
| [`_legacy/wc.py`](../_legacy/wc.py) | Standalone two-population Wilson–Cowan simulation. |

## Root project notes

| File | Purpose |
| --- | --- |
| `AGENTS.md` | Repository instructions and coding style. |

## Tooling and generated files

| Entry | Purpose |
| --- | --- |
| `.codex` | Codex tooling file. |
| `.git/` | Local Git metadata. |
| `.gitignore` | Git ignore rules. |
| `.matplotlib/` | Matplotlib font cache. |
| `.vscode/` | Local VS Code settings. |
| `__pycache__/` | Generated Python bytecode cache. |
