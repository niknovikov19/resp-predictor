import os
import sys
from pathlib import Path

DIRPATH_REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(DIRPATH_REPO))
print(DIRPATH_REPO)

import numpy as np
from scipy.linalg import eig

from rate_model import RateModel, RateModelLin, RateModelWC
from transfer_mat import TransferMatBuider

#ARTIFACT_DIR = Path(__file__).resolve().parent / 'artifacts' / 'rate_model_ker'
#ARTIFACT_PATH = ARTIFACT_DIR / 'rate_model_kernel_syn_demo.png'

ONE_COMPLEX = 1 + 0j


np.random.seed(0)

# Target operating point
#r0 = np.array([1, 3, 5, 7, 9]).reshape(-1, 1)
r0 = np.array([3, 4, 5, 6, 7]).reshape(-1, 1)
n_pop = r0.shape[0]

# Model parameters
tau = np.linspace(0.4, 1.8, n_pop).reshape(-1, 1)

# Model type
model_type = 'lin'
#model_type = 'wc'

# Create model
if model_type == 'lin':
    model, h0, J = RateModelLin.from_steady_state(
        r0=r0, tau=tau, R=0.5
    )
elif model_type == 'wc':
    model, h0, J = RateModelWC.from_steady_state(
        r0=r0, tau=tau, R=0.5,
        rmax=10, gain_slope=0.5, gain_center=0
    )
else:
    raise ValueError(f"Invalid model type: {model_type}")

# Simulation params
T = 10
dt = 0.01
nsteps = int(T / dt)

# Probe params
probe_freq = 5
probe_amp = 2

# Add input noise
sigma_h = 0.1
#H = h0 + sigma_h * np.random.randn(n_pop, nsteps) / np.sqrt(dt)


def get_target_mode(C, target=ONE_COMPLEX):
    """
    Find eigenvalue/eigenvectors of C closest to target=1+0j.

    scipy convention:
        C @ v = lam * v
        C.H @ u = lam.conj() * u
    so u is the left eigenvector in the sense u^H C = lam u^H.
    """
    vals, left_vecs, right_vecs = eig(C, left=True, right=True)

    #print(vals, target)
    idx = np.argmin(np.abs(vals - target))

    lam = vals[idx]
    u = left_vecs[:, idx]
    v = right_vecs[:, idx]

    return lam, u, v, idx, vals


def multiplicative_sensitivity(C, u, v):
    """
    Sensitivity of selected eigenvalue to multiplicative edge changes:
        C_mn -> C_mn * (1 + alpha_mn)
    First-order:
        delta_lambda = sum_mn S_mn alpha_mn
    where
        S_mn = u_m^* C_mn v_n / (u^H v)
    """
    denom = np.vdot(u, v)  # u^H v
    if abs(denom) < 1e-12:
        raise RuntimeError("Left/right eigenvectors nearly orthogonal; sensitivity unstable.")
    S = (np.conj(u)[:, None] * C * v[None, :]) / denom
    return S


def solve_real_alpha(S, mask, delta_lambda, beta=1e-3, max_abs_alpha=0.02):
    """
    Solve for real multiplicative changes alpha_mn.
    Minimize:
        |sum S_mn alpha_mn - delta_lambda|^2 + beta * ||alpha||^2
    alpha is real because weights are real.
    """
    #print('S:', S)
    #print('mask:', mask)
    S_arr = np.asarray(S)

    if mask is None:
        s = S_arr.reshape(-1)
    else:
        s = S_arr[mask].reshape(-1)

    # Convert one complex equation into two real equations
    A = np.vstack([s.real, s.imag])  # shape: (2, n_edges)
    b = np.array([delta_lambda.real, delta_lambda.imag])

    # Ridge/minimum-norm solution:
    # alpha = A.T @ inv(A A.T + beta I) @ b
    M = A @ A.T + beta * np.eye(2)
    alpha_vec = A.T @ np.linalg.solve(M, b)

    # Trust-region clipping
    alpha_vec = np.clip(alpha_vec, -max_abs_alpha, max_abs_alpha)

    alpha = np.zeros_like(S_arr.real)
    if mask is None:
        alpha = alpha_vec.reshape(S_arr.shape)
    else:
        alpha[mask] = alpha_vec

    return alpha


def calc_model_C(model, h0):
    """ Compute transfer matrix of a model for static input h0. """
    tmb = TransferMatBuider(
        model,
        freqs=np.atleast_1d(probe_freq),
        amps=np.atleast_1d(probe_amp),
        h0=h0,
        dt=dt,
        sigma_h=sigma_h
    )
    tmb.compute_input_traces()
    tmb.compute_input_gains()
    C = tmb.compute_transfer_mat()
    return C


def shape_eigenvalue_to_one(
    model: RateModel, r0,
    allowed_mask=None,
    target=ONE_COMPLEX,
    eta=0.5,
    beta=1e-3,
    max_abs_alpha=0.02,
    max_iter=100,
    tol=1e-4,
    verbose=True,
):
    """
    Iteratively modify C and optionally W so that one eigenvalue approaches 1+0j.

    Assumption:
        C_mn scales multiplicatively with W_mn:
            W_mn -> W_mn * (1 + alpha_mn)
            C_mn -> C_mn * (1 + alpha_mn)

    Parameters
    ----------
    C0 : complex ndarray, shape (N, N)
        Open-loop transfer matrix at target frequency omega0.

    W0 : real ndarray, optional
        Connectivity matrix. If given, it is updated together with C.

    allowed_mask : bool ndarray, optional
        True for edges allowed to change.
        If None, all nonzero C entries are allowed.

    eta : float
        Fraction of the predicted eigenvalue correction to apply per iteration.

    beta : float
        Regularization strength. Larger beta = smaller, more distributed updates.

    max_abs_alpha : float
        Maximum multiplicative change per iteration per edge.
        Example: 0.02 means at most 2 percent per iteration.

    Returns
    -------
    C : complex ndarray
        Updated open-loop transfer matrix.

    W : real ndarray or None
        Updated connectivity matrix, if W0 was provided.

    history : list of dict
        Diagnostic information.
    """

    # Which weights we can change
    if allowed_mask is None:
        allowed_mask = np.abs(model.W) > 0
    allowed_mask = allowed_mask.astype(bool)

    history = []

    # Compute transfer matrix using the current W
    h0 = model.h0_by_r0(r0)   # recompute h0 to keep r0=const for any W
    C = calc_model_C(model, h0).squeeze(drop=True)

    for it in range(max_iter):
        # Find eigmode with eigval closest to target 
        lam, u, v, idx, vals = get_target_mode(C, target=target)

        # Distance to the target
        err = target - lam
        delta_lambda = eta * err   # step size

        history.append({
            "iter": it,
            "lambda": lam,
            "abs_lambda": abs(lam),
            "phase_lambda": np.angle(lam),
            "error": abs(err),
            "mode_index": idx,
        })

        if verbose:
            print(
                f"{it:03d}  "
                f"lambda={lam.real:+.4f}{lam.imag:+.4f}j  "
                f"|lambda|={abs(lam):.4f}  "
                f"phase={np.angle(lam):+.6f}  "
                f"err={abs(err):.4e}"
            )

        if abs(err) < tol:
            break

        # Sensitivity of eigval to multiplicative weight changes
        # S_mn = d_lambda / alpha_mn
        S = multiplicative_sensitivity(C, u, v)

        # Find weight multipliers that yield a step toward target
        alpha = solve_real_alpha(
            S=S,
            mask=None,   #allowed_mask,
            delta_lambda=delta_lambda,
            beta=beta,
            max_abs_alpha=max_abs_alpha,
        )

        # Update the model by applying the multipliers
        #update_model_W(model, model.W * (1 + alpha))
        C *= (1 + alpha)

    return model, C, history


model, C, history = shape_eigenvalue_to_one(
    model, r0,
    eta=0.5,   # step size multiplier
    beta=0,   # regularization
    max_abs_alpha=1,   # max relative step size
    max_iter=100,
    tol=1e-4,
    verbose=1
)
