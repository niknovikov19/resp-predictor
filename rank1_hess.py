import numpy as np


def estimate_rank1_hessian(F, x0, dx=1e-4):
    """
    Estimate the Hessian of F at x0 knowing that   H = λ v vᵀ   has rank 1.

    Parameters
    ----------
    F : callable
        The costly scalar function y = F(x) that accepts an (N,) ndarray.
    x0 : array_like, shape (N,)
        Expansion point.
    h : float, optional
        Finite–difference step.
    assume_psd : bool, optional
        If True (default) the Hessian is assumed positive-semidefinite,
        so all vᵢ carry the same sign and we can skip the extra “sign-finding”
        probes, reducing the call count from 4 N–1 to 2 N+1.

    Returns
    -------
    H : ndarray, shape (N, N)
        Rank-1 Hessian estimate  λ v vᵀ.
    n_calls : int
        Number of evaluations of F carried out.
    """
    
    x0   = np.asarray(x0, dtype=float)
    N    = x0.size
    f0   = F(x0)

    # Diagonal elements
    diag = np.empty(N)        # λ v_i²
    for i in range(N):        # 2 calls per axis  → 2 N calls
        e_i      = np.zeros_like(x0);  e_i[i] = 1.0
        f_plus   = F(x0 + dx*e_i)
        f_minus  = F(x0 - dx*e_i)
        diag[i]  = (f_plus + f_minus - 2.0*f0) / dx**2

    # Absolute size of v-components
    v_abs = np.sqrt(np.abs(diag))

    ref = int(np.argmax(np.abs(diag)))     # use the strongest axis as reference
    sign_lambda = np.sign(diag[ref])       # λ shares the sign of that diagonal
    v           = np.empty(N)
    v[ref]      = v_abs[ref]               # v_ref is chosen positive

    #  Probe along (e_ref + e_i) for i ≠ ref to get the relative signs of v_i
    for i in range(N):
        if i == ref:
            continue
        d          = np.zeros_like(x0)
        d[[ref,i]] = 1.0                   # direction  e_ref + e_i
        f_plus     = F(x0 + dx*d)
        f_minus    = F(x0 - dx*d)
        s_mix      = (f_plus + f_minus - 2.0*f0) / dx**2     # λ(v_ref+v_i)²

        # 2 λ v_ref v_i  =  s_mix - diag[ref] - diag[i]
        rel_sign   = np.sign(s_mix - diag[ref] - diag[i]) * sign_lambda
        v[i]       = rel_sign * v_abs[i]

    H = sign_lambda * np.outer(v, v)
    return H
