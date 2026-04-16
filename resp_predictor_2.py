import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from tqdm import tqdm

from rank1_hess import estimate_rank1_hessian
from rate_model import RateModel, RateModelWC


class ResponsePredictor_2:

    model: RateModel
    h0: np.ndarray   # (npops x 1)

    dt: float
    nsteps: int

    r0: np.ndarray   # (npops x 1)

    J1: np.ndarray   # J1kn = dfk / drn
    Q1: np.ndarray   # Q1n = dfn / dhn

    J2: np.ndarray   # J2knm = d2fk / (drn * drm)
    Q2: np.ndarray   # Q2n = d2fn / dhn^2
    JQ11: np.ndarray   # JQ11nm = d2fn / (dhn * drm)

    def __init__(
            self,
            model: RateModel | None = None,
            h0: np.ndarray | None = None,
            dt: float = 0.5,
            nsteps: int = 20
            ):
        self.model = model
        self.h0 = h0
        self.dt = dt
        self.nsteps = nsteps
        self._reset()
    
    def _reset(self):
        self.r0 = None
        self.J1 = None
        self.Q1 = None
        self.J2 = None
    
    def set_model(self, model: RateModel):
        self.model = model
        self.h0 = None
        self._reset()
    
    def set_h0(self, h0: np.ndarray):
        self.h0 = h0
        self._reset()

    def _calc_r0(self) -> None:
        """Find unperturbed steady-state. """
        N = self.model.npops    
        self.r0 = self.model.run(self.h0, r0=np.zeros((N, 1)),
                                 dt=self.dt, nsteps=self.nsteps)
    
    def _calc_J1(self, dr: float) -> None:
        """Calculate J1: J1kn = dfk / drn. """
        N = self.model.npops
        sim_par = self.dt, self.nsteps
        D = np.eye(N, N)
        self.J1 = np.zeros((N, N))
        for k in range(N):   # population index
            for n in range(N):   # axis to perturb
                dr_n = D[:, [n]] * dr   # perturbations along the n-th axis
                r_pert_p = self.model.run_1pop(k, self.h0[k], self.r0 + dr_n, *sim_par)
                r_pert_n = self.model.run_1pop(k, self.h0[k], self.r0 - dr_n, *sim_par)
                self.J1[k, n] = (r_pert_p - r_pert_n) / (2 * dr)

    def _calc_Q1(self, dh: float) -> None:
        """Calculate Q1: Q1n = dfn / dhn. """
        N = self.model.npops
        sim_par = self.dt, self.nsteps
        self.Q1 = np.zeros((N, 1))
        for n in range(N):   # population index
            r_pert_p = self.model.run_1pop(n, self.h0[n] + dh, self.r0, *sim_par)
            r_pert_n = self.model.run_1pop(n, self.h0[n] - dh, self.r0, *sim_par)
            self.Q1[n] = (r_pert_p - r_pert_n) / (2 * dh)
    
    def _calc_J2(self, dr: float) -> None:
        """Calculate J2: J2knm = d2fk / (drn * drm) """
        N = self.model.npops
        sim_par = self.dt, self.nsteps
        D = np.eye(N, N)
        self.J2 = np.zeros((N, N, N))
        for k in range(N):
            for m in range(N):
                for n in range(N):
                    dr_m = D[:, [m]] * dr
                    dr_n = D[:, [n]] * dr
                    r_pert_pp = self.model.run_1pop(k, self.h0[k], self.r0 + dr_m + dr_n, *sim_par)
                    r_pert_pn = self.model.run_1pop(k, self.h0[k], self.r0 + dr_m - dr_n, *sim_par)
                    r_pert_np = self.model.run_1pop(k, self.h0[k], self.r0 - dr_m + dr_n, *sim_par)
                    r_pert_nn = self.model.run_1pop(k, self.h0[k], self.r0 - dr_m - dr_n, *sim_par)
                    self.J2[k, m, n] = (r_pert_pp - r_pert_pn - r_pert_np + r_pert_nn) / (4 * dr**2)
    
    def _calc_Q2(self, dh: float) -> None:
        """Calculate Q2: Q2n = d2fn / dhn^2 """
        N = self.model.npops
        sim_par = self.dt, self.nsteps
        self.Q2 = np.zeros((N, 1))
        for n in range(N):
            r_pert_p = self.model.run_1pop(n, self.h0[n] + dh, self.r0, *sim_par)
            r_pert_0 = self.model.run_1pop(n, self.h0[n], self.r0, *sim_par)
            r_pert_n = self.model.run_1pop(n, self.h0[n] - dh, self.r0, *sim_par)
            self.Q2[n] = (r_pert_p - 2 * r_pert_0 + r_pert_n) / (dh**2)
    
    def _calc_JQ11(self, dr: float, dh: float) -> None:
        """Calculate JQ11: JQ11nm = d2f_n / (dh_n * dr_m) """
        N = self.model.npops
        sim_par = self.dt, self.nsteps
        D = np.eye(N, N)
        self.JQ11 = np.zeros((N, N))
        for n in range(N):        # population whose f_n we read out
            for m in range(N):    # axis in r we perturb
                dr_m = D[:, [m]] * dr
                f_pp = self.model.run_1pop(n, self.h0[n] + dh, self.r0 + dr_m, *sim_par)
                f_pn = self.model.run_1pop(n, self.h0[n] + dh, self.r0 - dr_m, *sim_par)
                f_np = self.model.run_1pop(n, self.h0[n] - dh, self.r0 + dr_m, *sim_par)
                f_nn = self.model.run_1pop(n, self.h0[n] - dh, self.r0 - dr_m, *sim_par)
                self.JQ11[n, m] = (f_pp - f_pn - f_np + f_nn) / (4 * dh * dr)
    
    def _calc_J2_estim(self, dr: float) -> None:
        """Calculate J2: J2kmn = d2fk / (drm * drn) """
        N = self.model.npops
        sim_par = self.dt, self.nsteps
        self.J2 = np.zeros((N, N, N))
        for k in range(N):
            F = lambda r: self.model.run_1pop(k, self.h0[k], r, *sim_par)
            self.J2[k, :, :] = estimate_rank1_hessian(F, self.r0, dr)

    def train(self, dh: float, dr: float) -> None:        
        self._calc_r0()        
        self._calc_Q1(dh)
        self._calc_J1(dr)
        self._calc_J2(dr)
        self._calc_Q2(dh)
        self._calc_JQ11(dr, dh)

    """ def predict_r(
            self,
            Dh: np.ndarray,   # (npops, 1) 
            ) -> np.ndarray:
        
        N = self.model.npops
        Dh = Dh.reshape((N, 1))
        
        if dh_train is None:
            #dh_train = np.linalg.norm(Dh, 1)
            Q = self._get_Q_by_Dh(Dh)
        else:
            dh_train = self._clip_dh(dh_train)
            Q = self.Q.interp(dh=dh_train).values
        
        Dr_pre = Q @ Dh
        Dr_post = Dr_pre

        if dr_train is None:
            #dr_train = dh_train * np.sqrt(np.mean(np.diag(Q)**2))
            #dr_train = dh_train * np.mean(np.diag(Q))
            for n in range(10):
                J = self._get_J_by_Dr(Dr_post)
                Dr_post = np.linalg.inv(np.eye(N) - J) @ Dr_pre
                #P = self._get_P_by_Dr(Dr_post)
                #Dr_post = P @ Dr_pre
        else:
            dr_train = self._clip_dr(dr_train)
            #J = self.J.interp(dr=dr_train).values
            #Dr_post = np.linalg.inv(np.eye(N) - J) @ Dr_pre
            P = self.P.interp(dr=dr_train).values
            Dr_post = P @ Dr_pre

        r_hat = self.r0 + Dr_post
        return r_hat """

    def predict_r(self, Dh: np.ndarray,
                  tol=1e-8, max_iter=30,
                  damping=1.0, backtrack=True,
                  use_dh2=True
                  ) -> np.ndarray:
        """
        Solve for r* given Dh using a 2nd-order Taylor implicit model and Newton's method.
        Returns r_pred (N, 1).
        """
        if any(x is None for x in [self.r0, self.J1, self.Q1, self.Q2, self.J2, self.JQ11]):
            raise ValueError("Call train(...) first to populate r0, J1, Q1, Q2, J2, JQ11.")

        N = self.model.npops
        Dh = np.asarray(Dh).reshape(N, 1)

        # Precompute constants
        J1   = self.J1.copy()
        JQ   = self.JQ11.copy()
        J2   = self.J2.copy()
        Q1   = self.Q1.reshape(N, 1).copy()
        Q2   = self.Q2.reshape(N, 1).copy()
        I    = np.eye(N)

        # Symmetrize J2
        J2 = 0.5 * (J2 + np.transpose(J2, (0, 2, 1)))

        # Right-hand side terms depending only on Dh
        rhs = Q1 * Dh          # (N,1)
        if use_dh2:
            rhs += 0.5 * Q2 * (Dh * Dh)

        # Good initial guess: ignore quadratic-in-Δr and JQ term
        M0 = I - J1
        try:
            dr = np.linalg.solve(M0, rhs)             # (N,1)
        except np.linalg.LinAlgError:
            lam = 1e-6 * np.linalg.norm(M0)
            dr = np.linalg.solve(M0 + lam * I, rhs)

        def quad_vec(J2, dr_flat):
            # q_k = dr^T J2[k] dr
            return np.einsum('kij,i,j->k', J2, dr_flat, dr_flat).reshape(N, 1)

        def H_rows(J2, dr_flat):
            # Row k = (J2[k] @ dr)^T
            return np.einsum('kij,j->ki', J2, dr_flat)

        # Newton iterations
        for _ in range(max_iter):
            drf = dr.ravel()

            quad = quad_vec(J2, drf)                                   # (N,1)
            g = dr - J1 @ dr - (Dh * (JQ @ dr)) - 0.5 * quad - rhs     # residual (N,1)

            # Convergence check (relative to rhs scale)
            if np.linalg.norm(g) <= tol * (1.0 + np.linalg.norm(rhs)):
                return self.r0 + dr

            # Build Jacobian
            A = I - J1 - np.diagflat(Dh.ravel()) @ JQ                  # (N,N)
            A -= H_rows(J2, drf)                                       # subtract row-wise (N,N)

            # Newton step
            try:
                delta = np.linalg.solve(A, -g)
            except np.linalg.LinAlgError:
                lam = 1e-6 * np.linalg.norm(A)
                delta = np.linalg.solve(A + lam * I, -g)

            # Damping / backtracking (simple Armijo-style)
            step = float(damping)
            if backtrack:
                g_norm = np.linalg.norm(g)
                for _ in range(10):
                    dr_try = dr + step * delta
                    drf_try = dr_try.ravel()
                    quad_try = quad_vec(J2, drf_try)
                    g_try = dr_try - J1 @ dr_try - (Dh * (JQ @ dr_try)) - 0.5 * quad_try - rhs
                    if np.linalg.norm(g_try) < g_norm:
                        break
                    step *= 0.5
            dr = dr + step * delta

        # Not converged: return best attempt
        return self.r0 + dr




