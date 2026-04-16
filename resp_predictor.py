import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from tqdm import tqdm

from rate_model import RateModel, RateModelWC


class ResponsePredictor:

    model: RateModel
    h0: np.ndarray   # (npops x 1)

    dt: float
    nsteps: int

    r0: np.ndarray   # (npops x 1)
    J: xr.DataArray   # (npops, npops, n_dr)
    P: xr.DataArray   # (npops, npops, n_dr)
    Q: xr.DataArray   # (npops, npops, n_dh)

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
        self.J = None
        self.Q = None
    
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
        r_start = np.zeros((N, 1))
        rr = self.model.run(self.h0, r_start,
                            dt=self.dt, nsteps=self.nsteps)
        self.r0 = rr.isel(time=-1).values.reshape((N, 1))
    
    def _calc_J_slice(self, dr: float) -> np.ndarray:
        """Calculate J slice for a given dr: Jkn = dfk / drn. """
        N = self.model.npops
        D = np.eye(N, N)
        J = np.zeros((N, N))
        for k in range(N):   # population index
            for n in range(N):   # axis to perturb
                # Vector of rate perturbations along the n-th axis
                dr_n = D[:, [n]] * dr
                # Jkn = dfk / drn
                rr = self.model.run_1pop(
                    k, self.h0[k], self.r0 + dr_n,
                    self.dt, self.nsteps
                )
                r_pert = rr.isel(time=-1).item()
                J[k, n] = (r_pert - self.r0[k, 0]) / dr
        return J

    def _calc_Q_slice(self, dh: float) -> np.ndarray:
        """Calculate Q slice for a given dh: Qnn = dfn / dhn. """
        N = self.model.npops
        Q = np.zeros((N, N))
        for n in range(N):   # population index
            # Qnn = dfn / dhn
            rr = self.model.run_1pop(
                n, self.h0[n] + dh, self.r0,
                self.dt, self.nsteps
            )
            r_pert = rr.isel(time=-1).item()
            Q[n, n] = (r_pert - self.r0[n, 0]) / dh
        return Q

    def _calc_J(self, dr_vals: np.ndarray) -> None:
        """Calculate J matrix for a range of dr values. """
        N = self.model.npops
        n_dr = len(dr_vals)
        J = np.zeros((N, N, n_dr))
        for k, dr in enumerate(dr_vals):
            J[:, :, k] = self._calc_J_slice(dr)
        self.J = xr.DataArray(
            J,
            dims=('pop_post', 'pop_pre', 'dr'),
            coords={
                'pop_post': np.arange(N),
                'pop_pre': np.arange(N),
                'dr': dr_vals
            }
        )
    
    def _calc_Q(self, dh_vals: np.ndarray) -> None:
        """Calculate Q matrix for a range of dh values. """
        N = self.model.npops
        n_dh = len(dh_vals)
        Q = np.zeros((N, N, n_dh))
        for n, dh in enumerate(dh_vals):
            Q[:, :, n] = self._calc_Q_slice(dh)
        self.Q = xr.DataArray(
            Q,
            dims=('pop_post', 'pop_pre', 'dh'),
            coords={
                'pop_post': np.arange(N),
                'pop_pre': np.arange(N),
                'dh': dh_vals
            }
        )
    
    def _calc_P(self):
        N = self.model.npops
        n_dr = len(self.J.dr)
        self.P = xr.full_like(self.J, np.nan)
        for n in range(n_dr):
            self.P[:, :, n] = np.linalg.inv(
                np.eye(N) - self.J[:, :, n])

    def train(
            self,
            dh_vals: np.ndarray,
            dr_vals: np.ndarray | None = None
            ) -> None:
        
        self._calc_r0()
        
        self._calc_Q(dh_vals)

        if dr_vals is None:
            dr_vals = np.full_like(dh_vals, np.nan)
            for n, dh in enumerate(dh_vals):
                Q = self.Q.sel(dh=dh, drop=True).values
                #dr_vals[n] = dh * np.sqrt(np.mean(np.diag(Q)**2))
                dr_vals[n] = dh * np.mean(np.diag(Q))

        self._calc_J(dr_vals)
        self._calc_P()
    
    def _clip_dh(self, dh):
        dh_vals = self.Q.dh.values
        return np.clip(dh, dh_vals.min(), dh_vals.max())
    
    def _clip_dr(self, dr):
        dr_vals = self.J.dr.values
        return np.clip(dr, dr_vals.min(), dr_vals.max())
    
    def _get_Q_by_Dh(self, Dh: np.ndarray) -> np.ndarray:
        N = self.model.npops
        Q = np.zeros((N, N))
        for n in range(N):
            dh_ = self._clip_dh(Dh[n])
            Q[n, n] = self.Q[n, n, :].interp(dh=dh_).values.item()
        return Q
    
    def _get_J_by_Dr(self, Dr: np.ndarray) -> np.ndarray:
        N = self.model.npops
        J = np.zeros((N, N))
        for n in range(N):
            dr_ = self._clip_dr(Dr[n])
            J[:, [n]] = self.J[:, n, :].interp(dr=dr_).values
        return J
    
    def _get_P_by_Dr(self, Dr: np.ndarray) -> np.ndarray:
        N = self.model.npops
        P = np.zeros((N, N))
        for n in range(N):
            dr_ = self._clip_dr(Dr[n])
            P[:, [n]] = self.P[:, n, :].interp(dr=dr_).values
        return P

    def predict_r(
            self,
            Dh: np.ndarray,   # (npops, 1) 
            dh_train: float | None = None,
            dr_train: float | None = None
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
        return r_hat

    def run_model(self, h: np.ndarray) -> np.ndarray:
        R = self.model.run(
            h, self.r0, self.dt, self.nsteps)
        r = R.isel(time=-1).values.reshape((self.model.npops, 1))
        return r


#if __name__ == '__main__':


def pert_test(
        model: RateModel,
        h0: np.ndarray,
        dh_vals_train, dh_vals_test,
        n_trials,
        dr_train=None,
        dt=0.5, nsteps=20
        ) -> tuple[np.ndarray, xr.DataArray, xr.DataArray, xr.DataArray]:

    # Allocate the result containers
    N = model.npops
    R = xr.DataArray(
        np.full((N, len(dh_vals_test), n_trials), np.nan),
        dims=['pop', 'dh_test', 'trial'],
        coords={
            'pop': np.arange(N),
            'dh_test': dh_vals_test,
            'trial': np.arange(n_trials)
        }
    )
    Rhat0 = R.copy()
    Rhat = xr.DataArray(
        np.full((N, len(dh_vals_train), len(dh_vals_test), n_trials), np.nan),
        dims=['pop', 'dh_train', 'dh_test', 'trial'],
        coords={
            'pop': np.arange(N),
            'dh_train': dh_vals_train,
            'dh_test': dh_vals_test,
            'trial': np.arange(n_trials)
        }
    )

    # Initialize the response predictor
    pred = ResponsePredictor(model, h0, dt=dt, nsteps=nsteps)
    pred.train(dh_vals_train)

    for n, dh_test in tqdm(enumerate(dh_vals_test),
                           total=len(dh_vals_test)):
        for m in range(n_trials):
            # Random input perturbation
            Dh = np.random.random((N, 1)) * dh_test

            # Get perturbation result from simulation
            r = pred.run_model(h0 + Dh)
            R.isel(dh_test=n, trial=m)[...] = r.ravel()

            # Predict perturbation result with flexible dh_train
            r_hat = pred.predict_r(Dh, None, dr_train)
            Rhat0.isel(dh_test=n, trial=m)[...] = r_hat.ravel()

            # Predict perturbation result with fixed dh_train
            for k, dh_train in enumerate(dh_vals_train):                
                r_hat = pred.predict_r(Dh, dh_train, dr_train)
                Rhat.isel(dh_train=k, dh_test=n, trial=m)[...] = r_hat.ravel()                
    
    return pred.r0, R, Rhat, Rhat0


def plot_pert_err(R, Rhat, Rhat0):
    
    # Compute error mean and std. over trials (fixed dh_train)
    err = np.abs(R - Rhat) / R
    err = err.mean(dim='pop')
    err_mean = err.mean(dim='trial')
    err_std = err.std(dim='trial')

    # Compute error mean and std. over trials (flexible dh_train)
    err0 = np.abs(R - Rhat0) / R
    err0 = err0.mean(dim='pop')
    err0_mean = err0.mean(dim='trial')
    err0_std = err0.std(dim='trial')

    dh_vals_test_ = R.dh_test.values

    plt.figure(figsize=(8, 6))

    # Plot mean error as a function of dh_test for each dh_train
    for k, dh_train in enumerate(Rhat.dh_train):
        err_mean_ = err_mean.isel(dh_train=k).values
        err_std_ = err_std.values
        plt.plot(dh_vals_test_, err_mean_, '.-',
                 label=f'dh_train={dh_train.item():.05f}')
        """ plt.fill_between(
            dh_abs_vals,
            err_mean_ - err_std_,
            err_mean_ + err_std_,
            color='blue',
            alpha=0.2
        ) """
    
    # Plot mean error as a function of dh_test with flexible dh_train
    plt.plot(dh_vals_test_, err0_mean.values, 'k.-',
             label=f'dh_train: flexible')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('dh_test')
    plt.ylabel('Error')
    plt.title(f'Error |r - r_hat| vs. dh_test (avg. over pops.)')
    plt.legend()


#np.random.seed(111)

# Model
N = 5  # Number of populations
W = 0.2 * np.random.randn(N, N)  # Weight matrix
model = RateModelWC(W, tau=1, rmax=10, gain_slope=0.5, gain_center=0)

# External input
h0 = 1 * np.random.randn(N, 1)

n_trials = 1
dh_vals_train = 10. ** np.linspace(-4, 0.5, 5)
dh_vals_test = 10. ** np.linspace(-3.5, 1, 13)

r0, R, Rhat, Rhat0 = pert_test(model, h0, dh_vals_train, 
                               dh_vals_test, n_trials=n_trials)
plot_pert_err(R, Rhat, Rhat0)

plt.show()

