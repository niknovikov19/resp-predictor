import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from rate_model import RateModel, RateModelWC


np.random.seed(111)

# Model
N = 5  # Number of populations
W = 0.2 * np.random.randn(N, N)  # Weight matrix
model = RateModelWC(W, tau=1, rmax=10, gain_slope=0.5, gain_center=0)

# External input
h0 = 1 * np.random.randn(N, 1)

def test_run():
    # Run
    r_start = np.zeros((N, 1))
    rr = model.run(h0, r_start, dt=0.5, nsteps=20)
    # Plot
    plt.figure()
    for n in range(N):
        plt.plot(rr.time, rr.isel(pop=n))
    plt.xlabel('Time')
    plt.ylabel('Firing rate')
    plt.title('Simulation (full model)')
    plt.show()

def calc_r0(model: RateModel, h0, dt=0.5, nsteps=20, r_start=None):
    """ Find unperturbed steady-state. """
    if r_start is None:
        r_start = np.zeros((N, 1))
    rr = model.run(h0, r_start, dt=dt, nsteps=nsteps)
    r0 = rr.isel(time=-1).values.reshape((N, 1))
    return r0

def calc_pert_mats(model: RateModel, h0, r0, eps_r, eps_h,
                   dt=0.5, nsteps=20):
    # Unit perturbations
    N = model.npops
    D = np.eye(N, N)

    # Matrix of dfk / drn
    J = np.zeros((N, N))
    # Diagonal matrix of dfn / dhn
    Q = np.zeros((N, N))

    # Calculate J
    for k in range(N):   # population index
        for n in range(N):   # axis to perturb
            # Vector of rate perturbations along the n-th axis
            dr_n = D[:, [n]] * eps_r
            # Jkn = dfk / drn
            rr = model.run_1pop(k, h0[k], r0 + dr_n, dt, nsteps)
            r_pert = rr.isel(time=-1).item()
            J[k, n] = (r_pert - r0[k, 0]) / eps_r

    # Calculate Q
    for n in range(N):   # population index
        # Qnn = dfn / dhn
        rr = model.run_1pop(n, h0[n] + eps_h, r0, dt, nsteps)
        r_pert = rr.isel(time=-1).item()
        Q[n, n] = (r_pert - r0[n, 0]) / eps_h
    
    return J, Q

def perturb_h(model: RateModel, h0, r0, J, Q, dh_abs,
              dt=0.5, nsteps=20):
    # Random input perturbation
    N = model.npops
    dh = np.random.random(N)[:, None] * dh_abs

    # Estimated rate under perturbed input
    dr = np.linalg.inv(np.eye(N) - J) @ Q @ dh
    r_hat = r0 + dr

    # Actual rate under perturbed input
    rr = model.run(h0 + dh, r0, dt, nsteps)
    r = rr.isel(time=-1).values.reshape((N, 1))

    return r, r_hat

""" dh_abs = 0.1

# Get perturbation result
r, r_hat = perturb_h(model, h0, r0, J, Q, dh_abs)

# Build and print the results table
df_result = pd.DataFrame({
    'pop': np.arange(N),
    'r': r.ravel(),
    'r_hat': r_hat.ravel(),
    'err': np.abs(r_hat.ravel() - r.ravel())
})
print(df_result) """

def pert_test(model, h0, dh_vals_train, dh_vals_test, dr, n_trials,
              dt=0.5, nsteps=20):
    # Allocate the result containers
    R = xr.DataArray(
        np.full((N, len(dh_vals_train), len(dh_vals_test), n_trials), np.nan),
        dims=['pop', 'dh_train', 'dh_test', 'trial'],
        coords={
            'pop': np.arange(N),
            'dh_train': dh_vals_train,
            'dh_test': dh_vals_test,
            'trial': np.arange(n_trials)
        }
    )
    Rhat = R.copy()

    # Find unperturbed steady state
    r0 = calc_r0(model, h0)

    for k, dh_train in enumerate(dh_vals_train):
        # Calculate perturbation matrices
        J, Q = calc_pert_mats(model, h0, r0, eps_r=dr, eps_h=dh_train)

        for n, dh_test in enumerate(dh_vals_test):
            for m in range(n_trials):
                # Get perturbation result (simulation and estimation)
                r, r_hat = perturb_h(model, h0, r0, J, Q, dh_test)
                # Store the result
                R.isel(dh_train=k, dh_test=n, trial=m)[...] = r.ravel()
                Rhat.isel(dh_train=k, dh_test=n, trial=m)[...] = r_hat.ravel()
    
    return r0, R, Rhat

def plot_pert_err(R, Rhat):
    # Compute error mean and std. over trials
    err = np.abs(R - Rhat) / R
    err = err.mean(dim='pop')
    err_mean = err.mean(dim='trial')
    #err_std = err.std(dim='trial')

    dh_vals_test_ = R.dh_test.values

    plt.figure(figsize=(8, 6))

    # Plot mean error as a function of dh_test for each dh_train
    for k, dh_train in enumerate(R.dh_train):
        err_mean_ = err_mean.isel(dh_train=k).values
        #err_std_ = err_std.values
        plt.plot(dh_vals_test_, err_mean_, '.-',
                 label=f'dh_train={dh_train.item():.05f}')
        """ plt.fill_between(
            dh_abs_vals,
            err_mean_ - err_std_,
            err_mean_ + err_std_,
            color='blue',
            alpha=0.2
        ) """
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('dh_test')
    plt.ylabel('Error')
    plt.title(f'Error |r - r_hat| vs. dh_test (avg. over pops.)')
    plt.legend()
    #plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    #plt.show()


n_trials = 15   # number of random trials for each dh_abs
dh_vals_train = 10. ** np.linspace(-2.5, 0.5, 5)
dh_vals_test = 10. ** np.linspace(-2.5, 0.5, 20)
#dr = 0.2

r0, R, Rhat = pert_test(model, h0, dh_vals_train, dh_vals_test, dr=0.1, n_trials=n_trials)
plot_pert_err(R, Rhat)

r0, R, Rhat = pert_test(model, h0, dh_vals_train, dh_vals_test, dr=0.01, n_trials=n_trials)
plot_pert_err(R, Rhat)

plt.show()
