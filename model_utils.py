import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import xarray as xr

from rate_model import RateModelWC



def sample_disk_points_conjugate_symmetric(N: int, R: float):
    vals = []
    k = 0
    while k < N:
        if k == N - 1:
            x = np.random.uniform(-R, R)
            vals.append(complex(x, 0.0))
            k += 1
        else:
            u = np.random.rand()
            theta = 2 * np.pi * np.random.rand()
            lam = R * np.sqrt(u) * np.exp(1j * theta)

            if abs(lam.imag) < 1e-12 or np.random.rand() < 0.3:
                vals.append(complex(lam.real, 0.0))
                k += 1
            else:
                vals.append(lam)
                vals.append(np.conj(lam))
                k += 2
    return vals[:N]


def block_diag(blocks):
    n = sum(b.shape[0] for b in blocks)
    out = np.zeros((n, n), dtype=float)
    i = 0
    for b in blocks:
        m = b.shape[0]
        out[i:i+m, i:i+m] = b
        i += m
    return out


def random_W_real(N: int, R: float) -> np.ndarray:
    eigvals = sample_disk_points_conjugate_symmetric(N, R)

    blocks = []
    i = 0
    while i < N:
        lam = eigvals[i]
        if abs(lam.imag) < 1e-12:
            blocks.append(np.array([[lam.real]]))
            i += 1
        else:
            a, b = lam.real, lam.imag
            blocks.append(np.array([[a, -b],
                                    [b,  a]]))
            i += 2

    J = block_diag(blocks)

    Q = np.random.randn(N, N)
    while np.linalg.matrix_rank(Q) < N:
        Q = np.random.randn(N, N)

    W = Q @ J @ np.linalg.inv(Q)
    return W


def make_W_h0(model, r0, tau, R, J0=-1.0):
    r0 = np.asarray(r0).reshape(-1, 1)
    N = len(r0)

    # Find the input that produces r0
    mu0 = model.gain_inv(r0)
    
    # Compute gain derivative at the fixed point
    gp = model.gain_deriv(mu0)
    
    Ginv = np.diag(1 / gp.ravel())

    # Random Jacobian with spectral radius R
    J = random_W_real(N, R=R) + J0 * np.eye(N)

    # Handle scalar or per-population tau
    tau_arr = np.asarray(tau).reshape(-1, 1)
    if tau_arr.size == 1:
        Tau = tau_arr[0, 0] * np.eye(N)
    else:
        Tau = np.diag(tau_arr.ravel())

    # Solve for W from J = (G W - I)/tau
    W = Ginv @ (np.eye(N) + Tau @ J)

    # Constant input making r0 a fixed point
    h0 = model.gain_inv(r0) - W @ r0
    return W, h0, J


def make_W_h0_ei_2d(model, r0, tau, f, tau_decay, q_ie=1, phi_ie=0, tol=1e-12):
    """
    Assumes:
        dr/dt = T^{-1}[-r + gain(W @ r + h0)]

    tau and tau_decay must be in seconds if f is in Hz.
    """

    r0 = np.asarray(r0, dtype=float).reshape(-1, 1)
    if r0.shape != (2, 1):
        raise ValueError("r0 must have shape (2,) or (2,1)")

    tau = np.asarray(tau, dtype=float).ravel()
    if tau.size == 1:
        tau = np.repeat(tau.item(), 2)
    if tau.size != 2:
        raise ValueError("tau must be scalar or length 2")

    tau_e, tau_i = tau
    if tau_e <= 0 or tau_i <= 0:
        raise ValueError("tau values must be positive")
    if tau_decay <= 0:
        raise ValueError("tau_decay must be positive")
    if f < 0:
        raise ValueError("f must be nonnegative")

    mu0 = np.asarray(model.gain_inv(r0), dtype=float).reshape(2, 1)
    gp = np.asarray(model.gain_deriv(r0), dtype=float).ravel()

    if gp.size != 2:
        raise ValueError("gain_deriv must return length 2")
    if np.any(gp <= 0):
        raise ValueError(f"gain derivatives must be positive; got {gp}")

    g_e, g_i = gp

    a = -1.0 / tau_decay
    b = 2.0 * np.pi * f

    alpha_min = max(
        -1.0 / tau_e,
        2.0 * a + 1.0 / tau_i,
    )

    alpha = a + b / np.tan(phi_ie)

    if alpha < alpha_min - tol:
        raise ValueError(
            f"alpha violates Dale constraint: alpha={alpha}, "
            f"required alpha >= {alpha_min}"
        )

    K = (alpha - a) ** 2 + b ** 2
    sqrtK = np.sqrt(K)

    beta = sqrtK / q_ie
    gamma = q_ie * sqrtK

    J_target = np.array([
        [alpha, -beta],
        [gamma, 2.0 * a - alpha],
    ])

    T = np.diag([tau_e, tau_i])
    Ginv = np.diag([1.0 / g_e, 1.0 / g_i])

    W = Ginv @ (np.eye(2) + T @ J_target)
    h0 = mu0 - W @ r0

    G = np.diag([g_e, g_i])
    J = np.diag(1.0 / tau) @ (G @ W - np.eye(2))

    if not (
        W[0, 0] >= -tol and
        W[0, 1] <=  tol and
        W[1, 0] >= -tol and
        W[1, 1] <=  tol
    ):
        raise RuntimeError(f"Dale sign violation:\nW=\n{W}")

    return W, h0, J


def main_rand():
    #np.random.seed(114)

    # Num. populations
    N = 5

    tau = np.linspace(0.2, 2, N)
    rho = 0.95   # spectral radius   

    # Baseline rate vector
    r0 = np.linspace(2, 8, N).reshape(-1, 1)

    # Model
    dummy_W = np.zeros((N, N))
    model = RateModelWC(
        dummy_W, tau=tau, rmax=10, gain_slope=0.5, gain_center=0)
    W, h0, J = make_W_h0(model, r0, tau, rho)
    model.W = W

    # Simulation params
    T = 10
    dt = 0.05
    nsteps = int(T / dt)

    # Add zero-mean noise around that operating point
    sigma_h = 0.2
    H = h0 + sigma_h * np.random.randn(N, nsteps) / np.sqrt(dt)

    # Initial state
    r_start = r0.copy()

    # Run simulation
    R = model.run(
        H, r_start, dt, nsteps
    )
    tt = R.coords['time']

    # Plot result
    plt.figure(figsize=(10, 5))
    for n in range(N):
        plt.plot(tt, R.isel(pop=n))
    plt.xlabel('Time')
    plt.ylabel('Rate')
    plt.show()


def main_ei():
    # 2-population E-I model with oscillations
    tau = np.array([1.0, 0.5])  # E and I time constants
    f = 10
    tau_decay = 0.5
    r0 = np.array([5.0, 5.0])  # E and I baseline rates
    
    # Model
    model = RateModelWC(
        np.zeros((2, 2)), tau=tau, rmax=10, gain_slope=0.5, gain_center=0)
    W, h0, J = make_W_h0_ei_2d(model, r0, tau, f, tau_decay, q_ie=1, phi_ie=-pi*1.6)
    model.W = W
    
    print(f"W (Dale's law: E+ col, I- col):\n{W}")
    print(f"\nJacobian eigenvalues: {np.linalg.eigvals(J)}")
    
    # Simulation
    T = 10
    dt = 0.001
    nsteps = int(T / dt)

    # Add zero-mean noise
    sigma_h = 0.2
    H = h0 + sigma_h * np.random.randn(2, nsteps) / np.sqrt(dt)
    
    # Initial state
    r_start = r0.reshape(-1, 1)   # + np.array([[0.5], [-0.3]])
    
    R = model.run(H, r_start, dt, nsteps)
    tt = R.coords['time']
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(tt, R.isel(pop=0), label='Excitatory')
    plt.plot(tt, R.isel(pop=1), label='Inhibitory')
    plt.axhline(r0[0], color='C0', linestyle='--', alpha=0.3)
    plt.axhline(r0[1], color='C1', linestyle='--', alpha=0.3)
    plt.xlabel('Time')
    plt.ylabel('Rate')
    plt.legend()
    plt.title(f'E-I oscillations (f={f} Hz, τ_decay={tau_decay})')
    plt.show()


if __name__ == "__main__":
    main_ei()
