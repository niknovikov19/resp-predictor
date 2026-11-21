import numpy as np


def make_W_h0_ei_2d(model, r0, tau, f, tau_decay, q_ie=1, phi_ie=0, tol=1e-12):
    """
    Assumes:
        dr/dt = T^{-1}[-r + gain(W @ r + h0)]

    tau and tau_decay must be in seconds if f is in Hz.
    """

    # Validate the EI rate-model inputs
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

    # Extract the local gain linearization
    mu0 = np.asarray(model.gain_inv(r0), dtype=float).reshape(2, 1)
    gp = np.asarray(model.gain_deriv(r0), dtype=float).ravel()

    if gp.size != 2:
        raise ValueError("gain_deriv must return length 2")
    if np.any(gp <= 0):
        raise ValueError(f"gain derivatives must be positive; got {gp}")

    g_e, g_i = gp

    # Build the target Jacobian eigenstructure
    a = -1.0 / tau_decay
    b = 2.0 * np.pi * f

    alpha_min = max(
        -1.0 / tau_e,
        2.0 * a + 1.0 / tau_i,
    )

    alpha = a + b / np.tan(phi_ie)

    if alpha < alpha_min - tol:
        if b == 0:
            raise ValueError("phi_ie violates Dale constraint: no valid phi_ie")
        phi_max = np.arctan2(b, alpha_min - a)
        n = np.floor(phi_ie / np.pi)
        phi_lo = n * np.pi
        phi_hi = phi_lo + phi_max
        _raise_phi_error(phi_ie, phi_lo, phi_hi)

    K = (alpha - a) ** 2 + b ** 2
    sqrtK = np.sqrt(K)

    beta = sqrtK / q_ie
    gamma = q_ie * sqrtK

    J_target = np.array([
        [alpha, -beta],
        [gamma, 2.0 * a - alpha],
    ])

    # Convert the target Jacobian into W and h0
    T = np.diag([tau_e, tau_i])
    Ginv = np.diag([1.0 / g_e, 1.0 / g_i])

    W = Ginv @ (np.eye(2) + T @ J_target)
    h0 = mu0 - W @ r0

    G = np.diag([g_e, g_i])
    J = np.diag(1.0 / tau) @ (G @ W - np.eye(2))

    # Enforce the EI sign structure
    if not (
        W[0, 0] >= -tol and
        W[0, 1] <= tol and
        W[1, 0] >= -tol and
        W[1, 1] <= tol
    ):
        raise RuntimeError(f"Dale sign violation:\nW=\n{W}")

    return W, h0, J


def _syn_eig_factors(tau_syn, f, tau_decay):
    # Build the synaptic factors for the target eigenvalue
    tau_e, tau_i = np.asarray(tau_syn, dtype=float).ravel()
    a = -1.0 / tau_decay
    b = 2.0 * np.pi * f
    lam = a + 1j * b
    Ae = 1.0 + lam * tau_e
    Ai = 1.0 + lam * tau_i
    return tau_e, tau_i, a, b, Ae, Ai


def rate_qphi_to_eig_qphi(q_rate, phi_rate, tau_syn, f, tau_decay):
    # Convert rate-space q and phi into eig-space q and phi
    tau_e, tau_i, a, b, Ae, Ai = _syn_eig_factors(tau_syn, f, tau_decay)
    q_eig = q_rate * np.abs(Ae) / np.abs(Ai)
    phi_eig = phi_rate - np.angle(Ae) + np.angle(Ai)
    return q_eig, phi_eig


def _eig_phi_shift(tau_syn, f, tau_decay):
    # Compute the rate-to-eig phase shift
    tau_e, tau_i, a, b, Ae, Ai = _syn_eig_factors(tau_syn, f, tau_decay)
    return np.angle(Ae) - np.angle(Ai)


def _eig_phi_upper(tau_syn, f, tau_decay, tol=1e-12):
    # Compute the upper EI phase bound on the displayed branch
    tau_e, tau_i, a, b, Ae, Ai = _syn_eig_factors(tau_syn, f, tau_decay)
    alpha_min = max(-1.0 / tau_e, 2.0 * a + 1.0 / tau_i)
    if b == 0:
        if a >= alpha_min - tol:
            return np.pi
        return np.nan
    phi_upper = np.arctan2(b, alpha_min - a)
    if phi_upper > 0:
        return float(phi_upper)
    return np.nan


def rate_phi_bounds(tau_syn, f, tau_decay):
    # Keep the displayed branch around zero
    phi_upper = _eig_phi_upper(tau_syn, f, tau_decay)
    if not np.isfinite(phi_upper):
        return np.nan, np.nan
    lower = np.mod(_eig_phi_shift(tau_syn, f, tau_decay), np.pi)
    upper = lower + phi_upper
    if upper > np.pi:
        lower -= np.pi
        upper -= np.pi
    return lower, upper


def wrap_rate_phi_to_zero_branch(phi_ie, tau_syn, f, tau_decay):
    # Shift phi_ie onto the branch that contains zero
    lower, upper = rate_phi_bounds(tau_syn, f, tau_decay)
    if not (np.isfinite(lower) and np.isfinite(upper)):
        return np.nan
    return lower + np.mod(phi_ie - lower, np.pi)


def _raise_phi_error(phi_ie, lower, upper):
    raise ValueError(
        f"phi_ie violates Dale constraint: use ({lower:.3f}, {upper:.3f}], got {phi_ie:.3f}"
    )


def validate_rate_phi_ie(phi_ie, tau_syn, f, tau_decay, tol=1e-12):
    # Check phi_ie in rate coordinates before any eig conversion
    lower, upper = rate_phi_bounds(tau_syn, f, tau_decay)
    if not (np.isfinite(lower) and np.isfinite(upper)):
        raise ValueError("phi_ie violates Dale constraint: no valid phi_ie")
    phi_zero = wrap_rate_phi_to_zero_branch(phi_ie, tau_syn, f, tau_decay)
    if not (lower < phi_zero <= upper + tol):
        _raise_phi_error(phi_ie, lower, upper)
    return phi_zero


def _build_W_h0_ei_2d_syn_from_eig_qphi(
    model, r0, tau_syn, f, tau_decay, q_ie=1, phi_ie=0, tol=1e-12
):
    """
    Assumes almost instantaneous rate dynamics:
        x' = T_syn^{-1}[-x + r]
        r ~= gain(W @ x + h0)

    tau_syn and tau_decay must be in seconds if f is in Hz.
    Negative tau_decay is allowed and gives a positive real part
    in the target eigenvalues.
    tau_syn = [tau_e, tau_i] means synapses from E have tau_e and
    synapses from I have tau_i.

    Returns:
        W, h0, J, tau_syn_mat
    where tau_syn_mat is the 2x2 matrix to assign to model.tau_syn.
    """

    # Validate the EI synaptic-model inputs
    r0 = np.asarray(r0, dtype=float).reshape(-1, 1)
    if r0.shape != (2, 1):
        raise ValueError("r0 must have shape (2,) or (2,1)")

    tau_syn = np.asarray(tau_syn, dtype=float).ravel()
    if tau_syn.size == 1:
        tau_syn = np.repeat(tau_syn.item(), 2)
    if tau_syn.size != 2:
        raise ValueError("tau_syn must be scalar or length 2")

    tau_e, tau_i = tau_syn
    if tau_e <= 0 or tau_i <= 0:
        raise ValueError("tau_syn values must be positive")
    if tau_decay == 0:
        raise ValueError("tau_decay must be nonzero")
    if f < 0:
        raise ValueError("f must be nonnegative")

    # Extract the local gain linearization
    mu0 = np.asarray(model.gain_inv(r0), dtype=float).reshape(2, 1)
    gp = np.asarray(model.gain_deriv(r0), dtype=float).ravel()

    if gp.size != 2:
        raise ValueError("gain_deriv must return length 2")
    if np.any(gp <= 0):
        raise ValueError(f"gain derivatives must be positive; got {gp}")

    g_e, g_i = gp

    # Build the target synaptic Jacobian
    tau_e, tau_i, a, b, Ae, Ai = _syn_eig_factors(tau_syn, f, tau_decay)
    alpha = a + b / np.tan(phi_ie)

    K = (alpha - a) ** 2 + b ** 2
    sqrtK = np.sqrt(K)

    beta = sqrtK / q_ie
    gamma = q_ie * sqrtK

    J_target = np.array([
        [alpha, -beta],
        [gamma, 2.0 * a - alpha],
    ])

    # Convert the target Jacobian into W and h0
    T = np.diag([tau_e, tau_i])
    Ginv = np.diag([1.0 / g_e, 1.0 / g_i])

    W = Ginv @ (np.eye(2) + T @ J_target)
    h0 = mu0 - W @ r0

    J = np.diag(1.0 / tau_syn) @ (np.diag([g_e, g_i]) @ W - np.eye(2))

    # Enforce the EI sign structure
    if not (
        W[0, 0] >= -tol and
        W[0, 1] <= tol and
        W[1, 0] >= -tol and
        W[1, 1] <= tol
    ):
        raise RuntimeError(f"Dale sign violation:\nW=\n{W}")

    # Expand the synaptic time constants into the model matrix form
    tau_syn_mat = np.array([
        [tau_e, tau_i],
        [tau_e, tau_i],
    ])

    return W, h0, J, tau_syn_mat


def make_W_h0_ei_2d_syn(
    model, r0, tau_syn, f, tau_decay, q_ie=1, phi_ie=0, tol=1e-12
):
    # Validate phi_ie in rate space before converting to eig coordinates
    phi_zero = validate_rate_phi_ie(phi_ie, tau_syn, f, tau_decay, tol=tol)

    # Build the EI system from the equivalent eig-space q and phi
    q_eig, phi_eig = rate_qphi_to_eig_qphi(q_ie, phi_zero, tau_syn, f, tau_decay)
    return _build_W_h0_ei_2d_syn_from_eig_qphi(
        model, r0, tau_syn, f, tau_decay, q_ie=q_eig, phi_ie=phi_eig, tol=tol
    )


def make_eigmode_start_ei_2d_syn(
    W,
    r0,
    tau_syn,
    f,
    tau_decay,
    q_ie=1,
    phi_ie=0,
    dre_start=0.1,
    phase_start=0.0,
    tol=1e-12,
):
    # Validate the state-space inputs
    r0 = np.asarray(r0, dtype=float).reshape(2, 1)
    W = np.asarray(W, dtype=float).reshape(2, 2)
    tau_e, tau_i, a, b, Ae, Ai = _syn_eig_factors(tau_syn, f, tau_decay)
    phi_zero = validate_rate_phi_ie(
        phi_ie, np.array([tau_e, tau_i]), f, tau_decay, tol=tol
    )

    # Build the rate amplitudes at t=0
    re_amp = dre_start * np.exp(1j * phase_start)
    ri_amp = dre_start * q_ie * np.exp(1j * (phase_start - phi_zero))
    r_amp = np.array([re_amp, ri_amp], dtype=complex)
    r_start = r0 + np.real(r_amp).reshape(2, 1)

    # Convert the rate amplitudes to the filtered synaptic state
    x_amp = np.array([re_amp / Ae, ri_amp / Ai], dtype=complex)
    x0 = r0[:, 0] + np.real(x_amp)
    tau_syn_mat = np.array([
        [tau_e, tau_i],
        [tau_e, tau_i],
    ])
    s_start = np.zeros((2, 2), dtype=float)
    is_dyn = tau_syn_mat > 0
    s_hat = W * x0.reshape(1, 2)
    s_start[is_dyn] = s_hat[is_dyn]

    return r_start, s_start
