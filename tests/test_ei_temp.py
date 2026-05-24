import numpy as np
from rate_model import RateModelWC
from dev_scratch.model_utils import make_W_h0_ei_2d


def main():
    tau = np.array([1.0, 0.5])
    f = 5.0
    tau_decay = 10.0
    r0 = np.array([5.0, 3.0])

    model = RateModelWC(
        np.zeros((2, 2)), tau=tau, rmax=10, gain_slope=0.5, gain_center=0
    )
    W, h0, J = make_W_h0_ei_2d(model, r0, tau, f, tau_decay, phi_ie=-np.pi * 1.6)
    model.W = W

    print("=" * 60)
    print("W matrix (Dale's law: col 0 E+, col 1 I-):")
    print(W)
    print()
    print("Check Dale's law:")
    print(f"  Column 0 (E): all ≥ 0? {np.all(W[:, 0] >= 0)}")
    print(f"  Column 1 (I): all ≤ 0? {np.all(W[:, 1] <= 0)}")
    print()
    print("Jacobian J:")
    print(J)
    print()
    eigenvalues = np.linalg.eigvals(J)
    print("Jacobian eigenvalues:", eigenvalues)
    print()
    print("Expected eigenvalues:")
    a = -1.0 / tau_decay
    b = 2 * np.pi * f
    print(f"  λ = {a} ± {b}i")
    print(f"  λ = {a + 1j*b}, {a - 1j*b}")
    print()
    print("Frequency from eigenvalues:")
    if np.iscomplex(eigenvalues[0]):
        freq_achieved = np.abs(eigenvalues[0].imag) / (2 * np.pi)
        print(f"  f = {freq_achieved:.4f} Hz (target: {f} Hz)")
    print("=" * 60)


if __name__ == '__main__':
    main()
