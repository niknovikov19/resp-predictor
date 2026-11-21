import numpy as np


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