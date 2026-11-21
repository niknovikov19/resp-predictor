import numpy as np

import matplotlib.pyplot as plt


def wilson_cowan(u, v, a, b, c, d, I, tau_u, tau_v):
    du = (-u + (a * u - b * u * v + I)) / tau_u
    dv = (-v + (c * u * v - d * v)) / tau_v
    return du, dv

def simulate_wilson_cowan(T, dt, u0, v0, a, b, c, d, I, tau_u, tau_v):
    num_steps = int(T / dt)
    u = np.zeros(num_steps)
    v = np.zeros(num_steps)
    u[0] = u0
    v[0] = v0

    for t in range(1, num_steps):
        du, dv = wilson_cowan(u[t-1], v[t-1], a, b, c, d, I, tau_u, tau_v)
        u[t] = u[t-1] + du * dt
        v[t] = v[t-1] + dv * dt

    return u, v


# Parameters
T = 100.0
dt = 0.01
u0 = 0.5
v0 = 0.5
a = 1.0
b = 1.0
c = 1.0
d = 1.0
I = 0.5
tau_u = 1.0
tau_v = 1.0

# Simulation
u, v = simulate_wilson_cowan(T, dt, u0, v0, a, b, c, d, I, tau_u, tau_v)

# Linearization around the fixed point (u*, v*)
u_star = (I + d) / (a + d)
v_star = (I + a) / (b + c)

# Jacobian matrix
J = np.array([
    [(-1 + a - b * v_star) / tau_u, (-b * u_star) / tau_u],
    [(c * v_star) / tau_v, (c * u_star - d) / tau_v]
])

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(J)

print("Fixed point (u*, v*):", (u_star, v_star))
print("Jacobian matrix at the fixed point:\n", J)
print("Eigenvalues of the Jacobian matrix:", eigenvalues)
print("Eigenvectors of the Jacobian matrix:\n", eigenvectors)

# Plotting
time = np.arange(0, T, dt)
plt.plot(time, u, label='u (Excitatory)')
plt.plot(time, v, label='v (Inhibitory)')
plt.xlabel('Time')
plt.ylabel('Activity')
plt.legend()
plt.title('Wilson-Cowan Model Simulation')
plt.show()