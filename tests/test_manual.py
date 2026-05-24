# Manual test of the W construction logic

# Given parameters
tau_e, tau_i = 1.0, 0.5
tau_decay = 10.0
f = 5.0
g_e, g_i = 2.0, 1.5  # example gain derivatives

# Target eigenvalues
a = -1.0 / tau_decay  # -0.1
b = 2 * 3.14159 * f   # ~31.4

print(f"Target eigenvalues: {a} ± {b}i")
print()

# Current implementation
j_ee = a
j_ii = a
j_ei = -b
j_ie = b

w_ee = (1 + tau_e * j_ee) / g_e
w_ei = (-tau_e * j_ei) / g_e
w_ie = (tau_i * j_ie) / g_i
w_ii = (1 + tau_i * j_ii) / g_i

print("Computed w values:")
print(f"  w_ee = {w_ee:.4f}")
print(f"  w_ei = {w_ei:.4f}")
print(f"  w_ie = {w_ie:.4f}")
print(f"  w_ii = {w_ii:.4f}")
print()

print("W matrix (with Dale's law signs):")
W_00 = w_ee
W_01 = -w_ei
W_10 = w_ie
W_11 = -w_ii

print(f"  W = [[{W_00:8.4f}, {W_01:8.4f}],")
print(f"       [{W_10:8.4f}, {W_11:8.4f}]]")
print()

print("Dale's law check:")
print(f"  Column 0 (E+): W[0,0]={W_00:.4f} (≥0?{W_00>=0}), W[1,0]={W_10:.4f} (≥0?{W_10>=0})")
print(f"  Column 1 (I-): W[0,1]={W_01:.4f} (≤0?{W_01<=0}), W[1,1]={W_11:.4f} (≤0?{W_11<=0})")
print()

# Verify Jacobian
G = [[g_e, 0], [0, g_i]]
J_00 = (g_e * W_00 - 1) / tau_e
J_01 = (g_e * W_01) / tau_e
J_10 = (g_i * W_10) / tau_i
J_11 = (g_i * W_11 - 1) / tau_i

print("Reconstructed Jacobian:")
print(f"  J = [[{J_00:8.4f}, {J_01:8.4f}],")
print(f"       [{J_10:8.4f}, {J_11:8.4f}]]")
print()

print("Expected Jacobian:")
print(f"  J = [[{a:8.4f}, {-b:8.4f}],")
print(f"       [{b:8.4f}, {a:8.4f}]]")
