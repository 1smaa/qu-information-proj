import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
# Identity matrix
I = sp.Matrix([[1, 0], [0, 1]])

def partial_trace_B(rho: sp.Matrix):
    """Computes the partial trace over qubit B (tr_B(rho))."""
    return (
        (rho[0:2, 0:2]) + (rho[2:4, 2:4])
    )  # Summing over 2x2 blocks corresponding to traced-out qubit

def partial_trace_A(rho: sp.Matrix):
    """Computes the partial trace over qubit A (tr_A(rho))."""
    return (
        (rho[0:4:2, 0:4:2]) + (rho[1:4:2, 1:4:2])
    )  # Summing over alternating 2x2 blocks

def von_neumann(rho: sp.Matrix):
    """Computes the von Neumann entropy S(rho) = -Tr(rho log2(rho))"""
    eigenvals = rho.eigenvals()
    entropy = 0
    for eig in eigenvals:
        if eig != 0:  # Avoid log(0)
            entropy += eig * sp.log(eig, 2)  # Use SymPy log base 2
    return -sp.simplify(entropy)

# Define parameter p
p = sp.symbols("p")

# Density matrix
rho = sp.Matrix([
    [2 - p + 2 * sp.sqrt(1 - p), 0, p, 0],
    [0, p, 0, p],
    [p, 0, 2 - p - 2 * sp.sqrt(1 - p), 0],
    [0, p, 0, p]
])

# Compute reduced density matrices
rho_A = sp.simplify(partial_trace_B(rho))
rho_B = sp.simplify(partial_trace_A(rho))

# Compute Quantum Discord Entropy Formula: S(A) + S(B) - S(AB)
result = sp.simplify(von_neumann(rho_A) + von_neumann(rho_B) - von_neumann(rho))
p_space=np.linspace(0,1,100)
d=[result.subs(p,p_v) for p_v in p_space]
plt.plot(p_space,d)
plt.grid(True)
plt.show()