import sympy as sp
from sympy import diff
from sympy.utilities.lambdify import lambdify
from sympy.abc import t

# ──────────────────────────────────────────────────────────────────────
# Strain history test cases
# ──────────────────────────────────────────────────────────────────────

def epsilon_tensor_case1(t):
    """Uniaxial — proportional, fixed phi (the simplest verification case)."""
    t_peak = 4  # half of t_end=8

    envelope = sp.Piecewise(
        (t / t_peak, t <= t_peak),
        ((2 - t / t_peak), t > t_peak)
    )

    eps_11 = envelope * t_peak / 100
    return sp.Matrix([[eps_11,      0,           0],
                      [0,           -eps_11/2,   0],
                      [0,           0,           -eps_11/2]])


def epsilon_tensor_case2(t):
    """Equibiaxial — proportional loading at a different phi."""
    t_peak = 4

    envelope = sp.Piecewise(
        (t / t_peak, t <= t_peak),
        ((2 - t / t_peak), t > t_peak)
    )

    eps_11 = envelope * t_peak / 100
    return sp.Matrix([[eps_11,      0,           0],
                      [0,            eps_11,     0],
                      [0,            0,         -2 * eps_11]])


def epsilon_tensor_case3(t):
    """Non-proportional biaxial — stress direction rotates in the pi-plane."""
    Amplitude = 4/100

    omega = 6.0 / (2 * sp.pi)  # rad/s

    s = sp.pi/2
    theta = omega * t

    eps_11 = Amplitude * sp.cos(theta - s)*sp.sin(theta - s)
    eps_22 = Amplitude * sp.cos(theta - s)
    eps_33 = -(eps_11 + eps_22)
    return sp.Matrix([[eps_11,    0,        0],
                      [0,         eps_22,   0],
                      [0,         0,        eps_33]])


def epsilon_tensor_case4(t):
    """Rotating principal axes — true 3D test with load–unload cycle.
    Ramps up to peak at t=t_end/2, then unloads back to zero."""
    t_peak = 4  # half of t_end=8

    # Triangular envelope: 0 → 1 → 0
    envelope = sp.Piecewise(
        (t / t_peak, t <= t_peak),
        ((2 - t / t_peak), t > t_peak)
    )

    omega = 16.0 / (2 * sp.pi) # rad/s

    eps_11 = envelope * t_peak / 200
    gamma_12 = envelope * (t_peak / 200) * sp.sin(omega*t)

    return sp.Matrix([[eps_11,     gamma_12,    0],
                      [gamma_12,  -eps_11/2,    0],
                      [0,          0,          -eps_11/2]])


def epsilon_tensor_case5(t):
    """The 'crazy' case — heavy off-diagonal entries, no symmetry in time.
    Eigenvalues and eigenvectors both evolve non-trivially."""

    # Normal strain components
    eps_11 = 0
    eps_22 = 0
    eps_33 = 0

    # Shear strain components
    eps_12 = sp.sin(t)
    eps_13 = -2*sp.sin(t)
    eps_23 = sp.cos(t)-1

    # Assemble the symmetric matrix
    return sp.Matrix([
        [eps_11, eps_12, eps_13],
        [eps_12, eps_22, eps_23],
        [eps_13, eps_23, eps_33]
    ]) / 100

def epsilon_tensor_case6(t):
    """Cyclic biaxial — eps_11 = A*sin(t), eps_22 = A*(1-cos(t)).
    The strain trajectory traces a circle in (eps_11, eps_22) space,
    producing a complex non-proportional path in the pi-plane. Tests
    formulation under continuous, smooth principal-axis rotation."""
    A = 0.01
    eps_11 = A * sp.sin(t)
    eps_22 = A * (1 - sp.cos(t))
    return sp.Matrix([[eps_11,    0,         0],
                      [0,         eps_22,    0],
                      [0,         0,         0]])


# ──────────────────────────────────────────────────────────────────────
# Case selection — change this to switch between strain histories
# ──────────────────────────────────────────────────────────────────────
STRAIN_CASES = {
    1: epsilon_tensor_case1,
    2: epsilon_tensor_case2,
    3: epsilon_tensor_case3,
    4: epsilon_tensor_case4,
    5: epsilon_tensor_case5,
    6: epsilon_tensor_case6
}

CASE = 5   # <-- change this (1 through 5) to select strain history
epsilon_tensor = STRAIN_CASES[CASE]
print(f"Using strain test case {CASE}: {epsilon_tensor.__doc__.splitlines()[0]}")

OD = "output_epsilon_tensor"

# Symbolic deviatoric strain and its rate (now uses the selected case)
eps = epsilon_tensor(t)
eps_dev = eps - (eps.trace() / 3) * sp.eye(3)
d_eps_dev_dt = diff(eps_dev, t)

eps_dev_func = lambdify(t, eps_dev, modules='numpy')
d_eps_dev_dt_func = lambdify(t, d_eps_dev_dt, modules='numpy')