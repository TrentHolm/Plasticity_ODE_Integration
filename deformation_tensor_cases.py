"""
Deformation gradient test cases (small-strain approximation).

Each case defines a deformation gradient F(t) = I + H(t) where H is small.
The strain is computed as the symmetric small-strain tensor:
    ε = (H + Hᵀ) / 2
and its deviatoric part and rate are exported for the plasticity solver,
making this a drop-in replacement for epsilon_tensor_cases.
"""

import sympy as sp
from sympy import diff, eye, Rational
from sympy.utilities.lambdify import lambdify
from sympy.abc import t

# ──────────────────────────────────────────────────────────────────────
# Deformation gradient test cases
# ──────────────────────────────────────────────────────────────────────

def F_case1(t, nu=sp.Rational(1, 2)):
    """Uniaxial extension — transverse contraction set by Poisson's ratio."""
    t_peak = 4
    envelope = sp.Piecewise(
        (t / t_peak, t <= t_peak),
        ((2 - t / t_peak), t > t_peak)
    )
    lam = 1 + envelope * t_peak
    lam_t = lam**(-nu)                      # ν=1/2 → 1/sqrt(lam)
    return sp.Matrix([[lam,   0,     0],
                      [0,     lam_t, 0],
                      [0,     0,     lam_t]])


def F_case2(t, nu=sp.Rational(1, 2)):
    """Equibiaxial extension — out-of-plane contraction set by Poisson's ratio."""
    t_peak = 4
    envelope = sp.Piecewise(
        (t / t_peak, t <= t_peak),
        ((2 - t / t_peak), t > t_peak)
    )
    lam = 1 + envelope * t_peak
    lam_3 = lam**(-2 * nu / (1 - nu))       # ν=1/2 → 1/lam**2
    return sp.Matrix([[lam,   0,     0],
                      [0,     lam,   0],
                      [0,     0,     lam_3]])


def F_case3(t, nu=sp.Rational(1, 2)):
    """Non-proportional biaxial — two stretches evolve independently."""
    A = sp.Rational(20, 100)
    omega = 6 / (2 * sp.pi)
    s = sp.pi / 2
    theta = omega * t
    lam_1 = 1 + A * sp.cos(theta - s) * sp.sin(theta - s)  # range: A*sin(2x)/2 → ±A/2
    lam_2 = 1 + A * sp.cos(theta - s)                       # range: ±A
    lam_3 = (lam_1 * lam_2)**(-nu / (1 - nu))               # ν=1/2 → 1/(lam_1*lam_2)
    return sp.Matrix([[lam_1, 0,     0],
                      [0,     lam_2, 0],
                      [0,     0,     lam_3]])


def F_case4(t, nu=sp.Rational(1, 2)):
    """Rotating principal axes — stretch + simple shear with load-unload."""
    t_peak = 4

    A = sp.Rational(20, 100)

    envelope = sp.Piecewise(
        (t / t_peak, t <= t_peak),
        ((2 - t / t_peak), (t > t_peak) & (t <= 2 * t_peak)),
        (0, True)
    )

    omega = 16 / (2 * sp.pi)

    lam = 1 + envelope * t_peak * A
    gamma = envelope * t_peak * A * sp.sin(omega * t)

    lam_t = lam**(-nu)                      # ν=1/2 → 1/sqrt(lam)

    return sp.Matrix([[lam,   gamma, 0],
                      [0,     lam_t, 0],
                      [0,     0,     lam_t]])


def F_case5(t, nu=sp.Rational(1, 2)):
    """Combined shear — off-diagonal dominated, upper triangular F.

    Simple shear is isochoric for *any* material (det F = 1 regardless
    of the off-diagonal terms), and there are no normal stretches to
    contract. Poisson's ratio therefore has no kinematic role here; nu
    is accepted only to keep a uniform signature across the cases.
    """
    A = sp.Rational(20, 100)
    gamma_12 = A * sp.sin(t)
    gamma_13 = -2 * A * sp.sin(t)
    gamma_23 = A * (sp.cos(t) - 1)
    return sp.Matrix([[1,          gamma_12,   gamma_13],
                      [0,          1,          gamma_23],
                      [0,          0,          1]])


def F_case6(t, nu=sp.Rational(1, 2)):
    """Cyclic biaxial — stretches trace a circle in (λ₁, λ₂) space."""
    A = sp.Rational(20, 100)
    lam_1 = 1 + A * sp.sin(t)
    lam_2 = 1 + A * (1 - sp.cos(t))
    lam_3 = (lam_1 * lam_2)**(-nu / (1 - nu))   # ν=1/2 → 1/(lam_1*lam_2)
    return sp.Matrix([[lam_1, 0,     0],
                      [0,     lam_2, 0],
                      [0,     0,     lam_3]])


# ──────────────────────────────────────────────────────────────────────
# Case selection
# ──────────────────────────────────────────────────────────────────────
F_CASES = {
    1: F_case1,
    2: F_case2,
    3: F_case3,
    4: F_case4,
    5: F_case5,
    6: F_case6,
}

CASE = 3  # <-- change this to select case
F_func_sym = F_CASES[CASE]
print(f"Using deformation gradient case {CASE}: {F_func_sym.__doc__.splitlines()[0]}")

OD = "output_def_tensor"

# ──────────────────────────────────────────────────────────────────────
# Symbolic derivation: F → ε → ε_dev → dε_dev/dt
# ──────────────────────────────────────────────────────────────────────
F = F_func_sym(t, sp.Rational(3, 10))
H = F - eye(3)

# Small-strain tensor: ε = (H + Hᵀ) / 2
eps = Rational(1, 2) * (H + H.T)

# Deviatoric strain
eps_dev = eps - (eps.trace() / 3) * eye(3)

# Strain rate
d_eps_dev_dt = diff(eps_dev, t)

# ──────────────────────────────────────────────────────────────────────
# Lambdified numerical functions (drop-in compatible)
# ──────────────────────────────────────────────────────────────────────
eps_dev_func = lambdify(t, eps_dev, modules='numpy')
d_eps_dev_dt_func = lambdify(t, d_eps_dev_dt, modules='numpy')