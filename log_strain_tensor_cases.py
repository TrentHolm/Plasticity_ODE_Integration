"""
Deformation gradient test cases (Hencky / logarithmic strain).

Each case defines a deformation gradient F(t) as a 3×3 sympy Matrix.
The Hencky (logarithmic) strain is:
    e = (1/2) ln(B)    where B = F Fᵀ

The strain rate is the material time derivative ė, computed numerically
via the logarithmic spin so that ė is exactly the rate of e = ½ ln(B).
This is the correct rate for a hyperelastic-plastic Hencky formulation
where τ = 2G · e.

Exports eps_dev_func(t) and d_eps_dev_dt_func(t) for drop-in compatibility.
"""

import sympy as sp
import numpy as np
from sympy import diff, eye, Rational
from sympy.utilities.lambdify import lambdify
from sympy.abc import t

# ──────────────────────────────────────────────────────────────────────
# Deformation gradient test cases
# ──────────────────────────────────────────────────────────────────────

def F_case1(t):
    """Uniaxial extension — proportional, incompressible."""
    t_peak = 4
    envelope = sp.Piecewise(
        (t / t_peak, t <= t_peak),
        ((2 - t / t_peak), t > t_peak)
    )
    lam = 1 + envelope * t_peak
    lam_t = 1 / sp.sqrt(lam)
    return sp.Matrix([[lam,   0,     0],
                      [0,     lam_t, 0],
                      [0,     0,     lam_t]])


def F_case2(t):
    """Equibiaxial extension — incompressible."""
    t_peak = 4
    envelope = sp.Piecewise(
        (t / t_peak, t <= t_peak),
        ((2 - t / t_peak), t > t_peak)
    )
    lam = 1 + envelope * t_peak
    lam_3 = 1 / lam**2
    return sp.Matrix([[lam,   0,     0],
                      [0,     lam,   0],
                      [0,     0,     lam_3]])


def F_case3(t):
    """Non-proportional biaxial — two stretches evolve independently."""
    A = sp.Rational(0.0001, 100) # sp.Rational(20, 100)
    omega = 6 / (2 * sp.pi)
    s = sp.pi / 2
    theta = omega * t
    lam_1 = 1 + A * sp.cos(theta - s) * sp.sin(theta - s)  # range: A*sin(2x)/2 → ±A/2
    lam_2 = 1 + A * sp.cos(theta - s)                       # range: ±A
    lam_3 = 1 / (lam_1 * lam_2)
    return sp.Matrix([[lam_1, 0,     0],
                      [0,     lam_2, 0],
                      [0,     0,     lam_3]])


def F_case4(t):
    """Rotating principal axes — stretch + simple shear with load-unload."""
    t_peak = 4

    A = sp.Rational(20, 100) # sp.Rational(20, 100)

    envelope = sp.Piecewise(
        (t / t_peak, t <= t_peak),
        ((2 - t / t_peak), (t > t_peak) & (t <= 2 * t_peak)),
        (0, True)
    )

    omega = 16 / (2 * sp.pi)

    lam = 1 + envelope * t_peak * A
    gamma = envelope * t_peak * A * sp.sin(omega * t)

    lam_t = 1 / sp.sqrt(lam)

    return sp.Matrix([[lam,   gamma, 0],
                      [0,     lam_t, 0],
                      [0,     0,     lam_t]])


def F_case5(t):
    """Combined shear — off-diagonal dominated, upper triangular F."""
    A = sp.Rational(70, 100) # sp.Rational(20, 100)
    gamma_12 = A * sp.sin(t)
    gamma_13 = -2 * A * sp.sin(t)
    gamma_23 = A * (sp.cos(t) - 1)
    return sp.Matrix([[1,          gamma_12,   gamma_13],
                      [0,          1,          gamma_23],
                      [0,          0,          1]])


def F_case6(t):
    """Cyclic biaxial — stretches trace a circle in (λ₁, λ₂) space."""
    A = sp.Rational(70, 100) # sp.Rational(20, 100)
    lam_1 = 1 + A * sp.sin(t)
    lam_2 = 1 + A * (1 - sp.cos(t))
    lam_3 = 1 / (lam_1 * lam_2)
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
print(f"Using Hencky strain case {CASE}: {F_func_sym.__doc__.splitlines()[0]}")

OD = "output_log_strain"

# ──────────────────────────────────────────────────────────────────────
# Symbolic: F, Ḟ, L, D, W (for diagnostics and the D component of ė)
# ──────────────────────────────────────────────────────────────────────
F_sym = F_func_sym(t)
F_dot_sym = diff(F_sym, t)
F_inv_sym = F_sym.inv()

L_sym = sp.simplify(F_dot_sym * F_inv_sym)
D_sym = sp.simplify(Rational(1, 2) * (L_sym + L_sym.T))
W_sym = sp.simplify(Rational(1, 2) * (L_sym - L_sym.T))

B_sym = sp.simplify(F_sym * F_sym.T)

# Lambdified functions
F_func = lambdify(t, F_sym, modules='numpy')
F_inv_func = lambdify(t, F_inv_sym, modules='numpy')
L_func = lambdify(t, L_sym, modules='numpy')
D_func = lambdify(t, D_sym, modules='numpy')
W_func = lambdify(t, W_sym, modules='numpy')
B_func = lambdify(t, B_sym, modules='numpy')


# ──────────────────────────────────────────────────────────────────────
# Hencky strain: e = (1/2) ln(B),  B = F Fᵀ
# ──────────────────────────────────────────────────────────────────────

def hencky_strain(t_val):
    """Compute e = (1/2) ln(B) via eigendecomposition."""
    B = np.array(B_func(t_val), dtype=float)
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals = np.maximum(eigvals, 1e-30)
    log_B = np.zeros((3, 3))
    for i in range(3):
        n = eigvecs[:, i]
        log_B += np.log(eigvals[i]) * np.outer(n, n)
    return 0.5 * log_B


def hencky_strain_dev(t_val):
    """Deviatoric Hencky strain."""
    e = hencky_strain(t_val)
    return e - (np.trace(e) / 3.0) * np.eye(3)


def eps_dev_func(t_val):
    """Drop-in: deviatoric Hencky strain at time t."""
    return hencky_strain_dev(t_val)


# ──────────────────────────────────────────────────────────────────────
# Logarithmic strain rate: ė = D + Ω^log e - e Ω^log
#
# The logarithmic spin Ω^log is defined (Xiao, Bruhns, Meyers 1997) as:
#
#   Ω^log = W + Σ_{i≠j} [ (1 + (λⱼ/λᵢ)) / (1 - (λⱼ/λᵢ))
#                          + 2/ln(λⱼ/λᵢ) ] · (nᵢ⊗nⱼ) · (nⱼᵀ D nᵢ)
#
# where λᵢ are eigenvalues of B and nᵢ the corresponding eigenvectors.
# When two eigenvalues coalesce (λᵢ ≈ λⱼ), the bracket → 0 (removable
# singularity), so the spin reduces to W.
# ──────────────────────────────────────────────────────────────────────

def _log_spin_coefficient(lam_i, lam_j):
    """Compute the coefficient for the Ω^log correction term.

    For eigenvalues λᵢ, λⱼ of B (not squared stretches — these ARE λ² already
    from B = FFᵀ):

        coeff = (1 + r) / (1 - r) + 2 / ln(r)

    where r = λⱼ / λᵢ. When r → 1 the expression → 0 (L'Hôpital).
    """
    r = lam_j / lam_i
    if abs(r - 1.0) < 1e-10:
        # Taylor expansion: coeff ≈ -(1/3)(r-1) + O((r-1)²)
        return -(r - 1.0) / 3.0
    log_r = np.log(r)
    return (1.0 + r) / (1.0 - r) + 2.0 / log_r


def logarithmic_spin(D_val, W_val, B_val):
    """Compute the logarithmic spin Ω^log from D, W, and B at a single time."""
    eigvals, eigvecs = np.linalg.eigh(B_val)
    eigvals = np.maximum(eigvals, 1e-30)

    Omega_log = W_val.copy()

    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            ni = eigvecs[:, i]
            nj = eigvecs[:, j]
            coeff = _log_spin_coefficient(eigvals[i], eigvals[j])
            # nⱼᵀ D nᵢ
            D_ji = nj @ D_val @ ni
            # Accumulate: coeff * (nᵢ ⊗ nⱼ) * D_ji
            Omega_log += coeff * np.outer(ni, nj) * D_ji

    return Omega_log


def hencky_strain_rate(t_val):
    """Compute the material time derivative of e = ½ ln(B).

    ė = D + Ω^log · e - e · Ω^log

    This is the logarithmic rate of the Hencky strain, which is
    exactly D when expressed in the log-rate framework:
        e̊^log = D
    but the MATERIAL time derivative ė ≠ D in general.
    """
    D_val = np.array(D_func(t_val), dtype=float)
    W_val = np.array(W_func(t_val), dtype=float)
    B_val = np.array(B_func(t_val), dtype=float)
    e_val = hencky_strain(t_val)

    Omega_log = logarithmic_spin(D_val, W_val, B_val)

    e_dot = D_val + Omega_log @ e_val - e_val @ Omega_log
    return e_dot


def hencky_strain_rate_dev(t_val):
    """Deviatoric part of the Hencky strain rate."""
    e_dot = hencky_strain_rate(t_val)
    return e_dot - (np.trace(e_dot) / 3.0) * np.eye(3)


def d_eps_dev_dt_func(t_val):
    """Drop-in: deviatoric Hencky strain rate at time t.

    Returns ė_dev, the material time derivative of the deviatoric
    logarithmic strain. For a Hencky material (τ = 2G·e), the
    elastic relation in rate form is:
        τ̊^log = 2G · ė_dev
    which has the same structure as the small-strain relation
        σ̇ = 2G · ε̇_dev
    making the solver code identical.
    """
    return hencky_strain_rate_dev(t_val)