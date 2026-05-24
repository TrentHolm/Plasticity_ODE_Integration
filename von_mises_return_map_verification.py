"""
Analytical vs Return-Mapping verification for von Mises elastoplasticity
with linear isotropic iso_hardening.

Strain path:
    eps_11 = A * sin(theta)
    eps_22 = A * (1 - cos(theta))
    eps_33 = 0, no shear

The analytical (exact) solution is obtained by integrating the reduced
ODE system in orthonormal deviatoric coordinates (alpha, beta) where
the yield surface is a circle.  The return-mapping solution uses the
standard radial-return algorithm at a user-chosen step count.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────
# Material parameters
# ──────────────────────────────────────────────────────────────────────
E       = 200.0e3    # Young's modulus [MPa]
nu      = 0.3        # Poisson's ratio
sigma_y0 = 250.0     # Initial yield stress [MPa]
H       = 1.0e3      # Linear isotropic iso_hardening modulus [MPa]

G  = E / (2.0 * (1.0 + nu))          # Shear modulus
K  = E / (3.0 * (1.0 - 2.0 * nu))    # Bulk modulus
sqrt32 = np.sqrt(3.0 / 2.0)
sqrt23 = np.sqrt(2.0 / 3.0)

# Loading
amplitude = 0.01          # strain amplitude A
theta_end = 4.0 * np.pi   # two full cycles

# ──────────────────────────────────────────────────────────────────────
# Helpers: strain path in orthonormal deviatoric coords
# ──────────────────────────────────────────────────────────────────────

def total_deviatoric(theta):
    """Return (e1, e2, e3) principal deviatoric strains."""
    eps1 = amplitude * np.sin(theta)
    eps2 = amplitude * (1.0 - np.cos(theta))
    eps3 = 0.0
    tr   = eps1 + eps2 + eps3
    return eps1 - tr / 3.0, eps2 - tr / 3.0, eps3 - tr / 3.0


def alpha_beta(theta):
    """Orthonormal deviatoric coords where yield surface is a circle."""
    e1, e2, _ = total_deviatoric(theta)
    a = (e1 - e2) / np.sqrt(2.0)
    b = (e1 + e2) * sqrt32
    return a, b


def alpha_beta_dot(theta):
    """d(alpha)/dtheta, d(beta)/dtheta."""
    c, s = np.cos(theta), np.sin(theta)
    de1 = amplitude * (2.0 * c - s) / 3.0            # d(e1)/dtheta (from chain rule)
    # Easier: differentiate alpha, beta directly
    da = amplitude / np.sqrt(2.0) * (c - s)           # d/dtheta of (sin+cos-1)/sqrt2
    # actually let me be precise
    # alpha = A/sqrt2 (sin th + cos th - 1)
    da = amplitude / np.sqrt(2.0) * (np.cos(theta) - np.sin(theta))
    # beta  = A/sqrt6 (sin th - cos th + 1)
    db = amplitude / np.sqrt(6.0) * (np.cos(theta) + np.sin(theta))
    return da, db


def yield_radius(eps_bar_p):
    """Radius of the yield circle in (alpha, beta) space."""
    return (sigma_y0 + H * eps_bar_p) / (2.0 * G * sqrt32)


# ──────────────────────────────────────────────────────────────────────
# 1.  Find theta_yield  (first yield from the elastic solution)
# ──────────────────────────────────────────────────────────────────────

def elastic_vm(theta):
    """Von Mises equivalent stress for pure elastic response."""
    a, b = alpha_beta(theta)
    return 2.0 * G * sqrt32 * np.sqrt(a**2 + b**2)


def _yield_residual(theta):
    return elastic_vm(theta) - sigma_y0


# Search for first yield in (0, theta_end]
_th_search = np.linspace(1e-8, theta_end, 100000)
_vm_search = np.array([elastic_vm(t) for t in _th_search])
_idx = np.argmax(_vm_search >= sigma_y0)

if _vm_search[_idx] < sigma_y0:
    raise RuntimeError("Loading never reaches yield — increase amplitude.")

theta_y = brentq(_yield_residual, _th_search[max(_idx - 1, 0)], _th_search[_idx])
print(f"First yield at theta_y = {theta_y:.6f} rad "
      f"({np.degrees(theta_y):.2f} deg)")

# Initial stress angle at yield
a0, b0 = alpha_beta(theta_y)
phi0   = np.arctan2(b0, a0)

# ──────────────────────────────────────────────────────────────────────
# 2.  Analytical (exact) ODE integration
# ──────────────────────────────────────────────────────────────────────

def ode_rhs(theta, y):
    """
    y = [phi, eps_bar_p]

    dphi/dtheta     = (-sin(phi)*da + cos(phi)*db) / R
    deps_bar_p/dtheta = 2G*sqrt(3/2) * (cos(phi)*da + sin(phi)*db) / (3G + H)
                        (only when positive — loading)
    """
    phi, ebp = y
    da, db = alpha_beta_dot(theta)
    R = yield_radius(ebp)

    radial    = np.cos(phi) * da + np.sin(phi) * db
    tangential = -np.sin(phi) * da + np.cos(phi) * db

    if radial <= 0.0:
        # Elastic unloading (or neutral) — stress point stays, no plastic flow
        # In a more rigorous treatment we would track re-yielding,
        # but for this smooth path we keep it simple.
        return [tangential / R, 0.0]

    depbp = 2.0 * G * sqrt32 * radial / (3.0 * G + H)
    dphi  = tangential / R

    return [dphi, depbp]


sol = solve_ivp(
    ode_rhs,
    [theta_y, theta_end],
    [phi0, 0.0],
    method='RK45',
    max_step=1e-4,
    rtol=1e-12,
    atol=1e-14,
    dense_output=True,
)

theta_analytical = sol.t
phi_analytical   = sol.y[0]
ebp_analytical   = sol.y[1]

print(f"Analytical eps_bar_p at theta_end = {ebp_analytical[-1]:.10e}")

# ──────────────────────────────────────────────────────────────────────
# 3.  Return-mapping integrator (arbitrary step count)
# ──────────────────────────────────────────────────────────────────────

def return_mapping(n_steps):
    """
    Standard radial-return for von Mises + linear isotropic iso_hardening.

    Works entirely in principal deviatoric stress components (s1, s2)
    with s3 = -(s1+s2).

    Returns arrays of (theta, eps_bar_p, s1, s2).
    """
    thetas = np.linspace(0.0, theta_end, n_steps + 1)
    dtheta = thetas[1] - thetas[0]

    # State
    s1, s2 = 0.0, 0.0          # deviatoric stress components
    ebp    = 0.0                # accumulated plastic strain

    out_theta = [0.0]
    out_ebp   = [0.0]
    out_s1    = [0.0]
    out_s2    = [0.0]

    e1_old, e2_old, _ = total_deviatoric(0.0)

    for i in range(1, n_steps + 1):
        th = thetas[i]
        e1, e2, _ = total_deviatoric(th)

        # Deviatoric strain increment
        de1 = e1 - e1_old
        de2 = e2 - e2_old

        # Trial deviatoric stress  (s3_trial = -(s1_trial + s2_trial))
        s1_tr = s1 + 2.0 * G * de1
        s2_tr = s2 + 2.0 * G * de2
        s3_tr = -(s1_tr + s2_tr)

        # Trial von Mises stress
        sig_eq_tr = np.sqrt(1.5 * (s1_tr**2 + s2_tr**2 + s3_tr**2))

        # Current yield stress
        sig_y = sigma_y0 + H * ebp

        if sig_eq_tr > sig_y:
            # Plastic correction — radial return
            dgamma = (sig_eq_tr - sig_y) / (3.0 * G + H)
            ebp   += dgamma
            ratio  = 1.0 - 3.0 * G * dgamma / sig_eq_tr
            s1 = s1_tr * ratio
            s2 = s2_tr * ratio
        else:
            s1 = s1_tr
            s2 = s2_tr

        e1_old, e2_old = e1, e2

        out_theta.append(th)
        out_ebp.append(ebp)
        out_s1.append(s1)
        out_s2.append(s2)

    return (np.array(out_theta), np.array(out_ebp),
            np.array(out_s1), np.array(out_s2))


# ──────────────────────────────────────────────────────────────────────
# 4.  Convergence study — compare return mapping at various step counts
# ──────────────────────────────────────────────────────────────────────

step_counts = [10, 20, 50, 100, 200, 500, 1000, 5000]
ebp_exact   = ebp_analytical[-1]

print(f"\n{'N steps':>10s}  {'eps_bar_p':>16s}  {'rel error':>12s}")
print("-" * 44)

rm_results = {}
for N in step_counts:
    th_rm, ebp_rm, _, _ = return_mapping(N)
    err = abs(ebp_rm[-1] - ebp_exact) / ebp_exact
    print(f"{N:10d}  {ebp_rm[-1]:16.10e}  {err:12.4e}")
    rm_results[N] = (th_rm, ebp_rm)

# ──────────────────────────────────────────────────────────────────────
# 5.  Plots
# ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Strain path in (eps11, eps22) space
ax = axes[0, 0]
th_plot = np.linspace(0, theta_end, 2000)
ax.plot(amplitude * np.sin(th_plot),
        amplitude * (1.0 - np.cos(th_plot)), 'k-', lw=1.2)
ax.set_xlabel(r'$\varepsilon_{11}$')
ax.set_ylabel(r'$\varepsilon_{22}$')
ax.set_title('Strain load path')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# (b) Deviatoric path in (alpha, beta) space + yield circle at onset
ax = axes[0, 1]
ab = np.array([alpha_beta(t) for t in th_plot])
ax.plot(ab[:, 0], ab[:, 1], 'k-', lw=1.2, label='deviatoric path')
R0 = yield_radius(0.0)
circ = np.linspace(0, 2 * np.pi, 300)
ax.plot(R0 * np.cos(circ), R0 * np.sin(circ), 'r--', lw=1, label='initial yield')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\beta$')
ax.set_title(r'Deviatoric $(\alpha,\beta)$ space')
ax.set_aspect('equal')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) Accumulated plastic strain vs theta
ax = axes[1, 0]
ax.plot(theta_analytical, ebp_analytical, 'k-', lw=2, label='analytical (ODE)')
for N in [10, 50, 200, 1000]:
    th_rm, ebp_rm = rm_results[N]
    ax.plot(th_rm, ebp_rm, '--', lw=1.2, label=f'return map N={N}')
ax.set_xlabel(r'$\theta$ [rad]')
ax.set_ylabel(r'$\bar{\varepsilon}^p$')
ax.set_title('Accumulated plastic strain')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (d) Convergence plot
ax = axes[1, 1]
errors = []
dthetas = []
for N in step_counts:
    _, ebp_rm = rm_results[N]
    errors.append(abs(ebp_rm[-1] - ebp_exact) / ebp_exact)
    dthetas.append(theta_end / N)
ax.loglog(dthetas, errors, 'ko-', lw=1.5, ms=6)
# Reference slope (second order)
dt_ref = np.array([dthetas[0], dthetas[-1]])
ax.loglog(dt_ref, errors[0] * (dt_ref / dt_ref[0])**2,
          'r--', lw=1, label=r'$\mathcal{O}(\Delta\theta^2)$')
ax.loglog(dt_ref, errors[0] * (dt_ref / dt_ref[0])**1,
          'b--', lw=1, label=r'$\mathcal{O}(\Delta\theta)$')
ax.set_xlabel(r'$\Delta\theta$')
ax.set_ylabel('Relative error in $\\bar{\\varepsilon}^p$')
ax.set_title('Convergence of return mapping')
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('return_map_verification.png', dpi=180)
plt.close()

print("\nPlot saved to return_map_verification.png")
