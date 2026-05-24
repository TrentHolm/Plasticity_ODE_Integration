"""
Test harness for verifying stress-return-mapping implementations
against the analytical ODE solution for von Mises + linear isotropic iso_hardening.

Strain path
-----------
    eps_11 = A sin(theta)
    eps_22 = A (1 - cos(theta))
    eps_33 = 0,  no shear

Usage
-----
1.  Implement your return-mapping as a callable with the signature:

        stress_new, eps_bar_p_new = my_return_map(
            stress_old,      # (6,) Voigt stress  [s11 s22 s33 s12 s23 s13]
            eps_bar_p_old,   # scalar accumulated plastic strain
            delta_eps,       # (6,) Voigt strain increment
            mat,             # dict with keys E, nu, G, K, sigma_y0, H
        )

2.  Pass it to  run_convergence_study(my_return_map)  and you're done.
    The harness will sweep over step counts, compare against the ODE
    reference, print a table, and produce diagnostic plots.

Two example implementations are provided:
    - radial_return_principal   (works in principal deviatoric stress)
    - radial_return_voigt       (works in full 6-component Voigt notation)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════
# Material & loading configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Material:
    E: float        = 200.0e3
    nu: float       = 0.3
    sigma_y0: float = 250.0
    H: float        = 1.0e3

    def __post_init__(self):
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))
        self.lam = self.K - 2.0 * self.G / 3.0   # Lamé lambda

    def as_dict(self):
        return dict(E=self.E, nu=self.nu, G=self.G, K=self.K,
                    lam=self.lam, sigma_y0=self.sigma_y0, H=self.H)


@dataclass
class LoadPath:
    amplitude: float  = 0.01
    theta_end: float  = 4.0 * np.pi   # 2 full cycles

    def strain_voigt(self, theta):
        """Full 6-component Voigt strain [e11 e22 e33 e12 e23 e13]."""
        return np.array([
            self.amplitude * np.sin(theta),
            self.amplitude * (1.0 - np.cos(theta)),
            0.0, 0.0, 0.0, 0.0,
        ])


# ═══════════════════════════════════════════════════════════════════════
# Analytical ODE reference  (in orthonormal deviatoric space)
# ═══════════════════════════════════════════════════════════════════════

sqrt32 = np.sqrt(1.5)
sqrt23 = np.sqrt(2.0 / 3.0)


class AnalyticalSolution:
    """Integrate the reduced (phi, eps_bar_p) ODE system."""

    def __init__(self, mat: Material, load: LoadPath):
        self.mat  = mat
        self.load = load
        self._solve()

    # -- deviatoric helpers ------------------------------------------
    def _deviatoric(self, theta):
        e = self.load.strain_voigt(theta)[:3]
        tr = e.sum()
        return e - tr / 3.0

    def _alpha_beta(self, theta):
        d = self._deviatoric(theta)
        return np.array([(d[0] - d[1]) / np.sqrt(2.0),
                         (d[0] + d[1]) * sqrt32])

    def _alpha_beta_dot(self, theta):
        A = self.load.amplitude
        c, s = np.cos(theta), np.sin(theta)
        da = A / np.sqrt(2.0) * (c - s)
        db = A / np.sqrt(6.0) * (c + s)
        return da, db

    def _yield_radius(self, ebp):
        return (self.mat.sigma_y0 + self.mat.H * ebp) / (2.0 * self.mat.G * sqrt32)

    # -- find first yield --------------------------------------------
    def _elastic_vm(self, theta):
        ab = self._alpha_beta(theta)
        return 2.0 * self.mat.G * sqrt32 * np.linalg.norm(ab)

    def _find_theta_yield(self):
        th = np.linspace(1e-8, self.load.theta_end, 200000)
        vm = np.array([self._elastic_vm(t) for t in th])
        idx = np.argmax(vm >= self.mat.sigma_y0)
        if vm[idx] < self.mat.sigma_y0:
            raise RuntimeError("Loading never reaches yield.")
        return brentq(lambda t: self._elastic_vm(t) - self.mat.sigma_y0,
                      th[max(idx - 1, 0)], th[idx])

    # -- ODE integration ---------------------------------------------
    def _ode_rhs(self, theta, y):
        phi, ebp = y
        da, db   = self._alpha_beta_dot(theta)
        R        = self._yield_radius(ebp)
        G, H     = self.mat.G, self.mat.H

        radial     = np.cos(phi) * da + np.sin(phi) * db
        tangential = -np.sin(phi) * da + np.cos(phi) * db

        if radial <= 0.0:          # elastic unloading / neutral
            return [tangential / R, 0.0]

        depbp = 2.0 * G * sqrt32 * radial / (3.0 * G + H)
        return [tangential / R, depbp]

    def _solve(self):
        self.theta_y = self._find_theta_yield()
        ab0  = self._alpha_beta(self.theta_y)
        phi0 = np.arctan2(ab0[1], ab0[0])

        sol = solve_ivp(
            self._ode_rhs,
            [self.theta_y, self.load.theta_end],
            [phi0, 0.0],
            method='RK45', max_step=1e-4, rtol=1e-12, atol=1e-14,
            dense_output=True,
        )
        self._sol   = sol
        self.theta  = sol.t
        self.phi    = sol.y[0]
        self.ebp    = sol.y[1]
        self.ebp_final = sol.y[1, -1]

    def evaluate(self, theta_arr):
        """Interpolate eps_bar_p onto an arbitrary theta array."""
        out = np.zeros_like(theta_arr)
        mask = theta_arr >= self.theta_y
        if mask.any():
            vals = self._sol.sol(theta_arr[mask])
            out[mask] = vals[1]
        return out


# ═══════════════════════════════════════════════════════════════════════
# Example return-mapping implementations
# ═══════════════════════════════════════════════════════════════════════

def _deviatoric_from_voigt(sig):
    """Extract deviatoric part of a Voigt stress vector."""
    p = (sig[0] + sig[1] + sig[2]) / 3.0
    dev = sig.copy()
    dev[:3] -= p
    return dev


def _von_mises(sig):
    """Von Mises equivalent stress from 6-component Voigt stress."""
    s = _deviatoric_from_voigt(sig)
    # s:s  =  s11^2 + s22^2 + s33^2 + 2*(s12^2 + s23^2 + s13^2)
    J2 = 0.5 * (s[0]**2 + s[1]**2 + s[2]**2) + s[3]**2 + s[4]**2 + s[5]**2
    return np.sqrt(3.0 * J2)


def radial_return_principal(stress_old, ebp_old, deps, mat):
    """
    Radial return in principal deviatoric stress.
    Assumes diagonal loading (no shear) so principal dirs = coord axes.
    """
    G, H, sy0 = mat['G'], mat['H'], mat['sigma_y0']
    lam = mat['lam']

    # Full elastic trial stress (isotropic Hooke)
    tr_deps = deps[0] + deps[1] + deps[2]
    sig_tr  = stress_old.copy()
    sig_tr[:3] += lam * tr_deps + 2.0 * G * deps[:3]
    sig_tr[3]  += 2.0 * G * deps[3]        # sig_12
    sig_tr[4]  += 2.0 * G * deps[4]        # sig_23
    sig_tr[5]  += 2.0 * G * deps[5]        # sig_13

    sig_eq_tr = _von_mises(sig_tr)
    sig_y     = sy0 + H * ebp_old

    if sig_eq_tr <= sig_y * (1.0 + 1e-14):
        return sig_tr, ebp_old              # elastic step

    dgamma  = (sig_eq_tr - sig_y) / (3.0 * G + H)
    s_trial = _deviatoric_from_voigt(sig_tr)
    ratio   = 1.0 - 3.0 * G * dgamma / sig_eq_tr

    sig_new       = sig_tr.copy()
    sig_new[:3]   = s_trial[:3] * ratio + (sig_tr[0] + sig_tr[1] + sig_tr[2]) / 3.0
    sig_new[3:]   = s_trial[3:] * ratio

    return sig_new, ebp_old + dgamma


def radial_return_voigt(stress_old, ebp_old, deps, mat):
    """
    Identical algorithm, just to show a second implementation that can
    be swapped in.  Works with full Voigt notation throughout.
    """
    # Intentionally identical to the above — replace this body with
    # YOUR implementation to test.
    return radial_return_principal(stress_old, ebp_old, deps, mat)


# ═══════════════════════════════════════════════════════════════════════
# Test driver
# ═══════════════════════════════════════════════════════════════════════

def run_return_mapping(return_map_fn, mat: Material, load: LoadPath, n_steps: int):
    """
    Drive an arbitrary return-mapping function along the prescribed
    strain path and record the full state history.
    """
    thetas  = np.linspace(0.0, load.theta_end, n_steps + 1)
    stress  = np.zeros(6)
    ebp     = 0.0
    eps_old = load.strain_voigt(0.0)
    m       = mat.as_dict()

    hist_theta = [0.0]
    hist_ebp   = [0.0]
    hist_sig   = [stress.copy()]
    hist_vm    = [0.0]

    for i in range(1, n_steps + 1):
        eps_new = load.strain_voigt(thetas[i])
        deps    = eps_new - eps_old

        stress, ebp = return_map_fn(stress, ebp, deps, m)

        hist_theta.append(thetas[i])
        hist_ebp.append(ebp)
        hist_sig.append(stress.copy())
        hist_vm.append(_von_mises(stress))

        eps_old = eps_new

    return {
        'theta':  np.array(hist_theta),
        'ebp':    np.array(hist_ebp),
        'stress': np.array(hist_sig),
        'vm':     np.array(hist_vm),
    }


def run_convergence_study(
    return_map_fn,
    mat:  Material  = None,
    load: LoadPath  = None,
    step_counts:    list = None,
    label:          str  = "user return map",
    plot_filename:  str  = "return_map_verification.png",
):
    """
    Full verification suite:
      1. Compute the analytical ODE reference
      2. Run the supplied return mapping at each step count
      3. Print convergence table
      4. Generate diagnostic plots

    Returns the AnalyticalSolution and a dict of results.
    """
    if mat  is None: mat  = Material()
    if load is None: load = LoadPath()
    if step_counts is None:
        step_counts = [10, 20, 50, 100, 200, 500, 1000, 5000]

    # ── analytical reference ────────────────────────────────────────
    ref = AnalyticalSolution(mat, load)
    print(f"Material: E={mat.E}, nu={mat.nu}, "
          f"sigma_y0={mat.sigma_y0}, H={mat.H}")
    print(f"Loading:  amplitude={load.amplitude}, "
          f"theta_end={load.theta_end:.4f} ({load.theta_end/np.pi:.1f}π)")
    print(f"First yield at theta = {ref.theta_y:.6f} rad "
          f"({np.degrees(ref.theta_y):.2f}°)")
    print(f"Analytical eps_bar_p(end) = {ref.ebp_final:.12e}\n")

    # ── convergence sweep ───────────────────────────────────────────
    results = {}
    errors_ebp  = []
    errors_path = []

    print(f"{'N':>7s}  {'eps_bar_p':>18s}  "
          f"{'rel err (final)':>15s}  {'max path err':>15s}")
    print("-" * 62)

    for N in step_counts:
        res = run_return_mapping(return_map_fn, mat, load, N)
        results[N] = res

        # Error at final theta
        err_final = abs(res['ebp'][-1] - ref.ebp_final) / max(ref.ebp_final, 1e-30)

        # Max error along the path (interpolate reference onto RM thetas)
        ref_interp = ref.evaluate(res['theta'])
        err_path   = np.max(np.abs(res['ebp'] - ref_interp)) / max(ref.ebp_final, 1e-30)

        errors_ebp.append(err_final)
        errors_path.append(err_path)

        print(f"{N:7d}  {res['ebp'][-1]:18.12e}  "
              f"{err_final:15.6e}  {err_path:15.6e}")

    # ── estimate convergence rate ───────────────────────────────────
    log_dt  = np.log(load.theta_end / np.array(step_counts, dtype=float))
    log_err = np.log(np.array(errors_ebp) + 1e-30)
    # Fit slope over the finer half of the data
    n_fit   = max(len(step_counts) // 2, 2)
    coeffs  = np.polyfit(log_dt[-n_fit:], log_err[-n_fit:], 1)
    print(f"\nEstimated convergence order ≈ {coeffs[0]:.2f}")

    # ── helper: yield ellipse in (s1, s2) deviatoric stress space ──
    def _yield_ellipse_s1s2(sigma_y):
        """
        Von Mises in principal deviatoric space:
            s1^2 + s1*s2 + s2^2 = sigma_y^2 / 3
        Parametrize via orthonormal coords (alpha_s, beta_s) where
        the surface is a circle of radius R = sigma_y * sqrt(2/3),
        then invert to (s1, s2).
        """
        R = sigma_y * sqrt23              # sqrt(2/3)
        t = np.linspace(0, 2 * np.pi, 500)
        a_s = R * np.cos(t)
        b_s = R * np.sin(t)
        # Invert:  alpha_s = (s1-s2)/sqrt2,  beta_s = sqrt(3/2)(s1+s2)
        #   =>  s1 = alpha_s/sqrt2 + beta_s/sqrt6
        #        s2 = -alpha_s/sqrt2 + beta_s/sqrt6
        s1 = a_s / np.sqrt(2.0) + b_s / np.sqrt(6.0)
        s2 = -a_s / np.sqrt(2.0) + b_s / np.sqrt(6.0)
        return s1, s2, a_s, b_s, R

    def _stress_to_dev(sig_arr):
        """(N,6) Voigt stress → (s1, s2) principal deviatoric."""
        p  = (sig_arr[:, 0] + sig_arr[:, 1] + sig_arr[:, 2]) / 3.0
        s1 = sig_arr[:, 0] - p
        s2 = sig_arr[:, 1] - p
        return s1, s2

    def _dev_to_ortho(s1, s2):
        """(s1,s2) principal deviatoric → orthonormal (alpha_s, beta_s)."""
        return (s1 - s2) / np.sqrt(2.0), (s1 + s2) * sqrt32

    # ── plots ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    # (a) Strain path
    ax = axes[0, 0]
    th = np.linspace(0, load.theta_end, 2000)
    ax.plot(load.amplitude * np.sin(th),
            load.amplitude * (1.0 - np.cos(th)), 'k-', lw=1.2)
    ax.set_xlabel(r'$\varepsilon_{11}$')
    ax.set_ylabel(r'$\varepsilon_{22}$')
    ax.set_title('Strain load path')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # (b) Von Mises stress vs theta
    ax = axes[0, 1]
    for N in [step_counts[0], step_counts[len(step_counts)//2], step_counts[-1]]:
        r = results[N]
        ax.plot(r['theta'], r['vm'], lw=1.2, label=f'N={N}')
    sy_line = mat.sigma_y0 + mat.H * ref.ebp
    ax.plot(ref.theta, sy_line, 'k--', lw=1, alpha=0.5, label=r'$\sigma_y(\bar\varepsilon^p)$')
    ax.set_xlabel(r'$\theta$ [rad]')
    ax.set_ylabel(r'$\sigma_{\rm eq}$ [MPa]')
    ax.set_title('Von Mises stress')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Stress path in (s1, s2) deviatoric stress space
    ax = axes[1, 0]
    # Initial yield ellipse
    s1_y0, s2_y0, _, _, _ = _yield_ellipse_s1s2(mat.sigma_y0)
    ax.plot(s1_y0, s2_y0, 'b-', lw=1.5, label=rf'$\sigma_{{y0}}={mat.sigma_y0:.0f}$')
    # Final yield ellipse (from finest run)
    N_fine = step_counts[-1]
    ebp_end = results[N_fine]['ebp'][-1]
    sy_final = mat.sigma_y0 + mat.H * ebp_end
    s1_yf, s2_yf, _, _, _ = _yield_ellipse_s1s2(sy_final)
    ax.plot(s1_yf, s2_yf, 'r-', lw=1.5,
            label=rf'$\sigma_{{y,final}}={sy_final:.1f}$')
    # Stress path (finest run)
    s1_path, s2_path = _stress_to_dev(results[N_fine]['stress'])
    ax.plot(s1_path, s2_path, 'k-', lw=0.8, alpha=0.8, label=f'stress path (N={N_fine})')
    ax.plot(s1_path[0], s2_path[0], 'go', ms=8, zorder=5, label='start')
    ax.plot(s1_path[-1], s2_path[-1], 'rs', ms=8, zorder=5, label='end')
    ax.set_xlabel(r'$s_1$ [MPa]')
    ax.set_ylabel(r'$s_2$ [MPa]')
    ax.set_title(r'Stress path in $(s_1, s_2)$ deviatoric space')
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # (d) Stress path in orthonormal deviatoric space (circles)
    ax = axes[1, 1]
    _, _, a0, b0, R0 = _yield_ellipse_s1s2(mat.sigma_y0)
    circ_t = np.linspace(0, 2*np.pi, 500)
    ax.plot(R0 * np.cos(circ_t), R0 * np.sin(circ_t),
            'b-', lw=1.5, label=rf'$\sigma_{{y0}}={mat.sigma_y0:.0f}$')
    Rf = sy_final * sqrt23
    ax.plot(Rf * np.cos(circ_t), Rf * np.sin(circ_t),
            'r-', lw=1.5, label=rf'$\sigma_{{y,final}}={sy_final:.1f}$')
    # Stress path in orthonormal coords
    a_path, b_path = _dev_to_ortho(s1_path, s2_path)
    ax.plot(a_path, b_path, 'k-', lw=0.8, alpha=0.8, label=f'stress path (N={N_fine})')
    ax.plot(a_path[0], b_path[0], 'go', ms=8, zorder=5)
    ax.plot(a_path[-1], b_path[-1], 'rs', ms=8, zorder=5)
    ax.set_xlabel(r'$\alpha_s$ [MPa]')
    ax.set_ylabel(r'$\beta_s$ [MPa]')
    ax.set_title(r'Stress path in orthonormal deviatoric space (yield = circles)')
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # (e) Accumulated plastic strain
    ax = axes[2, 0]
    ax.plot(ref.theta, ref.ebp, 'k-', lw=2.5, label='analytical (ODE)')
    for N in [step_counts[0], step_counts[len(step_counts)//2], step_counts[-1]]:
        r = results[N]
        ax.plot(r['theta'], r['ebp'], '--', lw=1.2, label=f'{label} N={N}')
    ax.set_xlabel(r'$\theta$ [rad]')
    ax.set_ylabel(r'$\bar{\varepsilon}^p$')
    ax.set_title('Accumulated plastic strain')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (f) Convergence
    ax = axes[2, 1]
    dthetas = load.theta_end / np.array(step_counts, dtype=float)
    ax.loglog(dthetas, errors_ebp,  'ko-', lw=1.5, ms=6, label='final-point error')
    ax.loglog(dthetas, errors_path, 's--', color='0.4', lw=1.2, ms=5,
              label='max path error')
    dt_ref = np.array([dthetas[0], dthetas[-1]])
    ax.loglog(dt_ref, errors_ebp[0] * (dt_ref / dt_ref[0])**1,
              'b:', lw=1, label=r'$O(\Delta\theta)$')
    ax.loglog(dt_ref, errors_ebp[0] * (dt_ref / dt_ref[0])**2,
              'r:', lw=1, label=r'$O(\Delta\theta^2)$')
    ax.set_xlabel(r'$\Delta\theta$')
    ax.set_ylabel('Relative error')
    ax.set_title(f'Convergence  (rate ≈ {coeffs[0]:.2f})')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    plt.suptitle(f'Return-mapping verification: {label}', fontsize=13, y=1.005)
    plt.tight_layout()
    plt.savefig(f'{plot_filename}', dpi=180,
                bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved → {plot_filename}")

    return ref, results


# ═══════════════════════════════════════════════════════════════════════
# Run if executed directly
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 62)
    print("  EXAMPLE 1: radial return (principal deviatoric)")
    print("=" * 62)
    run_convergence_study(
        radial_return_principal,
        label="radial return (principal)",
        plot_filename="verify_principal.png",
    )

    print("\n")
    print("=" * 62)
    print("  EXAMPLE 2: same algorithm, different H")
    print("=" * 62)
    run_convergence_study(
        radial_return_principal,
        mat=Material(H=5.0e3),
        label="radial return (H=5000)",
        plot_filename="verify_high_hardening.png",
    )

    print("\n")
    print("=" * 62)
    print("  EXAMPLE 3: perfect plasticity (H=0)")
    print("=" * 62)
    run_convergence_study(
        radial_return_principal,
        mat=Material(H=0.0),
        label="radial return (perfect plasticity)",
        plot_filename="verify_perfect_plasticity.png",
    )