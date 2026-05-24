import numpy as np
import sympy as sp
from sympy import diff, Function
from sympy.utilities.lambdify import lambdify
from sympy.abc import t

from scipy.integrate import solve_ivp

# ──────────────────────────────────────────────────────────────────────
# Strain history test cases
# ──────────────────────────────────────────────────────────────────────

def epsilon_tensor_case1(t):
    """Uniaxial — proportional, fixed phi (the simplest verification case)."""
    eps_11 = t / 100
    return sp.Matrix([[eps_11,      0,           0],
                      [0,           -eps_11/2,   0],
                      [0,           0,           -eps_11/2]])


def epsilon_tensor_case2(t):
    """Equibiaxial — proportional loading at a different phi.
    Tests whether the formulation is correct at angles other than phi_0 = pi/2."""
    eps_11 = t / 100
    return sp.Matrix([[eps_11,      0,           0],
                      [0,            eps_11,     0],
                      [0,            0,         -2 * eps_11]])


def epsilon_tensor_case3(t):
    """Non-proportional biaxial — stress direction rotates in the pi-plane,
    exercising phi_dot. Eigenvectors stay axis-aligned, so principal-value
    rates reduce to simple diagonal time derivatives."""
    eps_11 = t / 100
    eps_22 = (t / 100) * sp.sin(t / 2)
    eps_33 = -(eps_11 + eps_22)
    return sp.Matrix([[eps_11,    0,        0],
                      [0,         eps_22,   0],
                      [0,         0,        eps_33]])


def epsilon_tensor_case4(t):
    """Rotating principal axes — true 3D test. Shear develops over time,
    rotating the principal frame. Exercises the full n_i^T A_dot n_i
    projection and the principal-axis spin coupling."""
    eps_11 = t / 100
    gamma_12 = (t / 200) * sp.sin(t)
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

CASE = 3   # <-- change this (1 through 5) to select strain history
epsilon_tensor = STRAIN_CASES[CASE]
print(f"Using strain test case {CASE}: {epsilon_tensor.__doc__.splitlines()[0]}")


# Symbolic deviatoric strain and its rate (now uses the selected case)
eps = epsilon_tensor(t)
eps_dev = eps - (eps.trace() / 3) * sp.eye(3)
d_eps_dev_dt = diff(eps_dev, t)

eps_dev_func = lambdify(t, eps_dev, modules='numpy')
d_eps_dev_dt_func = lambdify(t, d_eps_dev_dt, modules='numpy')


def principal_values(t_val):
    """Sorted (descending) principal values of the deviatoric strain at t_val."""
    A = np.array(eps_dev_func(t_val), dtype=float)
    return np.sort(np.linalg.eigvalsh(A))[::-1]


def principal_value_rates_analytical(t_val):
    """d(lambda_i)/dt = n_i^T A_dot n_i, sorted to match principal_values."""
    A = np.array(eps_dev_func(t_val), dtype=float)
    A_dot = np.array(d_eps_dev_dt_func(t_val), dtype=float)
    lambdas, N = np.linalg.eigh(A)
    order = np.argsort(lambdas)[::-1]
    N = N[:, order]
    return np.einsum('ji,jk,ki->i', N, A_dot, N)

# Module-level state for continuous eigenvector tracking
_prev_eigvecs = None

def principal_value_rates_analytical(t_val, reset=False):
    """d(lambda_i)/dt = n_i^T A_dot n_i, with eigenvectors tracked
    continuously across calls (no magnitude-based sort)."""
    global _prev_eigvecs

    A = np.array(eps_dev_func(t_val), dtype=float)
    A_dot = np.array(d_eps_dev_dt_func(t_val), dtype=float)

    lambdas, N = np.linalg.eigh(A)  # ascending order, arbitrary

    if reset or _prev_eigvecs is None:
        # First call: sort once descending and remember
        order = np.argsort(lambdas)[::-1]
        N = N[:, order]
        lambdas = lambdas[order]
    else:
        # Match new eigenvectors to previous ones by max |dot product|
        # Build a 3x3 alignment matrix |N_prev^T @ N_new|
        align = np.abs(_prev_eigvecs.T @ N)
        # Greedy assignment: pick best match for each previous column
        assignment = np.zeros(3, dtype=int)
        used = set()
        for i in range(3):
            row = align[i].copy()
            for j in used:
                row[j] = -1
            best = int(np.argmax(row))
            assignment[i] = best
            used.add(best)
        # Reorder new eigvecs/eigvals to match previous labeling
        N = N[:, assignment]
        lambdas = lambdas[assignment]
        # Fix signs so each new eigvec aligns with previous (not anti-aligned)
        for i in range(3):
            if np.dot(_prev_eigvecs[:, i], N[:, i]) < 0:
                N[:, i] = -N[:, i]

    _prev_eigvecs = N.copy()
    return np.einsum('ji,jk,ki->i', N, A_dot, N)


def principal_value_rates_fd(t_val, h=1e-3):
    """5-point central finite difference of the sorted principal values."""
    stencil = [-2, -1, 1, 2]
    coeffs = [1, -8, 8, -1]
    return sum(c * principal_values(t_val + s * h)
               for c, s in zip(coeffs, stencil)) / (12 * h)


def compare_analytical_vs_fd(t_vals=(0.5, 2.0, 5.0), h=1e-3):
    """Compare analytical principal-value rates against 5-point FD."""
    for t_val in t_vals:
        analytical = principal_value_rates_analytical(t_val)
        fd = principal_value_rates_fd(t_val, h=h)
        print(f"\n--- t = {t_val} ---")
        print(f"Analytical: {analytical}")
        print(f"5-pt FD:    {fd}")
        print(f"Difference: {analytical - fd}")


def project_onto_pi_plane(principal_values):
    alpha = (principal_values[1] - principal_values[2]) / np.sqrt(2)
    beta = (2 * principal_values[0] - principal_values[1] - principal_values[2]) / np.sqrt(6)
    return alpha, beta


def project_onto_tangential_and_radial_basis(alpha_eps_dots, beta_eps_dot, phi):
    alpha_r_eps_dot = alpha_eps_dots * np.cos(phi) + beta_eps_dot * np.sin(phi)
    beta_theta_eps_dot = -alpha_eps_dots * np.sin(phi) + beta_eps_dot * np.cos(phi)

    return alpha_r_eps_dot, beta_theta_eps_dot


# == == == == == == == == == == == == == == == ==
# --------------- Voce Hardening ----------------
# == == == == == == == == == == == == == == == ==
b = 20.0
R_inf = 130.0
sigma_y0 = 200.0  # initial yield stress

def kappa_voce(eps_bar):
    return R_inf * (1 - np.exp(-b * eps_bar))

def H_iso_voce(eps_bar):
    return b * R_inf * np.exp(-b * eps_bar)

def sigma_y_voce(eps_bar):
    """von Mises yield stress with Voce isotropic iso_hardening."""
    return sigma_y0 + kappa_voce(eps_bar)

# == == == == == == == == == == == == == == == ==
# ------------ Linear Isotropic -----------------
# == == == == == == == == == == == == == == == ==
H_lin_iso = 500.0  # constant iso_hardening modulus

def kappa_linear(eps_bar):
    return H_lin_iso * eps_bar

def H_iso_linear():
    return H_lin_iso

def sigma_y_linear(eps_bar):
    """von Mises yield stress with linear isotropic iso_hardening."""
    return sigma_y0 + kappa_linear(eps_bar)

# == == == == == == == == == == == == == == == ==
# --------------- Swift Isotropic ---------------
# == == == == == == == == == == == == == == == ==

K = 55.0
n = 0.07
eps_0 = 0.001
def kappa_swift(eps_bar):
    return K * (eps_0 + eps_bar)**n

def H_iso_swift(eps_bar):
    return K * n * (eps_0 + eps_bar)**(n - 1)  # fixed: exponent should be n-1

def sigma_y_swift(eps_bar):
    """von Mises yield stress with Swift isotropic iso_hardening."""
    return kappa_swift(eps_bar)


# == == == == == == == == == == == == == == == ==
# ---- Voce-Swift Combined Isotropic ------------
# == == == == == == == == == == == == == == == ==
alpha_vs = 0.5  # mixing parameter, 0 = pure Swift, 1 = pure Voce

def sigma_y_voce_swift(eps_bar):
    """Combined Voce-Swift isotropic iso_hardening yield stress."""
    return alpha_vs * sigma_y_voce(eps_bar) + (1 - alpha_vs) * sigma_y_swift(eps_bar)

def H_iso_voce_swift(eps_bar):
    """Hardening modulus for combined Voce-Swift."""
    return alpha_vs * H_iso_voce(eps_bar) + (1 - alpha_vs) * H_iso_swift(eps_bar)

# == == == == == == == == == == == == == == == ==
# ------------ Linear Kinematic -----------------
# == == == == == == == == == == == == == == == ==
H_lin_kin = 500
b_kin = 0.5

def beta_dot_proportional(N, lam_dot, H_kin):
    """Linear (Prager) kinematic hardening: beta_dot = (2/3) H_kin * eps_p_dot.
       With eps_p_dot = sqrt(3/2) * lam_dot * N, this becomes
       beta_dot = (2/3) * H_kin * sqrt(3/2) * lam_dot * N."""
    return (2.0 / 3.0) * H_kin * lam_dot * N

def H_kin_linear():
    return H_lin_kin

# == == == == == == == == == == == == == == == ==
# ------- Armstrong-Frederick Kinematic ---------
# == == == == == == == == == == == == == == == ==
def beta_dot_armstrong_frederick(N, lam_dot, beta, H_kin):
    """Armstrong-Frederick: beta_dot = (2/3) H_kin * eps_p_dot - b_kin * beta * eps_bar_p_dot.
       Use lam_dot as the equivalent plastic strain rate (== eps_bar_p_dot in this convention)."""
    return (2.0 / 3.0) * H_kin * lam_dot * N - b_kin * beta * lam_dot

# ====================================================
# - - - - - - - Dispatcher Functions - - - - - - - - -
# ====================================================

def H_iso_dispatcher(eps_bar, iso_hardening):
    if iso_hardening == 'voce': return H_iso_voce(eps_bar)
    if iso_hardening == 'swift': return H_iso_swift(eps_bar)
    if iso_hardening == 'voce_swift': return H_iso_voce_swift(eps_bar)
    if iso_hardening == 'linear': return H_iso_linear()
    if iso_hardening == 'none': return 0.0
    else: raise ValueError(f"Unknown isotropic hardening model: {iso_hardening}")

def H_kin_dispatcher(eps_bar, kin_hardening):
    if kin_hardening == 'linear': return H_kin_linear()
    if kin_hardening == 'none': return 0.0
    else: raise ValueError(f"Unknown kinematic hardening model: {kin_hardening}")

def sigma_y_dispatcher(eps_bar, iso_hardening):
    if iso_hardening == 'linear': return sigma_y_linear(eps_bar)
    if iso_hardening == 'voce': return sigma_y_voce(eps_bar)
    if iso_hardening == 'swift': return sigma_y_swift(eps_bar)
    if iso_hardening == 'voce_swift': return sigma_y_voce_swift(eps_bar)
    if iso_hardening == 'none': return sigma_y0
    else: raise ValueError(f"Unknown isotropic iso_hardening model: {iso_hardening}")

def beta_dot_dispatcher(N, lam_dot, beta, H_kin, kin_hardening):
    if kin_hardening == "proportional": return beta_dot_proportional(N, lam_dot, H_kin)
    if kin_hardening == "armstrong-frederick": return beta_dot_armstrong_frederick(N, lam_dot, beta, H_kin)
    if kin_hardening == "none": return np.zeros([3,3])
    else: raise ValueError(f"Unknown kinematic iso_hardening model: {kin_hardening}")

# == == == == == == == == == == == == == == == ==
# --------------- Coupled ODEs ------------------
# == == == == == == == == == == == == == == == ==
E       = 200.0e3    # Young's modulus [MPa]
nu      = 0.3        # Poisson's ratio
H       = 500        # Linear isotropic hardening modulus [MPa]

G  = E / (2.0 * (1.0 + nu))          # Shear modulus

def gamma_dot(hardening_term_sum, alpha_r_eps_dot):
    """d(eps_bar_p)/dt under associative von Mises flow."""
    return (np.sqrt(6) * G) / (3 * G + hardening_term_sum) * alpha_r_eps_dot

def phi_dot(R_sigma_val, beta_theta_eps_dot):
    t1 = 2 * G / R_sigma_val
    return t1 * beta_theta_eps_dot

def ode_rhs_stress(t_val, y,
                   iso_hardening='linear',
                   kin_hardening_H='none',
                   kin_hardening_beta='none'):
    """RHS for solve_ivp tracking full deviatoric stress and backstress.
    y = [s11, s22, s33, s12, s13, s23, eps_bar_p, b11, b22, b33, b12, b13, b23]
    """
    s = np.array([[y[0], y[3], y[4]],
                  [y[3], y[1], y[5]],
                  [y[4], y[5], y[2]]])
    eps_bar = y[6]
    beta = np.array([[y[7], y[10], y[11]],
                     [y[10], y[8], y[12]],
                     [y[11], y[12], y[9]]])

    eta = s - beta
    e_dot = np.array(d_eps_dev_dt_func(t_val), dtype=float)
    eta_norm = np.sqrt(np.tensordot(eta, eta))

    sigma_y = sigma_y_dispatcher(eps_bar, iso_hardening)
    sigma_eq = np.sqrt(1.5) * eta_norm

    eps_bar_dot = 0.0
    beta_dot = np.zeros((3, 3))

    f = sigma_eq - sigma_y

    N = np.zeros((3, 3))
    if eta_norm > 1e-10:
        N = eta / eta_norm

    N_dot_e = np.tensordot(N, e_dot)

    if f < -1e-12 or N_dot_e < 0.0: # If not at the yield surface or non-plastic loading then pure elasticity
        s_dot = 2 * G * e_dot
        eps_bar_dot = 0.0
        beta_dot.fill(0.0)

    else:   # If at the yield surface and loading then plastic yielding
        if N_dot_e <= 0:
            s_dot = 2 * G * e_dot
            eps_bar_dot = 0.0
        else:
            H_iso = H_iso_dispatcher(eps_bar, iso_hardening)
            H_kin = H_kin_dispatcher(eps_bar, kin_hardening_H)

            lam_dot = 3 * G * N_dot_e / (3 * G + H_iso + H_kin)
            eps_bar_dot = np.sqrt(2/3) * lam_dot

            # Plastic strain rate (note: factor sqrt(3/2) for unit-N convention)
            eps_p_dot = lam_dot * N
            s_dot = 2 * G * (e_dot - eps_p_dot)
            beta_dot = beta_dot_dispatcher(N, lam_dot, beta, H_kin, kin_hardening_beta)

    return [s_dot[0,0], s_dot[1,1], s_dot[2,2],
            s_dot[0,1], s_dot[0,2], s_dot[1,2],
            eps_bar_dot,
            beta_dot[0,0], beta_dot[1,1], beta_dot[2,2],
            beta_dot[0,1], beta_dot[0,2], beta_dot[1,2]]

# ──────────────────────────────────────────────────────────────────────
# Return-mapping integrator (arbitrary step count)
# ──────────────────────────────────────────────────────────────────────

def return_mapping(n_steps, t_end,
                   iso_hardening='linear',
                   kin_hardening_H='none',
                   kin_hardening_beta='none'):
    """Radial return for von Mises + isotropic + kinematic hardening."""
    ts = np.linspace(0.0, t_end, n_steps + 1)

    sigma_dev = np.zeros((3, 3))
    beta = np.zeros((3, 3))  # backstress
    ebp = 0.0

    ebp_history = np.zeros(n_steps + 1)
    s_norm_hist = np.zeros(n_steps + 1)
    sigma_hist = np.zeros((n_steps + 1, 3, 3))
    beta_hist = np.zeros((n_steps + 1, 3, 3))

    eps_dev_old = np.array(eps_dev_func(ts[0]), dtype=float)

    for i in range(1, n_steps + 1):
        eps_dev_new = np.array(eps_dev_func(ts[i]), dtype=float)
        delta_eps_dev = eps_dev_new - eps_dev_old

        # Elastic trial deviatoric stress and relative stress eta = s - beta
        sigma_trial = sigma_dev + 2.0 * G * delta_eps_dev
        eta_trial = sigma_trial - beta
        eta_norm_tr = np.sqrt(np.tensordot(eta_trial, eta_trial))
        sig_eq_tr = np.sqrt(1.5) * eta_norm_tr

        sig_y = sigma_y_dispatcher(ebp, iso_hardening=iso_hardening)

        if sig_eq_tr > sig_y:
            # Newton iteration on dgamma
            dgamma = 0.0
            for _ in range(50):
                ebp_new = ebp + dgamma
                sy = sigma_y_dispatcher(ebp_new, iso_hardening=iso_hardening)
                H_iso = H_iso_dispatcher(ebp_new, iso_hardening=iso_hardening)
                H_kin = H_kin_dispatcher(ebp_new, kin_hardening=kin_hardening_H)

                # NOTE: for linear (Prager) kinematic only.
                # Armstrong-Frederick requires a tensor return mapping;
                # this scalar equation is exact only for linear kinematic.
                f = sig_eq_tr - 3.0 * G * dgamma - H_kin * dgamma - sy
                if abs(f) < 1e-10:
                    break
                df = -3.0 * G - H_iso - H_kin
                dgamma -= f / df

            # Update direction (in eta-space, not sigma-space)
            N_hat = eta_trial / eta_norm_tr
            sigma_dev = sigma_trial - 2.0 * G * dgamma * np.sqrt(3.0/2.0) * N_hat

            # Update backstress (linear / Prager: beta_dot = (2/3) H_kin * eps_p_dot)
            if kin_hardening_beta == 'proportional':
                beta = beta + (2.0 / 3.0) * H_kin * dgamma * N_hat  # <-- removed sqrt(1.5), use N_hat
            elif kin_hardening_beta == 'armstrong-frederick':
                beta = beta + dgamma * ((2.0 / 3.0) * H_kin * N_hat - b_kin * beta)
            # 'none': beta stays zero

            ebp += dgamma
        else:
            sigma_dev = sigma_trial
            # beta unchanged in elastic step

        eps_dev_old = eps_dev_new

        ebp_history[i] = ebp
        s_norm_hist[i] = np.sqrt(np.tensordot(sigma_dev, sigma_dev))
        sigma_hist[i] = sigma_dev
        beta_hist[i] = beta

    return ts, ebp_history, s_norm_hist, sigma_hist, beta_hist

def plot_results(sol, t_analytical, gamma_analytical, ebp_exact, rm_results,
                 t_end, step_counts, iso_hardening, kin_hardening_H, kin_hardening_beta,
                 sy0, t_plot, eps_path, alpha_s, beta_s, alpha_eps, beta_eps,
                 rm_stress_paths,
                 save_path='return_map_verification.png',
                 t_now=None, anim_axes=None):
    """
    Draw the 4-panel summary plot. If `anim_axes` is provided, draw into those
    axes (for animation reuse). If `t_now` is provided, animate the paths up to
    that time only (the yield-circle visuals reflect the state at t_now).

    Returns the figure and axes so callers can save or animate.
    """
    import matplotlib.pyplot as plt

    if anim_axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        axes = anim_axes
        fig = axes[0, 0].get_figure()
        for ax in axes.flat:
            ax.clear()

    hardening_label = f"iso={iso_hardening}, kin H={kin_hardening_H}, kin β={kin_hardening_beta}"
    if t_now is None:
        fig.suptitle(f"Case {CASE} — {hardening_label}", fontsize=11)
    else:
        fig.suptitle(f"Case {CASE} — {hardening_label}   |   t = {t_now:.3f}", fontsize=11)

    # If animating, mask data beyond t_now
    if t_now is not None:
        path_mask = t_plot <= t_now
        ode_mask = t_analytical <= t_now
    else:
        path_mask = np.ones_like(t_plot, dtype=bool)
        ode_mask = np.ones_like(t_analytical, dtype=bool)

    # (a) Strain pi-plane projection
    ax = axes[0, 0]
    ax.plot(alpha_eps[path_mask], beta_eps[path_mask], 'k-', lw=1.2,
            label='strain path')

    R_init = np.sqrt(2.0 / 3.0) * sy0

    # Initial yield circle in strain-space (radius = R_init / (2G))
    R_init_eps = R_init / (2.0 * G)
    circ = np.linspace(0, 2 * np.pi, 300)

    ax.plot(R_init_eps * np.cos(circ), R_init_eps * np.sin(circ),
            'r', lw=0.5, dashes=[4,2], label='initial yield (strain)')

    # Lock axes for animation consistency
    pad = 1.2
    lim_eps = max(np.abs(alpha_eps).max(), np.abs(beta_eps).max(), R_init_eps) * pad
    ax.set_xlim(-lim_eps, lim_eps)
    ax.set_ylim(-lim_eps, lim_eps)
    ax.set_xlabel(r'$\alpha_\varepsilon$')
    ax.set_ylabel(r'$\beta_\varepsilon$')
    ax.set_title(f'Strain $(\\alpha,\\beta)$ pi-plane (c{CASE})')
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # (b) Stress pi-plane + yield surfaces
    ax = axes[0, 1]
    ax.plot(alpha_s[ode_mask], beta_s[ode_mask], 'k-', lw=1.2, label='stress path (ODE)')

    rm_colors = {10: 'tab:orange', 50: 'tab:green', 200: 'tab:purple', 1000: 'tab:cyan'}
    for N, (ts_rm_N, alpha_rm, beta_rm) in rm_stress_paths.items():
        if t_now is not None:
            mask = ts_rm_N <= t_now
        else:
            mask = np.ones_like(ts_rm_N, dtype=bool)
        ax.plot(alpha_rm[mask], beta_rm[mask], '-.', lw=1.0,
                color=rm_colors.get(N, 'gray'),
                label=f'RM N={N}', alpha=0.7)

    # Determine current eps_bar_p for sizing the "current" yield circle
    if t_now is not None:
        idx_now = np.searchsorted(t_analytical, t_now)
        idx_now = min(idx_now, len(t_analytical) - 1)
        ebp_now = gamma_analytical[idx_now]
        b_now = sol.y[7:13, idx_now]
    else:
        ebp_now = ebp_exact
        b_now = sol.y[7:13, -1]

    ax.plot(R_init * np.cos(circ), R_init * np.sin(circ),
            'r', lw=0.5, dashes=[4,2], label='initial yield')

    sy_now = sigma_y_dispatcher(ebp_now, iso_hardening=iso_hardening)
    R_now = np.sqrt(2.0 / 3.0) * sy_now
    label_now = ('final yield' if t_now is None else 'current yield')
    ax.plot(R_now * np.cos(circ), R_now * np.sin(circ),
            'b', lw=0.5, dashes=[4,2], label=f'{label_now} ($\\bar\\varepsilon^p$={ebp_now:.3f})')

    if kin_hardening_beta != 'none':
        alpha_b = (b_now[0] - b_now[1]) / np.sqrt(2.0)
        beta_b = (2 * b_now[2] - b_now[0] - b_now[1]) / np.sqrt(6.0)
        ax.plot(R_now * np.cos(circ) + alpha_b,
                R_now * np.sin(circ) + beta_b,
                'g', lw=0.5, dashes=[4,2], label=f'{label_now} (with backstress)')

    # Lock axes for consistent animation frames
    pad = 1.2
    lim = max(np.abs(alpha_s).max(), np.abs(beta_s).max(), R_init) * pad
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r'$\alpha$ [stress]')
    ax.set_ylabel(r'$\beta$ [stress]')
    ax.set_title(fr'Stress $(\alpha,\beta)$ space (c{CASE})')
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # (c) Accumulated plastic strain vs t
    ax = axes[1, 0]
    ax.plot(t_analytical[ode_mask], gamma_analytical[ode_mask], 'k-', lw=2,
            label='ODE (reference)')
    if t_now is None:
        for N in [10, 50, 200, 1000]:
            if N in rm_results:
                ts_rm, ebp_rm, _ = rm_results[N]
                ax.plot(ts_rm, ebp_rm, '--', lw=1.2, label=f'return map N={N}')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\bar{\varepsilon}^p$')
    ax.set_title(f'Accumulated plastic strain (c{CASE})')
    ax.set_xlim(0, t_end)
    ax.set_ylim(0, gamma_analytical.max() * 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Convergence
    ax = axes[1, 1]
    if t_now is None:
        errors, dts = [], []
        for N in step_counts:
            _, ebp_rm, _ = rm_results[N]
            errors.append(abs(ebp_rm[-1] - ebp_exact) / max(ebp_exact, 1e-30))
            dts.append(t_end / N)
        ax.loglog(dts, errors, 'ko-', lw=1.5, ms=6, label='RM vs ODE')
        dt_ref = np.array([dts[0], dts[-1]])
        ax.loglog(dt_ref, errors[0] * (dt_ref / dt_ref[0])**2,
                  'r--', lw=1, label=r'$\mathcal{O}(\Delta t^2)$')
        ax.loglog(dt_ref, errors[0] * (dt_ref / dt_ref[0])**1,
                  'b--', lw=1, label=r'$\mathcal{O}(\Delta t)$')
        ax.set_xlabel(r'$\Delta t$')
        ax.set_ylabel(r'Relative error in $\bar{\varepsilon}^p$')
        ax.set_title(f'Convergence (c{CASE})')
        ax.legend(fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
    else:
        # In animation mode, swap convergence for an info panel
        ax.text(0.5, 0.5,
                f"t = {t_now:.3f}\n"
                f"$\\bar\\varepsilon^p$ = {ebp_now:.4f}\n"
                f"$\\sigma_y$ = {sy_now:.1f}\n"
                f"R = {R_now:.1f}",
                ha='center', va='center', fontsize=14,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightyellow'))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('Current state')

    plt.tight_layout()

    if anim_axes is None:
        plt.savefig(save_path, dpi=180)
        print(f"Plot saved to {save_path}")

    return fig, axes


def animate_results(sol, t_analytical, gamma_analytical, ebp_exact, rm_results,
                    t_end, step_counts, iso_hardening, kin_hardening_H, kin_hardening_beta,
                    sy0, t_plot, eps_path, alpha_s, beta_s, alpha_eps, beta_eps,
                    rm_stress_paths,
                    save_path='return_map_animation.mp4',
                    n_frames=180, fps=30, dpi=180):
    """
    Animate the four-panel plot over t in [0, t_end] and save as .mp4 or .gif.
    Use save_path ending in '.mp4' for video, '.gif' for animated GIF.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    frame_times = np.linspace(0, t_end, n_frames)

    def update(frame_idx):
        t_now = frame_times[frame_idx]
        plot_results(sol, t_analytical, gamma_analytical, ebp_exact, rm_results,
                     t_end, step_counts, iso_hardening, kin_hardening_H, kin_hardening_beta,
                     sy0, t_plot, eps_path, alpha_s, beta_s, alpha_eps, beta_eps,
                     rm_stress_paths,
                     t_now=t_now, anim_axes=axes)
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)

    if save_path.lower().endswith('.gif'):
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, bitrate=4000, codec='libx264')

    print(f"Rendering {n_frames} frames at {fps} fps → {save_path}")
    anim.save(save_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Animation saved to {save_path}")

def main():
    import matplotlib.pyplot as plt

    t_end = 14.0
    iso_hardening = 'linear'      # 'linear' | 'voce' | 'swift' | 'voce_swift' | 'none'
    kin_hardening_H = 'linear'      # 'linear' | 'none'
    kin_hardening_beta = 'armstrong-frederick'   # 'none' | 'proportional' | 'armstrong-frederick'

    # ──────────────────────────────────────────────────────────────────
    # 1.  Yield detection (informational only — ODE starts elastic at t=0)
    # ──────────────────────────────────────────────────────────────────
    sy0 = sigma_y_dispatcher(0.0, iso_hardening=iso_hardening)

    # ──────────────────────────────────────────────────────────────────
    # 2.  Reference solution: full tensor ODE
    # ──────────────────────────────────────────────────────────────────

    print("Performing ODE integration:")

    y0 = [0.0] * 13   # [s_dev (6), eps_bar_p (1), beta (6)]
    sol = solve_ivp(
        ode_rhs_stress, [0.0, t_end], y0,
        method='RK45', max_step=1e-3, rtol=1e-8, atol=1e-10,
        dense_output=True,
        args=(iso_hardening, kin_hardening_H, kin_hardening_beta),
    )
    assert sol.success, f"ODE integration failed: {sol.message}"

    print("Solved!")

    t_analytical = sol.t
    gamma_analytical = sol.y[6]
    ebp_exact = gamma_analytical[-1]
    print(f"ODE integration: {len(t_analytical)} steps, "
          f"final eps_bar_p = {ebp_exact:.10e}")

    if ebp_exact < 1e-10:
        print("WARNING: integration stayed essentially elastic; "
              "convergence study won't be meaningful.")

    # ──────────────────────────────────────────────────────────────────
    # 3.  Return-mapping convergence study
    # ──────────────────────────────────────────────────────────────────
    step_counts = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000]

    print(f"\n{'N steps':>10s}  {'eps_bar_p':>16s}  {'rel error':>12s}")
    print("-" * 44)

    rm_results = {}
    for N in step_counts:
        ts_rm, ebp_rm, s_norm_rm, _, _ = return_mapping(
            N, t_end,
            iso_hardening=iso_hardening,
            kin_hardening_H=kin_hardening_H,
            kin_hardening_beta=kin_hardening_beta,
        )
        err = abs(ebp_rm[-1] - ebp_exact) / max(ebp_exact, 1e-30)
        print(f"{N:10d}  {ebp_rm[-1]:16.10e}  {err:12.4e}")
        rm_results[N] = (ts_rm, ebp_rm, s_norm_rm)

    # Pre-compute RM stress path in pi-plane for visualization (only N=1000)
    rm_stress_paths = {}
    N_vis = 1000
    if N_vis in rm_results:
        _, _, _, sigma_hist, _ = return_mapping(
            N_vis, t_end,
            iso_hardening=iso_hardening,
            kin_hardening_H=kin_hardening_H,
            kin_hardening_beta=kin_hardening_beta,
        )
        alpha_rm = np.zeros(len(sigma_hist))
        beta_rm = np.zeros(len(sigma_hist))
        for i in range(len(sigma_hist)):
            p = np.sort(np.linalg.eigvalsh(sigma_hist[i]))[::-1]
            alpha_rm[i] = (p[0] - p[1]) / np.sqrt(2.0)
            beta_rm[i] = (2 * p[2] - p[0] - p[1]) / np.sqrt(6.0)
        ts_rm_N = np.linspace(0.0, t_end, N_vis + 1)
        rm_stress_paths[N_vis] = (ts_rm_N, alpha_rm, beta_rm)

    # Pre-compute the shared plotting data once
    t_plot = np.linspace(0, t_end, 2000)
    eps_path = np.array([eps_dev_func(tt) for tt in t_plot], dtype=float)

    # ODE stress path — principal deviatoric stress pi-plane
    n_t = len(sol.t)
    alpha_s = np.zeros(n_t)
    beta_s = np.zeros(n_t)
    for i in range(n_t):
        s11, s22, s33, s12, s13, s23 = sol.y[0:6, i]
        s_mat = np.array([[s11, s12, s13],
                          [s12, s22, s23],
                          [s13, s23, s33]])
        p = np.sort(np.linalg.eigvalsh(s_mat))[::-1]
        alpha_s[i] = (p[0] - p[1]) / np.sqrt(2.0)
        beta_s[i] = (2 * p[2] - p[0] - p[1]) / np.sqrt(6.0)

    # Strain pi-plane — principal deviatoric strain
    alpha_eps = np.zeros(len(t_plot))
    beta_eps = np.zeros(len(t_plot))
    for i in range(len(t_plot)):
        p = np.sort(np.linalg.eigvalsh(eps_path[i]))[::-1]
        alpha_eps[i] = (p[0] - p[1]) / np.sqrt(2.0)
        beta_eps[i] = (2 * p[2] - p[0] - p[1]) / np.sqrt(6.0)

    # Static plot
    plot_results(sol, t_analytical, gamma_analytical, ebp_exact, rm_results,
                 t_end, step_counts, iso_hardening, kin_hardening_H, kin_hardening_beta,
                 sy0, t_plot, eps_path, alpha_s, beta_s, alpha_eps, beta_eps,
                 rm_stress_paths,
                 save_path=f'return_map_verification_{CASE}.png')

    # Animation
    animate_results(sol, t_analytical, gamma_analytical, ebp_exact, rm_results,
                    t_end, step_counts, iso_hardening, kin_hardening_H, kin_hardening_beta,
                    sy0, t_plot, eps_path, alpha_s, beta_s, alpha_eps, beta_eps,
                    rm_stress_paths,
                    save_path=f'return_map_animation_{CASE}.mp4',
                    n_frames=120, fps=24, dpi=120)

if __name__ == "__main__":
    main()



