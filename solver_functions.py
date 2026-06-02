import numpy as np
import plasticity_models as pm

# == == == == == == == == == == == == == == == ==
# --------------- Coupled ODEs ------------------
# == == == == == == == == == == == == == == == ==
E       = 210.0e3    # Young's modulus [MPa]
nu      = 0.3        # Poisson's ratio
H       = 1000.0     # Linear isotropic hardening modulus [MPa]

G  = E / (2.0 * (1.0 + nu))          # Shear modulus

def ode_rhs_stress(t_val, y,
                   iso_hardening,
                   kin_hardening_H,
                   kin_hardening_beta,
                   strain_rate_func):
    s = np.array([[y[0], y[3], y[4]],
                  [y[3], y[1], y[5]],
                  [y[4], y[5], y[2]]])
    eps_bar = max(y[6], 0.0)  # guard against negative overshoot
    beta = np.array([[y[7], y[10], y[11]],
                     [y[10], y[8], y[12]],
                     [y[11], y[12], y[9]]])

    eta = s - beta
    e_dot = np.array(strain_rate_func(t_val), dtype=float)
    eta_norm = np.sqrt(np.tensordot(eta, eta))

    sigma_y = pm.sigma_y_dispatcher(eps_bar, iso_hardening)
    sigma_eq = np.sqrt(1.5) * eta_norm
    f = sigma_eq - sigma_y

    # Default: elastic
    s_dot = 2 * G * e_dot
    eps_bar_dot = 0.0
    beta_dot = np.zeros((3, 3))

    if eta_norm > 1e-10 and f > -1e-8:
        N = eta / eta_norm

        # Elastic trial rate of f
        # f_dot_trial = sqrt(3/2) * N : (2G * e_dot) - H_iso * eps_bar_dot
        # With eps_bar_dot = 0 for elastic trial:
        N_dot_e = np.tensordot(N, e_dot)
        f_dot_trial = np.sqrt(1.5) * 2 * G * N_dot_e

        if f_dot_trial > 0 or f > 1e-8:
            H_iso = pm.H_iso_dispatcher(eps_bar, iso_hardening)
            H_kin = pm.H_kin_dispatcher(eps_bar, kin_hardening_H)

            # Standard consistency: lam_dot from f_dot = 0
            lam_dot = np.sqrt(1.5) * 2 * G * N_dot_e / (3 * G + H_iso + H_kin)
            lam_dot = max(lam_dot, 0.0)

            # Penalty drift correction
            # This enforces f = 0 as a DAE constraint
            # The correction drives f → 0 exponentially with rate c_drift
            # lam_correction satisfies: f_dot = -c_drift * f
            # f_dot = sqrt(3/2)*2G*N:e_dot - (3G + H_iso + H_kin)*lam_dot_total = -c_drift*f
            # lam_dot_total = [sqrt(3/2)*2G*N:e_dot + c_drift*f] / (3G + H_iso + H_kin)
            if f > 0:
                c_drift = 1.0 / 1e-4  # 1/max_step — correction timescale
                lam_dot = (np.sqrt(1.5) * 2 * G * N_dot_e + c_drift * f) / (3 * G + H_iso + H_kin)
                lam_dot = max(lam_dot, 0.0)

            eps_bar_dot = np.sqrt(2.0 / 3.0) * lam_dot
            eps_p_dot = lam_dot * N
            s_dot = 2 * G * (e_dot - eps_p_dot)
            beta_dot = pm.beta_dot_dispatcher(N, lam_dot, beta, H_kin, kin_hardening_beta)

    return [s_dot[0, 0], s_dot[1, 1], s_dot[2, 2],
            s_dot[0, 1], s_dot[0, 2], s_dot[1, 2],
            eps_bar_dot,
            beta_dot[0, 0], beta_dot[1, 1], beta_dot[2, 2],
            beta_dot[0, 1], beta_dot[0, 2], beta_dot[1, 2]]

# ──────────────────────────────────────────────────────────────────────
# Return-mapping integrator (arbitrary step count)
# ──────────────────────────────────────────────────────────────────────

def return_mapping(n_steps, t_end,
                   iso_hardening,
                   kin_hardening_H,
                   kin_hardening_beta,
                   eps_dev_function):
    """Radial return for von Mises + isotropic + kinematic hardening."""
    ts = np.linspace(0.0, t_end, n_steps + 1)

    sigma_dev = np.zeros((3, 3))
    beta = np.zeros((3, 3))  # backstress
    ebp = 0.0

    ebp_history = np.zeros(n_steps + 1)
    s_norm_hist = np.zeros(n_steps + 1)
    sigma_hist = np.zeros((n_steps + 1, 3, 3))
    beta_hist = np.zeros((n_steps + 1, 3, 3))

    eps_dev_old = np.array(eps_dev_function(ts[0]), dtype=float)

    for i in range(1, n_steps + 1):
        eps_dev_new = np.array(eps_dev_function(ts[i]), dtype=float)
        delta_eps_dev = eps_dev_new - eps_dev_old

        # Elastic trial deviatoric stress and relative stress eta = s - beta
        sigma_trial = sigma_dev + 2.0 * G * delta_eps_dev
        eta_trial = sigma_trial - beta
        eta_norm_tr = np.sqrt(np.tensordot(eta_trial, eta_trial))
        sig_eq_tr = np.sqrt(1.5) * eta_norm_tr

        sig_y = pm.sigma_y_dispatcher(ebp, iso_hardening=iso_hardening)

        if sig_eq_tr > sig_y:
            # Newton iteration on dgamma
            dgamma = 0.0
            for _ in range(50):
                ebp_new = ebp + dgamma
                sy = pm.sigma_y_dispatcher(ebp_new, iso_hardening=iso_hardening)
                H_iso = pm.H_iso_dispatcher(ebp_new, iso_hardening=iso_hardening)
                H_kin = pm.H_kin_dispatcher(ebp_new, kin_hardening=kin_hardening_H)

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
                beta = beta + dgamma * ((2.0 / 3.0) * H_kin * N_hat - pm.b_kin * beta)
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





