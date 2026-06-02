import numpy as np

# == == == == == == == == == == == == == == == ==
# --------------- Voce Hardening ----------------
# == == == == == == == == == == == == == == == ==
b = 10.0
R_inf = 100.0  # [MPa]
sigma_y0 = 450.0  # initial yield stress

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
H_lin_iso = 1000.0  # constant iso_hardening modulus  [MPa]

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

K = 500.0  # [MPa]
n = 0.3
eps_0 = 0.01
def kappa_swift(eps_bar):
    return K * (max(eps_0 + eps_bar, eps_0))**n

def H_iso_swift(eps_bar):
    return K * n * (max(eps_0 + eps_bar, eps_0))**(n - 1)  # fixed: exponent should be n-1

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
H_lin_kin = 1000.0 # [MPa]
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