"""
fem_overlay.py
──────────────────────────────────────────────────────────────────────
Load the single-element deal.II stress/strain CSV and overlay the FEM
path on the existing return-map / ODE pi-plane verification plots.

Drop-in companion to the plotting module. Three pieces:

  1. load_fem_data(case)          -> reads + parses the CSV into tensors
  2. process_tensor_history(...) -> projects all histories consistently
  3. patched _plot_*_pi_plane      -> accept an optional `fem` overlay

The CSV is parsed positionally as:
    <6 sigma components>; <6 epsilon components>;
    optional <6 beta components>; optional <1 epsilon_b_bar component>
in Voigt-like order [00, 11, 22, 01, 02, 12] for each tensor block.
"""

import os
import numpy as np

import principal_radial_and_angular_values as pv


# ──────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────

def deformation_case_name(test_case):
    return {
        1: "uniaxial_extension",
        2: "equibiaxial_extension",
        3: "nonproportional_biaxial",
        4: "rotating_principal_axes",
        5: "combined_multi_shear",
        6: "cyclic_biaxial",
    }.get(test_case, "unknown")


def load_fem_data(case, data_dir='.', verbose=True):
    """Read stress_strain_out_<case>.csv if it exists.

    Returns a dict with sigma, eps, beta, and epsilon_b_bar. Older
    12-column files are accepted; beta defaults to zero and
    epsilon_b_bar is left as None.
    """
    fname = os.path.join(
        data_dir, f"fem_out_c{case}.csv")

    if not os.path.exists(fname):
        if verbose:
            print(f"[fem_overlay] no FEM file: {fname}")
        return None

    if verbose:
        print("Data file found:", f"stress_strain_out_{deformation_case_name(case)}.csv")

    # semicolon-delimited, one header row to skip
    data = np.genfromtxt(fname, delimiter=';', skip_header=1)
    if data.ndim == 1:                       # single data row
        data = data.reshape(1, -1)

    # strip any all-NaN columns produced by a trailing ';'
    data = data[:, ~np.all(np.isnan(data), axis=0)]

    if data.shape[1] < 12:
        raise ValueError(
            f"[fem_overlay] expected >=12 columns, got {data.shape[1]} "
            f"in {fname}")

    sig_v = data[:, 0:6]
    eps_v = data[:, 6:12]
    has_beta = data.shape[1] >= 18
    beta_v = data[:, 12:18] if has_beta else np.zeros_like(sig_v)
    epsilon_b_bar = data[:, 18] if data.shape[1] >= 19 else None

    if verbose:
        print(f"[fem_overlay] loaded {data.shape[0]} steps from {fname}")

    return dict(sigma=pv.voigt_to_tensor(sig_v),
                eps=pv.voigt_to_tensor(eps_v),
                beta=pv.voigt_to_tensor(beta_v),
                epsilon_b_bar=epsilon_b_bar,
                has_beta=has_beta)


def build_fem_overlay(case, t_end, data_dir='.', verbose=True):
    """Load + project the FEM data, ready to hand to the plot panels.

    Returns dict {sa, sb, ea, eb, t, Vs, Ve, processed} or None.
    """
    fem = load_fem_data(case, data_dir=data_dir, verbose=verbose)
    if fem is None:
        return None
    # rows are evenly spaced load steps from t=0 to t=t_end
    t = np.linspace(0.0, t_end, fem['sigma'].shape[0])
    processed = pv.process_tensor_history(
        t=t,
        sigma=fem['sigma'],
        epsilon=fem['eps'],
        beta=fem['beta'],
        epsilon_b_bar=fem['epsilon_b_bar'],
        stress_scale=1000.0)

    return dict(sa=processed.sigma_pp.alpha,
                sb=processed.sigma_pp.beta,
                ea=processed.epsilon_pp.alpha,
                eb=processed.epsilon_pp.beta,
                ba=processed.backstress_pp.alpha,
                bb=processed.backstress_pp.beta,
                t=t,
                Vs=processed.sigma_pp.V,
                Ve=processed.epsilon_pp.V,
                epsilon_b_bar=processed.epsilon_b_bar,
                has_beta=fem['has_beta'],
                processed=processed)


def _fem_mask(fem, t_now):
    if t_now is None:
        return np.ones_like(fem['t'], dtype=bool)
    return fem['t'] <= t_now
