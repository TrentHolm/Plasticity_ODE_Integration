import numpy as np
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

# def principal_values(t_val):
#     """Sorted (descending) principal values of the deviatoric strain at t_val."""
#     A = np.array(etc.eps_dev_func(t_val), dtype=float)
#     return np.sort(np.linalg.eigvalsh(A))[::-1]
#
# def project_onto_pi_plane(pv):
#     alpha = (pv[1] - pv[2]) / np.sqrt(2)
#     beta = (2 * pv[0] - pv[1] - pv[2]) / np.sqrt(6)
#     return alpha, beta
#
# def principal_value_rates_analytical(t_val):
#     """d(lambda_i)/dt = n_i^T A_dot n_i, sorted to match principal values."""
#     A = np.array(etc.eps_dev_func(t_val), dtype=float)
#     A_dot = np.array(etc.d_eps_dev_dt_func(t_val), dtype=float)
#     lambdas, N = np.linalg.eigh(A)
#     order = np.argsort(lambdas)[::-1]
#     N = N[:, order]
#     return np.einsum('ji,jk,ki->i', N, A_dot, N)

# Module-level state_ivp for continuous eigenvector tracking
# _prev_eigvecs = None
#
# def principal_value_rates_fd(t_val, h=1e-3):
#     """5-point central finite difference of the sorted principal values."""
#     stencil = [-2, -1, 1, 2]
#     coeffs = [1, -8, 8, -1]
#     return sum(c * principal_values(t_val + s * h)
#                for c, s in zip(coeffs, stencil)) / (12 * h)
#
#
# def compare_analytical_vs_fd(t_vals=(0.5, 2.0, 5.0), h=1e-3):
#     """Compare analytical principal-value rates against 5-point FD."""
#     for t_val in t_vals:
#         analytical = principal_value_rates_analytical(t_val)
#         fd = principal_value_rates_fd(t_val, h=h)
#         print(f"\n--- t = {t_val} ---")
#         print(f"Analytical: {analytical}")
#         print(f"5-pt FD:    {fd}")
#         print(f"Difference: {analytical - fd}")

@dataclass
class PiPlanePath:
    alpha: np.ndarray
    beta: np.ndarray
    V: np.ndarray
    principal_values: np.ndarray


@dataclass
class ProcessedTensorHistory:
    t: np.ndarray = None
    sigma: np.ndarray = None
    epsilon: np.ndarray = None
    backstress: np.ndarray = None
    epsilon_b_bar: np.ndarray = None
    sigma_pp: PiPlanePath = None
    epsilon_pp: PiPlanePath = None
    backstress_pp: PiPlanePath = None
    backstress_in_sigma_pp: object = None
    backstress_in_epsilon_pp: object = None


def project_onto_tangential_and_radial_basis(alpha_eps_dots, beta_eps_dot, phi):
    alpha_r_eps_dot = alpha_eps_dots * np.cos(phi) + beta_eps_dot * np.sin(phi)
    beta_theta_eps_dot = -alpha_eps_dots * np.sin(phi) + beta_eps_dot * np.cos(phi)

    return alpha_r_eps_dot, beta_theta_eps_dot


def voigt_to_tensor(v):
    """Convert [00, 11, 22, 01, 02, 12] Voigt rows to symmetric tensors."""
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    if v.shape[1] != 6:
        raise ValueError(f"Expected 6 Voigt components, got shape {v.shape}")

    n = v.shape[0]
    T = np.zeros((n, 3, 3))
    T[:, 0, 0] = v[:, 0]
    T[:, 1, 1] = v[:, 1]
    T[:, 2, 2] = v[:, 2]
    T[:, 0, 1] = v[:, 3]
    T[:, 1, 0] = v[:, 3]
    T[:, 0, 2] = v[:, 4]
    T[:, 2, 0] = v[:, 4]
    T[:, 1, 2] = v[:, 5]
    T[:, 2, 1] = v[:, 5]
    return T


def voigt_history_to_tensor(v_hist):
    """Convert history columns [s11,s22,s33,s12,s13,s23] to tensors."""
    return voigt_to_tensor(np.asarray(v_hist, dtype=float))


def pi_plane_from_principal_values(p):
    """Project principal values to the pi-plane convention used here."""
    p = np.asarray(p, dtype=float)
    alpha = (p[..., 0] - p[..., 1]) / np.sqrt(2.0)
    beta = (2.0 * p[..., 2] - p[..., 0] - p[..., 1]) / np.sqrt(6.0)
    return alpha, beta


def tensor_principal_history(tensor_hist):
    """Track principal values and unit directions through a tensor history."""
    tensor_hist = np.asarray(tensor_hist, dtype=float)
    n = len(tensor_hist)
    principal_values = np.zeros((n, 3))
    V_hist = np.zeros((n, 3, 3))

    eigvals, eigvecs = np.linalg.eigh(tensor_hist[0])
    idx = np.argsort(eigvals)[::-1]
    principal_values[0] = eigvals[idx]
    V_prev = eigvecs[:, idx]
    V_hist[0] = V_prev

    for i in range(1, n):
        eigvals, eigvecs = np.linalg.eigh(tensor_hist[i])
        dots = np.abs(V_prev.T @ eigvecs)
        _, col_ind = linear_sum_assignment(1.0 - dots)
        principal_values[i] = eigvals[col_ind]
        V_prev = eigvecs[:, col_ind]
        V_hist[i] = V_prev

    return principal_values, V_hist


def pi_plane_path(tensor_hist):
    """Process one tensor history into pi-plane coordinates and frames."""
    principal_values, V_hist = tensor_principal_history(tensor_hist)
    alpha, beta = pi_plane_from_principal_values(principal_values)
    return PiPlanePath(alpha=alpha, beta=beta, V=V_hist,
                       principal_values=principal_values)


def pi_plane_coords_unfolded(tensor_hist):
    """Backward-compatible wrapper returning alpha, beta, tracked frames."""
    path = pi_plane_path(tensor_hist)
    return path.alpha, path.beta, path.V


def project_tensor_onto_principal_directions(tensor, principal_directions):
    """Return tensor principal components in a supplied principal frame.

    `principal_directions` is the 3x3 matrix whose columns are the unit
    directions from a stress or strain eigensolve. Passing beta here gives
    the backstress coordinates in that same stress/strain frame.
    """
    tensor = np.asarray(tensor, dtype=float)
    V = np.asarray(principal_directions, dtype=float)
    projected = V.T @ tensor @ V
    return np.array([projected[0, 0], projected[1, 1], projected[2, 2]])


def tensor_pi_plane_in_principal_directions(tensor, principal_directions):
    principal_components = project_tensor_onto_principal_directions(
        tensor, principal_directions)
    return pi_plane_from_principal_values(principal_components)


def tensor_history_pi_plane_in_frames(tensor_hist, frame_hist):
    tensor_hist = np.asarray(tensor_hist, dtype=float)
    frame_hist = np.asarray(frame_hist, dtype=float)
    principal_components = np.array([
        project_tensor_onto_principal_directions(T, V)
        for T, V in zip(tensor_hist, frame_hist)
    ])
    return pi_plane_from_principal_values(principal_components)


def backstress_pi_plane_from_principal_directions(beta, principal_directions):
    """Project backstress beta using stress/strain principal unit directions."""
    return tensor_pi_plane_in_principal_directions(beta, principal_directions)


def process_tensor_history(t=None, sigma=None, epsilon=None, beta=None,
                           epsilon_b_bar=None, stress_scale=1.0,
                           beta_scale=None):
    """Run IVP, return-map, or FEM tensors through one processing pipeline."""
    beta_scale = stress_scale if beta_scale is None else beta_scale
    sigma = None if sigma is None else np.asarray(sigma, dtype=float) * stress_scale
    epsilon = None if epsilon is None else np.asarray(epsilon, dtype=float)
    backstress = None if beta is None else np.asarray(beta, dtype=float) * beta_scale
    epsilon_b_bar = (None if epsilon_b_bar is None
                     else np.asarray(epsilon_b_bar, dtype=float).reshape(-1))

    sigma_pp = pi_plane_path(sigma) if sigma is not None else None
    epsilon_pp = pi_plane_path(epsilon) if epsilon is not None else None
    backstress_pp = pi_plane_path(backstress) if backstress is not None else None

    backstress_in_sigma_pp = None
    backstress_in_epsilon_pp = None
    if backstress is not None and sigma_pp is not None:
        backstress_in_sigma_pp = tensor_history_pi_plane_in_frames(
            backstress, sigma_pp.V)
    if backstress is not None and epsilon_pp is not None:
        backstress_in_epsilon_pp = tensor_history_pi_plane_in_frames(
            backstress, epsilon_pp.V)

    return ProcessedTensorHistory(
        t=None if t is None else np.asarray(t, dtype=float),
        sigma=sigma,
        epsilon=epsilon,
        backstress=backstress,
        epsilon_b_bar=epsilon_b_bar,
        sigma_pp=sigma_pp,
        epsilon_pp=epsilon_pp,
        backstress_pp=backstress_pp,
        backstress_in_sigma_pp=backstress_in_sigma_pp,
        backstress_in_epsilon_pp=backstress_in_epsilon_pp)
