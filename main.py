import os
import pickle
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import plasticity_models as pm
import solver_functions as sf
import plot_and_animate_results as par
import principal_radial_and_angular_values as pv
from fem_overlay import build_fem_overlay   # loaded HERE, passed into the plotter

USE_HENCKY = True
USE_CACHE = True   # if a matching cache file exists, load it instead of re-solving

if USE_HENCKY:
    import log_strain_tensor_cases as et
else:
    import deformation_tensor_cases as et

# Solver settings live here so they can also feed the cache key: changing a
# tolerance changes the solution, so it must invalidate the cache.
SOLVER_OPTS = dict(method='Radau', max_step=1e-4, rtol=1e-10, atol=1e-12)


def _cache_key(t_end, iso_hardening, kin_hardening_H, kin_hardening_beta,
               step_counts, N_vis):
    """Hash the parameters that determine the solution.

    NOTE: this keys on the *parameters*, not on the deformation-case source
    code. If you edit the case functions in log_strain_tensor_cases /
    deformation_tensor_cases without changing any of the values below, the
    cache won't notice -- delete the cache file (or the cache folder) to
    force a re-solve.
    """
    key_data = dict(
        use_hencky=USE_HENCKY,
        case=et.CASE,
        t_end=t_end,
        iso_hardening=iso_hardening,
        kin_hardening_H=kin_hardening_H,
        kin_hardening_beta=kin_hardening_beta,
        step_counts=list(step_counts),
        N_vis=N_vis,
        solver=sorted(SOLVER_OPTS.items()),
        pipeline_version=2,
    )
    blob = repr(sorted(key_data.items())).encode()
    return hashlib.md5(blob).hexdigest()[:16]


def compute_solution(t_end, iso_hardening, kin_hardening_H, kin_hardening_beta,
                     step_counts, N_vis, sy0):
    """Solve the problem: reference ODE, RM convergence study, RM
    visualization data, and the derived pi-plane plotting arrays.

    Returns a plain (pickle-friendly) dict of numpy arrays / dicts. The
    full solve_ivp result object is not stored; only sol.t and sol.y are
    kept, since that is all the plotting code needs.
    """
    # ------------------------------------------------------------------
    # 2.  Reference solution: full tensor ODE
    # ------------------------------------------------------------------
    print("Performing ODE integration:")

    y0 = [0.0] * 13   # [s_dev (6), eps_bar_p (1), beta (6)]
    sol = solve_ivp(
        sf.ode_rhs_stress, [0.0, t_end], y0,
        dense_output=True,
        args=(iso_hardening, kin_hardening_H, kin_hardening_beta, et.d_eps_dev_dt_func),
        **SOLVER_OPTS,
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

    # ------------------------------------------------------------------
    # 3.  Return-mapping convergence study
    # ------------------------------------------------------------------
    print(f"\n{'N steps':>10s}  {'eps_bar_p':>16s}  {'rel error':>12s}")
    print("-" * 44)

    rm_results = {}
    for N in step_counts:
        ts_rm, ebp_rm, s_norm_rm, _, _ = sf.return_mapping(
            N, t_end,
            iso_hardening,
            kin_hardening_H,
            kin_hardening_beta,
            et.eps_dev_func
        )
        err = abs(ebp_rm[-1] - ebp_exact) / max(ebp_exact, 1e-30)
        print(f"{N:10d}  {ebp_rm[-1]:16.10e}  {err:12.4e}")
        rm_results[N] = (ts_rm, ebp_rm, s_norm_rm)

    # ------------------------------------------------------------------
    # 4.  RM visualization data (stress path + yield surface state)
    # ------------------------------------------------------------------
    rm_stress_paths = {}
    rm_full_data = {}

    if N_vis in rm_results:
        ts_rm, ebp_rm, s_norm_rm, sigma_hist, beta_hist = sf.return_mapping(
            N_vis, t_end,
            iso_hardening,
            kin_hardening_H,
            kin_hardening_beta,
            et.eps_dev_func)

        eps_rm = np.array([et.eps_dev_func(tt) for tt in ts_rm], dtype=float)
        rm_processed = pv.process_tensor_history(
            t=ts_rm, sigma=sigma_hist, epsilon=eps_rm, beta=beta_hist,
            epsilon_b_bar=ebp_rm)
        rm_stress_paths[N_vis] = (
            rm_processed.t,
            rm_processed.sigma_pp.alpha,
            rm_processed.sigma_pp.beta)
        rm_full_data[N_vis] = (rm_processed.t, ebp_rm, s_norm_rm,
                               sigma_hist, beta_hist,
                               rm_processed.sigma_pp.V,
                               rm_processed)

    # ------------------------------------------------------------------
    # 5.  Pre-compute shared plotting data
    # ------------------------------------------------------------------
    t_plot = np.linspace(0, t_end, 2000)
    eps_path = np.array([et.eps_dev_func(tt) for tt in t_plot], dtype=float)

    # Shared tensor-history processing for the IVP and strain path.
    s_hist_ode = pv.voigt_history_to_tensor(sol.y[0:6, :].T)
    beta_hist_ode = pv.voigt_history_to_tensor(sol.y[7:13, :].T)
    eps_hist_ode = np.array([et.eps_dev_func(tt) for tt in sol.t], dtype=float)
    ivp_processed = pv.process_tensor_history(
        t=sol.t, sigma=s_hist_ode, epsilon=eps_hist_ode, beta=beta_hist_ode,
        epsilon_b_bar=gamma_analytical)
    alpha_s = ivp_processed.sigma_pp.alpha
    beta_s = ivp_processed.sigma_pp.beta
    V_hist = ivp_processed.sigma_pp.V

    # Strain pi-plane -- principal deviatoric strain
    strain_plot_processed = pv.process_tensor_history(t=t_plot, epsilon=eps_path)
    alpha_eps = strain_plot_processed.epsilon_pp.alpha
    beta_eps = strain_plot_processed.epsilon_pp.beta

    return dict(
        sol_t=sol.t, sol_y=sol.y,
        t_analytical=t_analytical,
        gamma_analytical=gamma_analytical,
        ebp_exact=ebp_exact,
        rm_results=rm_results,
        rm_stress_paths=rm_stress_paths,
        rm_full_data=rm_full_data,
        t_plot=t_plot,
        alpha_s=alpha_s, beta_s=beta_s, V_hist=V_hist,
        alpha_eps=alpha_eps, beta_eps=beta_eps,
        ivp_processed=ivp_processed,
        strain_plot_processed=strain_plot_processed,
        sy0=sy0,
    )

def main():
    OUTPUT_DIR = Path.cwd() / et.OD
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    CACHE_DIR = Path.cwd() / 'cache'
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    t_end = 8.0
    iso_hardening = 'voce_swift'      # 'linear' | 'voce' | 'swift' | 'voce_swift' | 'none'
    kin_hardening_H = 'linear'      # 'linear' | 'none'
    kin_hardening_beta = 'armstrong-frederick'   # 'none' | 'proportional' | 'armstrong-frederick'

    step_counts = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000]
    N_vis = 1000

    # ------------------------------------------------------------------
    # 1.  Yield detection (informational only -- ODE starts elastic at t=0)
    # ------------------------------------------------------------------
    sy0 = pm.sigma_y_dispatcher(0.0, iso_hardening=iso_hardening)

    # ------------------------------------------------------------------
    # 2-5.  Solve (or load a cached solution if one already exists)
    # ------------------------------------------------------------------
    key = _cache_key(t_end, iso_hardening, kin_hardening_H,
                     kin_hardening_beta, step_counts, N_vis)
    cache_file = CACHE_DIR / f"solution_C{et.CASE}_{iso_hardening}_{key}.pkl"

    if USE_CACHE and cache_file.exists():
        print(f"Loading cached solution from {cache_file}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
    else:
        data = compute_solution(t_end, iso_hardening, kin_hardening_H,
                                kin_hardening_beta, step_counts, N_vis, sy0)
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Solution cached to {cache_file}")

    # Unpack the (computed or loaded) solution. sol is rebuilt as a light
    # namespace exposing only .t and .y, which is all the plotting needs.
    sol = SimpleNamespace(t=data['sol_t'], y=data['sol_y'])
    t_analytical    = data['t_analytical']
    gamma_analytical = data['gamma_analytical']
    ebp_exact       = data['ebp_exact']
    rm_results      = data['rm_results']
    rm_stress_paths = data['rm_stress_paths']
    rm_full_data    = data['rm_full_data']
    t_plot          = data['t_plot']
    alpha_s         = data['alpha_s']
    beta_s          = data['beta_s']
    V_hist          = data['V_hist']
    alpha_eps       = data['alpha_eps']
    beta_eps        = data['beta_eps']
    sy0             = data['sy0']

    # FEM overlay is loaded HERE (externally) and passed in -- the plotter
    # never touches the filesystem.
    fem = build_fem_overlay(et.CASE, t_end, data_dir=str(Path.cwd()), verbose=True)

    hardening_label = (f"iso={iso_hardening}, kin H={kin_hardening_H}, "
                       f"kin β={kin_hardening_beta}")
    suptitle = f"Case {et.CASE} — {hardening_label}"

    # ------------------------------------------------------------------
    # 6.  Static plot — ONE plot_results call per dataset, overlaid.
    # ------------------------------------------------------------------
    fig, axes = par.make_panels()

    # (i) IVP / ODE reference  -> fresh draw (clears the axes)
    ivp_sol = par.build_ivp_solution(
        sol, t_analytical, gamma_analytical, ebp_exact,
        iso_hardening, kin_hardening_beta, V_hist, sy0,
        t_plot, alpha_s, beta_s, alpha_eps, beta_eps)
    convergence = par.build_convergence(rm_results, step_counts, ebp_exact, t_end)
    par.plot_results(axes, [ivp_sol], convergence=convergence, case=et.CASE,
                     suptitle=suptitle, t_end=t_end, clear_axes=True)

    # (ii) Return mapping  -> overlay
    rm_sols = par.build_rm_stress_path_solutions(rm_stress_paths)
    rm_yield = par.build_rm_yield_solution(rm_full_data, iso_hardening,
                                           kin_hardening_beta, alpha_eps,
                                           beta_eps, t_plot)
    if rm_yield is not None:
        rm_sols.append(rm_yield)
    rm_sols += par.build_rm_plastic_solutions(rm_results)
    par.plot_results(axes, rm_sols, case=et.CASE, t_end=t_end, clear_axes=False)

    # (iii) FEM (deal.II)  -> overlay
    fem_sol = par.build_fem_solution(fem, iso_hardening)
    if fem_sol is not None:
        par.plot_results(axes, [fem_sol], case=et.CASE, t_end=t_end, clear_axes=False)

    fig.tight_layout()
    save_path = os.path.join(OUTPUT_DIR,
                             f'plot_C{et.CASE}.png')
    fig.savefig(save_path, dpi=200)
    print(f"Plot saved to {save_path}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 7.  Animation (also one plot_results call per dataset, per frame)
    # ------------------------------------------------------------------
    par.animate_results(sol, t_analytical, gamma_analytical, ebp_exact, rm_results,
                        t_end, iso_hardening, kin_hardening_H, kin_hardening_beta, V_hist,
                        sy0, t_plot, alpha_s, beta_s, alpha_eps, beta_eps,
                        rm_stress_paths, et.CASE, fem=fem, rm_full_data=rm_full_data,
                        save_path=os.path.join(OUTPUT_DIR, f'animation_C{et.CASE}.mp4'),
                        n_frames=120, fps=24, dpi=200)


if __name__ == "__main__":
    main()
