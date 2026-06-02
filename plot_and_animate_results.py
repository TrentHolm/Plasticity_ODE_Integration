import numpy as np
from dataclasses import dataclass, field

import solver_functions as sf
import plasticity_models as pm
import principal_radial_and_angular_values as pv


# ──────────────────────────────────────────────────────────────────────
#  Case naming + figure factory
# ──────────────────────────────────────────────────────────────────────
def deformation_case_name(test_case):
    match test_case:
        case 1:
            return "uniaxial_extension"
        case 2:
            return "equibiaxial_extension"
        case 3:
            return "nonproportional_biaxial"
        case 4:
            return "rotating_principal_axes"
        case 5:
            return "combined_multi_shear"
        case 6:
            return "cyclic_biaxial"
        case _:
            return "unknown"


def make_panels(figsize=(14, 10)):
    """Create and return the (fig, axes) 2x2 grid used by plot_results.

    The caller owns the figure: it decides whether to save a static image
    or hand the axes to an animation. plot_results never creates or saves.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    return fig, axes


# ──────────────────────────────────────────────────────────────────────
#  Plain data containers — the *only* thing plot_results understands.
#
#  A `Solution` is one independent dataset to overlay (the IVP/ODE
#  reference, a return-mapping result, the deal.II FEM run, ...). Every
#  field is optional; whatever is None/empty is simply not drawn.
#
#      eps_pp[0] = alpha_eps,  eps_pp[1] = beta_eps   (strain pi-plane)
#      s_pp[0]   = alpha,      s_pp[1]   = beta        (stress pi-plane)
#      plastic_strain = (t, eps_bar_p)                 (panel c)
#      yield_surfaces = [YieldSurface, ...]            (drawn on a & b)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class YieldSurface:
    """A single yield circle, drawn in stress space and/or strain space.

    Provide R_s for the stress pi-plane and/or R_eps for the strain
    pi-plane; whichever is None is skipped in that panel.
    """
    R_s: float = None
    center_s: tuple = (0.0, 0.0)
    R_eps: float = None
    center_eps: tuple = (0.0, 0.0)
    label: str = ''
    color: str = 'gray'
    color_s: str = None     # overrides `color` in the stress pi-plane
    color_eps: str = None   # overrides `color` in the strain pi-plane
    lw: float = 0.5
    dashes: tuple = (4, 2)


@dataclass
class Solution:
    label: str = ''
    color: str = None
    linestyle: str = '-'
    lw: float = 1.2
    alpha: float = 1.0
    marker: str = None
    ms: float = 2.5
    zorder: float = None
    mark_endpoint: bool = False
    endpoint_color: str = None
    endpoint_size: float = 18.0
    endpoint_edgecolor: str = None
    endpoint_lw: float = 0.4
    eps_pp: object = None          # [alpha_eps, beta_eps]
    s_pp: object = None            # [alpha, beta]
    plastic_strain: object = None  # (t, eps_bar_p)
    yield_surfaces: list = field(default_factory=list)


@dataclass
class Convergence:
    """Optional data for the convergence panel (d)."""
    dts: object
    errors: object
    label: str = 'RM vs ODE'
    reference_slopes: tuple = (1, 2)


# ──────────────────────────────────────────────────────────────────────
#  Low-level drawing / limit helpers
# ──────────────────────────────────────────────────────────────────────
def _circle_xy(R, center, n=300):
    th = np.linspace(0, 2 * np.pi, n)
    return R * np.cos(th) + center[0], R * np.sin(th) + center[1]


def _refit_pi(ax, pad=1.2):
    """Square, symmetric limits covering everything currently on `ax`.

    Reads all lines + scatter collections, so repeated (overlay) calls
    keep growing the view rather than clipping earlier data.
    """
    vals = []
    for ln in ax.lines:
        xd = np.asarray(ln.get_xdata(), dtype=float)
        yd = np.asarray(ln.get_ydata(), dtype=float)
        if xd.size:
            vals.append(np.max(np.abs(xd)))
            vals.append(np.max(np.abs(yd)))
    for col in ax.collections:
        off = np.asarray(col.get_offsets(), dtype=float)
        if off.size:
            vals.append(np.max(np.abs(off)))
    lim = (max(vals) if vals else 1.0) * pad
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


def _refit_plastic(ax, t_end=None):
    xmax = ymax = 0.0
    for ln in ax.lines:
        xd = np.asarray(ln.get_xdata(), dtype=float)
        yd = np.asarray(ln.get_ydata(), dtype=float)
        if xd.size:
            xmax = max(xmax, float(np.max(xd)))
            ymax = max(ymax, float(np.max(yd)))
    ax.set_xlim(0, t_end if t_end is not None else (xmax or 1.0))
    ax.set_ylim(0, ymax * 1.1 if ymax > 0 else 1.0)


# ──────────────────────────────────────────────────────────────────────
#  Panel renderers. Each draws the new artists for this call, then styles
#  the panel from *everything* present (so overlay calls accumulate). On a
#  fresh (clearing) call a panel left with no content is switched off.
# ──────────────────────────────────────────────────────────────────────
def _plot_pi_plane(ax, solutions, which, case, fresh):
    is_eps = (which == 'eps')

    for s in solutions:
        path = s.eps_pp if is_eps else s.s_pp
        if path is not None:
            a = np.asarray(path[0])
            b = np.asarray(path[1])
            if a.size:
                kw = dict(color=s.color, ls=s.linestyle, lw=s.lw,
                          alpha=s.alpha, label=(s.label or None))
                if s.marker:
                    kw.update(marker=s.marker, ms=s.ms, mfc=s.color, mec='none')
                if s.zorder is not None:
                    kw['zorder'] = s.zorder
                ax.plot(a, b, **kw)
                if s.mark_endpoint:
                    sc = dict(color=(s.endpoint_color or s.color),
                              s=s.endpoint_size, zorder=6)
                    if s.endpoint_edgecolor is not None:
                        sc.update(edgecolors=s.endpoint_edgecolor,
                                  linewidths=s.endpoint_lw)
                    ax.scatter(a[-1], b[-1], **sc)

        for ys in s.yield_surfaces:
            R = ys.R_eps if is_eps else ys.R_s
            if R is None:
                continue
            center = ys.center_eps if is_eps else ys.center_s
            color = (ys.color_eps if is_eps else ys.color_s) or ys.color
            x, y = _circle_xy(R, center)
            kw = dict(color=color, lw=ys.lw, label=(ys.label or None))
            if ys.dashes is not None:
                kw['dashes'] = list(ys.dashes)
            ax.plot(x, y, **kw)

    has_content = bool(ax.lines) or bool(ax.collections)
    if not has_content:
        if fresh:
            ax.set_axis_off()
        return

    ax.set_axis_on()
    _refit_pi(ax)
    if is_eps:
        ax.set_xlabel(r'$\alpha_\varepsilon$')
        ax.set_ylabel(r'$\beta_\varepsilon$')
        ax.set_title(f'Strain $(\\alpha,\\beta)$ pi-plane (c{case})')
    else:
        ax.set_xlabel(r'$\alpha$ [stress]')
        ax.set_ylabel(r'$\beta$ [stress]')
        ax.set_title(fr'Stress $(\alpha,\beta)$ space (c{case})')
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)


def _plot_plastic_strain(ax, solutions, case, fresh, t_end=None):
    for s in solutions:
        if s.plastic_strain is None:
            continue
        t = np.asarray(s.plastic_strain[0])
        ebp = np.asarray(s.plastic_strain[1])
        if not t.size:
            continue
        ax.plot(t, ebp, color=s.color, ls=s.linestyle, lw=s.lw,
                label=(s.label or None))

    if not ax.lines:
        if fresh:
            ax.set_axis_off()
        return

    ax.set_axis_on()
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\bar{\varepsilon}^p$')
    ax.set_title(f'Accumulated plastic strain (c{case})')
    _refit_plastic(ax, t_end=t_end)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_convergence(ax, convergence, case):
    if convergence is None:
        return False
    dts = np.asarray(convergence.dts)
    errors = np.asarray(convergence.errors)
    ax.set_axis_on()
    ax.loglog(dts, errors, 'ko-', lw=1.5, ms=6, label=convergence.label)
    dt_ref = np.array([dts[0], dts[-1]])
    slope_styles = {1: ('b--', r'$\mathcal{O}(\Delta t)$'),
                    2: ('r--', r'$\mathcal{O}(\Delta t^2)$')}
    for p in convergence.reference_slopes:
        style, lbl = slope_styles.get(p, ('g--', f'O(dt^{p})'))
        ax.loglog(dt_ref, errors[0] * (dt_ref / dt_ref[0]) ** p, style, lw=1, label=lbl)
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'Relative error in $\bar{\varepsilon}^p$')
    ax.set_title(f'Convergence (c{case})')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    return True


def _plot_info(ax, info_text):
    ax.set_axis_on()
    ax.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=14,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Current state')


# ──────────────────────────────────────────────────────────────────────
#  The renderer: draws ONLY what it is given. No file access.
#
#  Designed to be called once PER dataset, overlaying onto shared axes:
#      plot_results(axes, ivp_sols, convergence=conv, ...)   # clears first
#      plot_results(axes, rm_sols,  clear_axes=False, ...)   # overlays
#      plot_results(axes, fem_sols, clear_axes=False, ...)   # overlays
# ──────────────────────────────────────────────────────────────────────
def plot_results(axes, solutions, *, convergence=None, info_text=None,
                 case='', suptitle=None, t_end=None, clear_axes=True):
    """Render the four-panel summary from data passed in.

    Parameters
    ----------
    axes : 2x2 array of matplotlib Axes (e.g. from make_panels()).
    solutions : iterable of Solution
        Each independent dataset to draw on this call. For every Solution
        only the set fields are drawn:
            eps_pp          -> strain pi-plane path        (panel a)
            s_pp            -> stress pi-plane path         (panel b)
            yield_surfaces  -> circles on panels a and b
            plastic_strain  -> accumulated plastic strain  (panel c)
        Anything None/empty is skipped.
    convergence : Convergence or None    -> panel (d) if given.
    info_text : str or None              -> panel (d) text box (animation).
    case, suptitle, t_end : labels / limits only.
    clear_axes : bool
        True  -> fresh draw: clear all axes first; empty panels are hidden.
        False -> OVERLAY: add to whatever is already on the axes without
                 disturbing panels this call doesn't touch. Limits and
                 legends grow to include the new data.

    Does no I/O and computes no model state — give it ready arrays.
    """
    solutions = list(solutions or [])
    fresh = clear_axes
    if fresh:
        for ax in axes.flat:
            ax.clear()

    fig = axes[0, 0].get_figure()
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=11)

    _plot_pi_plane(axes[0, 0], solutions, 'eps', case, fresh)
    _plot_pi_plane(axes[0, 1], solutions, 's', case, fresh)
    _plot_plastic_strain(axes[1, 0], solutions, case, fresh, t_end=t_end)

    ax_d = axes[1, 1]
    if convergence is not None:
        _plot_convergence(ax_d, convergence, case)
    elif info_text is not None:
        _plot_info(ax_d, info_text)
    elif fresh:
        ax_d.set_axis_off()   # nothing for panel (d) on a fresh draw

    return axes


# ──────────────────────────────────────────────────────────────────────
#  Dataset builders (OPTIONAL helpers).
#
#  These turn raw solver output into Solution / Convergence objects, doing
#  the yield-surface state computation that used to live inside the plot.
#  plot_results does NOT need them — you can build Solution objects however
#  you like. They are provided so the IVP / RM / FEM datasets can be
#  prepared *outside* the renderer and passed in.
# ──────────────────────────────────────────────────────────────────────
RM_COLORS = {10: 'tab:orange', 50: 'tab:green', 200: 'tab:purple', 1000: 'tab:cyan'}


def build_ivp_solution(sol, t_analytical, gamma_analytical, ebp_exact,
                       iso_hardening, kin_hardening_beta, V_hist, sy0,
                       t_plot, alpha_s, beta_s, alpha_eps, beta_eps,
                       t_now=None, label='ODE', color='k'):
    """Build the IVP/ODE reference Solution (paths + yield surfaces).

    alpha_s/beta_s and V_hist are sampled on the ODE time grid
    (t_analytical); alpha_eps/beta_eps on the plotting grid (t_plot).
    """
    if t_now is not None:
        path_mask = t_plot <= t_now
        ode_mask = t_analytical <= t_now
        idx_now = min(np.searchsorted(t_analytical, t_now), len(t_analytical) - 1)
        ebp_now = gamma_analytical[idx_now]
        b_now = sol.y[7:13, idx_now]
        V_now = V_hist[idx_now]
        idx_s = idx_now
        idx_eps = min(np.searchsorted(t_plot, t_now), len(t_plot) - 1)
    else:
        path_mask = np.ones_like(t_plot, dtype=bool)
        ode_mask = np.ones_like(t_analytical, dtype=bool)
        ebp_now = ebp_exact
        b_now = sol.y[7:13, -1]
        V_now = V_hist[-1]
        idx_s = -1
        idx_eps = -1

    sy_now = pm.sigma_y_dispatcher(ebp_now, iso_hardening=iso_hardening)
    R_init = np.sqrt(2.0 / 3.0) * sy0
    R_now = np.sqrt(2.0 / 3.0) * sy_now
    R_init_eps = R_init / (2.0 * sf.G)
    R_now_eps = R_now / (2.0 * sf.G)

    alpha_b = beta_b = 0.0
    if kin_hardening_beta != 'none':
        b_mat = pv.voigt_to_tensor(b_now)[0]
        alpha_b, beta_b = pv.backstress_pi_plane_from_principal_directions(
            b_mat, V_now)

    alpha_p = alpha_eps[idx_eps] - alpha_s[idx_s] / (2.0 * sf.G)
    beta_p = beta_eps[idx_eps] - beta_s[idx_s] / (2.0 * sf.G)

    ys = [
        YieldSurface(R_s=R_init, R_eps=R_init_eps, color='gray',
                     label='initial yield'),
        YieldSurface(R_s=R_now, center_s=(0.0, 0.0),
                     R_eps=R_now_eps, center_eps=(alpha_p, beta_p),
                     color_s='yellow', color_eps='b',
                     label=f'ODE yield ($\\bar\\varepsilon^p$={ebp_now:.3f})'),
    ]
    if kin_hardening_beta != 'none':
        ys.append(YieldSurface(
            R_s=R_now, center_s=(alpha_b, beta_b),
            R_eps=R_now_eps,
            center_eps=(alpha_p + alpha_b / (2.0 * sf.G),
                        beta_p + beta_b / (2.0 * sf.G)),
            color_s='deeppink', color_eps='g', lw=1.0,
            label='ODE yield (with backstress)'))

    return Solution(
        label=label, color=color, linestyle='-', lw=1.2,
        eps_pp=np.array([alpha_eps[path_mask], beta_eps[path_mask]]),
        s_pp=np.array([alpha_s[ode_mask], beta_s[ode_mask]]),
        plastic_strain=(t_analytical[ode_mask], gamma_analytical[ode_mask]),
        yield_surfaces=ys,
        mark_endpoint=True, endpoint_color='b', endpoint_size=2.0,
    )


def build_rm_yield_solution(rm_full_data, iso_hardening, kin_hardening_beta,
                            alpha_eps, beta_eps, t_plot, t_now=None):
    """Build a Solution carrying only the return-mapping yield surface(s)."""
    if not rm_full_data:
        return None

    N = max(rm_full_data.keys())
    rm_data = rm_full_data[N]
    ts_rm, ebp_rm, _, sigma_hist, beta_hist, V_hist_rm = rm_data[:6]
    rm_processed = rm_data[6] if len(rm_data) > 6 else None
    idx = (min(np.searchsorted(ts_rm, t_now), len(ts_rm) - 1)
           if t_now is not None else -1)

    ebp_now = ebp_rm[idx]
    sy_now = pm.sigma_y_dispatcher(ebp_now, iso_hardening=iso_hardening)
    R_now = np.sqrt(2.0 / 3.0) * sy_now
    R_now_eps = R_now / (2.0 * sf.G)

    if rm_processed is not None:
        alpha_s_rm = rm_processed.sigma_pp.alpha[idx]
        beta_s_rm = rm_processed.sigma_pp.beta[idx]
        V_now = rm_processed.sigma_pp.V[idx]
    else:
        sigma_now = sigma_hist[idx]
        V_now = V_hist_rm[idx]
        alpha_s_rm, beta_s_rm = pv.tensor_pi_plane_in_principal_directions(
            sigma_now, V_now)

    idx_eps = (min(np.searchsorted(t_plot, ts_rm[idx]), len(t_plot) - 1)
               if t_now is not None else -1)
    alpha_p = alpha_eps[idx_eps] - alpha_s_rm / (2.0 * sf.G)
    beta_p = beta_eps[idx_eps] - beta_s_rm / (2.0 * sf.G)

    alpha_b = beta_b = 0.0
    if kin_hardening_beta != 'none' and beta_hist is not None:
        if rm_processed is not None and rm_processed.backstress_in_sigma_pp is not None:
            alpha_b = rm_processed.backstress_in_sigma_pp[0][idx]
            beta_b = rm_processed.backstress_in_sigma_pp[1][idx]
        else:
            alpha_b, beta_b = pv.backstress_pi_plane_from_principal_directions(
                beta_hist[idx], V_now)

    ys = [YieldSurface(R_s=R_now, center_s=(0.0, 0.0),
                       R_eps=R_now_eps, center_eps=(alpha_p, beta_p),
                       color_s='green', color_eps='c', dashes=(2, 2),
                       label=f'RM yield ($\\bar\\varepsilon^p$={ebp_now:.3f})')]
    if kin_hardening_beta != 'none':
        ys.append(YieldSurface(
            R_s=R_now, center_s=(alpha_b, beta_b),
            R_eps=R_now_eps,
            center_eps=(alpha_p + alpha_b / (2.0 * sf.G),
                        beta_p + beta_b / (2.0 * sf.G)),
            color_s='purple', color_eps='m', lw=1.0, dashes=(2, 2),
            label='RM yield (with backstress)'))

    return Solution(label='', yield_surfaces=ys)


def build_rm_stress_path_solutions(rm_stress_paths, t_now=None):
    """One Solution per N: a return-mapping stress path (panel b only)."""
    sols = []
    for N, (ts_rm_N, alpha_rm, beta_rm) in rm_stress_paths.items():
        mask = (ts_rm_N <= t_now) if t_now is not None else np.ones_like(ts_rm_N, dtype=bool)
        sols.append(Solution(
            label=f'RM N={N}', color=RM_COLORS.get(N, 'gray'),
            linestyle='-.', lw=1.0, alpha=0.7,
            s_pp=np.array([alpha_rm[mask], beta_rm[mask]])))
    return sols


def build_rm_plastic_solutions(rm_results, which_N=(10, 50, 200, 1000)):
    """One Solution per N: a return-mapping plastic-strain curve (panel c)."""
    sols = []
    for N in which_N:
        if N in rm_results:
            ts_rm, ebp_rm, _ = rm_results[N]
            sols.append(Solution(label=f'return map N={N}', linestyle='--', lw=1.2,
                                 plastic_strain=(ts_rm, ebp_rm)))
    return sols


def build_fem_solution(fem, iso_hardening=None, t_now=None, label='FEM (deal.II)'):
    """Build the deal.II FEM Solution from a PRE-LOADED overlay dict.

    `fem` is whatever build_fem_overlay returned (loaded by the caller),
    or None. No file access happens here.
    """
    if fem is None:
        return None
    mask = (fem['t'] <= t_now) if t_now is not None else np.ones_like(fem['t'], dtype=bool)
    if not np.any(mask):
        return None

    yield_surfaces = []
    epsilon_b_bar = fem.get('epsilon_b_bar')
    if iso_hardening is not None and epsilon_b_bar is not None:
        idx_now = np.flatnonzero(mask)[-1]
        ebp_now = float(epsilon_b_bar[idx_now])
        sy_now = pm.sigma_y_dispatcher(ebp_now, iso_hardening=iso_hardening)
        R_now = np.sqrt(2.0 / 3.0) * sy_now

        center_s = (0.0, 0.0)
        processed = fem.get('processed')
        if (fem.get('has_beta') and processed is not None
                and processed.backstress_in_sigma_pp is not None):
            center_s = (float(processed.backstress_in_sigma_pp[0][idx_now]),
                        float(processed.backstress_in_sigma_pp[1][idx_now]))

        yield_surfaces.append(YieldSurface(
            R_s=R_now, center_s=center_s, color='crimson',
            lw=1.0, dashes=(1, 2),
            label=f'FEM yield ($\\bar\\varepsilon^p$={ebp_now:.3f})'))

    return Solution(
        label=label, color='crimson', linestyle=':', lw=1.1, ms=2.5, alpha=0.85, zorder=5,
        eps_pp=np.array([fem['ea'][mask], fem['eb'][mask]]),
        s_pp=np.array([fem['sa'][mask], fem['sb'][mask]]),
        plastic_strain=(fem['t'][mask], fem['epsilon_b_bar'][mask])
        if fem.get('epsilon_b_bar') is not None else None,
        yield_surfaces=yield_surfaces,

        mark_endpoint=True, endpoint_color='crimson', endpoint_size=18,
        endpoint_edgecolor='k', endpoint_lw=0.4)


def build_convergence(rm_results, step_counts, ebp_exact, t_end, label='RM vs ODE'):
    errors, dts = [], []
    for N in step_counts:
        _, ebp_rm, _ = rm_results[N]
        errors.append(abs(ebp_rm[-1] - ebp_exact) / max(ebp_exact, 1e-30))
        dts.append(t_end / N)
    return Convergence(dts=np.array(dts), errors=np.array(errors), label=label)


# ──────────────────────────────────────────────────────────────────────
#  Animation — orchestrates per-frame dataset building + rendering.
#  Mirrors the static flow: one plot_results call per dataset, per frame.
# ──────────────────────────────────────────────────────────────────────
def animate_results(sol, t_analytical, gamma_analytical, ebp_exact, rm_results,
                    t_end, iso_hardening, kin_hardening_H, kin_hardening_beta, V_hist,
                    sy0, t_plot, alpha_s, beta_s, alpha_eps, beta_eps,
                    rm_stress_paths, case, fem=None, rm_full_data=None,
                    save_path='return_map_animation.mp4',
                    n_frames=180, fps=30, dpi=180, color='k'):
    """Animate the four-panel plot over t in [0, t_end].

    The FEM overlay (if any) is passed in pre-loaded; nothing is read from
    disk here.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    fig, axes = make_panels()
    frame_times = np.linspace(0, t_end, n_frames)
    hardening_label = f"iso={iso_hardening}, kin H={kin_hardening_H}, kin β={kin_hardening_beta}"

    def update(frame_idx):
        t_now = frame_times[frame_idx]

        # current-state info box for panel (d)
        idx_now = min(np.searchsorted(t_analytical, t_now), len(t_analytical) - 1)
        ebp_now = gamma_analytical[idx_now]
        sy_now = pm.sigma_y_dispatcher(ebp_now, iso_hardening=iso_hardening)
        R_now = np.sqrt(2.0 / 3.0) * sy_now
        info = (f"t = {t_now:.3f}\n"
                f"$\\bar\\varepsilon^p$ = {ebp_now:.4f}\n"
                f"$\\sigma_y$ = {sy_now:.1f}\n"
                f"R = {R_now:.1f}")
        suptitle = f"CASE {case} — {hardening_label}   |   t = {t_now:.3f}"

        # 1) IVP / ODE  (fresh: clears + sets the info box)
        ivp = build_ivp_solution(
            sol, t_analytical, gamma_analytical, ebp_exact,
            iso_hardening, kin_hardening_beta, V_hist, sy0,
            t_plot, alpha_s, beta_s, alpha_eps, beta_eps,
            t_now=t_now, color=color)
        plot_results(axes, [ivp], info_text=info, case=case, t_end=t_end,
                     suptitle=suptitle, clear_axes=True)

        # 2) Return mapping  (overlay)
        rm_sols = build_rm_stress_path_solutions(rm_stress_paths, t_now=t_now)
        rm_y = build_rm_yield_solution(rm_full_data, iso_hardening, kin_hardening_beta,
                                       alpha_eps, beta_eps, t_plot, t_now=t_now)
        if rm_y is not None:
            rm_sols.append(rm_y)
        plot_results(axes, rm_sols, case=case, t_end=t_end, clear_axes=False)

        # 3) FEM  (overlay)
        fem_sol = build_fem_solution(fem, iso_hardening, t_now=t_now)
        if fem_sol is not None:
            plot_results(axes, [fem_sol], case=case, t_end=t_end, clear_axes=False)

        fig.tight_layout()
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
