"""
Animated strain and stress path plotter in principal spaces.

Given a prescribed strain path (eps_11, eps_22, eps_33=0, all shear=0),
computes the corresponding stress via 3D isotropic Hooke's law and animates:
  - The evolving path in principal strain space (eps_11 vs eps_22)
  - The evolving path in principal stress space (sigma_11 vs sigma_22)

Output: strain_stress_path.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter


# ---------------------------------------------------------------------------
# Material parameters
# ---------------------------------------------------------------------------
E  = 200e9   # Young's modulus [Pa]
nu = 0.3     # Poisson's ratio

sigma_y = 200  # Von mises yield stress [MPa]

lam = E * nu / ((1 + nu) * (1 - 2 * nu))
mu  = E / (2 * (1 + nu))


def hooke_3d(eps_11, eps_22, eps_33=0.0):
    """
    sigma_ij = lam * tr(eps) * delta_ij + 2*mu * eps_ij
    Returns sigma_11, sigma_22 for zero-shear, eps_33=0 input.
    """
    tr_eps   = eps_11 + eps_22 + eps_33
    sigma_11 = lam * tr_eps + 2 * mu * eps_11
    sigma_22 = lam * tr_eps + 2 * mu * eps_22

    return sigma_11, sigma_22

def von_mises_surface_parameterize(n_points_plot):
    theta = np.linspace(0, 2 * np.pi, n_points_plot)

    sigma_1 = (2.0 / np.sqrt(3)) * sigma_y * np.sin(theta + np.pi / 6)
    sigma_2 = (2.0 / np.sqrt(3)) * sigma_y * np.sin(theta - np.pi / 6)

    return sigma_1, sigma_2


# ---------------------------------------------------------------------------
# Full path (precompute everything)
# ---------------------------------------------------------------------------
n_steps   = 200
amplitude = 0.001

theta = 2.0 * np.pi * np.arange(n_steps + 1) / n_steps

eps_11_path = amplitude * np.sin(theta)
eps_22_path = amplitude * (1.0 - np.cos(theta))

sigma_11_path, sigma_22_path = hooke_3d(eps_11_path, eps_22_path)
sigma_11_MPa = sigma_11_path / 1e6
sigma_22_MPa = sigma_22_path / 1e6

sigma_1_vm, sigma_2_vm = von_mises_surface_parameterize(n_steps)

# Fixed axis limits with a small margin
def padded_lim(data, frac=0.12):
    lo, hi = data.min(), data.max()
    pad = (hi - lo) * frac or abs(lo) * frac or frac
    return lo - pad, hi + pad

e1_lim  = padded_lim(eps_11_path)
e2_lim  = padded_lim(eps_22_path)
s1_lim  = padded_lim(sigma_11_MPa)
s2_lim  = padded_lim(sigma_22_MPa)

# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(12, 5.5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

for ax in (ax1, ax2):
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", zorder=0)
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--", zorder=0)
    ax.grid(True, linewidth=0.4, alpha=0.5)

# Ghost (full future path)
ax1.plot(eps_11_path, eps_22_path,
         color="#B5D4F4", linewidth=1.0, linestyle="--", zorder=1)
ax2.plot(sigma_11_MPa, sigma_22_MPa,
         color="#B5D4F4", linewidth=1.0, linestyle="--", zorder=1)

ax2.plot(sigma_1_vm, sigma_2_vm, color="#D85A30", linewidth=1.0, linestyle="-", zorder=3, label="Von Mises yield surface")

# Start marker
ax1.scatter([eps_11_path[0]], [eps_22_path[0]],
            color="#3B6D11", s=45, marker="^", zorder=5, label="Start")
ax2.scatter([sigma_11_MPa[0]], [sigma_22_MPa[0]],
            color="#3B6D11", s=45, marker="^", zorder=5, label="Start")

# Animated artists
hist_line1, = ax1.plot([], [], color="#378ADD", linewidth=1.8, zorder=2, label="History")
hist_line2, = ax2.plot([], [], color="#378ADD", linewidth=1.8, zorder=2, label="History")
point1,     = ax1.plot([], [], "o", color="#D85A30", ms=7, zorder=6, label="Current")
point2,     = ax2.plot([], [], "o", color="#D85A30", ms=7, zorder=6, label="Current")
annot1      = ax1.annotate("", xy=(0, 0), xytext=(8, 8),
                            textcoords="offset points", fontsize=7.5,
                            color="#D85A30",
                            arrowprops=dict(arrowstyle="-", color="#D85A30", lw=0.7))
annot2      = ax2.annotate("", xy=(0, 0), xytext=(8, 8),
                            textcoords="offset points", fontsize=7.5,
                            color="#D85A30",
                            arrowprops=dict(arrowstyle="-", color="#D85A30", lw=0.7))

# Fixed labels and limits
ax1.set_xlabel(r"$\varepsilon_{11}$", fontsize=11)
ax1.set_ylabel(r"$\varepsilon_{22}$", fontsize=11)
ax1.set_title("Principal strain space", fontsize=10)
ax1.set_xlim(e1_lim); ax1.set_ylim(e2_lim)
ax1.legend(fontsize=8, loc="upper left")
ax1.set_aspect("equal", adjustable="box")

ax2.set_xlabel(r"$\sigma_{11}$ [MPa]", fontsize=11)
ax2.set_ylabel(r"$\sigma_{22}$ [MPa]", fontsize=11)
ax2.set_title("Principal stress space", fontsize=10)
ax2.set_xlim(s1_lim); ax2.set_ylim(s2_lim)
ax2.legend(fontsize=8, loc="upper left")
ax2.set_aspect("equal", adjustable="box")

title = fig.suptitle("", fontsize=10)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
def init():
    hist_line1.set_data([], [])
    hist_line2.set_data([], [])
    point1.set_data([], [])
    point2.set_data([], [])
    return hist_line1, hist_line2, point1, point2, annot1, annot2, title


def update(frame):
    i = frame  # frame index == load step

    # History slices
    hist_line1.set_data(eps_11_path[:i+1], eps_22_path[:i+1])
    hist_line2.set_data(sigma_11_MPa[:i+1], sigma_22_MPa[:i+1])

    # Current point
    e1, e2 = eps_11_path[i], eps_22_path[i]
    s1, s2 = sigma_11_MPa[i], sigma_22_MPa[i]

    point1.set_data([e1], [e2])
    point2.set_data([s1], [s2])

    annot1.xy = (e1, e2)
    annot1.set_text(f"({e1:.4f}, {e2:.4f})")

    annot2.xy = (s1, s2)
    annot2.set_text(f"({s1:.2f}, {s2:.2f}) MPa")

    t_deg = np.degrees(2.0 * np.pi * i / n_steps)
    title.set_text(
        f"E = {E/1e9:.0f} GPa   ν = {nu}   amplitude = {amplitude}   "
        f"step {i}/{n_steps}   θ = {t_deg:.1f}°"
    )

    return hist_line1, hist_line2, point1, point2, annot1, annot2, title


anim = FuncAnimation(
    fig, update,
    frames=n_steps + 1,
    init_func=init,
    interval=30,       # ms per frame → ~33 fps
    repeat=True
)

# out_file = "strain_stress_path.gif"
# anim.save(out_file, writer=PillowWriter(fps=33))
# print(f"Saved to {out_file}")
plt.show()