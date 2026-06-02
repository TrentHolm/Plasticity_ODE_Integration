# ──────────────────────────────────────────────────────────────────
# Compare logarithmic vs engineering strain on the pi-plane
# ──────────────────────────────────────────────────────────────────
import log_strain_tensor_cases as hsc
import principal_radial_and_angular_values as pv

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path.cwd() / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

t_end = 8.0

t_plot = np.linspace(0, t_end, 2000)

# Engineering (small) strain: ε = (F - I + (F - I)ᵀ) / 2
eps_small = np.zeros((len(t_plot), 3, 3))
for i, tv in enumerate(t_plot):
    F = np.array(hsc.F_func(tv), dtype=float)
    H = F - np.eye(3)
    eps_small[i] = 0.5 * (H + H.T)
    eps_small[i] -= (np.trace(eps_small[i]) / 3.0) * np.eye(3)

# Hencky (logarithmic) strain: e = ½ ln(B)
eps_hencky = np.zeros((len(t_plot), 3, 3))
for i, tv in enumerate(t_plot):
    eps_hencky[i] = hsc.hencky_strain_dev(tv)

# Green-Lagrange strain: E = ½(FᵀF - I)
eps_green = np.zeros((len(t_plot), 3, 3))
for i, tv in enumerate(t_plot):
    F = np.array(hsc.F_func(tv), dtype=float)
    C = F.T @ F
    E = 0.5 * (C - np.eye(3))
    eps_green[i] -= (np.trace(E) / 3.0) * np.eye(3)
    eps_green[i] = E - (np.trace(E) / 3.0) * np.eye(3)

# Pi-plane coords for each
alpha_small, beta_small, _ = pv.pi_plane_coords_unfolded(eps_small)
alpha_hencky, beta_hencky, _ = pv.pi_plane_coords_unfolded(eps_hencky)
alpha_green, beta_green, _ = pv.pi_plane_coords_unfolded(eps_green)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.plot(alpha_small, beta_small, 'b-', lw=1.5, label='Engineering (small)')
ax.plot(alpha_hencky, beta_hencky, 'r--', lw=1.5, label='Hencky (logarithmic)')
ax.plot(alpha_green, beta_green, 'g:', lw=1.5, label='Green-Lagrange')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\beta$')
ax.set_title(f'Strain measures in $\\pi$-plane (c{hsc.CASE})')
ax.set_aspect('equal')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Component comparison vs time
ax = axes[1]
ax.plot(t_plot, [eps_small[i][0, 0] for i in range(len(t_plot))],
        'b-', lw=1.5, label=r'$\varepsilon_{11}$ (small)')
ax.plot(t_plot, [eps_hencky[i][0, 0] for i in range(len(t_plot))],
        'r--', lw=1.5, label=r'$e_{11}$ (Hencky)')
ax.plot(t_plot, [eps_green[i][0, 0] for i in range(len(t_plot))],
        'g:', lw=1.5, label=r'$E_{11}$ (Green-Lagrange)')
ax.set_xlabel(r'$t$')
ax.set_ylabel('Strain component')
ax.set_title(f'$\\varepsilon_{{11}}$ vs $e_{{11}}$ vs $E_{{11}}$ (c{hsc.CASE})')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle(f'Strain measure comparison — Case {hsc.CASE}', fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'strain_measure_comparison_{hsc.CASE}.png', dpi=180)
print("Strain comparison plot saved")
plt.close()