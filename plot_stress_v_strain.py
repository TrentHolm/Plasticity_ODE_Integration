"""
plot_fem_results.py
───────────────────
Utilities for reading FEM CSV output and plotting von Mises stress versus
equivalent (von Mises) strain.

File format expected
────────────────────
Semicolon-delimited, one header row, trailing semicolons tolerated:

  sig[0,0]; sig[1,1]; sig[2,2]; sig[0,1]; sig[0,2]; sig[1,2];
  eps[0,0]; eps[1,1]; eps[2,2]; eps[0,1]; eps[0,2]; eps[1,2];

Off-diagonal stress/strain components are assumed to be the *tensor*
components (i.e. eps[0,1] = ε₁₂, not the engineering shear γ₁₂ = 2ε₁₂).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _read_fem_csv(path: Path) -> pd.DataFrame:
    """Read a semicolon-delimited FEM CSV; strip whitespace from column names
    and drop the spurious empty column produced by a trailing semicolon."""
    df = pd.read_csv(path, sep=";", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    # Drop any unnamed columns (artifact of the trailing semicolon)
    df = df.loc[:, ~df.columns.str.fullmatch(r"Unnamed.*")]
    return df


def _von_mises_stress(df: pd.DataFrame) -> np.ndarray:
    """
    σ_VM = √( ½ [(σ₁₁-σ₂₂)² + (σ₂₂-σ₃₃)² + (σ₃₃-σ₁₁)²
                 + 6(σ₁₂² + σ₁₃² + σ₂₃²)] )
    """
    s11 = df["sig[0,0]"].to_numpy(float)
    s22 = df["sig[1,1]"].to_numpy(float)
    s33 = df["sig[2,2]"].to_numpy(float)
    s12 = df["sig[0,1]"].to_numpy(float)
    s13 = df["sig[0,2]"].to_numpy(float)
    s23 = df["sig[1,2]"].to_numpy(float)

    return np.sqrt(
        0.5 * (
            (s11 - s22) ** 2
            + (s22 - s33) ** 2
            + (s33 - s11) ** 2
            + 6.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)
        )
    )


def _equiv_strain(df: pd.DataFrame) -> np.ndarray:
    """
    ε̄ = √( ⅔ e_ij e_ij )   with   e_ij = ε_ij − ⅓ ε_kk δ_ij

    Expanded for the symmetric tensor stored column-wise:

        ε̄ = √( ⅔ [ d₁₁² + d₂₂² + d₃₃² + 2(ε₁₂² + ε₁₃² + ε₂₃²) ] )

    The factor of 2 on the off-diagonal terms comes from summing both
    upper and lower triangle of the tensor contraction.
    """
    e11 = df["eps[0,0]"].to_numpy(float)
    e22 = df["eps[1,1]"].to_numpy(float)
    e33 = df["eps[2,2]"].to_numpy(float)
    e12 = df["eps[0,1]"].to_numpy(float)
    e13 = df["eps[0,2]"].to_numpy(float)
    e23 = df["eps[1,2]"].to_numpy(float)

    # Deviatoric part (diagonal only; off-diagonals are already traceless)
    tr_over_3 = (e11 + e22 + e33) / 3.0
    d11 = e11 - tr_over_3
    d22 = e22 - tr_over_3
    d33 = e33 - tr_over_3

    return np.sqrt(
        (2.0 / 3.0) * (
            d11 ** 2 + d22 ** 2 + d33 ** 2
            + 2.0 * (e12 ** 2 + e13 ** 2 + e23 ** 2)
        )
    )


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def plot_vm_vs_eqstrain(
    cases: int | Sequence[int],
    data_dir: str | Path = ".",
    filename_pattern: str = "fem_out_c{case}.csv",
    ax: plt.Axes | None = None,
    labels: Sequence[str] | None = None,
    stress_scale: float = 1.0,
    stress_unit: str = "Pa",
    **plot_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Load FEM CSV output for one or more cases and plot von Mises stress
    versus equivalent strain.

    Parameters
    ----------
    cases : int or list of int
        Case number(s). Files are resolved as
        ``data_dir / filename_pattern.format(case=N)``.
    data_dir : str or Path
        Directory containing the CSV files. Defaults to the current
        working directory.
    filename_pattern : str
        Filename template; ``{case}`` is replaced by the case number.
    ax : Axes, optional
        Existing Axes to draw into. A new figure is created if omitted.
    labels : list of str, optional
        Legend labels, one per case. Defaults to ``'Case N'``.
    stress_scale : float
        Multiply raw stress values before plotting (e.g. ``1e-6`` to
        convert Pa → MPa).
    stress_unit : str
        Unit string shown on the y-axis label (e.g. ``'MPa'``).
    **plot_kwargs
        Forwarded to ``ax.plot()`` for every case (e.g. ``linewidth=2``).

    Returns
    -------
    fig : Figure
    ax  : Axes
    """
    if isinstance(cases, int):
        cases = [cases]

    if labels is None:
        labels = [f"Case {c}" for c in cases]

    data_dir = Path(data_dir)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    for case, label in zip(cases, labels):
        path = data_dir / filename_pattern.format(case=case)
        df = _read_fem_csv(path)

        sigma_vm = _von_mises_stress(df) * stress_scale
        eps_eq   = _equiv_strain(df)

        ax.plot(eps_eq, sigma_vm, label=label, **plot_kwargs)

    ax.set_xlabel(r"Equivalent strain $\bar{\varepsilon}$")
    ax.set_ylabel(rf"Von Mises stress $\sigma_{{\mathrm{{VM}}}}$ [{stress_unit}]")
    ax.set_title("Von Mises stress vs. equivalent strain")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    return fig, ax

if __name__ == "__main__":
    # Single case
    fig, ax = plot_vm_vs_eqstrain(1, data_dir="./")

    # Multiple cases on one plot, scaled to MPa
    fig, ax = plot_vm_vs_eqstrain(
        [1],
        data_dir="./",
        stress_scale=1e-6,
        stress_unit="MPa",
        linewidth=1.5,
    )
    plt.show()

    # Into an existing subplot grid
    fig, axes = plt.subplots(2, 3)
    for i, ax in enumerate(axes.flat, start=1):
        plot_vm_vs_eqstrain(i, data_dir="./", ax=ax)