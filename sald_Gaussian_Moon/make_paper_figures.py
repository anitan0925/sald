#!/usr/bin/env python3
"""Regenerate the synthetic-data figure panels used by the paper."""

from __future__ import annotations

import shutil
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / os.environ.get("SALD_OUTPUT_DIR", "outputs")
FIG_DIR = ROOT / os.environ.get("SALD_FIG_DIR", "Figs")

METHODS = [
    ("SALD", "#5f7890", "o"),
    ("VA-SALD", "#1f5c7a", "s"),
    ("DOIT", "#b45f06", "^"),
]


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 26,
            "axes.titlesize": 28,
            "axes.labelsize": 26,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 20,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "lines.linewidth": 4.0,
            "lines.markersize": 12,
        }
    )


def _read_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"missing required CSV: {path}")
    return pd.read_csv(path)


def _copy_notebook_pngs() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    names = [
        "target_family.png",
        "sald_overview_safe_mu225.png",
        "sald_samples_safe_mu225.png",
        "guided_overview_guided_moons_lam1.png",
        "guided_samples_guided_moons_lam1.png",
        "velocity_aware_overview_velocity_aware_moons_lam1.png",
        "velocity_aware_samples_velocity_aware_moons_lam1.png",
        "doit_overview_doit_moons_lam1.png",
        "three_algorithm_comparison_two_moons_guided_moons_lam1.png",
        "eight_gaussian_target_setup_eight_gaussian_left_penalty_lam1.png",
        "eight_gaussian_sald_overview_eight_gaussian_left_penalty_lam1.png",
        "eight_gaussian_va_overview_eight_gaussian_left_penalty_lam1.png",
        "eight_gaussian_doit_overview_eight_gaussian_left_penalty_lam1.png",
        "three_algorithm_comparison_eight_gaussian_eight_gaussian_left_penalty_lam1.png",
    ]
    for name in names:
        src = OUTPUT_DIR / name
        if src.exists():
            shutil.copy2(src, FIG_DIR / name)


def _plot_kl(summary: dict[str, pd.DataFrame], title: str, ylabel: str, output_name: str) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 7.2), constrained_layout=True)
    for method, color, marker in METHODS:
        df = summary[method].sort_values("r")
        ax.plot(df["r"], df["kl_to_target"], label=method, color=color, marker=marker)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(r"$r$ (Computational Budget)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(FIG_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_guidance(trajectory: dict[str, pd.DataFrame], title: str, ylabel: str, output_name: str) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 7.2), constrained_layout=True)
    for method, color, _marker in METHODS:
        df = trajectory[method]
        curve = df.groupby("u", as_index=False)["mean_guidance"].mean().sort_values("u")
        ax.plot(curve["u"], curve["mean_guidance"], label=method, color=color)
    ax.set_title(title)
    ax.set_xlabel(r"Normalized Progress $u$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(FIG_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(f"outputs directory does not exist: {OUTPUT_DIR}")
    FIG_DIR.mkdir(exist_ok=True)
    _setup_style()
    _copy_notebook_pngs()

    two_moons_summary = {
        "SALD": _read_csv("guided_summary_guided_moons_lam1.csv"),
        "VA-SALD": _read_csv("velocity_aware_summary_velocity_aware_moons_lam1.csv"),
        "DOIT": _read_csv("doit_summary_doit_moons_lam1.csv"),
    }
    two_moons_traj = {
        "SALD": _read_csv("guided_trajectory_guided_moons_lam1.csv"),
        "VA-SALD": _read_csv("velocity_aware_trajectory_velocity_aware_moons_lam1.csv"),
        "DOIT": _read_csv("doit_trajectory_doit_moons_lam1.csv"),
    }
    eight_summary = {
        "SALD": _read_csv("eight_gaussian_sald_summary_eight_gaussian_left_penalty_lam1.csv"),
        "VA-SALD": _read_csv("eight_gaussian_va_summary_eight_gaussian_left_penalty_lam1.csv"),
        "DOIT": _read_csv("eight_gaussian_doit_summary_eight_gaussian_left_penalty_lam1.csv"),
    }
    eight_traj = {
        "SALD": _read_csv("eight_gaussian_sald_trajectory_eight_gaussian_left_penalty_lam1.csv"),
        "VA-SALD": _read_csv("eight_gaussian_va_trajectory_eight_gaussian_left_penalty_lam1.csv"),
        "DOIT": _read_csv("eight_gaussian_doit_trajectory_eight_gaussian_left_penalty_lam1.csv"),
    }

    _plot_kl(
        two_moons_summary,
        "KL vs. r: Two-Moons Guided 2-Gaussian",
        "Terminal-Target KL",
        "main_panel_a_kl_two_moons.png",
    )
    _plot_guidance(
        two_moons_traj,
        "Guidance Value: Two-Moons Guided 2-Gaussian",
        "Mean Guidance Value",
        "main_panel_b_guidance_two_moons.png",
    )
    _plot_kl(
        eight_summary,
        "KL vs. r: 8-Gaussian with Mode Penalty",
        "Terminal-Target KL",
        "main_panel_c_kl_eight_gaussian.png",
    )
    _plot_guidance(
        eight_traj,
        "Guidance Value: 8-Gaussian with Mode Penalty",
        "Mean Guidance Value",
        "main_panel_d_guidance_eight_gaussian.png",
    )
    print(f"wrote paper figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
