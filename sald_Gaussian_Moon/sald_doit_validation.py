"""DOIT-style training-free adaptation for the guided VP toy experiments."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sald_guided_validation import (
    GuidedFieldConfig,
    GuidedFieldState,
    GuidedRunProfile,
    discrete_kl,
    empirical_hist_prob_2d,
    guided_target_mass_grid,
    interpolate_guidance_potential,
    prepare_guidance_field,
    prior_score_2d,
)
from sald_guided_predictor_validation import reverse_beta_scalar


@dataclass(frozen=True)
class DoitConfig:
    m_proposals: int = 4
    tau: float = 0.6
    gamma: float = 1.0
    chunk_size: int = 32_768


def doit_reward_from_guidance(guidance_value: torch.Tensor) -> torch.Tensor:
    """DOIT maximizes reward; the guided target exp(-f) requires reward = -f."""
    return -guidance_value


def vp_reverse_sde_base_drift(x: torch.Tensor, t: float, model_cfg: Any) -> torch.Tensor:
    beta_t = reverse_beta_scalar(t, model_cfg)
    base_score = prior_score_2d(x, t, model_cfg)
    return 0.5 * beta_t * x + beta_t * base_score


def equal_budget_doit_profile(
    guided_profile: GuidedRunProfile,
    doit_cfg: DoitConfig,
    *,
    name: str = "doit",
) -> GuidedRunProfile:
    n_particles = max(1, int(guided_profile.n_particles) // max(1, int(doit_cfg.m_proposals)))
    return replace(guided_profile, name=name, n_particles=n_particles)


def _checkpoint_steps(n_steps: int, checkpoint_count: int) -> list[int]:
    fractions = np.linspace(0.0, 1.0, checkpoint_count)
    return sorted({int(round(frac * n_steps)) for frac in fractions})


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    gen_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)
    return generator


def _doit_correction(
    x: torch.Tensor,
    t: float,
    eta: float,
    model_cfg: Any,
    state: GuidedFieldState,
    doit_cfg: DoitConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    beta_t = reverse_beta_scalar(t, model_cfg)
    sigma_step = math.sqrt(max(eta * beta_t, 1e-12))
    base_drift = vp_reverse_sde_base_drift(x, t, model_cfg)
    mean = x + eta * base_drift

    out = torch.empty_like(x)
    m = int(doit_cfg.m_proposals)
    tau = max(float(doit_cfg.tau), 1e-12)
    for start in range(0, x.shape[0], doit_cfg.chunk_size):
        stop = min(start + doit_cfg.chunk_size, x.shape[0])
        mean_chunk = mean[start:stop]
        z = torch.randn((m, stop - start, x.shape[1]), device=x.device, dtype=x.dtype, generator=generator)
        candidates = mean_chunk.unsqueeze(0) + sigma_step * z
        reward = doit_reward_from_guidance(
            interpolate_guidance_potential(candidates.reshape(-1, x.shape[1]), state)
        ).reshape(m, stop - start)
        logits = (reward - reward.max(dim=0, keepdim=True).values) / tau
        weights = torch.softmax(logits, dim=0)
        out[start:stop] = (weights.unsqueeze(-1) * z).sum(dim=0) / sigma_step
    return out


def _capture_doit_row(
    x: torch.Tensor,
    completed_steps: int,
    eta: float,
    budget_r: float,
    q_target: torch.Tensor,
    model_cfg: Any,
    state: GuidedFieldState,
    doit_cfg: DoitConfig,
) -> tuple[dict[str, float], np.ndarray]:
    s = completed_steps * eta
    t = min(model_cfg.T, s)
    u = s / model_cfg.T
    hist_prob = empirical_hist_prob_2d(x, state.x_edges, state.y_edges)
    q_path = guided_target_mass_grid(t, model_cfg, state)
    guidance_values = interpolate_guidance_potential(x, state)
    mean_penalty = float(guidance_values.mean().item())
    row = {
        "r": float(budget_r),
        "budget_r": float(budget_r),
        "completed_steps": float(completed_steps),
        "s": float(s),
        "u": float(u),
        "t": float(t),
        "kl_to_path": float(discrete_kl(hist_prob, q_path).item()),
        "kl_to_target": float(discrete_kl(hist_prob, q_target).item()),
        "mean_x1": float(x[:, 0].mean().item()),
        "mean_x2": float(x[:, 1].mean().item()),
        "var_x1": float(x[:, 0].var(unbiased=False).item()),
        "var_x2": float(x[:, 1].var(unbiased=False).item()),
        "mean_guidance": -mean_penalty,
        "mean_penalty": mean_penalty,
        "mean_beta_reverse": float(reverse_beta_scalar(t, model_cfg)),
        "doit_m": float(doit_cfg.m_proposals),
        "doit_tau": float(doit_cfg.tau),
        "doit_gamma": float(doit_cfg.gamma),
    }
    return row, hist_prob.detach().cpu().numpy()


def run_doit_guided_vp_em_2d(
    budget_r: float,
    x0: torch.Tensor,
    model_cfg: Any,
    profile: GuidedRunProfile,
    state: GuidedFieldState,
    doit_cfg: DoitConfig,
    *,
    seed: int,
    checkpoint_count: int | None = None,
) -> dict[str, Any]:
    if budget_r <= 0:
        raise ValueError("budget_r must be positive")

    q_target = guided_target_mass_grid(model_cfg.T, model_cfg, state)
    checkpoint_steps = _checkpoint_steps(
        n_steps=max(1, math.ceil(budget_r * model_cfg.T / profile.eta_s)),
        checkpoint_count=checkpoint_count or profile.checkpoint_count,
    )
    n_steps = checkpoint_steps[-1]
    eta = model_cfg.T / n_steps

    x = x0.clone()
    generator = _make_generator(x.device, seed)
    trajectory_rows: list[dict[str, float]] = []
    final_hist: np.ndarray | None = None

    if x.device.type == "cuda":
        torch.cuda.synchronize(device=x.device)
    start_time = time.perf_counter()

    checkpoint_index = 0
    for completed_steps in range(n_steps + 1):
        if checkpoint_index < len(checkpoint_steps) and completed_steps == checkpoint_steps[checkpoint_index]:
            row, hist_prob = _capture_doit_row(x, completed_steps, eta, budget_r, q_target, model_cfg, state, doit_cfg)
            row["eta"] = float(eta)
            row["n_steps"] = float(n_steps)
            row["n_particles"] = float(profile.n_particles)
            row["effective_budget_particles"] = float(profile.n_particles * doit_cfg.m_proposals)
            trajectory_rows.append(row)
            if completed_steps == n_steps:
                final_hist = hist_prob
            checkpoint_index += 1

        if completed_steps == n_steps:
            break

        t = min(model_cfg.T, completed_steps * eta)
        beta_t = reverse_beta_scalar(t, model_cfg)
        base_drift = vp_reverse_sde_base_drift(x, t, model_cfg)
        doob_grad = _doit_correction(x, t, eta, model_cfg, state, doit_cfg, generator)
        drift = base_drift + doit_cfg.gamma * beta_t * doob_grad
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        x = x + eta * drift + math.sqrt(eta * beta_t) * noise

    if x.device.type == "cuda":
        torch.cuda.synchronize(device=x.device)
    wall_clock_sec = time.perf_counter() - start_time

    if final_hist is None:
        final_row, final_hist = _capture_doit_row(x, n_steps, eta, budget_r, q_target, model_cfg, state, doit_cfg)
        final_row["eta"] = float(eta)
        final_row["n_steps"] = float(n_steps)
        final_row["n_particles"] = float(profile.n_particles)
        final_row["effective_budget_particles"] = float(profile.n_particles * doit_cfg.m_proposals)
        trajectory_rows.append(final_row)

    summary = dict(trajectory_rows[-1])
    summary["wall_clock_sec"] = float(wall_clock_sec)
    summary["r"] = int(budget_r)
    summary["n_steps"] = int(summary["n_steps"])

    subset_count = min(profile.scatter_points, x.shape[0])
    return {
        "trajectory_rows": trajectory_rows,
        "summary_row": summary,
        "sample_subset": x[:subset_count].detach().cpu().numpy(),
        "hist_final": final_hist,
    }


def run_doit_guided_experiment_suite(
    r_values: Sequence[float],
    model_cfg: Any,
    profile: GuidedRunProfile,
    state: GuidedFieldState,
    doit_cfg: DoitConfig,
    *,
    seed: int = 0,
    checkpoint_count: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    device = state.flat_points.device
    generator = _make_generator(device, seed)
    x0 = torch.randn(profile.n_particles, 2, device=device, dtype=torch.float32, generator=generator)

    trajectory_rows: list[dict[str, float]] = []
    summary_rows: list[dict[str, Any]] = []
    sample_subsets: dict[int, np.ndarray] = {}
    histograms: dict[int, np.ndarray] = {}

    for run_index, r in enumerate(r_values):
        if verbose:
            print(
                f"running DOIT for budget_r={r} with eta_s={profile.eta_s:.4f}, "
                f"particles={profile.n_particles}, M={doit_cfg.m_proposals}"
            )
        result = run_doit_guided_vp_em_2d(
            r,
            x0,
            model_cfg,
            profile,
            state,
            doit_cfg,
            seed=seed + 20_000 + 131 * run_index,
            checkpoint_count=checkpoint_count,
        )
        trajectory_rows.extend(result["trajectory_rows"])
        summary_rows.append(result["summary_row"])
        sample_subsets[int(r)] = result["sample_subset"]
        histograms[int(r)] = result["hist_final"]

    trajectory_df = pd.DataFrame(trajectory_rows).sort_values(["r", "u"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("r").reset_index(drop=True)
    target_prob_final = guided_target_mass_grid(model_cfg.T, model_cfg, state)
    target_density_final = (target_prob_final / (state.dx * state.dy)).reshape(state.config.bins_y, state.config.bins_x)

    return {
        "model_config": model_cfg,
        "profile": profile,
        "field_state": state,
        "doit_config": doit_cfg,
        "trajectory_df": trajectory_df,
        "summary_df": summary_df,
        "sample_subsets": sample_subsets,
        "histograms": histograms,
        "target_prob_final": target_prob_final.detach().cpu().numpy(),
        "target_density_final": target_density_final.detach().cpu().numpy(),
        "x_edges": state.x_edges.detach().cpu().numpy(),
        "y_edges": state.y_edges.detach().cpu().numpy(),
        "mesh_x": state.mesh_x.detach().cpu().numpy(),
        "mesh_y": state.mesh_y.detach().cpu().numpy(),
        "reference_plot_points": state.plot_reference_points,
    }


def make_three_algorithm_comparison_figure(
    guided_suite: dict[str, Any],
    va_suite: dict[str, Any],
    doit_suite: dict[str, Any],
    *,
    title_prefix: str,
    selected_r: Sequence[int] | None = None,
) -> "plt.Figure":
    if selected_r is None:
        selected_r = guided_suite["summary_df"]["r"].astype(int).tolist()
    selected = {float(r) for r in selected_r}
    specs = [
        ("SALD", guided_suite["trajectory_df"], "#5f7890"),
        ("VA-SALD", va_suite["trajectory_df"], "#1f5c7a"),
        ("DOIT", doit_suite["trajectory_df"], "#b45f06"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), constrained_layout=True)
    for name, df, color in specs:
        plot_df = df[df["r"].isin(selected)]
        final_df = plot_df.sort_values(["r", "u"]).groupby("r", as_index=False).tail(1).sort_values("r")
        axes[0].plot(final_df["r"], final_df["kl_to_target"], marker="o", lw=2.2, color=color, label=name)
        mean_by_u = plot_df.groupby("u", as_index=False)["mean_guidance"].mean()
        axes[1].plot(mean_by_u["u"], mean_by_u["mean_guidance"], lw=2.2, color=color, label=name)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title(f"{title_prefix}: Terminal-Target KL")
    axes[0].set_xlabel(r"$r$")
    axes[0].set_ylabel(r"final discrete KL")
    axes[1].set_title(f"{title_prefix}: Mean Guidance Objective")
    axes[1].set_xlabel(r"normalized progress $u$")
    axes[1].set_ylabel(r"mean guidance objective $\mathbb{E}[-f(X_s)]$")
    for ax in axes:
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.35)
    return fig


def make_doit_overview_figure(
    suite: dict[str, Any],
    *,
    selected_r: Sequence[int] | None = None,
) -> "plt.Figure":
    if selected_r is None:
        selected_r = suite["summary_df"]["r"].astype(int).tolist()
    all_r = suite["summary_df"]["r"].astype(int).tolist()

    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.contourf(suite["mesh_x"], suite["mesh_y"], suite["target_density_final"], levels=14, cmap="YlOrBr", alpha=0.35)
    ax.contour(suite["mesh_x"], suite["mesh_y"], suite["target_density_final"], levels=10, colors=["#1f5c7a"], linewidths=1.2, alpha=0.9)
    ref = suite.get("reference_plot_points")
    if ref is not None:
        ax.scatter(ref[:, 0], ref[:, 1], s=5.0, alpha=0.14, color="#3a3836", edgecolors="none")
    ax.set_title(r"DOIT Target $\pi_T$ and Two-Moons Reference")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal", adjustable="box")

    ax = axes[0, 1]
    for r in all_r:
        sub = suite["trajectory_df"][suite["trajectory_df"]["r"] == float(r)]
        ax.plot(sub["u"], sub["kl_to_target"], lw=1.8, alpha=0.9, label=fr"$r_b={r}$")
    ax.set_yscale("log")
    ax.set_title(r"Terminal-Target KL $KL(\rho_s \| \pi_T)$")
    ax.set_xlabel(r"normalized progress $u=s/T$")
    ax.set_ylabel(r"discrete KL to $\pi_T$")
    ax.legend(ncol=2, fontsize=9)

    ax = axes[1, 0]
    summary_df = suite["summary_df"]
    ax.plot(summary_df["r"], summary_df["kl_to_target"], marker="o", ms=6, lw=2.2, color="#b45f06")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"Final KL to the Guided Target")
    ax.set_xlabel(r"budget label $r_b$")
    ax.set_ylabel(r"final discrete KL")

    ax = axes[1, 1]
    for r in selected_r:
        sub = suite["trajectory_df"][suite["trajectory_df"]["r"] == float(r)]
        ax.plot(sub["u"], sub["mean_guidance"], lw=2.0, alpha=0.95, label=fr"$r_b={int(r)}$")
    ax.set_title(r"Mean Guidance Objective $\mathbb{E}[-f(X_s)]$")
    ax.set_xlabel(r"normalized progress $u=s/T$")
    ax.set_ylabel(r"mean guidance objective")
    ax.legend(ncol=2, fontsize=9)

    for ax in axes.flat:
        ax.grid(True, alpha=0.30)
    return fig


def run_doit_lambda_sweep(*args, **kwargs):
    raise RuntimeError("Lambda sweeps are disabled for this notebook; set lambda_guidance directly instead.")


run_doit_guided_sald_em_2d = run_doit_guided_vp_em_2d
