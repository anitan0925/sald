"""Eight-Gaussian VP guided generation experiments for SALD, VA-SALD, and DOIT."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from sald_em_validation import _palette_for_r
from sald_guided_validation import GuidedRunProfile, discrete_kl
from sald_doit_validation import DoitConfig, doit_reward_from_guidance, make_three_algorithm_comparison_figure


@dataclass(frozen=True)
class EightGaussianConfig:
    T: float
    beta_min: float
    beta_max: float
    centers: tuple[tuple[float, float], ...]
    penalty_mode_indices: tuple[int, ...]
    penalty_strength: float = 1.0
    penalty_width: float = 0.85


@dataclass(frozen=True)
class EightGaussianGrid:
    x_edges: torch.Tensor
    y_edges: torch.Tensor
    x_centers: torch.Tensor
    y_centers: torch.Tensor
    mesh_x: torch.Tensor
    mesh_y: torch.Tensor
    flat_points: torch.Tensor
    f_grid: torch.Tensor
    dx: float
    dy: float


def make_circle_gaussian_centers(radius: float = 3.0, start_angle: float = math.pi / 8.0) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            float(radius * math.cos(start_angle + 2.0 * math.pi * k / 8.0)),
            float(radius * math.sin(start_angle + 2.0 * math.pi * k / 8.0)),
        )
        for k in range(8)
    )


DEFAULT_EIGHT_GAUSSIAN_CENTERS = make_circle_gaussian_centers()
DEFAULT_LEFT_HALF_MODE_INDICES = tuple(
    idx for idx, center in enumerate(DEFAULT_EIGHT_GAUSSIAN_CENTERS) if center[0] < 0.0
)


def beta_tau_scalar(tau: float, cfg: EightGaussianConfig) -> float:
    tau = max(0.0, min(cfg.T, float(tau)))
    return cfg.beta_min + (cfg.beta_max - cfg.beta_min) * tau / cfg.T


def reverse_beta_scalar(t: float, cfg: EightGaussianConfig) -> float:
    return beta_tau_scalar(cfg.T - float(t), cfg)


def alpha_tau_scalar(tau: float, cfg: EightGaussianConfig) -> float:
    tau = max(0.0, min(cfg.T, float(tau)))
    integral_beta = cfg.beta_min * tau + 0.5 * (cfg.beta_max - cfg.beta_min) * tau * tau / cfg.T
    return math.exp(-0.5 * integral_beta)


def centers_tensor(cfg: EightGaussianConfig, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(cfg.centers, device=device, dtype=dtype)


def mixture_means_at_t(t: float, cfg: EightGaussianConfig, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tau = max(0.0, min(cfg.T, cfg.T - float(t)))
    return alpha_tau_scalar(tau, cfg) * centers_tensor(cfg, device=device, dtype=dtype)


def mixture_logpdf_2d(points: torch.Tensor, t: float, cfg: EightGaussianConfig) -> torch.Tensor:
    means = mixture_means_at_t(t, cfg, device=points.device, dtype=points.dtype)
    diff = points[:, None, :] - means[None, :, :]
    log_comp = -0.5 * diff.square().sum(dim=-1) - math.log(2.0 * math.pi)
    return torch.logsumexp(log_comp, dim=1) - math.log(means.shape[0])


def mixture_score_2d(points: torch.Tensor, t: float, cfg: EightGaussianConfig) -> torch.Tensor:
    means = mixture_means_at_t(t, cfg, device=points.device, dtype=points.dtype)
    diff = points[:, None, :] - means[None, :, :]
    log_comp = -0.5 * diff.square().sum(dim=-1)
    weights = torch.softmax(log_comp, dim=1)
    posterior_mean = weights @ means
    return posterior_mean - points


def mode_penalty_and_grad(points: torch.Tensor, cfg: EightGaussianConfig) -> tuple[torch.Tensor, torch.Tensor]:
    penalized = centers_tensor(cfg, device=points.device, dtype=points.dtype)[list(cfg.penalty_mode_indices)]
    width2 = float(cfg.penalty_width) ** 2
    diff = points[:, None, :] - penalized[None, :, :]
    bump = torch.exp(-0.5 * diff.square().sum(dim=-1) / width2)
    f = cfg.penalty_strength * bump.sum(dim=1)
    grad = cfg.penalty_strength * (-(bump[..., None] * diff).sum(dim=1) / width2)
    return f, grad


def make_eight_gaussian_grid(
    profile: GuidedRunProfile,
    cfg: EightGaussianConfig,
    *,
    device: torch.device,
) -> EightGaussianGrid:
    x_edges = torch.linspace(profile.x_min, profile.x_max, profile.bins_x + 1, device=device, dtype=torch.float32)
    y_edges = torch.linspace(profile.y_min, profile.y_max, profile.bins_y + 1, device=device, dtype=torch.float32)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    mesh_y, mesh_x = torch.meshgrid(y_centers, x_centers, indexing="ij")
    flat_points = torch.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], dim=-1)
    f_grid, _ = mode_penalty_and_grad(flat_points, cfg)
    return EightGaussianGrid(
        x_edges=x_edges,
        y_edges=y_edges,
        x_centers=x_centers,
        y_centers=y_centers,
        mesh_x=mesh_x,
        mesh_y=mesh_y,
        flat_points=flat_points,
        f_grid=f_grid.reshape(profile.bins_y, profile.bins_x),
        dx=float(x_edges[1].item() - x_edges[0].item()),
        dy=float(y_edges[1].item() - y_edges[0].item()),
    )


def target_mass_grid(t: float, cfg: EightGaussianConfig, grid: EightGaussianGrid) -> torch.Tensor:
    log_unnorm = mixture_logpdf_2d(grid.flat_points, t, cfg) - grid.f_grid.reshape(-1)
    return torch.softmax(log_unnorm, dim=0)


def empirical_hist_prob_2d(points: torch.Tensor, grid: EightGaussianGrid, eps: float = 1e-8) -> torch.Tensor:
    bins_x = grid.x_edges.numel() - 1
    bins_y = grid.y_edges.numel() - 1
    idx_x = torch.bucketize(points[:, 0].contiguous(), grid.x_edges) - 1
    idx_y = torch.bucketize(points[:, 1].contiguous(), grid.y_edges) - 1
    valid = (idx_x >= 0) & (idx_x < bins_x) & (idx_y >= 0) & (idx_y < bins_y)
    flat_idx = idx_y[valid] * bins_x + idx_x[valid]
    counts = torch.zeros(bins_x * bins_y, device=points.device, dtype=torch.float32)
    counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
    counts = counts + eps
    return counts / counts.sum()


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    gen_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)
    return generator


def _checkpoint_steps(n_steps: int, checkpoint_count: int) -> list[int]:
    fractions = np.linspace(0.0, 1.0, checkpoint_count)
    return sorted({int(round(frac * n_steps)) for frac in fractions})


def _capture_row(
    x: torch.Tensor,
    completed_steps: int,
    eta: float,
    r: float,
    q_target: torch.Tensor,
    cfg: EightGaussianConfig,
    grid: EightGaussianGrid,
    algorithm: str,
) -> tuple[dict[str, float], np.ndarray]:
    s = completed_steps * eta
    if algorithm == "DOIT":
        t = min(cfg.T, s)
        u = s / cfg.T
    else:
        t = min(cfg.T, s / r)
        u = s / (r * cfg.T)
    hist_prob = empirical_hist_prob_2d(x, grid)
    q_path = target_mass_grid(t, cfg, grid)
    f_val, _ = mode_penalty_and_grad(x, cfg)
    mean_penalty = float(f_val.mean().item())
    row = {
        "algorithm": algorithm,
        "r": float(r),
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
    }
    return row, hist_prob.detach().cpu().numpy()


def _doit_grad(
    x: torch.Tensor,
    t: float,
    eta: float,
    cfg: EightGaussianConfig,
    doit_cfg: DoitConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    beta_t = reverse_beta_scalar(t, cfg)
    sigma_step = math.sqrt(max(eta * beta_t, 1e-12))
    base_score = mixture_score_2d(x, t, cfg)
    base_drift = 0.5 * beta_t * x + beta_t * base_score
    mean = x + eta * base_drift
    out = torch.empty_like(x)
    m = int(doit_cfg.m_proposals)
    tau = max(float(doit_cfg.tau), 1e-12)
    for start in range(0, x.shape[0], doit_cfg.chunk_size):
        stop = min(start + doit_cfg.chunk_size, x.shape[0])
        z = torch.randn((m, stop - start, x.shape[1]), device=x.device, dtype=x.dtype, generator=generator)
        cand = mean[start:stop].unsqueeze(0) + sigma_step * z
        f_val, _ = mode_penalty_and_grad(cand.reshape(-1, x.shape[1]), cfg)
        reward = doit_reward_from_guidance(f_val).reshape(m, stop - start)
        logits = (reward - reward.max(dim=0, keepdim=True).values) / tau
        weights = torch.softmax(logits, dim=0)
        out[start:stop] = (weights.unsqueeze(-1) * z).sum(dim=0) / sigma_step
    return out


def run_eight_gaussian_algorithm(
    algorithm: Literal["SALD", "VA-SALD", "DOIT"],
    r_values: Sequence[int],
    cfg: EightGaussianConfig,
    profile: GuidedRunProfile,
    grid: EightGaussianGrid,
    *,
    seed: int,
    doit_cfg: DoitConfig | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    device = grid.flat_points.device
    x0 = torch.randn(profile.n_particles, 2, device=device, dtype=torch.float32, generator=_make_generator(device, seed))
    q_target = target_mass_grid(cfg.T, cfg, grid)
    trajectory_rows: list[dict[str, float]] = []
    summary_rows: list[dict[str, Any]] = []
    sample_subsets: dict[int, np.ndarray] = {}
    histograms: dict[int, np.ndarray] = {}

    for run_index, r in enumerate(r_values):
        if verbose:
            label = "budget_r" if algorithm == "DOIT" else "r"
            print(f"running 8-Gaussian {algorithm} for {label}={r}, particles={profile.n_particles}")
        n_steps = max(1, math.ceil(r * cfg.T / profile.eta_s))
        checkpoints = _checkpoint_steps(n_steps, profile.checkpoint_count)
        if algorithm == "DOIT":
            eta = cfg.T / checkpoints[-1]
        else:
            eta = r * cfg.T / checkpoints[-1]
        x = x0.clone()
        generator = _make_generator(device, seed + 20_000 + 131 * run_index)
        final_hist = None
        start_time = time.perf_counter()
        checkpoint_index = 0
        for completed_steps in range(checkpoints[-1] + 1):
            if checkpoint_index < len(checkpoints) and completed_steps == checkpoints[checkpoint_index]:
                row, hist_prob = _capture_row(x, completed_steps, eta, r, q_target, cfg, grid, algorithm)
                row["eta"] = float(eta)
                row["n_steps"] = float(checkpoints[-1])
                row["n_particles"] = float(profile.n_particles)
                if algorithm == "DOIT" and doit_cfg is not None:
                    row["doit_m"] = float(doit_cfg.m_proposals)
                    row["effective_budget_particles"] = float(profile.n_particles * doit_cfg.m_proposals)
                trajectory_rows.append(row)
                if completed_steps == checkpoints[-1]:
                    final_hist = hist_prob
                checkpoint_index += 1
            if completed_steps == checkpoints[-1]:
                break
            if algorithm == "DOIT":
                t = min(cfg.T, completed_steps * eta)
            else:
                t = min(cfg.T, (completed_steps * eta) / r)
            score = mixture_score_2d(x, t, cfg)
            _, guide_grad = mode_penalty_and_grad(x, cfg)
            noise = torch.randn(x.shape, device=device, dtype=x.dtype, generator=generator)
            if algorithm == "SALD":
                x = x + eta * (score - guide_grad) + math.sqrt(2.0 * eta) * noise
            elif algorithm == "VA-SALD":
                beta_t = reverse_beta_scalar(t, cfg)
                drift = 0.5 * beta_t * (x / r + (1.0 + 1.0 / r) * score - guide_grad)
                x = x + eta * drift + math.sqrt(eta * beta_t) * noise
            elif algorithm == "DOIT":
                if doit_cfg is None:
                    raise ValueError("doit_cfg is required for DOIT")
                beta_t = reverse_beta_scalar(t, cfg)
                base_drift = 0.5 * beta_t * x + beta_t * score
                doob_grad = _doit_grad(x, t, eta, cfg, doit_cfg, generator)
                x = x + eta * (base_drift + doit_cfg.gamma * beta_t * doob_grad) + math.sqrt(eta * beta_t) * noise
            else:
                raise ValueError(f"unknown algorithm {algorithm}")
        wall_clock_sec = time.perf_counter() - start_time
        summary = dict(trajectory_rows[-1])
        summary["wall_clock_sec"] = float(wall_clock_sec)
        summary["r"] = int(r)
        summary["n_steps"] = int(summary["n_steps"])
        summary_rows.append(summary)
        sample_subsets[int(r)] = x[: min(profile.scatter_points, x.shape[0])].detach().cpu().numpy()
        histograms[int(r)] = final_hist

    target_density_final = (q_target / (grid.dx * grid.dy)).reshape(profile.bins_y, profile.bins_x)
    base_density_final = torch.exp(mixture_logpdf_2d(grid.flat_points, cfg.T, cfg)).reshape(profile.bins_y, profile.bins_x)
    return {
        "model_config": cfg,
        "profile": profile,
        "grid": grid,
        "trajectory_df": pd.DataFrame(trajectory_rows).sort_values(["r", "u"]).reset_index(drop=True),
        "summary_df": pd.DataFrame(summary_rows).sort_values("r").reset_index(drop=True),
        "sample_subsets": sample_subsets,
        "histograms": histograms,
        "target_density_final": target_density_final.detach().cpu().numpy(),
        "base_density_final": base_density_final.detach().cpu().numpy(),
        "mesh_x": grid.mesh_x.detach().cpu().numpy(),
        "mesh_y": grid.mesh_y.detach().cpu().numpy(),
    }


def equal_budget_profile(profile: GuidedRunProfile, doit_cfg: DoitConfig, *, name: str) -> GuidedRunProfile:
    return replace(profile, name=name, n_particles=max(1, profile.n_particles // max(1, doit_cfg.m_proposals)))


def make_eight_gaussian_setup_figure(cfg: EightGaussianConfig, grid: EightGaussianGrid) -> "plt.Figure":
    target_density = (target_mass_grid(cfg.T, cfg, grid) / (grid.dx * grid.dy)).reshape(grid.f_grid.shape)
    base_density = torch.exp(mixture_logpdf_2d(grid.flat_points, cfg.T, cfg)).reshape(grid.f_grid.shape)
    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    ax.contourf(grid.mesh_x.cpu(), grid.mesh_y.cpu(), target_density.cpu(), levels=16, cmap="YlOrBr", alpha=0.42)
    ax.contour(grid.mesh_x.cpu(), grid.mesh_y.cpu(), base_density.cpu(), levels=10, colors=["#4d4d4d"], linewidths=1.0)
    centers = np.asarray(cfg.centers)
    penalized = centers[list(cfg.penalty_mode_indices)]
    unpenalized = np.delete(centers, list(cfg.penalty_mode_indices), axis=0)
    ax.scatter(unpenalized[:, 0], unpenalized[:, 1], s=90, color="#1f5c7a", label="unpenalized modes")
    ax.scatter(penalized[:, 0], penalized[:, 1], s=110, color="#b45f06", marker="x", linewidths=2.5, label="penalized modes")
    ax.set_title(r"8-Gaussian Terminal Target with Mode Penalty")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=10)
    return fig


def make_comparison_figure(sald_suite: dict[str, Any], va_suite: dict[str, Any], doit_suite: dict[str, Any]) -> "plt.Figure":
    return make_three_algorithm_comparison_figure(
        sald_suite,
        va_suite,
        doit_suite,
        title_prefix="8-Gaussian Mode-Penalty",
    )


def make_algorithm_overview_figure(
    suite: dict[str, Any],
    *,
    algorithm_name: str,
    selected_r: Sequence[int] | None = None,
) -> "plt.Figure":
    if selected_r is None:
        selected_r = suite["summary_df"]["r"].astype(int).tolist()
    all_r = suite["summary_df"]["r"].astype(int).tolist()
    palette = _palette_for_r(all_r)

    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.5), constrained_layout=True)

    ax = axes[0, 0]
    mesh_x = suite["mesh_x"]
    mesh_y = suite["mesh_y"]
    ax.contourf(mesh_x, mesh_y, suite["target_density_final"], levels=16, cmap="YlOrBr", alpha=0.42)
    ax.contour(mesh_x, mesh_y, suite["base_density_final"], levels=10, colors=["#4d4d4d"], linewidths=1.0)
    centers = np.asarray(suite["model_config"].centers)
    penalized = centers[list(suite["model_config"].penalty_mode_indices)]
    unpenalized = np.delete(centers, list(suite["model_config"].penalty_mode_indices), axis=0)
    ax.scatter(unpenalized[:, 0], unpenalized[:, 1], s=90, color="#1f5c7a", label="unpenalized modes")
    ax.scatter(penalized[:, 0], penalized[:, 1], s=110, color="#b45f06", marker="x", linewidths=2.5, label="penalized modes")
    ax.set_title(fr"{algorithm_name}: terminal target and circular modes")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    for r in all_r:
        sub = suite["trajectory_df"][suite["trajectory_df"]["r"] == float(r)]
        ax.plot(sub["u"], sub["kl_to_target"], color=palette[r], lw=1.8, alpha=0.9, label=fr"$r={r}$")
    ax.set_yscale("log")
    ax.set_title(r"Terminal-Target KL $KL(\rho_s \| \pi_T)$")
    ax.set_xlabel(r"normalized progress $u$")
    ax.set_ylabel(r"discrete KL to $\pi_T$")
    ax.legend(ncol=2, fontsize=9)

    ax = axes[1, 0]
    summary_df = suite["summary_df"]
    ax.plot(summary_df["r"], summary_df["kl_to_target"], marker="o", ms=6, lw=2.2, color="#1f5c7a")
    for _, row in summary_df.iterrows():
        r = int(row["r"])
        ax.scatter(row["r"], row["kl_to_target"], s=42, color=palette[r], zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"Final KL to the Guided Target")
    ax.set_xlabel(r"$r$ / budget label")
    ax.set_ylabel(r"final discrete KL")

    ax = axes[1, 1]
    for r in selected_r:
        sub = suite["trajectory_df"][suite["trajectory_df"]["r"] == float(r)]
        ax.plot(sub["u"], sub["mean_guidance"], color=palette[int(r)], lw=2.0, alpha=0.95, label=fr"$r={int(r)}$")
    ax.set_title(r"Mean Guidance Objective $\mathbb{E}[-f(X_s)]$")
    ax.set_xlabel(r"normalized progress $u$")
    ax.set_ylabel(r"mean guidance objective")
    ax.legend(ncol=2, fontsize=9)

    for ax in axes.flat:
        ax.grid(True, alpha=0.30)
    return fig
