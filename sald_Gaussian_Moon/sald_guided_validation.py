from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

from sald_em_validation import (
    FIXED_SAMPLE_POINT_COLOR,
    LOG_2PI,
    ModelConfig,
    _palette_for_r,
    mixture_radius_at_t,
    target_density_2d,
    target_logpdf_1d,
)


@dataclass(frozen=True)
class GuidedFieldConfig:
    lambda_guidance: float
    n_reference: int
    moon_noise: float
    moon_scale: float
    moon_shift_x: float
    moon_shift_y: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    bins_x: int
    bins_y: int
    distance_eps: float = 1e-3
    ref_chunk_size: int = 2048
    grid_chunk_size: int = 1024
    plot_reference_points: int = 12000


@dataclass(frozen=True)
class GuidedRunProfile:
    name: str
    n_particles: int
    eta_s: float
    checkpoint_count: int
    scatter_points: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    bins_x: int
    bins_y: int


@dataclass
class GuidedFieldState:
    config: GuidedFieldConfig
    x_edges: torch.Tensor
    y_edges: torch.Tensor
    x_centers: torch.Tensor
    y_centers: torch.Tensor
    mesh_x: torch.Tensor
    mesh_y: torch.Tensor
    flat_points: torch.Tensor
    f_grid: torch.Tensor
    grad_grid: torch.Tensor
    f_field: torch.Tensor
    grad_field: torch.Tensor
    dx: float
    dy: float
    plot_reference_points: np.ndarray


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    gen_device = device.type if device.type == 'cuda' else 'cpu'
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)
    return generator


def make_two_moons_samples(
    n_samples: int,
    *,
    noise: float,
    scale: float,
    shift_x: float,
    shift_y: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if n_samples <= 0:
        raise ValueError('n_samples must be positive')

    generator = _make_generator(device, seed)
    n_first = n_samples // 2
    n_second = n_samples - n_first

    theta_first = math.pi * torch.rand(n_first, device=device, dtype=torch.float32, generator=generator)
    theta_second = math.pi * torch.rand(n_second, device=device, dtype=torch.float32, generator=generator)

    moon_first = torch.stack([torch.cos(theta_first), torch.sin(theta_first)], dim=-1)
    moon_second = torch.stack([1.0 - torch.cos(theta_second), 1.0 - torch.sin(theta_second) - 0.5], dim=-1)
    points = torch.cat([moon_first, moon_second], dim=0)

    if noise > 0.0:
        points = points + noise * torch.randn(points.shape, device=device, dtype=torch.float32, generator=generator)

    points = points * scale
    points[:, 0] = points[:, 0] + shift_x
    points[:, 1] = points[:, 1] + shift_y

    permutation = torch.randperm(n_samples, device=device, generator=generator)
    return points[permutation]


def prepare_guidance_field(
    field_cfg: GuidedFieldConfig,
    *,
    seed: int,
    device: torch.device | None = None,
    verbose: bool = True,
) -> GuidedFieldState:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reference_points = make_two_moons_samples(
        field_cfg.n_reference,
        noise=field_cfg.moon_noise,
        scale=field_cfg.moon_scale,
        shift_x=field_cfg.moon_shift_x,
        shift_y=field_cfg.moon_shift_y,
        seed=seed,
        device=device,
    )

    x_edges = torch.linspace(field_cfg.x_min, field_cfg.x_max, field_cfg.bins_x + 1, device=device, dtype=torch.float32)
    y_edges = torch.linspace(field_cfg.y_min, field_cfg.y_max, field_cfg.bins_y + 1, device=device, dtype=torch.float32)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    mesh_y, mesh_x = torch.meshgrid(y_centers, x_centers, indexing='ij')
    flat_points = torch.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], dim=-1)

    n_grid = flat_points.shape[0]
    f_flat = torch.zeros(n_grid, device=device, dtype=torch.float32)
    grad_flat = torch.zeros(n_grid, 2, device=device, dtype=torch.float32)

    start_time = time.perf_counter()
    for g0 in range(0, n_grid, field_cfg.grid_chunk_size):
        g1 = min(g0 + field_cfg.grid_chunk_size, n_grid)
        grid_chunk = flat_points[g0:g1]
        f_chunk = torch.zeros(grid_chunk.shape[0], device=device, dtype=torch.float32)
        grad_chunk = torch.zeros(grid_chunk.shape[0], 2, device=device, dtype=torch.float32)

        for r0 in range(0, reference_points.shape[0], field_cfg.ref_chunk_size):
            r1 = min(r0 + field_cfg.ref_chunk_size, reference_points.shape[0])
            ref_chunk = reference_points[r0:r1]
            diff = grid_chunk[:, None, :] - ref_chunk[None, :, :]
            distance = torch.sqrt(diff.square().sum(dim=-1) + field_cfg.distance_eps**2)
            f_chunk += distance.sum(dim=1)
            grad_chunk += (diff / distance.unsqueeze(-1)).sum(dim=1)

        normalizer = float(reference_points.shape[0]) * field_cfg.lambda_guidance
        f_flat[g0:g1] = f_chunk / normalizer
        grad_flat[g0:g1] = grad_chunk / normalizer

    if verbose:
        elapsed = time.perf_counter() - start_time
        print(
            f'prepared guidance field on {device.type} with '
            f'{field_cfg.n_reference} reference moons in {elapsed:.2f}s'
        )

    f_grid = f_flat.reshape(field_cfg.bins_y, field_cfg.bins_x)
    grad_grid = grad_flat.reshape(field_cfg.bins_y, field_cfg.bins_x, 2).permute(2, 0, 1).contiguous()

    plot_count = min(field_cfg.plot_reference_points, reference_points.shape[0])
    plot_reference_points = reference_points[:plot_count].detach().cpu().numpy()

    return GuidedFieldState(
        config=field_cfg,
        x_edges=x_edges,
        y_edges=y_edges,
        x_centers=x_centers,
        y_centers=y_centers,
        mesh_x=mesh_x,
        mesh_y=mesh_y,
        flat_points=flat_points,
        f_grid=f_grid,
        grad_grid=grad_grid,
        f_field=f_grid.unsqueeze(0).unsqueeze(0),
        grad_field=grad_grid.unsqueeze(0),
        dx=float(x_edges[1].item() - x_edges[0].item()),
        dy=float(y_edges[1].item() - y_edges[0].item()),
        plot_reference_points=plot_reference_points,
    )


def _sample_field(field: torch.Tensor, points: torch.Tensor, state: GuidedFieldState) -> torch.Tensor:
    x_norm = 2.0 * (points[:, 0] - state.config.x_min) / (state.config.x_max - state.config.x_min) - 1.0
    y_norm = 2.0 * (points[:, 1] - state.config.y_min) / (state.config.y_max - state.config.y_min) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(field, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1)


def interpolate_guidance_potential(points: torch.Tensor, state: GuidedFieldState) -> torch.Tensor:
    return _sample_field(state.f_field, points, state).squeeze(-1)


def interpolate_guidance_grad(points: torch.Tensor, state: GuidedFieldState) -> torch.Tensor:
    return _sample_field(state.grad_field, points, state)


def prior_score_2d(points: torch.Tensor, t: Any, model_cfg: ModelConfig) -> torch.Tensor:
    m_t = mixture_radius_at_t(t, model_cfg).to(device=points.device, dtype=points.dtype)
    score = torch.empty_like(points)
    score[:, 0] = -points[:, 0] + m_t * torch.tanh(m_t * points[:, 0])
    score[:, 1] = -points[:, 1]
    return score


def prior_logpdf_2d(points: torch.Tensor, t: Any, model_cfg: ModelConfig) -> torch.Tensor:
    return target_logpdf_1d(points[:, 0], t, model_cfg) - 0.5 * (points[:, 1].square() + LOG_2PI)


def guided_score(points: torch.Tensor, t: Any, model_cfg: ModelConfig, state: GuidedFieldState) -> torch.Tensor:
    return prior_score_2d(points, t, model_cfg) - interpolate_guidance_grad(points, state)


def guided_log_unnormalized(points: torch.Tensor, t: Any, model_cfg: ModelConfig, state: GuidedFieldState) -> torch.Tensor:
    return prior_logpdf_2d(points, t, model_cfg) - interpolate_guidance_potential(points, state)


def guided_target_mass_grid(t: float, model_cfg: ModelConfig, state: GuidedFieldState) -> torch.Tensor:
    log_unnorm = prior_logpdf_2d(state.flat_points, t, model_cfg) - state.f_grid.reshape(-1)
    return torch.softmax(log_unnorm, dim=0)


def empirical_hist_prob_2d(
    points: torch.Tensor,
    x_edges: torch.Tensor,
    y_edges: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    bins_x = x_edges.numel() - 1
    bins_y = y_edges.numel() - 1
    idx_x = torch.bucketize(points[:, 0], x_edges) - 1
    idx_y = torch.bucketize(points[:, 1], y_edges) - 1
    valid = (idx_x >= 0) & (idx_x < bins_x) & (idx_y >= 0) & (idx_y < bins_y)
    flat_idx = idx_y[valid] * bins_x + idx_x[valid]
    counts = torch.zeros(bins_x * bins_y, device=points.device, dtype=torch.float32)
    counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
    counts = counts + eps
    return counts / counts.sum()


def discrete_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.sum(p * (torch.log(p) - torch.log(q)))


def _checkpoint_steps(n_steps: int, checkpoint_count: int) -> list[int]:
    fractions = np.linspace(0.0, 1.0, checkpoint_count)
    return sorted({int(round(frac * n_steps)) for frac in fractions})


def _capture_guided_row(
    x: torch.Tensor,
    completed_steps: int,
    eta: float,
    r: int,
    q_target: torch.Tensor,
    model_cfg: ModelConfig,
    state: GuidedFieldState,
) -> tuple[dict[str, float], np.ndarray]:
    s = completed_steps * eta
    t = min(model_cfg.T, s / r)
    u = s / (r * model_cfg.T)
    hist_prob = empirical_hist_prob_2d(x, state.x_edges, state.y_edges)
    q_path = guided_target_mass_grid(t, model_cfg, state)
    guidance_values = interpolate_guidance_potential(x, state)
    row = {
        'r': float(r),
        'completed_steps': float(completed_steps),
        's': float(s),
        'u': float(u),
        't': float(t),
        'kl_to_path': float(discrete_kl(hist_prob, q_path).item()),
        'kl_to_target': float(discrete_kl(hist_prob, q_target).item()),
        'mean_x1': float(x[:, 0].mean().item()),
        'mean_x2': float(x[:, 1].mean().item()),
        'var_x1': float(x[:, 0].var(unbiased=False).item()),
        'var_x2': float(x[:, 1].var(unbiased=False).item()),
        'mean_guidance': float(guidance_values.mean().item()),
    }
    return row, hist_prob.detach().cpu().numpy()


def run_guided_sald_em_2d(
    r: int,
    x0: torch.Tensor,
    model_cfg: ModelConfig,
    profile: GuidedRunProfile,
    state: GuidedFieldState,
    *,
    seed: int,
    checkpoint_count: int | None = None,
) -> dict[str, Any]:
    if r <= 0:
        raise ValueError('r must be positive')

    q_target = guided_target_mass_grid(model_cfg.T, model_cfg, state)
    checkpoint_steps = _checkpoint_steps(
        n_steps=max(1, math.ceil(r * model_cfg.T / profile.eta_s)),
        checkpoint_count=checkpoint_count or profile.checkpoint_count,
    )
    n_steps = checkpoint_steps[-1]
    eta = r * model_cfg.T / n_steps
    noise_scale = math.sqrt(2.0 * eta)

    x = x0.clone()
    generator = _make_generator(x.device, seed)
    trajectory_rows: list[dict[str, float]] = []
    final_hist: np.ndarray | None = None

    if x.device.type == 'cuda':
        torch.cuda.synchronize(device=x.device)
    start_time = time.perf_counter()

    checkpoint_index = 0
    for completed_steps in range(n_steps + 1):
        if checkpoint_index < len(checkpoint_steps) and completed_steps == checkpoint_steps[checkpoint_index]:
            row, hist_prob = _capture_guided_row(x, completed_steps, eta, r, q_target, model_cfg, state)
            row['eta'] = float(eta)
            row['n_steps'] = float(n_steps)
            trajectory_rows.append(row)
            if completed_steps == n_steps:
                final_hist = hist_prob
            checkpoint_index += 1

        if completed_steps == n_steps:
            break

        t = (completed_steps * eta) / r
        drift = guided_score(x, t, model_cfg, state)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        x = x + eta * drift + noise_scale * noise

    if x.device.type == 'cuda':
        torch.cuda.synchronize(device=x.device)
    wall_clock_sec = time.perf_counter() - start_time

    if final_hist is None:
        final_row, final_hist = _capture_guided_row(x, n_steps, eta, r, q_target, model_cfg, state)
        final_row['eta'] = float(eta)
        final_row['n_steps'] = float(n_steps)
        trajectory_rows.append(final_row)

    summary = dict(trajectory_rows[-1])
    summary['wall_clock_sec'] = float(wall_clock_sec)
    summary['r'] = int(r)
    summary['n_steps'] = int(summary['n_steps'])

    subset_count = min(profile.scatter_points, x.shape[0])
    sample_subset = x[:subset_count].detach().cpu().numpy()

    return {
        'trajectory_rows': trajectory_rows,
        'summary_row': summary,
        'sample_subset': sample_subset,
        'hist_final': final_hist,
    }


def run_guided_experiment_suite(
    r_values: Sequence[int],
    model_cfg: ModelConfig,
    profile: GuidedRunProfile,
    state: GuidedFieldState,
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
            print(f'running guided SALD for r={r} with eta_s={profile.eta_s:.4f}')
        result = run_guided_sald_em_2d(
            r,
            x0,
            model_cfg,
            profile,
            state,
            seed=seed + 20_000 + 131 * run_index,
            checkpoint_count=checkpoint_count,
        )
        trajectory_rows.extend(result['trajectory_rows'])
        summary_rows.append(result['summary_row'])
        sample_subsets[int(r)] = result['sample_subset']
        histograms[int(r)] = result['hist_final']

    trajectory_df = pd.DataFrame(trajectory_rows).sort_values(['r', 'u']).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values('r').reset_index(drop=True)

    target_prob_final = guided_target_mass_grid(model_cfg.T, model_cfg, state)
    target_density_final = (target_prob_final / (state.dx * state.dy)).reshape(state.config.bins_y, state.config.bins_x)
    base_density_final = target_density_2d(state.flat_points[:, 0], state.flat_points[:, 1], model_cfg.T, model_cfg)
    base_density_final = base_density_final.reshape(state.config.bins_y, state.config.bins_x)

    return {
        'model_config': model_cfg,
        'profile': profile,
        'field_state': state,
        'trajectory_df': trajectory_df,
        'summary_df': summary_df,
        'sample_subsets': sample_subsets,
        'histograms': histograms,
        'target_prob_final': target_prob_final.detach().cpu().numpy(),
        'target_density_final': target_density_final.detach().cpu().numpy(),
        'base_density_final': base_density_final.detach().cpu().numpy(),
        'x_edges': state.x_edges.detach().cpu().numpy(),
        'y_edges': state.y_edges.detach().cpu().numpy(),
        'mesh_x': state.mesh_x.detach().cpu().numpy(),
        'mesh_y': state.mesh_y.detach().cpu().numpy(),
        'reference_plot_points': state.plot_reference_points,
    }


def run_guided_lambda_sweep(
    lambda_values: Sequence[float],
    r_values: Sequence[int],
    model_cfg: ModelConfig,
    profile: GuidedRunProfile,
    field_cfg_template: GuidedFieldConfig,
    *,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[float, dict[str, Any]]]:
    rows: list[pd.DataFrame] = []
    suites: dict[float, dict[str, Any]] = {}

    for idx, lambda_value in enumerate(lambda_values):
        field_cfg = replace(field_cfg_template, lambda_guidance=float(lambda_value))
        state = prepare_guidance_field(field_cfg, seed=seed + 500 * idx, verbose=verbose)
        suite = run_guided_experiment_suite(
            r_values,
            model_cfg,
            profile,
            state,
            seed=seed + 1_000 * idx,
            verbose=verbose,
        )
        summary_df = suite['summary_df'].copy()
        summary_df['lambda_guidance'] = float(lambda_value)
        rows.append(summary_df)
        suites[float(lambda_value)] = suite

    return pd.concat(rows, ignore_index=True), suites


def plot_guided_target_setup(ax: plt.Axes, suite: dict[str, Any], *, scatter_size: float = 5.0) -> None:
    mesh_x = suite['mesh_x']
    mesh_y = suite['mesh_y']
    target_density = suite['target_density_final']
    base_density = suite['base_density_final']
    moon_points = suite['reference_plot_points']

    ax.contourf(mesh_x, mesh_y, target_density, levels=14, cmap='YlOrBr', alpha=0.35)
    ax.contour(mesh_x, mesh_y, target_density, levels=10, colors=['#1f5c7a'], linewidths=1.2, alpha=0.9)
    ax.contour(mesh_x, mesh_y, base_density, levels=8, colors=['#5f5c53'], linewidths=1.0, alpha=0.65)
    ax.scatter(moon_points[:, 0], moon_points[:, 1], s=scatter_size, alpha=0.14, color='#3a3836', edgecolors='none')
    ax.set_title(r'Guided Target $q_T^X$ and Two-Moons Reference')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_aspect('equal', adjustable='box')


def plot_guided_kl_trajectory(ax: plt.Axes, trajectory_df: pd.DataFrame, r_values: Sequence[int]) -> None:
    palette = _palette_for_r(r_values)
    for r in sorted(int(v) for v in r_values):
        sub = trajectory_df[trajectory_df['r'] == float(r)]
        ax.plot(sub['u'], sub['kl_to_target'], color=palette[r], lw=1.8, alpha=0.9, label=fr'$r={r}$')
    ax.set_yscale('log')
    ax.set_title(r'Terminal-Target KL $KL(\rho_s \| q_T^X)$')
    ax.set_xlabel(r'normalized progress $u = s / (rT)$')
    ax.set_ylabel(r'discrete KL to $q_T^X$')
    ax.legend(ncol=2, fontsize=9)


def plot_guided_final_kl(ax: plt.Axes, summary_df: pd.DataFrame) -> None:
    r_values = summary_df['r'].astype(int).tolist()
    palette = _palette_for_r(r_values)
    ax.plot(
        summary_df['r'],
        summary_df['kl_to_target'],
        marker='o',
        ms=6,
        lw=2.2,
        color='#1f5c7a',
        label=r'final $KL(\rho_{rT} \| q_T^X)$',
    )
    for _, row in summary_df.iterrows():
        r = int(row['r'])
        ax.scatter(row['r'], row['kl_to_target'], s=42, color=palette[r], zorder=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(r'Final KL to the Guided Target')
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'final discrete KL')
    ax.legend(fontsize=10)


def plot_guided_mean_objective(
    ax: plt.Axes,
    trajectory_df: pd.DataFrame,
    *,
    selected_r: Sequence[int],
) -> None:
    palette = _palette_for_r(selected_r)
    for r in selected_r:
        sub = trajectory_df[trajectory_df['r'] == float(r)]
        ax.plot(sub['u'], sub['mean_guidance'], color=palette[int(r)], lw=2.0, alpha=0.95, label=fr'$r={int(r)}$')
    ax.set_title(r'Mean Guidance Objective $\mathbb{E}[f(X_s)]$')
    ax.set_xlabel(r'normalized progress $u = s / (rT)$')
    ax.set_ylabel(r'mean guidance energy')
    ax.legend(ncol=2, fontsize=9)


def make_guided_overview_figure(
    suite: dict[str, Any],
    *,
    selected_r: Sequence[int] | None = None,
) -> plt.Figure:
    if selected_r is None:
        selected_r = suite['summary_df']['r'].astype(int).tolist()

    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.5), constrained_layout=True)
    plot_guided_target_setup(axes[0, 0], suite)
    plot_guided_kl_trajectory(axes[0, 1], suite['trajectory_df'], suite['summary_df']['r'].astype(int).tolist())
    plot_guided_final_kl(axes[1, 0], suite['summary_df'])
    plot_guided_mean_objective(axes[1, 1], suite['trajectory_df'], selected_r=selected_r)
    return fig


def make_guided_sample_grid(
    suite: dict[str, Any],
    *,
    selected_r: Sequence[int] | None = None,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
) -> plt.Figure:
    if selected_r is None:
        selected_r = suite['summary_df']['r'].astype(int).tolist()

    summary_df = suite['summary_df'].set_index('r')
    r_values = [int(r) for r in selected_r]
    target_density = suite['target_density_final']
    mesh_x = suite['mesh_x']
    mesh_y = suite['mesh_y']
    reference_points = suite['reference_plot_points']

    if x_lim is None:
        x_lim = (float(suite['x_edges'][0]), float(suite['x_edges'][-1]))
    if y_lim is None:
        y_lim = (float(suite['y_edges'][0]), float(suite['y_edges'][-1]))

    n_panels = len(r_values)
    ncols = min(4, max(1, n_panels))
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.7 * ncols, 4.5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.reshape(-1)

    for ax, r in zip(axes, r_values):
        sample_points = suite['sample_subsets'][int(r)]
        ax.contourf(mesh_x, mesh_y, target_density, levels=12, cmap='YlOrBr', alpha=0.28)
        ax.contour(mesh_x, mesh_y, target_density, levels=8, colors=['#444444'], linewidths=1.0, alpha=0.75)
        ax.scatter(reference_points[:, 0], reference_points[:, 1], s=4, alpha=0.08, color='#3a3836', edgecolors='none')
        ax.scatter(
            sample_points[:, 0],
            sample_points[:, 1],
            s=7,
            alpha=0.30,
            color=FIXED_SAMPLE_POINT_COLOR,
            edgecolors='none',
        )
        final_kl = float(summary_df.loc[int(r), 'kl_to_target'])
        ax.set_title(fr'$r={int(r)}$, final KL $\approx {final_kl:.4f}$')
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_aspect('equal', adjustable='box')

    for ax in axes[len(r_values):]:
        ax.axis('off')

    return fig


def make_guided_lambda_heatmap(ax: plt.Axes, sweep_df: pd.DataFrame) -> None:
    pivot = sweep_df.pivot(index='lambda_guidance', columns='r', values='kl_to_target').sort_index(ascending=False)
    sns.heatmap(pivot, ax=ax, cmap='mako_r', norm=None, cbar_kws={'label': 'final discrete KL'})
    ax.set_title(r'Final KL Heatmap over $(\lambda, r)$')
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$\lambda$')
