"""Velocity-aware guided SALD validation for the 2D OU-Gaussian mixture.

For the forward VP diffusion

    dY_tau = -0.5 * beta(tau) * Y_tau dtau + sqrt(beta(tau)) dW_tau,

the reverse-indexed coefficients are beta_rev(t) = beta(T - t) and

    dX_s = 0.5 * beta_rev(t(s)) *
           (X_s/r + (1+1/r) * grad log p_{t(s)}(X_s) - grad f(X_s)) ds
           + sqrt(beta_rev(t(s))) dW_s,

where t(s) = s/r. The moving target is pi_t(x) propto p_t(x) exp(-f(x)).
"""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from sald_em_validation import (
    FIXED_SAMPLE_POINT_COLOR,
    ModelConfig,
    RunProfile,
    _palette_for_r,
    target_density_2d,
)

from sald_guided_validation import (
    GuidedFieldConfig,
    GuidedFieldState,
    empirical_hist_prob_2d,
    interpolate_guidance_grad,
    interpolate_guidance_potential,
    prepare_guidance_field,
    prior_logpdf_2d,
    prior_score_2d,
)


# ---------------------------------------------------------------------------
# Reverse-time VP coefficients and VA-SALD drift
# ---------------------------------------------------------------------------


def beta_tau_scalar(tau: float, cfg: ModelConfig) -> float:
    """Linear VP beta schedule at forward time tau."""
    tau = max(0.0, min(cfg.T, float(tau)))
    return cfg.beta_min + (cfg.beta_max - cfg.beta_min) * tau / cfg.T


def reverse_beta_scalar(t: float, cfg: ModelConfig) -> float:
    """Reverse-indexed VP variance rate sigma_t^2 = beta(T - t)."""
    tau = max(0.0, min(cfg.T, cfg.T - float(t)))
    return beta_tau_scalar(tau, cfg)


def predictor_guided_score(
    points: torch.Tensor,
    t: Any,
    r: float,
    model_cfg: ModelConfig,
    state: GuidedFieldState,
) -> torch.Tensor:
    """VA-SALD drift for the reverse-indexed VP diffusion.

    For t(s) = s/r and beta_rev(t) = beta(T - t), the drift is
    0.5 * beta_rev(t) * [x/r + (1 + 1/r) * grad log p_t(x) - grad f(x)].
    """
    beta_t = reverse_beta_scalar(float(t), model_cfg)
    base_score = prior_score_2d(points, t, model_cfg)
    guide_grad = interpolate_guidance_grad(points, state)
    return 0.5 * beta_t * (points / r + (1.0 + 1.0 / r) * base_score - guide_grad)


def predictor_guided_log_unnormalized(
    points: torch.Tensor,
    t: Any,
    r: float,
    model_cfg: ModelConfig,
    state: GuidedFieldState,
) -> torch.Tensor:
    """Log of the unnormalized VA target pi_t(x) propto p_t(x) exp(-f(x)).

    The r argument is kept for compatibility with existing call sites; VA-SALD
    targets pi_t, not an r-dependent tilted family.
    """
    prior_lp = prior_logpdf_2d(points, t, model_cfg)
    f_val = interpolate_guidance_potential(points, state)
    return prior_lp - f_val


def predictor_guided_target_mass_grid(
    t: float,
    r: float,
    model_cfg: ModelConfig,
    state: GuidedFieldState,
) -> torch.Tensor:
    """Normalized discrete target mass for pi_t(x) propto p_t(x) exp(-f(x))."""
    log_unnorm = prior_logpdf_2d(state.flat_points, t, model_cfg) - state.f_grid.reshape(-1)
    return torch.softmax(log_unnorm, dim=0)


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------

def _checkpoint_steps(n_steps: int, checkpoint_count: int) -> list[int]:
    fractions = np.linspace(0.0, 1.0, checkpoint_count)
    return sorted({int(round(frac * n_steps)) for frac in fractions})


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    gen_device = device.type if device.type == 'cuda' else 'cpu'
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)
    return generator


def _capture_predictor_guided_row(
    x: torch.Tensor,
    completed_steps: int,
    eta: float,
    r: float,
    q_target: torch.Tensor,
    model_cfg: ModelConfig,
    state: GuidedFieldState,
) -> tuple[dict[str, float], np.ndarray]:
    s = completed_steps * eta
    t = min(model_cfg.T, s / r)
    u = s / (r * model_cfg.T)
    hist_prob = empirical_hist_prob_2d(x, state.x_edges, state.y_edges)
    q_path = predictor_guided_target_mass_grid(t, r, model_cfg, state)
    guidance_values = interpolate_guidance_potential(x, state)
    mean_penalty = float(guidance_values.mean().item())

    def _kl(p, q):
        return torch.sum(p * (torch.log(p) - torch.log(q)))

    row = {
        'r': float(r),
        'completed_steps': float(completed_steps),
        's': float(s),
        'u': float(u),
        't': float(t),
        'kl_to_path': float(_kl(hist_prob, q_path).item()),
        'kl_to_target': float(_kl(hist_prob, q_target).item()),
        'mean_x1': float(x[:, 0].mean().item()),
        'mean_x2': float(x[:, 1].mean().item()),
        'var_x1': float(x[:, 0].var(unbiased=False).item()),
        'var_x2': float(x[:, 1].var(unbiased=False).item()),
        'mean_guidance': -mean_penalty,
        'mean_penalty': mean_penalty,
        'mean_beta_reverse': float(reverse_beta_scalar(t, model_cfg)),
        'mean_sigma_reverse': float(math.sqrt(reverse_beta_scalar(t, model_cfg))),
    }
    return row, hist_prob.detach().cpu().numpy()


def run_predictor_guided_sald_em_2d(
    r: float,
    x0: torch.Tensor,
    model_cfg: ModelConfig,
    profile: RunProfile,
    state: GuidedFieldState,
    *,
    seed: int,
    checkpoint_count: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one velocity-aware guided SALD experiment for a given r."""
    if r <= 0:
        raise ValueError('r must be positive')

    q_target = predictor_guided_target_mass_grid(model_cfg.T, r, model_cfg, state)
    checkpoint_steps = _checkpoint_steps(
        n_steps=max(1, math.ceil(r * model_cfg.T / profile.eta_s)),
        checkpoint_count=checkpoint_count or profile.checkpoint_count,
    )
    n_steps = checkpoint_steps[-1]
    eta = r * model_cfg.T / n_steps

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
            row, hist_prob = _capture_predictor_guided_row(
                x, completed_steps, eta, r, q_target, model_cfg, state,
            )
            row['eta'] = float(eta)
            row['n_steps'] = float(n_steps)
            trajectory_rows.append(row)
            if completed_steps == n_steps:
                final_hist = hist_prob
            checkpoint_index += 1

        if completed_steps == n_steps:
            break

        t = (completed_steps * eta) / r
        beta_t = reverse_beta_scalar(min(model_cfg.T, t), model_cfg)

        drift = predictor_guided_score(x, t, r, model_cfg, state)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)

        x = x + eta * drift + math.sqrt(eta * beta_t) * noise

    if x.device.type == 'cuda':
        torch.cuda.synchronize(device=x.device)
    wall_clock_sec = time.perf_counter() - start_time

    if final_hist is None:
        final_row, final_hist = _capture_predictor_guided_row(
            x, n_steps, eta, r, q_target, model_cfg, state,
        )
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


def run_predictor_guided_experiment_suite(
    r_values: Sequence[float],
    model_cfg: ModelConfig,
    profile: RunProfile,
    state: GuidedFieldState,
    *,
    seed: int = 0,
    checkpoint_count: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run velocity-aware guided SALD experiments for multiple r values."""
    device = state.flat_points.device
    generator = _make_generator(device, seed)
    x0 = torch.randn(profile.n_particles, 2, device=device, dtype=torch.float32, generator=generator)

    trajectory_rows: list[dict[str, float]] = []
    summary_rows: list[dict[str, Any]] = []
    sample_subsets: dict[int, np.ndarray] = {}
    histograms: dict[int, np.ndarray] = {}

    for run_index, r in enumerate(r_values):
        if verbose:
            print(f'running velocity-aware SALD for r={r} with eta_s={profile.eta_s:.4f}')
        result = run_predictor_guided_sald_em_2d(
            r,
            x0,
            model_cfg,
            profile,
            state,
            seed=seed + 20_000 + 131 * run_index,
            checkpoint_count=checkpoint_count,
            verbose=verbose,
        )
        trajectory_rows.extend(result['trajectory_rows'])
        summary_rows.append(result['summary_row'])
        sample_subsets[int(r)] = result['sample_subset']
        histograms[int(r)] = result['hist_final']

    trajectory_df = pd.DataFrame(trajectory_rows).sort_values(['r', 'u']).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values('r').reset_index(drop=True)

    # Precompute final target density for the largest r
    r_max = max(r_values)
    target_prob_final = predictor_guided_target_mass_grid(model_cfg.T, r_max, model_cfg, state)
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


# ---------------------------------------------------------------------------
# Lambda sweep
# ---------------------------------------------------------------------------

def run_predictor_guided_lambda_sweep(
    lambda_values: Sequence[float],
    r_values: Sequence[float],
    model_cfg: ModelConfig,
    profile: RunProfile,
    field_cfg_template: GuidedFieldConfig,
    *,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[float, dict[str, Any]]]:
    """Sweep over lambda values for velocity-aware guided SALD."""
    from dataclasses import replace

    rows: list[pd.DataFrame] = []
    suites: dict[float, dict[str, Any]] = {}

    for idx, lambda_value in enumerate(lambda_values):
        field_cfg = replace(field_cfg_template, lambda_guidance=float(lambda_value))
        state = prepare_guidance_field(field_cfg, seed=seed + 700 * idx, verbose=verbose)
        suite = run_predictor_guided_experiment_suite(
            r_values,
            model_cfg,
            profile,
            state,
            seed=seed + 1_200 * idx,
            verbose=verbose,
        )
        summary_df = suite['summary_df'].copy()
        summary_df['lambda_guidance'] = float(lambda_value)
        rows.append(summary_df)
        suites[float(lambda_value)] = suite

    return pd.concat(rows, ignore_index=True), suites


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_predictor_guided_target_setup(ax, suite: dict[str, Any], *, scatter_size: float = 5.0) -> None:
    """Plot the velocity-aware target density setup."""
    import matplotlib.pyplot as plt

    mesh_x = suite['mesh_x']
    mesh_y = suite['mesh_y']
    target_density = suite['target_density_final']
    base_density = suite['base_density_final']
    moon_points = suite['reference_plot_points']

    ax.contourf(mesh_x, mesh_y, target_density, levels=14, cmap='YlOrBr', alpha=0.35)
    ax.contour(mesh_x, mesh_y, target_density, levels=10, colors=['#1f5c7a'], linewidths=1.2, alpha=0.9)
    ax.contour(mesh_x, mesh_y, base_density, levels=8, colors=['#5f5c53'], linewidths=1.0, alpha=0.65)
    ax.scatter(moon_points[:, 0], moon_points[:, 1], s=scatter_size, alpha=0.14, color='#3a3836', edgecolors='none')
    ax.set_title(r'Velocity-Aware Target $\pi_T$ and Two-Moons Reference')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_aspect('equal', adjustable='box')


def plot_predictor_guided_kl_trajectory(ax, trajectory_df: pd.DataFrame, r_values: Sequence[int]) -> None:
    """Plot KL trajectory for velocity-aware SALD."""
    import matplotlib.pyplot as plt

    palette = _palette_for_r(r_values)
    for r in sorted(int(v) for v in r_values):
        sub = trajectory_df[trajectory_df['r'] == float(r)]
        ax.plot(sub['u'], sub['kl_to_target'], color=palette[r], lw=1.8, alpha=0.9, label=fr'$r={r}$')
    ax.set_yscale('log')
    ax.set_title(r'Terminal-Target KL $KL(\rho_s \| \pi_T)$')
    ax.set_xlabel(r'normalized progress $u = s / (rT)$')
    ax.set_ylabel(r'discrete KL to $\pi_T$')
    ax.legend(ncol=2, fontsize=9)


def plot_predictor_guided_final_kl(ax, summary_df: pd.DataFrame) -> None:
    """Plot final KL vs r for velocity-aware SALD."""
    import matplotlib.pyplot as plt

    r_values = summary_df['r'].astype(int).tolist()
    palette = _palette_for_r(r_values)
    ax.plot(
        summary_df['r'],
        summary_df['kl_to_target'],
        marker='o',
        ms=6,
        lw=2.2,
        color='#1f5c7a',
        label=r'final $KL(\rho_{rT} \| \pi_T)$',
    )
    for _, row in summary_df.iterrows():
        r = int(row['r'])
        ax.scatter(row['r'], row['kl_to_target'], s=42, color=palette[r], zorder=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(r'Final KL to the Velocity-Aware Target')
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'final discrete KL')
    ax.legend(fontsize=10)


def plot_predictor_guided_mean_objective(ax, trajectory_df, *, selected_r) -> None:
    """Plot mean guidance objective over trajectory."""
    import matplotlib.pyplot as plt

    palette = _palette_for_r(selected_r)
    for r in selected_r:
        sub = trajectory_df[trajectory_df['r'] == float(r)]
        ax.plot(sub['u'], sub['mean_guidance'], color=palette[int(r)], lw=2.0, alpha=0.95, label=fr'$r={int(r)}$')
    ax.set_title(r'Mean Guidance Objective $\mathbb{E}[-f(X_s)]$')
    ax.set_xlabel(r'normalized progress $u = s / (rT)$')
    ax.set_ylabel(r'mean guidance objective')
    ax.legend(ncol=2, fontsize=9)


def plot_predictor_guided_alpha_schedule(ax, trajectory_df, *, selected_r) -> None:
    """Plot reverse VP beta(T - t(s)) over trajectory."""
    import matplotlib.pyplot as plt

    palette = _palette_for_r(selected_r)
    for r in selected_r:
        sub = trajectory_df[trajectory_df['r'] == float(r)]
        ax.plot(sub['u'], sub['mean_beta_reverse'], color=palette[int(r)], lw=2.0, alpha=0.95, label=fr'$r={int(r)}$')
    ax.set_title(r'Reverse VP $\beta(T-t(s))$ along trajectory')
    ax.set_xlabel(r'normalized progress $u = s / (rT)$')
    ax.set_ylabel(r'$\beta(T-t(s))$')
    ax.legend(ncol=2, fontsize=9)


def make_predictor_guided_overview_figure(
    suite: dict[str, Any],
    *,
    selected_r: Sequence[int] | None = None,
) -> 'plt.Figure':
    """Create the 2x2 overview figure for velocity-aware SALD."""
    import matplotlib.pyplot as plt

    if selected_r is None:
        selected_r = suite['summary_df']['r'].astype(int).tolist()

    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.5), constrained_layout=True)
    plot_predictor_guided_target_setup(axes[0, 0], suite)
    plot_predictor_guided_kl_trajectory(axes[0, 1], suite['trajectory_df'], suite['summary_df']['r'].astype(int).tolist())
    plot_predictor_guided_final_kl(axes[1, 0], suite['summary_df'])
    plot_predictor_guided_mean_objective(axes[1, 1], suite['trajectory_df'], selected_r=selected_r)
    return fig


def make_predictor_guided_sample_grid(
    suite: dict[str, Any],
    *,
    selected_r: Sequence[int] | None = None,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
) -> 'plt.Figure':
    """Create sample grid for velocity-aware SALD."""
    import matplotlib.pyplot as plt

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


def make_predictor_guided_lambda_heatmap(ax, sweep_df: pd.DataFrame) -> None:
    """Create KL heatmap over (lambda, r) for velocity-aware SALD."""
    import seaborn as sns

    pivot = sweep_df.pivot(index='lambda_guidance', columns='r', values='kl_to_target').sort_index(ascending=False)
    sns.heatmap(pivot, ax=ax, cmap='mako_r', norm=None, cbar_kws={'label': 'final discrete KL'})
    ax.set_title(r'Final KL Heatmap over $(\lambda, r)$ - Velocity-Aware')
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$\lambda$')


velocity_aware_guided_drift = predictor_guided_score
velocity_aware_guided_log_unnormalized = predictor_guided_log_unnormalized
velocity_aware_guided_target_mass_grid = predictor_guided_target_mass_grid
run_velocity_aware_guided_sald_em_2d = run_predictor_guided_sald_em_2d
run_velocity_aware_guided_experiment_suite = run_predictor_guided_experiment_suite
run_velocity_aware_guided_lambda_sweep = run_predictor_guided_lambda_sweep
make_velocity_aware_guided_overview_figure = make_predictor_guided_overview_figure
make_velocity_aware_guided_sample_grid = make_predictor_guided_sample_grid
make_velocity_aware_guided_lambda_heatmap = make_predictor_guided_lambda_heatmap
