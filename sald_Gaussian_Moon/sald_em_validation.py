from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


LOG_2PI = math.log(2.0 * math.pi)
SQRT_2 = math.sqrt(2.0)
FIXED_SAMPLE_POINT_COLOR = "#5f7890"


@dataclass(frozen=True)
class ModelConfig:
    T: float = 1.0
    beta_min: float = 0.1
    beta_max: float = 14.0
    x_mean: float = 1.0


@dataclass(frozen=True)
class RunProfile:
    name: str
    n_particles: int
    eta_s: float
    hist_bins: int
    x_min: float = -6.0
    x_max: float = 6.0
    checkpoint_count: int = 33
    scatter_points: int = 4000


PROFILES: dict[str, RunProfile] = {
    "safe": RunProfile(
        name="safe",
        n_particles=32_768,
        eta_s=0.05,
        hist_bins=768,
        checkpoint_count=33,
        scatter_points=4_000,
    ),
    "paper": RunProfile(
        name="paper",
        n_particles=131_072,
        eta_s=0.025,
        hist_bins=1024,
        checkpoint_count=41,
        scatter_points=8_000,
    ),
}

DEFAULT_R_VALUES = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
DEFAULT_SCATTER_R_VALUES = [1, 100, 500, 1000]


def configure_torch(seed: int = 0) -> torch.device:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def runtime_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }
    if torch.cuda.is_available():
        report.update(
            {
                "logical_device_count": torch.cuda.device_count(),
                "logical_device_name": torch.cuda.get_device_name(0),
            }
        )
    return report


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#fbfaf7",
            "axes.facecolor": "#fbfaf7",
            "savefig.facecolor": "#fbfaf7",
            "axes.edgecolor": "#3c3836",
            "axes.labelcolor": "#1f1d1a",
            "xtick.color": "#1f1d1a",
            "ytick.color": "#1f1d1a",
            "text.color": "#1f1d1a",
            "grid.color": "#d9d3c7",
            "grid.alpha": 0.45,
            "axes.titleweight": "semibold",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": "#fbfaf7",
            "legend.edgecolor": "#d9d3c7",
            "font.family": "DejaVu Serif",
        }
    )


def _as_tensor(x: Any, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if device is not None and x.device != device:
            x = x.to(device=device)
        if dtype is not None and x.dtype != dtype:
            x = x.to(dtype=dtype)
        return x
    return torch.as_tensor(x, device=device, dtype=dtype)


def alpha_tau_scalar(tau: float, cfg: ModelConfig) -> float:
    integral_beta = cfg.beta_min * tau + 0.5 * (cfg.beta_max - cfg.beta_min) * tau * tau / cfg.T
    return math.exp(-0.5 * integral_beta)


def mixture_radius_scalar(t: float, cfg: ModelConfig) -> float:
    tau = max(0.0, min(cfg.T, cfg.T - t))
    return cfg.x_mean * alpha_tau_scalar(tau, cfg)


def alpha_tau(tau: Any, cfg: ModelConfig) -> torch.Tensor:
    tau_t = _as_tensor(tau, dtype=torch.float32)
    integral_beta = cfg.beta_min * tau_t + 0.5 * (cfg.beta_max - cfg.beta_min) * tau_t.square() / cfg.T
    return torch.exp(-0.5 * integral_beta)


def mixture_radius_at_t(t: Any, cfg: ModelConfig) -> torch.Tensor:
    t_t = _as_tensor(t, dtype=torch.float32)
    tau = torch.clamp(cfg.T - t_t, min=0.0, max=cfg.T)
    return cfg.x_mean * alpha_tau(tau, cfg)


def log_cosh(x: torch.Tensor) -> torch.Tensor:
    return torch.logaddexp(x, -x) - math.log(2.0)


def score_x1(x: torch.Tensor, t: Any, cfg: ModelConfig) -> torch.Tensor:
    m_t = mixture_radius_at_t(t, cfg).to(device=x.device, dtype=x.dtype)
    return -x + m_t * torch.tanh(m_t * x)


def target_logpdf_1d(x: torch.Tensor, t: Any, cfg: ModelConfig) -> torch.Tensor:
    m_t = mixture_radius_at_t(t, cfg).to(device=x.device, dtype=x.dtype)
    return -0.5 * (x.square() + m_t.square()) - 0.5 * LOG_2PI + log_cosh(m_t * x)


def target_density_1d(x: torch.Tensor, t: Any, cfg: ModelConfig) -> torch.Tensor:
    return torch.exp(target_logpdf_1d(x, t, cfg))


def target_density_2d(x1: torch.Tensor, x2: torch.Tensor, t: Any, cfg: ModelConfig) -> torch.Tensor:
    log_phi_y = -0.5 * x2.square() - 0.5 * LOG_2PI
    return target_density_1d(x1, t, cfg) * torch.exp(log_phi_y)


def gaussian_bin_mass(edges: torch.Tensor, mu: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    z = (edges - mu) / (sigma * SQRT_2)
    cdf = 0.5 * (1.0 + torch.erf(z))
    return cdf[1:] - cdf[:-1]


def target_bin_mass(t: float, edges: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
    edges = edges.to(dtype=torch.float32)
    m_t = mixture_radius_at_t(t, cfg).to(device=edges.device, dtype=edges.dtype)
    q = 0.5 * gaussian_bin_mass(edges, m_t) + 0.5 * gaussian_bin_mass(edges, -m_t)
    q = q.clamp_min(1e-12)
    return q / q.sum()


def empirical_hist_prob(x: torch.Tensor, edges: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    bins = edges.numel() - 1
    bucket = torch.bucketize(x, edges) - 1
    valid = (bucket >= 0) & (bucket < bins)
    counts = torch.zeros(bins, device=x.device, dtype=torch.float32)
    valid_bucket = bucket[valid]
    counts.scatter_add_(0, valid_bucket, torch.ones_like(valid_bucket, dtype=torch.float32))
    counts = counts + eps
    return counts / counts.sum()


def discrete_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.sum(p * (torch.log(p) - torch.log(q)))


def _make_edges(profile: RunProfile, device: torch.device) -> torch.Tensor:
    return torch.linspace(profile.x_min, profile.x_max, profile.hist_bins + 1, device=device, dtype=torch.float32)


def _checkpoint_steps(n_steps: int, checkpoint_count: int) -> list[int]:
    fractions = np.linspace(0.0, 1.0, checkpoint_count)
    return sorted({int(round(frac * n_steps)) for frac in fractions})


def _capture_row(
    x: torch.Tensor,
    completed_steps: int,
    eta: float,
    r: int,
    edges: torch.Tensor,
    q_target: torch.Tensor,
    cfg: ModelConfig,
) -> tuple[dict[str, float], np.ndarray]:
    s = completed_steps * eta
    t = min(cfg.T, s / r)
    u = s / (r * cfg.T)
    hist_prob = empirical_hist_prob(x, edges)
    q_path = target_bin_mass(t, edges, cfg)
    m_t = mixture_radius_at_t(t, cfg).item()
    row = {
        "r": float(r),
        "completed_steps": float(completed_steps),
        "s": float(s),
        "u": float(u),
        "t": float(t),
        "kl_to_path": float(discrete_kl(hist_prob, q_path).item()),
        "kl_to_target": float(discrete_kl(hist_prob, q_target).item()),
        "mean": float(x.mean().item()),
        "var": float(x.var(unbiased=False).item()),
        "target_var": float(1.0 + m_t * m_t),
        "radius_m": float(m_t),
    }
    return row, hist_prob.detach().cpu().numpy()


def run_sald_em_1d(
    r: int,
    x0: torch.Tensor,
    cfg: ModelConfig,
    profile: RunProfile,
    *,
    seed: int,
    checkpoint_count: int | None = None,
) -> dict[str, Any]:
    if r <= 0:
        raise ValueError("r must be positive")

    device = x0.device
    edges = _make_edges(profile, device)
    q_target = target_bin_mass(cfg.T, edges, cfg)
    checkpoint_steps = _checkpoint_steps(
        n_steps=max(1, math.ceil(r * cfg.T / profile.eta_s)),
        checkpoint_count=checkpoint_count or profile.checkpoint_count,
    )
    n_steps = checkpoint_steps[-1]
    eta = r * cfg.T / n_steps
    noise_scale = math.sqrt(2.0 * eta)

    x = x0.clone()
    generator = torch.Generator(device=device).manual_seed(seed)
    trajectory_rows: list[dict[str, float]] = []
    final_hist: np.ndarray | None = None

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    start_time = time.perf_counter()

    checkpoint_index = 0
    for completed_steps in range(n_steps + 1):
        if checkpoint_index < len(checkpoint_steps) and completed_steps == checkpoint_steps[checkpoint_index]:
            row, hist_prob = _capture_row(x, completed_steps, eta, r, edges, q_target, cfg)
            row["eta"] = float(eta)
            row["n_steps"] = float(n_steps)
            trajectory_rows.append(row)
            if completed_steps == n_steps:
                final_hist = hist_prob
            checkpoint_index += 1

        if completed_steps == n_steps:
            break

        t = (completed_steps * eta) / r
        drift = score_x1(x, t, cfg)
        noise = torch.randn(x.shape, device=device, dtype=x.dtype, generator=generator)
        x = x + eta * drift + noise_scale * noise

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    wall_clock_sec = time.perf_counter() - start_time

    if final_hist is None:
        final_row, final_hist = _capture_row(x, n_steps, eta, r, edges, q_target, cfg)
        final_row["eta"] = float(eta)
        final_row["n_steps"] = float(n_steps)
        trajectory_rows.append(final_row)

    summary = dict(trajectory_rows[-1])
    summary["wall_clock_sec"] = float(wall_clock_sec)
    summary["r"] = int(r)
    summary["n_steps"] = int(summary["n_steps"])

    subset_count = min(profile.scatter_points, x.numel())
    sample_subset_x = x[:subset_count].detach().cpu().numpy()

    return {
        "trajectory_rows": trajectory_rows,
        "summary_row": summary,
        "sample_subset_x": sample_subset_x,
        "hist_final": final_hist,
        "edges": edges.detach().cpu().numpy(),
    }


def run_experiment_suite(
    r_values: Sequence[int],
    cfg: ModelConfig,
    profile: RunProfile,
    *,
    seed: int = 0,
    checkpoint_count: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0_generator = torch.Generator(device=device).manual_seed(seed)
    x0 = torch.randn(profile.n_particles, device=device, dtype=torch.float32, generator=x0_generator)

    trajectory_rows: list[dict[str, float]] = []
    summary_rows: list[dict[str, Any]] = []
    sample_subsets: dict[int, np.ndarray] = {}
    histograms: dict[int, np.ndarray] = {}
    edges_cpu: np.ndarray | None = None

    for run_index, r in enumerate(r_values):
        if verbose:
            print(f"running r={r} on {device.type} with eta_s={profile.eta_s:.4f}")
        result = run_sald_em_1d(
            r,
            x0,
            cfg,
            profile,
            seed=seed + 10_000 + 97 * run_index,
            checkpoint_count=checkpoint_count,
        )
        trajectory_rows.extend(result["trajectory_rows"])
        summary_rows.append(result["summary_row"])
        sample_subsets[int(r)] = result["sample_subset_x"]
        histograms[int(r)] = result["hist_final"]
        edges_cpu = result["edges"]

    trajectory_df = pd.DataFrame(trajectory_rows).sort_values(["r", "u"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("r").reset_index(drop=True)

    if edges_cpu is None:
        raise RuntimeError("no experiment results were produced")

    edges_cpu_tensor = torch.as_tensor(edges_cpu, dtype=torch.float32)
    target_hist_final = target_bin_mass(cfg.T, edges_cpu_tensor, cfg).cpu().numpy()

    return {
        "model_config": cfg,
        "profile": profile,
        "device": device.type,
        "trajectory_df": trajectory_df,
        "summary_df": summary_df,
        "sample_subsets": sample_subsets,
        "histograms": histograms,
        "hist_edges": edges_cpu,
        "target_hist_final": target_hist_final,
    }


def _palette_for_r(r_values: Sequence[int]) -> dict[int, tuple[float, float, float]]:
    ordered = sorted(int(r) for r in r_values)
    palette = sns.color_palette("mako", len(ordered))
    return {r: palette[i] for i, r in enumerate(ordered)}


def plot_target_family(
    ax: plt.Axes,
    cfg: ModelConfig,
    *,
    t_values: Sequence[float] | None = None,
    x_lim: tuple[float, float] | None = None,
) -> None:
    if t_values is None:
        t_values = [0.0, 0.25 * cfg.T, 0.5 * cfg.T, 0.75 * cfg.T, cfg.T]
    if x_lim is None:
        half_width = max(4.5, cfg.x_mean + 3.5)
        x_lim = (-half_width, half_width)

    x_grid = np.linspace(x_lim[0], x_lim[1], 1200)
    colors = sns.color_palette("flare", len(t_values))
    base_norm = 1.0 / math.sqrt(2.0 * math.pi)

    for color, t in zip(colors, t_values):
        m_t = mixture_radius_scalar(float(t), cfg)
        density = 0.5 * base_norm * np.exp(-0.5 * (x_grid - m_t) ** 2)
        density += 0.5 * base_norm * np.exp(-0.5 * (x_grid + m_t) ** 2)
        ax.plot(x_grid, density, color=color, lw=2.4, label=fr"$t={t:.2f}$")

    ax.set_title(r"Reverse Marginals $p_t$ Along the OU Family")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"density")
    ax.legend(ncol=1, fontsize=10)


def plot_kl_trajectory(ax: plt.Axes, trajectory_df: pd.DataFrame, r_values: Sequence[int]) -> None:
    palette = _palette_for_r(r_values)
    for r in sorted(int(v) for v in r_values):
        sub = trajectory_df[trajectory_df["r"] == float(r)]
        ax.plot(
            sub["u"],
            sub["kl_to_target"],
            color=palette[r],
            lw=2.1 if r in DEFAULT_SCATTER_R_VALUES else 1.3,
            alpha=0.95 if r in DEFAULT_SCATTER_R_VALUES else 0.7,
            label=fr"$r={r}$",
        )

    ax.set_yscale("log")
    ax.set_title(r"Terminal-Target KL $KL(\rho_s \| \pi_T)$")
    ax.set_xlabel(r"normalized progress $u = s / (rT)$")
    ax.set_ylabel(r"discrete KL to $\pi_T$")
    ax.legend(ncol=2, fontsize=9)


def plot_final_kl(ax: plt.Axes, summary_df: pd.DataFrame) -> None:
    r_values = summary_df["r"].astype(int).tolist()
    palette = _palette_for_r(r_values)

    ax.plot(
        summary_df["r"],
        summary_df["kl_to_target"],
        marker="o",
        ms=6,
        lw=2.2,
        color="#1f5c7a",
        label=r"final $KL(\rho_{rT} \| \pi_T)$",
    )
    anchor = float(summary_df.iloc[-1]["kl_to_target"]) * float(summary_df.iloc[-1]["r"])
    guide = anchor / summary_df["r"]
    ax.plot(summary_df["r"], guide, ls="--", lw=1.7, color="#8a6d3b", label=r"$c/r$ guide")

    for _, row in summary_df.iterrows():
        r = int(row["r"])
        ax.scatter(row["r"], row["kl_to_target"], s=45, color=palette[r], zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"Final KL Shrinkage Versus $r$")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"final discrete KL")
    ax.legend(fontsize=10)


def plot_final_density(
    ax: plt.Axes,
    suite: dict[str, Any],
    *,
    selected_r: Sequence[int] | None = None,
    x_lim: tuple[float, float] | None = None,
) -> None:
    if selected_r is None:
        selected_r = DEFAULT_SCATTER_R_VALUES

    edges = suite["hist_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = edges[1] - edges[0]
    target_density = suite["target_hist_final"] / dx
    r_values = suite["summary_df"]["r"].astype(int).tolist()
    palette = _palette_for_r(r_values)

    ax.plot(centers, target_density, color="#111111", lw=2.7, label=r"target $\pi_T$")

    for r in selected_r:
        density = suite["histograms"][int(r)] / dx
        ax.plot(centers, density, lw=2.0, color=palette[int(r)], alpha=0.95, label=fr"$r={int(r)}$")

    if x_lim is None:
        ax.set_xlim(float(edges[0]), float(edges[-1]))
    else:
        ax.set_xlim(*x_lim)
    ax.set_title(r"Final $x_1$ Marginal Versus the Target")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"density")
    ax.legend(ncol=2, fontsize=10)


def plot_variance_trajectory(
    ax: plt.Axes,
    trajectory_df: pd.DataFrame,
    *,
    selected_r: Sequence[int] | None = None,
) -> None:
    if selected_r is None:
        selected_r = DEFAULT_SCATTER_R_VALUES

    r_values = trajectory_df["r"].astype(int).unique().tolist()
    palette = _palette_for_r(r_values)

    for r in selected_r:
        sub = trajectory_df[trajectory_df["r"] == float(r)]
        ax.plot(sub["u"], sub["var"], lw=2.1, color=palette[int(r)], label=fr"$r={int(r)}$")

    template = trajectory_df[trajectory_df["r"] == float(int(selected_r[0]))]
    ax.plot(
        template["u"],
        template["target_var"],
        ls="--",
        lw=2.0,
        color="#5f5c53",
        label=r"target variance $1 + m(t)^2$",
    )
    ax.set_title(r"Variance Growth on the Informative Axis")
    ax.set_xlabel(r"normalized progress $u = s / (rT)$")
    ax.set_ylabel(r"$\mathrm{Var}(X^{(1)}_s)$")
    ax.legend(fontsize=10)


def make_overview_figure(
    suite: dict[str, Any],
    *,
    selected_r: Sequence[int] | None = None,
    target_family_xlim: tuple[float, float] | None = None,
    final_density_xlim: tuple[float, float] | None = None,
) -> plt.Figure:
    if selected_r is None:
        selected_r = DEFAULT_SCATTER_R_VALUES

    cfg = suite["model_config"]
    summary_df = suite["summary_df"]
    trajectory_df = suite["trajectory_df"]

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.0), constrained_layout=True)
    plot_target_family(axes[0, 0], cfg, x_lim=target_family_xlim)
    plot_kl_trajectory(axes[0, 1], trajectory_df, summary_df["r"].astype(int).tolist())
    plot_final_kl(axes[1, 0], summary_df)
    plot_final_density(axes[1, 1], suite, selected_r=selected_r, x_lim=final_density_xlim)
    return fig


def _target_density_2d_numpy(x: np.ndarray, y: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    m_t = cfg.x_mean
    base = 1.0 / (2.0 * math.pi)
    density = 0.5 * base * np.exp(-0.5 * ((x - m_t) ** 2 + y**2))
    density += 0.5 * base * np.exp(-0.5 * ((x + m_t) ** 2 + y**2))
    return density


def make_sample_grid(
    suite: dict[str, Any],
    *,
    selected_r: Sequence[int] | None = None,
    seed: int = 0,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
) -> plt.Figure:
    if selected_r is None:
        selected_r = DEFAULT_SCATTER_R_VALUES

    cfg = suite["model_config"]
    summary_df = suite["summary_df"].set_index("r")
    r_values = [int(r) for r in selected_r]

    if x_lim is None:
        x_half_width = max(4.2, cfg.x_mean + 2.75)
        x_lim = (-x_half_width, x_half_width)
    if y_lim is None:
        y_lim = (-3.25, 3.25)

    grid_x = np.linspace(x_lim[0], x_lim[1], 220)
    grid_y = np.linspace(y_lim[0], y_lim[1], 200)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    contour_density = _target_density_2d_numpy(mesh_x, mesh_y, cfg)

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
        subset_x = suite["sample_subsets"][int(r)]
        rng = np.random.default_rng(seed + int(r))
        subset_y = rng.standard_normal(subset_x.shape[0])

        ax.contourf(
            mesh_x,
            mesh_y,
            contour_density,
            levels=10,
            cmap="YlOrBr",
            alpha=0.28,
        )
        ax.contour(
            mesh_x,
            mesh_y,
            contour_density,
            levels=7,
            linewidths=1.1,
            colors=["#444444"],
            alpha=0.65,
        )
        ax.scatter(
            subset_x,
            subset_y,
            s=7,
            alpha=0.30,
            color=FIXED_SAMPLE_POINT_COLOR,
            edgecolors="none",
        )
        final_kl = float(summary_df.loc[int(r), "kl_to_target"])
        ax.set_title(fr"$r={int(r)}$, final KL $\approx {final_kl:.4f}$")
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_aspect("equal", adjustable="box")

    for ax in axes[len(r_values) :]:
        ax.axis("off")

    return fig
