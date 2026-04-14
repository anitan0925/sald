from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOK_PATH = Path(__file__).with_name("sald_validation_ou_mixture.ipynb")


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            r"""
            # SALD on a 2D OU-Gaussian Mixture

            This notebook validates the Euler-Maruyama implementation of SALD for $r = 1, 100, \ldots, 1000$ while keeping CPU pressure deliberately low and using exactly one GPU chosen from physical GPUs $2$ through $7$.

            The empirical trend of interest is that the terminal mismatch

            $$
            KL(\rho_{rT} \| \pi_T)
            $$

            should shrink as $r$ grows. The theorem we want to compare against gives the upper bound

            $$
            KL(\rho_{rT} \| \pi_T)
            \le
            \exp\left(- r \int_0^T \mathcal{LSI}_t \, dt\right)
            \exp\left(\frac{T}{2 r \alpha}\right)
            KL(\rho_0 \| \pi_0)
            +
            \frac{T}{2r}
            \exp\left(\frac{T}{2 r \alpha}\right)
            \mathcal{A}_{\alpha},
            $$

            so the practical questions are:

            - Does the final KL decrease as $r$ increases?
            - In the large-$r$ regime, does the curve behave roughly like a $1/r$ shrinkage law before hitting a discretization / finite-sample floor?

            The experimental setup in this notebook is:

            - The target $\pi_T = p_{\mathrm{data}}$ is a 2D Gaussian mixture with two clearly separated means on the $x_1$ axis and unit covariance per component.
            - The forward process is the OU / VP diffusion with a linear noise schedule $\beta(\tau)$.
            - The reverse initial law is a centered Gaussian.
            - The main quantitative metric is a discrete KL on the informative $x_1$ marginal, because all nontrivial multimodality lives on that axis.
            - The 2D sample figure augments the simulated $x_1$ samples with an independent $x_2 \sim \mathcal{N}(0,1)$, which is exact here because the second coordinate stays standard Gaussian.
            """
        ),
        md(
            r"""
            ## Run This Cell First

            This cell pins all CPU thread pools to `1` and binds the notebook to one chosen GPU. By default it uses physical GPU `2`. If you want to switch to GPU `3-7`, set `SALD_PHYSICAL_GPU` before starting the notebook or edit the value below.

            Important: this must be the first executed cell in the kernel. If `torch` has already been imported, restart the kernel first.
            """
        ),
        code(
            """
            import os

            PREFERRED_PHYSICAL_GPU = int(os.environ.get("SALD_PHYSICAL_GPU", "2"))
            ALLOWED_PHYSICAL_GPUS = {2, 3, 4, 5, 6, 7}
            if PREFERRED_PHYSICAL_GPU not in ALLOWED_PHYSICAL_GPUS:
                raise ValueError(
                    f"Refusing to use physical GPU {PREFERRED_PHYSICAL_GPU}. "
                    f"Allowed GPUs are {sorted(ALLOWED_PHYSICAL_GPUS)}."
                )
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(PREFERRED_PHYSICAL_GPU)

            for key in [
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ]:
                os.environ[key] = "1"

            print("CUDA_VISIBLE_DEVICES =", os.environ["CUDA_VISIBLE_DEVICES"])
            print("CPU threads are pinned to 1 for OMP/MKL/OpenBLAS/NumExpr")
            """
        ),
        md(
            r"""
            ## Closed-Form Structure

            In this 2D example, the OU forward marginals preserve a simple form: a bimodal Gaussian mixture along $x_1$ and a standard Gaussian along $x_2$. Let

            $$
            m(t) = \alpha(T-t), \qquad
            \alpha(\tau) = \exp\left(
                -\frac12 \int_0^\tau \beta(u) \, du
            \right),
            $$

            and let the terminal data distribution have means at $x_1 = \pm \mu$. Then the reverse-indexed marginals can be written as

            $$
            \pi_t(x_1, x_2)
            =
            \left[
                \frac12 \varphi(x_1 - \mu m(t))
                +
                \frac12 \varphi(x_1 + \mu m(t))
            \right]
            \varphi(x_2),
            $$

            where $\varphi$ is the one-dimensional standard Gaussian density. The only nontrivial score term is therefore on $x_1$, and it has the closed form

            $$
            \partial_{x_1} \log \pi_t(x_1)
            =
            -x_1 + \mu m(t) \tanh(\mu m(t) x_1).
            $$

            The notebook implements the Euler-Maruyama update

            $$
            X_{k+1}
            =
            X_k + \eta \, \nabla \log \pi_{t(k\eta)}(X_k)
            + \sqrt{2\eta}\,\xi_k,
            \qquad
            t(s)=s/r.
            $$

            To keep the reverse starting law close to a centered Gaussian, the default schedule uses $T=1$, $\beta_{\min}=0.1$, and $\beta_{\max}=14$, which gives

            $$
            \alpha(T) \approx 0.029,
            $$

            so $\pi_0$ is already very close to a centered Gaussian while the terminal target $\pi_T$ still has two visually distinct modes.
            """
        ),
        code(
            """
            from pathlib import Path
            from pprint import pprint

            import pandas as pd

            from sald_em_validation import (
                ModelConfig,
                RunProfile,
                alpha_tau_scalar,
                configure_torch,
                make_overview_figure,
                make_sample_grid,
                plot_target_family,
                run_experiment_suite,
                runtime_report,
                set_plot_style,
            )

            device = configure_torch(seed=0)
            set_plot_style()
            pprint(runtime_report())

            OUTPUT_DIR = Path("outputs")
            OUTPUT_DIR.mkdir(exist_ok=True)
            """
        ),
        code(
            """
            # Centralized experiment configuration:
            # adjust r, mean separation, particle count, plot ranges, and other knobs here.
            EXPERIMENT = {
                "run_name": "safe_mu225",
                "seed": 123,
                "r_values": [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                "selected_r_for_plots": [1, 100, 500, 1000],
                "T": 1.0,
                "beta_min": 0.1,
                "beta_max": 14.0,
                "target_x_mean_abs": 2.25,
                "n_particles": 32_768,
                "eta_s": 0.05,
                "hist_bins": 768,
                "checkpoint_count": 33,
                "scatter_points": 4_000,
                "target_family_xlim": (-6.0, 6.0),
                "final_density_xlim": (-6.0, 6.0),
                "sample_xlim": (-5.8, 5.8),
                "sample_ylim": (-3.25, 3.25),
            }

            model_cfg = ModelConfig(
                T=EXPERIMENT["T"],
                beta_min=EXPERIMENT["beta_min"],
                beta_max=EXPERIMENT["beta_max"],
                x_mean=EXPERIMENT["target_x_mean_abs"],
            )

            profile = RunProfile(
                name=EXPERIMENT["run_name"],
                n_particles=EXPERIMENT["n_particles"],
                eta_s=EXPERIMENT["eta_s"],
                hist_bins=EXPERIMENT["hist_bins"],
                checkpoint_count=EXPERIMENT["checkpoint_count"],
                scatter_points=EXPERIMENT["scatter_points"],
                x_min=EXPERIMENT["final_density_xlim"][0],
                x_max=EXPERIMENT["final_density_xlim"][1],
            )

            r_values = EXPERIMENT["r_values"]
            scatter_r = EXPERIMENT["selected_r_for_plots"]
            RUN_NAME = EXPERIMENT["run_name"]

            alpha_T = alpha_tau_scalar(model_cfg.T, model_cfg)

            pd.DataFrame(
                [{"parameter": k, "value": v} for k, v in EXPERIMENT.items()]
                + [{"parameter": "alpha(T)", "value": alpha_T}]
            )
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
            plot_target_family(ax, model_cfg, x_lim=EXPERIMENT["target_family_xlim"])
            fig.savefig(OUTPUT_DIR / "target_family.png", dpi=180, bbox_inches="tight")
            plt.show()
            """
        ),
        md(
            r"""
            ## Experimental Design

            The comparison is intentionally fair and conservative:

            - Every $r$ uses the same initial sample batch $X_0 \sim \mathcal{N}(0,1)$.
            - Every $r$ uses the same SALD physical stepsize $\eta_s$, so this is a comparison inside one discretized continuous-time family rather than a deliberately coarsened integrator for large $r$.
            - There is no dataloader, no worker pool, and no multiprocessing; the full run uses a single Python process.
            - The main metric is the discrete KL on the $x_1$ marginal, which is exactly where the bimodality lives.
            - All tunable experiment hyperparameters are collected in the single configuration cell above.

            If you want a stronger visual separation of the two modes, the main knob is `target_x_mean_abs`. If you want a lower large-$r$ error floor, the most direct knobs are `n_particles`, `eta_s`, and `hist_bins`.
            """
        ),
        code(
            """
            suite = run_experiment_suite(
                r_values=r_values,
                cfg=model_cfg,
                profile=profile,
                seed=EXPERIMENT["seed"],
                verbose=True,
            )

            summary_df = suite["summary_df"].copy()
            summary_df["r_times_kl"] = summary_df["r"] * summary_df["kl_to_target"]

            summary_path = OUTPUT_DIR / f"sald_summary_{RUN_NAME}.csv"
            traj_path = OUTPUT_DIR / f"sald_trajectory_{RUN_NAME}.csv"
            summary_df.to_csv(summary_path, index=False)
            suite["trajectory_df"].to_csv(traj_path, index=False)

            summary_df[
                ["r", "kl_to_target", "r_times_kl", "var", "n_steps", "wall_clock_sec"]
            ]
            """
        ),
        code(
            """
            fig = make_overview_figure(
                suite,
                selected_r=scatter_r,
                target_family_xlim=EXPERIMENT["target_family_xlim"],
                final_density_xlim=EXPERIMENT["final_density_xlim"],
            )
            fig.savefig(OUTPUT_DIR / f"sald_overview_{RUN_NAME}.png", dpi=220, bbox_inches="tight")
            fig
            """
        ),
        code(
            """
            fig = make_sample_grid(
                suite,
                selected_r=scatter_r,
                seed=EXPERIMENT["seed"],
                x_lim=EXPERIMENT["sample_xlim"],
                y_lim=EXPERIMENT["sample_ylim"],
            )
            fig.savefig(OUTPUT_DIR / f"sald_samples_{RUN_NAME}.png", dpi=220, bbox_inches="tight")
            fig
            """
        ),
        md(
            r"""
            ## How To Read The Figures

            The usual qualitative pattern is:

            - The pathwise KL is visibly higher for $r=1$, and the terminal sample cloud struggles to develop the two target modes.
            - As $r$ increases, the terminal samples move closer to the two-mode target $\pi_T = p_{\mathrm{data}}$.
            - The final KL curve versus $r$ should decrease overall; in the large-$r$ regime it often reaches a floor driven by Euler discretization and finite particle count.

            If you want to push that floor lower, the most direct changes are:

            - Increase `n_particles`.
            - Decrease `eta_s`.
            - Increase `hist_bins`.
            - Optionally add more $r$ values inside `r_values`.

            All outputs are written into the local `outputs/` directory: the summary CSV, the trajectory CSV, and the main figures.
            """
        ),
    ]

    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    }
    return nb


def main() -> None:
    nb = build_notebook()
    NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
