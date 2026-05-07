# Synthetic VP Experiments

This directory contains the anonymous supplementary code for the synthetic-data
experiments in the paper.  It reproduces the 2-Gaussian/two-moons guided VP
experiment and the 8-Gaussian mode-penalty VP experiment, including the
comparisons between SALD, Velocity-Aware SALD, and the DOIT-style baseline.

## Files

- `sald_validation_ou_mixture.ipynb`: main experiment notebook with derivations,
  configurations, and plotting code.
- `run_reproduce.py`: one-command reproduction entry point.  It executes the
  notebook code cells with the current Python interpreter and then regenerates
  the paper figure panels.
- `make_paper_figures.py`: rebuilds the four standalone synthetic-data panels
  from CSV files written by the notebook.
- `sald_*_validation.py`: implementation modules for the VP process, SALD,
  VA-SALD, DOIT, and the two guided target families.
- `Figs/`: precomputed figures corresponding to the submitted paper.  Running
  `run_reproduce.py` refreshes this directory.
- `requirements.txt`: minimal Python dependencies.

No external repository clone or dataset download is required for these
synthetic experiments.

## Environment

The experiments were tested with Python 3.10, PyTorch 2.6, CUDA 12.4, NumPy
1.26, Pandas 2.3, Matplotlib 3.10, and Seaborn 0.13.  Other recent PyTorch
versions should also work.

Create an environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build if GPU execution is desired.  CPU execution
works but is slower.

## Reproduce the Paper Figures

From this directory, run one of:

```bash
python3 run_reproduce.py --gpu 0
```

or, if CUDA is not available:

```bash
python3 run_reproduce.py --cpu
```

The full run writes CSV files and intermediate figures to `outputs/`, then
refreshes the figure files in `Figs/`.  The four main paper panels are:

- `Figs/main_panel_a_kl_two_moons.png`
- `Figs/main_panel_b_guidance_two_moons.png`
- `Figs/main_panel_c_kl_eight_gaussian.png`
- `Figs/main_panel_d_guidance_eight_gaussian.png`

To check the environment quickly without reproducing the full experiment, run:

```bash
python3 run_reproduce.py --cpu --smoke
```

This smoke test writes to `outputs_smoke/` and `Figs_smoke/`, so it does not
overwrite the precomputed paper figures in `Figs/`.

To regenerate only the paper figure panels from existing CSV files in
`outputs/`, run:

```bash
python3 make_paper_figures.py
```

## Runtime Notes

The default experiment uses `r = 1, 2, 4, 10, 50, 100`, fixed guidance scale
`lambda = 1`, and matched computational budgets for SALD, VA-SALD, and DOIT.
The notebook pins common CPU thread pools to one thread to avoid oversubscription
on shared machines.

GPU selection is controlled by either `--gpu` in `run_reproduce.py` or the
environment variable `SALD_PHYSICAL_GPU`.  Set `SALD_FORCE_CPU=1` or pass
`--cpu` to force CPU execution.
