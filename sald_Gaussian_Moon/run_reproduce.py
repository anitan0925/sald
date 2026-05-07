#!/usr/bin/env python3
"""Run the synthetic VP experiments from the supplementary notebook.

This runner executes the code cells of ``sald_validation_ou_mixture.ipynb``
with the current Python interpreter.  It avoids requiring a pre-registered
Jupyter kernel, which makes the supplementary material easier to run on a
reviewer's machine.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "sald_validation_ou_mixture.ipynb"


def _set_thread_env() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")


def _configure_device(args: argparse.Namespace) -> None:
    if args.cpu:
        os.environ["SALD_FORCE_CPU"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return
    if args.gpu is not None:
        os.environ["SALD_PHYSICAL_GPU"] = str(args.gpu)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


def _smoke_patch(source: str) -> str:
    """Reduce the notebook workload for an environment check."""
    replacements = {
        '"r_values": [1, 2, 4, 10, 50, 100]': '"r_values": [1, 2]',
        '"selected_r_for_plots": [1, 2, 4, 10, 50, 100]': '"selected_r_for_plots": [1, 2]',
        "'selected_r_for_overview': [1, 10, 50, 100]": "'selected_r_for_overview': [1, 2]",
        '"n_particles": 10_000': '"n_particles": 1024',
        "'n_moon_reference': 16_384": "'n_moon_reference': 1024",
        "'field_bins_x': 160": "'field_bins_x': 64",
        "'field_bins_y': 128": "'field_bins_y': 56",
        "'field_bins_x': 176": "'field_bins_x': 72",
        "'field_bins_y': 160": "'field_bins_y': 64",
        "'scatter_points': 6_000": "'scatter_points': 512",
        "'plot_reference_points': 12_000": "'plot_reference_points': 512",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def _display_fallback(obj=None, *args, **kwargs) -> None:
    if obj is not None:
        print(obj)


def run_notebook(*, smoke: bool) -> None:
    with NOTEBOOK.open("r", encoding="utf-8") as fh:
        notebook = json.load(fh)

    namespace: dict[str, object] = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK),
        "display": _display_fallback,
    }

    os.chdir(ROOT)
    cells = notebook.get("cells", [])
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if smoke:
            source = _smoke_patch(source)
        if not source.strip():
            continue
        print(f"[run_reproduce] executing code cell {index + 1}/{len(cells)}", flush=True)
        exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=None, help="Physical GPU id to expose through CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution. This is useful for a quick smoke test.")
    parser.add_argument("--smoke", action="store_true", help="Run a small workload to check the environment.")
    parser.add_argument(
        "--skip-notebook",
        action="store_true",
        help="Skip notebook execution and only regenerate paper figures from existing CSV files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _set_thread_env()
    _configure_device(args)
    if args.smoke:
        os.environ.setdefault("SALD_OUTPUT_DIR", "outputs_smoke")
        os.environ.setdefault("SALD_FIG_DIR", "Figs_smoke")

    start = time.perf_counter()
    if not args.skip_notebook:
        run_notebook(smoke=args.smoke)

    print("[run_reproduce] regenerating paper figure panels", flush=True)
    runpy.run_path(str(ROOT / "make_paper_figures.py"), run_name="__main__")

    elapsed = time.perf_counter() - start
    print(f"[run_reproduce] completed in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
