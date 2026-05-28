"""Run-directory layout helpers.

Encapsulates the on-disk layout for a single training run:

    <base_dir>/<run_id>/
        ckpt/
        plots/
        artifacts/

`make_run_paths` returns a frozen `RunPaths`; `ensure_run_dirs` materializes
the subdirectories on disk. Logger configuration lives in `train/logger.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RunPaths:
    run_dir: str
    ckpt_dir: str
    plots_dir: str
    artifacts_dir: str


def make_run_paths(*, run_id: str, base_dir: str = "./results/runs") -> RunPaths:
    run_dir = os.path.join(base_dir, run_id)
    return RunPaths(
        run_dir=run_dir,
        ckpt_dir=os.path.join(run_dir, "ckpt"),
        plots_dir=os.path.join(run_dir, "plots"),
        artifacts_dir=os.path.join(run_dir, "artifacts"),
    )


def ensure_run_dirs(paths: RunPaths) -> None:
    os.makedirs(paths.ckpt_dir, exist_ok=True)
    os.makedirs(paths.plots_dir, exist_ok=True)
    os.makedirs(paths.artifacts_dir, exist_ok=True)
