"""Training CLI package.

Public entrypoint:
    from train.cli import parse_train_config

parse_train_config(argv) parses argv (and optional YAML config) into a
frozen TrainConfig dataclass.

Internal modules:
    defaults.py — DEFAULTS hyperparameter table
    schema.py   — TrainConfig dataclass
    parser.py   — argparse parser builder
    loader.py   — YAML merge + parse_train_config orchestration
"""

from __future__ import annotations

from .loader import parse_train_config
from .schema import TrainConfig

__all__ = ["parse_train_config", "TrainConfig"]
