"""argparse parser builder for the training CLI.

build_parser(suppress_defaults=False) returns a configured ArgumentParser.
With suppress_defaults=True, only flags explicitly provided on the CLI are
captured in the resulting Namespace (used when YAML provides base values and
we want a way to distinguish "user typed this flag" from "user left it alone").
"""

from __future__ import annotations

import argparse

from .defaults import DEFAULTS


def build_parser(*, suppress_defaults: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Training script with configurable hyperparameters"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g. config.yaml). CLI args override YAML.",
    )

    dflt = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument("--PINN", action="store_true", default=dflt, help="Enable Physics-Informed Neural Networks")
    parser.add_argument("--BLOCK_DIAG", action="store_true", default=dflt, help="Use block diagonal mode")
    parser.add_argument("--NORMALIZE", action="store_true", default=dflt, help="Enable normalization")
    parser.add_argument("--PER_UNIT", action="store_true", default=dflt, help="Use per-unit scaling")
    parser.add_argument("--float64", action="store_true", default=dflt, help="Use float64")
    parser.add_argument(
        "--mode",
        type=str,
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["mode"]),
        help="train_valid_test | train | valid | test",
    )
    parser.add_argument(
        "--mag_ang_mse",
        action="store_true",
        default=dflt,
        help="normalised |V| + wrapped-angle loss",
    )

    parser.add_argument("--model", type=str, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["model"]), help="Model name")
    parser.add_argument("--d", type=int, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["d"]), help="model input dim")
    parser.add_argument("--d_hi", type=int, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["d_hi"]), help="model hidden dim")
    parser.add_argument("--num_attn_layers", type=int, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["num_attn_layers"]), help="number of attention layers")

    parser.add_argument("--K", type=int, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["K"]), help="K")
    parser.add_argument("--gamma", type=float, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["gamma"]), help="phys_loss decay over K")
    parser.add_argument("--use_armijo", action="store_true", default=dflt, help="use_armijo")
    parser.add_argument("--vlimit", action="store_true", default=dflt, help="vlimit disabled")
    parser.add_argument("--DthetaMax", type=float, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["DthetaMax"]), help="DthetaMax")
    parser.add_argument("--DvmFrac", type=float, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["DvmFrac"]), help="DvmFrac")
    parser.add_argument("--train_ratio", type=float, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["train_ratio"]), help="train_ratio")
    parser.add_argument("--valid_ratio", type=float, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["valid_ratio"]), help="valid_ratio")
    parser.add_argument(
        "--split_mode",
        type=str,
        choices=("ratio", "equal3"),
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["split_mode"]),
        help="Dataset split strategy: ratio (train_ratio/valid_ratio) or equal3 (1/3 each; remainder distributed)",
    )

    parser.add_argument(
        "--train_subset_frac",
        type=float,
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["train_subset_frac"]),
        help="Optional fraction of the training split to use (few-shot budget).",
    )
    parser.add_argument(
        "--train_subset_min_n",
        type=int,
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["train_subset_min_n"]),
        help="Minimum number of training samples when train_subset_frac is set.",
    )

    parser.add_argument("--ADJ_MODE", type=str, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["ADJ_MODE"]), help="Adjacency mode: real | cplx | other")

    parser.add_argument("--weight_init", type=str, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["weight_init"]), help="Weight initialization method (None, He, sd0.02)")
    parser.add_argument("--bias_init", type=float, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["bias_init"]), help="Bias initialization value")
    parser.add_argument("--weight_decay", type=float, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["weight_decay"]), help="Weight decay rate")

    parser.add_argument("--lr_scheduler", type=str, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["lr_scheduler"]), help="CosineAnnealingLR")
    parser.add_argument("--cosineRestartEpoch", type=int, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["cosineRestartEpoch"]), help="cosineRestartEpoch")

    parser.add_argument("--BATCH", type=int, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["BATCH"]), help="Batch size")
    parser.add_argument("--EPOCHS", type=int, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["EPOCHS"]), help="Number of training epochs")
    parser.add_argument("--LR", type=float, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["LR"]), help="Learning rate")
    parser.add_argument("--VAL_EVERY", type=int, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["VAL_EVERY"]), help="Validation frequency (in epochs)")

    parser.add_argument(
        "--PARQUET",
        type=str,
        nargs="+",
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["PARQUET"]),
        help="Path to Parquet data file(s)",
    )

    # MLflow
    parser.add_argument(
        "--mlflow",
        action="store_true",
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["mlflow"]),
        help="Enable MLflow logging",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["mlflow_tracking_uri"]),
        help="MLflow tracking URI (e.g. sqlite:///mlflow.db, file:./mlruns, http://localhost:5000)",
    )
    parser.add_argument(
        "--mlflow_artifact_location",
        type=str,
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["mlflow_artifact_location"]),
        help="MLflow experiment artifact location (e.g. file:./results/mlruns). Only used when creating a new experiment.",
    )
    parser.add_argument(
        "--mlflow_artifact_path",
        type=str,
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["mlflow_artifact_path"]),
        help="Artifact path prefix in MLflow (e.g. 'run').",
    )
    parser.add_argument(
        "--mlflow_keep_local_run_dir",
        action="store_true",
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["mlflow_keep_local_run_dir"]),
        help="Keep local results/runs/<run_id> directory even when MLflow is enabled (default: stage+delete).",
    )
    parser.add_argument(
        "--no_mlflow_strict",
        action="store_true",
        default=(argparse.SUPPRESS if suppress_defaults else False),
        help="Disable strict MLflow mode (if MLflow import fails, continue without MLflow).",
    )
    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["mlflow_experiment"]),
        help="MLflow experiment name",
    )

    parser.add_argument("--seed_value", type=int, default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["seed_value"]), help="Random seed")

    # Initialization
    parser.add_argument(
        "--init_ckpt_path",
        type=str,
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["init_ckpt_path"]),
        help="Optional checkpoint path to load before training (full fine-tune or PEFT)",
    )

    # PEFT / LoRA (optional; typically configured via YAML scenarios)
    parser.add_argument(
        "--peft",
        action="store_true",
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["peft"]),
        help="Enable parameter-efficient fine-tuning (currently: LoRA)",
    )
    parser.add_argument(
        "--peft_method",
        type=str,
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["peft_method"]),
        help="PEFT method (currently supported: lora)",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["lora_r"]),
        help="LoRA rank r",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["lora_alpha"]),
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["lora_dropout"]),
        help="LoRA dropout",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["lora_target_modules"]),
        help="LoRA target module attribute names (e.g. q k v out)",
    )
    parser.add_argument(
        "--peft_train_base",
        action="store_true",
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["peft_train_base"]),
        help="If set, also train base model weights (not just LoRA adapters)",
    )
    parser.add_argument(
        "--peft_base_ckpt_path",
        type=str,
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["peft_base_ckpt_path"]),
        help="Optional base checkpoint path to load before applying PEFT/LoRA",
    )

    parser.add_argument(
        "--peft_unfreeze_modules",
        type=str,
        nargs="+",
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["peft_unfreeze_modules"]),
        help=(
            "Optional list of (possibly dotted) submodule names to keep trainable when PEFT is enabled and base is frozen. "
            "Example: --peft_unfreeze_modules theta_head v_head m_head"
        ),
    )

    # Head-only fine-tuning (no LoRA)
    parser.add_argument(
        "--head_only_ft",
        action="store_true",
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["head_only_ft"]),
        help="Freeze all params and unfreeze only prediction heads (no LoRA).",
    )
    parser.add_argument(
        "--head_only_modules",
        type=str,
        nargs="+",
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["head_only_modules"]),
        help=(
            "Override head-only modules to unfreeze (possibly dotted names). "
            "Default: theta_head v_head m_head"
        ),
    )

    # Compare vs baseline MLflow run
    parser.add_argument(
        "--compare",
        action="store_true",
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["compare"]),
        help="Enable comparison vs a baseline MLflow run (logs % and x deltas)",
    )
    parser.add_argument(
        "--compare_baseline_run_id",
        type=str,
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["compare_baseline_run_id"]),
        help="Baseline MLflow run_id to compare against",
    )
    parser.add_argument(
        "--compare_metrics",
        type=str,
        nargs="+",
        default=(argparse.SUPPRESS if suppress_defaults else DEFAULTS["compare_metrics"]),
        help="Metric keys to compare (e.g. test/loss test/rmse best/score)",
    )
    return parser
