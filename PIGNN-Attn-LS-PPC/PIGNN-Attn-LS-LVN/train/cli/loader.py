"""Orchestration: argparse + optional YAML config -> nested TrainConfig.

parse_train_config(argv) is the public entrypoint. Two paths:
  - --config <path> provided: load YAML, merge DEFAULTS < YAML < explicit CLI overrides.
  - Otherwise: legacy CLI-only, all defaults from DEFAULTS.

Both paths converge on a single "merged" dict (flat, using legacy DEFAULTS keys),
then call _build_train_config to produce the nested TrainConfig.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, List, Mapping

from ..config_loader import as_path_list, env_expand_paths, get, load_yaml_config
from .defaults import DEFAULTS
from .parser import build_parser
from .schema import (
    CompareCfg,
    DataCfg,
    MlflowCfg,
    ModelCfg,
    OptimCfg,
    PeftCfg,
    RunCfg,
    TrainConfig,
)


def _short_dataset_name(parquet_paths: List[str]) -> str:
    parquet_filenames = [os.path.splitext(os.path.basename(p))[0] for p in parquet_paths]
    shortened_names = ["_".join(name.split("_")[:3]) for name in parquet_filenames]
    return "_and_".join(shortened_names)


def _str_or_none(v: Any) -> str | None:
    return str(v) if v not in (None, "", "null") else None


def _float_or_none(v: Any) -> float | None:
    return float(v) if v not in (None, "", "null") else None


def _compute_runname(merged: Mapping[str, Any]) -> str:
    dataset_name = _short_dataset_name(list(merged["PARQUET"]))
    split_tag = (
        "eq3"
        if str(merged.get("split_mode", "ratio")) == "equal3"
        else f"tr{float(merged['train_ratio']):g}"
    )
    model_tag = str(merged["model"]).replace("GNSMsg_", "")
    return f"{dataset_name}_{model_tag}_k{merged['K']}_d{merged['d']}_h{merged['d_hi']}_ep{merged['EPOCHS']}_{split_tag}"


def _args_to_merged(args: argparse.Namespace) -> dict[str, Any]:
    """Convert a fully-defaulted argparse Namespace into the legacy-flat 'merged' dict.

    The Namespace already contains DEFAULTS-resolved values (suppress_defaults=False
    path), so we just drop --config and translate --no_mlflow_strict to its
    semantic equivalent.
    """
    merged: dict[str, Any] = {k: v for k, v in vars(args).items() if k != "config"}
    no_strict = bool(merged.pop("no_mlflow_strict", False))
    if no_strict:
        merged["mlflow_strict"] = False
    return merged


def _build_train_config(merged: Mapping[str, Any], *, runname: str) -> TrainConfig:
    """Single source of truth for assembling a nested TrainConfig from the flat merged dict."""
    return TrainConfig(
        run=RunCfg(
            runname=runname,
            seed=int(merged["seed_value"]),
            mode=str(merged["mode"]),
            init_ckpt_path=_str_or_none(merged.get("init_ckpt_path")),
        ),
        data=DataCfg(
            parquet_paths=list(merged["PARQUET"]),
            per_unit=bool(merged.get("PER_UNIT", DEFAULTS["PER_UNIT"])),
            normalize=bool(merged.get("NORMALIZE", DEFAULTS["NORMALIZE"])),
            split_mode=str(merged.get("split_mode", DEFAULTS["split_mode"])),
            train_ratio=float(merged["train_ratio"]),
            valid_ratio=float(merged["valid_ratio"]),
            train_subset_frac=_float_or_none(merged.get("train_subset_frac")),
            train_subset_min_n=int(merged.get("train_subset_min_n", DEFAULTS["train_subset_min_n"])),
        ),
        model=ModelCfg(
            name=str(merged["model"]),
            d=int(merged["d"]),
            d_hi=int(merged["d_hi"]),
            num_attn_layers=int(merged["num_attn_layers"]),
            K=int(merged["K"]),
            gamma=float(merged["gamma"]),
            vlimit=bool(merged.get("vlimit", DEFAULTS["vlimit"])),
            use_armijo=bool(merged.get("use_armijo", DEFAULTS["use_armijo"])),
            dtheta_max=float(merged.get("DthetaMax", DEFAULTS["DthetaMax"])),
            dvm_frac=float(merged.get("DvmFrac", DEFAULTS["DvmFrac"])),
            pinn=bool(merged.get("PINN", DEFAULTS["PINN"])),
            block_diag=bool(merged.get("BLOCK_DIAG", DEFAULTS["BLOCK_DIAG"])),
            weight_init=str(merged["weight_init"]),
            bias_init=float(merged["bias_init"]),
        ),
        optim=OptimCfg(
            batch_size=int(merged["BATCH"]),
            epochs=int(merged["EPOCHS"]),
            lr=float(merged["LR"]),
            val_every=int(merged["VAL_EVERY"]),
            weight_decay=float(merged["weight_decay"]),
            lr_scheduler=str(merged["lr_scheduler"]),
            cosine_restart_epoch=int(merged["cosineRestartEpoch"]),
            mag_ang_mse=bool(merged.get("mag_ang_mse", DEFAULTS["mag_ang_mse"])),
        ),
        peft=PeftCfg(
            enabled=bool(merged.get("peft", DEFAULTS["peft"])),
            method=str(merged.get("peft_method", DEFAULTS["peft_method"])),
            lora_r=int(merged.get("lora_r", DEFAULTS["lora_r"])),
            lora_alpha=int(merged.get("lora_alpha", DEFAULTS["lora_alpha"])),
            lora_dropout=float(merged.get("lora_dropout", DEFAULTS["lora_dropout"])),
            lora_target_modules=list(
                merged.get("lora_target_modules") or DEFAULTS["lora_target_modules"]
            ),
            train_base=bool(merged.get("peft_train_base", DEFAULTS["peft_train_base"])),
            base_ckpt_path=_str_or_none(merged.get("peft_base_ckpt_path")),
            unfreeze_modules=list(merged.get("peft_unfreeze_modules") or []),
            head_only_ft=bool(merged.get("head_only_ft", DEFAULTS["head_only_ft"])),
            head_only_modules=list(
                merged.get("head_only_modules") or DEFAULTS["head_only_modules"]
            ),
        ),
        mlflow=MlflowCfg(
            enabled=bool(merged.get("mlflow", DEFAULTS["mlflow"])),
            tracking_uri=_str_or_none(merged.get("mlflow_tracking_uri")),
            experiment=str(merged.get("mlflow_experiment", DEFAULTS["mlflow_experiment"])),
            artifact_location=_str_or_none(merged.get("mlflow_artifact_location")),
            artifact_path=str(merged.get("mlflow_artifact_path", DEFAULTS["mlflow_artifact_path"])),
            keep_local_run_dir=bool(
                merged.get("mlflow_keep_local_run_dir", DEFAULTS["mlflow_keep_local_run_dir"])
            ),
            strict=bool(merged.get("mlflow_strict", DEFAULTS["mlflow_strict"])),
        ),
        compare=CompareCfg(
            enabled=bool(merged.get("compare", DEFAULTS["compare"])),
            baseline_run_id=_str_or_none(merged.get("compare_baseline_run_id")),
            metrics=list(merged.get("compare_metrics") or DEFAULTS["compare_metrics"]),
        ),
    )


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Legacy CLI-only path: Namespace -> merged dict -> nested TrainConfig."""
    merged = _args_to_merged(args)
    return _build_train_config(merged, runname=_compute_runname(merged))


def parse_train_config(argv: list[str] | None = None) -> tuple[TrainConfig, str | None]:
    """Parse CLI + optional YAML config.

    Returns (TrainConfig, config_path).
    """
    # Pass 1: detect --config without applying defaults for other args.
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", type=str, default=None)
    probe_ns, remaining = config_probe.parse_known_args(argv)

    if not probe_ns.config:
        # No config.yaml: preserve legacy CLI defaults.
        parser = build_parser(suppress_defaults=False)
        args = parser.parse_args(argv)
        return config_from_args(args), None

    raw = load_yaml_config(probe_ns.config)

    # Pass 2: only parse explicit CLI overrides.
    parser = build_parser(suppress_defaults=True)
    args = parser.parse_args(remaining)

    # Merge order: hard defaults -> YAML -> CLI overrides
    merged: dict[str, Any] = dict(DEFAULTS)

    # YAML (nested)
    merged["mode"] = get(raw, ("run", "mode"), merged["mode"])
    merged["seed_value"] = get(raw, ("run", "seed"), merged["seed_value"])
    init_ckpt = get(
        raw, ("run", "init_ckpt_path"), merged.get("init_ckpt_path", DEFAULTS["init_ckpt_path"])
    )
    merged["init_ckpt_path"] = str(init_ckpt) if init_ckpt not in (None, "", "null") else None

    merged["train_ratio"] = get(raw, ("split", "train_ratio"), merged["train_ratio"])
    merged["valid_ratio"] = get(raw, ("split", "valid_ratio"), merged["valid_ratio"])
    merged["split_mode"] = get(raw, ("split", "mode"), merged["split_mode"])
    tsf = get(
        raw,
        ("split", "train_subset_frac"),
        merged.get("train_subset_frac", DEFAULTS["train_subset_frac"]),
    )
    merged["train_subset_frac"] = float(tsf) if tsf not in (None, "", "null") else None
    merged["train_subset_min_n"] = int(
        get(
            raw,
            ("split", "train_subset_min_n"),
            merged.get("train_subset_min_n", DEFAULTS["train_subset_min_n"]),
        )
    )

    parquet_paths = env_expand_paths(
        as_path_list(get(raw, ("data", "parquet_paths"), merged["PARQUET"]))
    )
    merged["PARQUET"] = parquet_paths
    merged["PER_UNIT"] = bool(get(raw, ("data", "per_unit"), merged["PER_UNIT"]))

    merged["PINN"] = bool(get(raw, ("flags", "pinn"), merged["PINN"]))
    merged["BLOCK_DIAG"] = bool(get(raw, ("flags", "block_diag"), merged["BLOCK_DIAG"]))
    merged["NORMALIZE"] = bool(get(raw, ("flags", "normalize"), merged["NORMALIZE"]))
    merged["mag_ang_mse"] = bool(get(raw, ("flags", "mag_ang_mse"), merged["mag_ang_mse"]))
    merged["float64"] = bool(get(raw, ("flags", "float64"), merged["float64"]))

    merged["model"] = get(raw, ("model", "name"), merged["model"])
    merged["d"] = int(get(raw, ("model", "d"), merged["d"]))
    merged["d_hi"] = int(get(raw, ("model", "d_hi"), merged["d_hi"]))
    merged["num_attn_layers"] = int(
        get(raw, ("model", "num_attn_layers"), merged["num_attn_layers"])
    )
    merged["K"] = int(get(raw, ("model", "K"), merged["K"]))
    merged["gamma"] = float(get(raw, ("model", "gamma"), merged["gamma"]))
    merged["vlimit"] = bool(get(raw, ("model", "vlimit"), merged["vlimit"]))
    merged["use_armijo"] = bool(get(raw, ("model", "use_armijo"), merged["use_armijo"]))
    merged["DthetaMax"] = float(get(raw, ("model", "DthetaMax"), merged["DthetaMax"]))
    merged["DvmFrac"] = float(get(raw, ("model", "DvmFrac"), merged["DvmFrac"]))

    merged["BATCH"] = int(get(raw, ("train", "batch_size"), merged["BATCH"]))
    merged["EPOCHS"] = int(get(raw, ("train", "epochs"), merged["EPOCHS"]))
    merged["LR"] = float(get(raw, ("train", "lr"), merged["LR"]))
    merged["VAL_EVERY"] = int(get(raw, ("train", "val_every"), merged["VAL_EVERY"]))

    merged["weight_init"] = get(raw, ("optimizer", "weight_init"), merged["weight_init"])
    merged["bias_init"] = float(get(raw, ("optimizer", "bias_init"), merged["bias_init"]))
    merged["weight_decay"] = float(get(raw, ("optimizer", "weight_decay"), merged["weight_decay"]))
    merged["lr_scheduler"] = get(raw, ("optimizer", "lr_scheduler"), merged["lr_scheduler"])
    merged["cosineRestartEpoch"] = int(
        get(raw, ("optimizer", "cosine_restart_epoch"), merged["cosineRestartEpoch"])
    )

    merged["ADJ_MODE"] = get(raw, ("misc", "adj_mode"), merged["ADJ_MODE"])

    # MLflow
    merged["mlflow"] = bool(get(raw, ("mlflow", "enabled"), merged["mlflow"]))
    merged["mlflow_tracking_uri"] = get(
        raw, ("mlflow", "tracking_uri"), merged["mlflow_tracking_uri"]
    )
    merged["mlflow_experiment"] = str(
        get(raw, ("mlflow", "experiment"), merged["mlflow_experiment"])
    )
    merged["mlflow_artifact_location"] = get(
        raw, ("mlflow", "artifact_location"), merged.get("mlflow_artifact_location")
    )
    merged["mlflow_artifact_path"] = str(
        get(
            raw,
            ("mlflow", "artifact_path"),
            merged.get("mlflow_artifact_path", DEFAULTS["mlflow_artifact_path"]),
        )
    )
    merged["mlflow_keep_local_run_dir"] = bool(
        get(
            raw,
            ("mlflow", "keep_local_run_dir"),
            merged.get("mlflow_keep_local_run_dir", DEFAULTS["mlflow_keep_local_run_dir"]),
        )
    )
    merged["mlflow_strict"] = bool(
        get(raw, ("mlflow", "strict"), merged.get("mlflow_strict", DEFAULTS["mlflow_strict"]))
    )

    # PEFT / LoRA
    merged["peft"] = bool(get(raw, ("peft", "enabled"), merged.get("peft", DEFAULTS["peft"])))
    merged["peft_method"] = str(
        get(raw, ("peft", "method"), merged.get("peft_method", DEFAULTS["peft_method"]))
    )
    merged["lora_r"] = int(get(raw, ("peft", "lora_r"), merged.get("lora_r", DEFAULTS["lora_r"])))
    merged["lora_alpha"] = int(
        get(raw, ("peft", "lora_alpha"), merged.get("lora_alpha", DEFAULTS["lora_alpha"]))
    )
    merged["lora_dropout"] = float(
        get(raw, ("peft", "lora_dropout"), merged.get("lora_dropout", DEFAULTS["lora_dropout"]))
    )
    merged["lora_target_modules"] = list(
        get(
            raw,
            ("peft", "lora_target_modules"),
            merged.get("lora_target_modules", DEFAULTS["lora_target_modules"]),
        )
    )
    merged["peft_train_base"] = bool(
        get(raw, ("peft", "train_base"), merged.get("peft_train_base", DEFAULTS["peft_train_base"]))
    )
    base_ckpt = get(
        raw,
        ("peft", "base_ckpt_path"),
        merged.get("peft_base_ckpt_path", DEFAULTS["peft_base_ckpt_path"]),
    )
    merged["peft_base_ckpt_path"] = (
        str(base_ckpt) if base_ckpt not in (None, "", "null") else None
    )

    merged["peft_unfreeze_modules"] = list(
        get(
            raw,
            ("peft", "unfreeze_modules"),
            merged.get("peft_unfreeze_modules", DEFAULTS["peft_unfreeze_modules"]),
        )
    )

    # Head-only fine-tuning (no LoRA)
    merged["head_only_ft"] = bool(
        get(raw, ("head_only", "enabled"), merged.get("head_only_ft", DEFAULTS["head_only_ft"]))
    )
    merged["head_only_modules"] = list(
        get(
            raw,
            ("head_only", "modules"),
            merged.get("head_only_modules", DEFAULTS["head_only_modules"]),
        )
    )

    # Compare vs baseline run (MLflow)
    merged["compare"] = bool(
        get(raw, ("compare", "enabled"), merged.get("compare", DEFAULTS["compare"]))
    )
    cmp_id = get(
        raw,
        ("compare", "baseline_run_id"),
        merged.get("compare_baseline_run_id", DEFAULTS["compare_baseline_run_id"]),
    )
    merged["compare_baseline_run_id"] = (
        str(cmp_id) if cmp_id not in (None, "", "null") else None
    )
    merged["compare_metrics"] = list(
        get(
            raw,
            ("compare", "metrics"),
            merged.get("compare_metrics", DEFAULTS["compare_metrics"]),
        )
    )

    # CLI overrides (only what the user explicitly provided; argparse with
    # suppress_defaults omits unset flags from the Namespace entirely).
    for k, v in vars(args).items():
        if k == "config":
            continue
        if k == "no_mlflow_strict":
            merged["mlflow_strict"] = False
            continue
        merged[k] = v

    # Optional explicit runname via YAML run.name; otherwise derive from final merged.
    yaml_runname = get(raw, ("run", "name"), None)
    runname = str(yaml_runname) if yaml_runname else _compute_runname(merged)

    return _build_train_config(merged, runname=runname), str(probe_ns.config)
