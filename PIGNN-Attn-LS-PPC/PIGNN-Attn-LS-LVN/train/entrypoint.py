from __future__ import annotations

import csv
import json
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Callable

import torch

from .cli import parse_train_config
from .cli.schema import CompareCfg, PeftCfg, TrainConfig
from .data import build_dataloaders
from .logger import configure_logging, log
from .run_paths import ensure_run_dirs, make_run_paths
from .loop import evaluate_test, train_validate
from .mlflow_utils import (
    add_basic_tags,
    log_params_safe,
    log_run_artifacts,
    mlflow_run,
    snapshot_code,
)
from .modeling import count_parameters, create_model, init_weights
from .optim_utils import build_optimizer_and_scheduler
from .peft_utils import (
    apply_lora_to_linear_modules,
    count_trainable_params,
    freeze_all,
    freeze_all_except_lora,
    unfreeze_modules,
)
from .run_naming import make_run_id, make_run_slug, safe_param_dict


def main(argv: list[str] | None = None) -> int:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    cfg, config_path = parse_train_config(argv)
    repo_root = Path(__file__).resolve().parents[1]

    run_slug = make_run_slug(
        parquet_paths=cfg.data.parquet_paths,
        model_name=cfg.model.name,
        K=cfg.model.K,
        d=cfg.model.d,
        d_hi=cfg.model.d_hi,
        pinn=cfg.model.pinn,
        block_diag=cfg.model.block_diag,
        per_unit=cfg.data.per_unit,
        split_mode=cfg.data.split_mode,
    )
    run_id = make_run_id(run_slug=run_slug)
    run_name = f"{run_id}_{run_slug}"

    # MLflow disabled OR keep_local_run_dir=true: persist under results/runs/<run_id>.
    # MLflow enabled and keep_local_run_dir=false: stage in temp dir, upload, then delete.
    keep_local = (not cfg.mlflow.enabled) or cfg.mlflow.keep_local_run_dir

    with ExitStack() as stack:
        paths = _setup_run_paths(stack, run_id=run_id, keep_local=keep_local)
        ensure_run_dirs(paths)
        log_file = str(Path(paths.run_dir) / "train.log")
        configure_logging(log_file=log_file)
        if config_path:
            log.info("Loaded config: %s", config_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _log_config_summary(cfg, device)

        splits = build_dataloaders(
            parquet_paths=cfg.data.parquet_paths,
            per_unit=cfg.data.per_unit,
            device=device,
            batch_size=cfg.optim.batch_size,
            block_diag=cfg.model.block_diag,
            seed=cfg.run.seed,
            split_mode=cfg.data.split_mode,
            train_ratio=cfg.data.train_ratio,
            valid_ratio=cfg.data.valid_ratio,
            train_subset_frac=cfg.data.train_subset_frac,
            train_subset_min_n=cfg.data.train_subset_min_n,
        )
        log.info(
            "Dataset sizes | train %s  valid %s  test %s",
            splits.n_train,
            splits.n_val,
            splits.n_test,
        )

        model = create_model(
            model_name=cfg.model.name,
            d=cfg.model.d,
            d_hi=cfg.model.d_hi,
            K=cfg.model.K,
            pinn=cfg.model.pinn,
            gamma=cfg.model.gamma,
            v_limit=cfg.model.vlimit,
            use_armijo=cfg.model.use_armijo,
            dtheta_max=cfg.model.dtheta_max,
            dvm_frac=cfg.model.dvm_frac,
            num_attn_layers=cfg.model.num_attn_layers,
            device=device,
        )
        _load_or_init_weights(model, cfg)
        _apply_peft_and_freezing(model, cfg.peft)

        param_eff = _compute_param_efficiency(model)

        optim_bundle = None
        if "train" in cfg.run.mode:
            optim_bundle = build_optimizer_and_scheduler(
                model=model,
                lr=cfg.optim.lr,
                weight_decay=cfg.optim.weight_decay,
                lr_scheduler=cfg.optim.lr_scheduler,
                cosine_restart_epoch=cfg.optim.cosine_restart_epoch,
                steps_per_epoch=len(splits.train_loader),
            )

        tags = _build_mlflow_tags(
            cfg=cfg,
            repo_root=repo_root,
            device=device,
            run_id=run_id,
            run_slug=run_slug,
            config_path=config_path,
            param_eff=param_eff,
        )
        _write_meta_json(
            paths,
            run_id=run_id,
            run_slug=run_slug,
            run_name=run_name,
            config_path=config_path,
            log_file=log_file,
            mlflow_enabled=cfg.mlflow.enabled,
            keep_local=keep_local,
        )

        with mlflow_run(
            enabled=cfg.mlflow.enabled,
            strict=cfg.mlflow.strict,
            tracking_uri=cfg.mlflow.tracking_uri,
            experiment=cfg.mlflow.experiment,
            artifact_location=cfg.mlflow.artifact_location,
            run_name=run_name,
            tags=tags,
        ) as mlf:
            if mlf is None:
                log.info("MLflow disabled; writing artifacts to %s", paths.run_dir)
            else:
                log.info("MLflow enabled; logging metrics + artifacts to MLflow")

            _log_initial_mlflow_metadata(mlf, cfg, config_path, param_eff)
            _snapshot_code_safe(repo_root, paths.artifacts_dir)

            rows: dict[tuple[int, str], dict[str, float]] = {}
            on_epoch_metrics = _make_epoch_metrics_callback(mlf, rows)
            final_metrics: dict[str, float] = {}

            history = _run_train_phase(
                cfg=cfg,
                model=model,
                splits=splits,
                optim_bundle=optim_bundle,
                device=device,
                paths=paths,
                on_epoch_metrics=on_epoch_metrics,
                mlf=mlf,
                final_metrics=final_metrics,
            )
            _run_eval_phase(
                cfg=cfg,
                model=model,
                splits=splits,
                device=device,
                mlf=mlf,
                final_metrics=final_metrics,
            )
            _compare_with_baseline(mlf, cfg.compare, final_metrics)
            _write_history_csv(rows, paths.artifacts_dir)
            _plot_history_safe(history, cfg.model.pinn, str(paths.plots_dir))

            if mlf is not None:
                log_run_artifacts(
                    mlflow=mlf,
                    run_dir=Path(paths.run_dir),
                    artifact_path=cfg.mlflow.artifact_path,
                )

    return 0


# --------------------------------------------------------------------------- #
# Phase helpers                                                               #
# --------------------------------------------------------------------------- #


def _setup_run_paths(stack: ExitStack, *, run_id: str, keep_local: bool):
    if keep_local:
        return make_run_paths(run_id=run_id, base_dir="./results/runs")

    staging_root = Path("./results/.mlflow_staging")
    staging_root.mkdir(parents=True, exist_ok=True)
    tmp_base = stack.enter_context(
        tempfile.TemporaryDirectory(prefix=f"{run_id}_", dir=str(staging_root))
    )
    return make_run_paths(run_id=run_id, base_dir=tmp_base)


def _log_config_summary(cfg: TrainConfig, device: torch.device) -> None:
    log.info(
        "MODEL:%s, PINN:%s, Block:%s, d:%s, d_hi:%s, attn_layers:%s, K:%s, "
        "Runname:%s, PARQUET:%s, BATCH:%s, EP:%s, LR:%s",
        cfg.model.name,
        cfg.model.pinn,
        cfg.model.block_diag,
        cfg.model.d,
        cfg.model.d_hi,
        cfg.model.num_attn_layers,
        cfg.model.K,
        cfg.run.runname,
        cfg.data.parquet_paths,
        cfg.optim.batch_size,
        cfg.optim.epochs,
        cfg.optim.lr,
    )
    log.info("Using device: %s", device)


def _load_or_init_weights(model, cfg: TrainConfig) -> None:
    # Full fine-tuning uses run.init_ckpt_path; PEFT may use either init_ckpt_path
    # or peft.base_ckpt_path. Fall back to fresh init when neither is set.
    ckpt_path = cfg.run.init_ckpt_path
    if ckpt_path is None and cfg.peft.enabled and cfg.peft.base_ckpt_path:
        ckpt_path = cfg.peft.base_ckpt_path

    if not ckpt_path:
        init_weights(
            model,
            weight_init=cfg.model.weight_init,
            bias_init=cfg.model.bias_init,
            exclude_modules=[],
        )
        return

    try:
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=True)
        log.info("Loaded checkpoint: %s", ckpt_path)
    except Exception as e:
        log.exception("Failed to load checkpoint '%s': %s", ckpt_path, e)
        raise


def _apply_peft_and_freezing(model, peft: PeftCfg) -> None:
    if peft.enabled:
        method = peft.method.lower().strip()
        if method != "lora":
            raise ValueError(f"Unsupported peft_method: {method}. Supported: lora")

        wrapped = apply_lora_to_linear_modules(
            model,
            target_module_names=list(peft.lora_target_modules),
            r=peft.lora_r,
            alpha=peft.lora_alpha,
            dropout=peft.lora_dropout,
        )
        log.info("Applied LoRA to %d Linear modules", len(wrapped))
        if wrapped:
            log.info("LoRA wrapped modules (first 12): %s", wrapped[:12])

        if not peft.train_base:
            freeze_all_except_lora(model)
            if peft.unfreeze_modules:
                unfrozen = unfreeze_modules(model, list(peft.unfreeze_modules))
                if unfrozen:
                    log.info("PEFT: additionally unfroze modules: %s", unfrozen)
                else:
                    log.warning(
                        "PEFT: peft_unfreeze_modules was set but no modules were unfrozen: %s",
                        peft.unfreeze_modules,
                    )

    if peft.enabled and peft.head_only_ft:
        log.warning("Head-only FT is enabled with PEFT; head-only freezing will override PEFT freezing.")

    if peft.head_only_ft:
        freeze_all(model)
        if not peft.head_only_modules:
            log.warning("Head-only FT enabled but no head_only_modules provided; all params frozen.")
            return

        unfrozen = unfreeze_modules(model, list(peft.head_only_modules))
        if unfrozen:
            log.info("Head-only FT: unfroze modules: %s", unfrozen)
        else:
            log.warning(
                "Head-only FT: head_only_modules was set but no modules were unfrozen: %s",
                peft.head_only_modules,
            )


def _compute_param_efficiency(model) -> dict[str, float]:
    total = int(count_parameters(model))
    trainable = int(count_trainable_params(model))
    frac = (float(trainable) / float(total)) if total > 0 else 0.0
    trainable_pct = 100.0 * frac
    reduction_pct = 100.0 * (1.0 - frac)
    reduction_x = (float(total) / float(trainable)) if trainable > 0 else float("inf")

    log.info("Total parameters: %s", total)
    log.info("Trainable parameters: %s", trainable)
    log.info(
        "Parameter efficiency: trainable %.2f%% | reduced %.2f%% | %.2fx fewer trainable params",
        trainable_pct,
        reduction_pct,
        reduction_x,
    )

    return {
        "params_total": float(total),
        "params_trainable": float(trainable),
        "params_trainable_pct": float(trainable_pct),
        "params_reduction_pct": float(reduction_pct),
        "params_reduction_x": float(reduction_x),
    }


def _build_mlflow_tags(
    *,
    cfg: TrainConfig,
    repo_root: Path,
    device: torch.device,
    run_id: str,
    run_slug: str,
    config_path: str | None,
    param_eff: dict[str, float],
) -> dict[str, str]:
    tags = add_basic_tags(repo_root=repo_root)
    tags.update(
        {
            "device": str(device),
            "run_id": run_id,
            "run_slug": run_slug,
            "seed": str(cfg.run.seed),
        }
    )
    if cfg.data.train_subset_frac is not None:
        tags["target_budget"] = str(cfg.data.train_subset_frac)
    if config_path:
        tags["config"] = str(config_path)

    if cfg.peft.enabled:
        tags.update(
            {
                "peft": "true",
                "peft_method": cfg.peft.method,
                "lora_r": str(cfg.peft.lora_r),
                "lora_alpha": str(cfg.peft.lora_alpha),
                "lora_dropout": str(cfg.peft.lora_dropout),
                "peft_train_base": str(cfg.peft.train_base).lower(),
                "peft_unfreeze_modules": ",".join(cfg.peft.unfreeze_modules),
            }
        )

    if cfg.peft.head_only_ft:
        tags.update(
            {
                "head_only_ft": "true",
                "head_only_modules": ",".join(cfg.peft.head_only_modules),
            }
        )

    tags.update(
        {
            "params_total": str(int(param_eff["params_total"])),
            "params_trainable": str(int(param_eff["params_trainable"])),
            "params_trainable_pct": f"{param_eff['params_trainable_pct']:.4f}",
            "params_reduction_pct": f"{param_eff['params_reduction_pct']:.4f}",
            "params_reduction_x": f"{param_eff['params_reduction_x']:.6f}",
        }
    )
    return tags


def _write_meta_json(
    paths,
    *,
    run_id: str,
    run_slug: str,
    run_name: str,
    config_path: str | None,
    log_file: str,
    mlflow_enabled: bool,
    keep_local: bool,
) -> None:
    try:
        meta = {
            "run_id": run_id,
            "run_slug": run_slug,
            "run_name": run_name,
            "config_path": config_path,
            "log_file": log_file,
            "mlflow_enabled": bool(mlflow_enabled),
            "keep_local_run_dir": bool(keep_local),
        }
        (Path(paths.run_dir) / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass


def _log_initial_mlflow_metadata(
    mlf,
    cfg: TrainConfig,
    config_path: str | None,
    param_eff: dict[str, float],
) -> None:
    if mlf is None:
        return

    log_params_safe(mlf, safe_param_dict(cfg))

    try:
        for k, v in param_eff.items():
            mlf.log_metric(k, float(v))
    except Exception:
        pass

    if cfg.data.train_subset_frac is not None:
        try:
            mlf.log_metric("target_budget", float(cfg.data.train_subset_frac))
        except Exception:
            pass

    if config_path:
        try:
            mlf.log_artifact(config_path, artifact_path="config")
        except Exception:
            pass


def _snapshot_code_safe(repo_root: Path, artifacts_dir: str) -> None:
    code_zip = Path(artifacts_dir) / "code_snapshot.zip"
    try:
        snapshot_code(
            repo_root=repo_root,
            out_zip=code_zip,
            include_globs=(
                "train/**/*.py",
                "models/**/*.py",
                "data_loading/**/*.py",
                "configs/**/*.yaml",
                "pyproject.toml",
                "README.md",
            ),
        )
    except Exception:
        pass


def _make_epoch_metrics_callback(
    mlf,
    rows: dict[tuple[int, str], dict[str, float]],
) -> Callable[[int, str, object], None]:
    def on_epoch_metrics(epoch: int, split: str, m) -> None:
        metrics = {
            f"{split}/loss": float(m.loss),
            f"{split}/rmse": float(m.rmse),
            f"{split}/rmse_mag": float(m.rmse_mag),
            f"{split}/rmse_ang_deg": float(m.rmse_ang_deg),
        }
        if mlf is not None:
            for k, v in metrics.items():
                try:
                    mlf.log_metric(k, v, step=int(epoch))
                except Exception:
                    pass
        rows[(int(epoch), split)] = {
            "epoch": float(epoch),
            "split": split,
            "loss": float(m.loss),
            "rmse": float(m.rmse),
            "rmse_mag": float(m.rmse_mag),
            "rmse_ang_deg": float(m.rmse_ang_deg),
            "phys": float(m.phys),
        }

    return on_epoch_metrics


def _run_train_phase(
    *,
    cfg: TrainConfig,
    model,
    splits,
    optim_bundle,
    device: torch.device,
    paths,
    on_epoch_metrics: Callable[[int, str, object], None],
    mlf,
    final_metrics: dict[str, float],
):
    if "train" not in cfg.run.mode:
        return None
    assert optim_bundle is not None

    best_ckpt_path = str(Path(paths.ckpt_dir) / "best.ckpt")
    history = train_validate(
        model=model,
        train_loader=splits.train_loader,
        val_loader=splits.val_loader,
        device=device,
        pinn=cfg.model.pinn,
        block_diag=cfg.model.block_diag,
        optim=optim_bundle.optim,
        scheduler=optim_bundle.scheduler,
        epochs=cfg.optim.epochs,
        val_every=cfg.optim.val_every,
        best_ckpt_path=best_ckpt_path,
        on_epoch_metrics=on_epoch_metrics,
    )

    if mlf is not None:
        try:
            mlf.log_metric("best/epoch", float(history.best_epoch))
            mlf.log_metric("best/score", float(history.best_score))
            mlf.log_metric("best/val_rmse_mag", float(history.best_val_rmse_mag))
            mlf.log_metric("best/val_rmse_ang_deg", float(history.best_val_rmse_ang_deg))
        except Exception:
            pass

    final_metrics["best/epoch"] = float(history.best_epoch)
    final_metrics["best/score"] = float(history.best_score)
    final_metrics["best/val_rmse_mag"] = float(history.best_val_rmse_mag)
    final_metrics["best/val_rmse_ang_deg"] = float(history.best_val_rmse_ang_deg)
    return history


def _run_eval_phase(
    *,
    cfg: TrainConfig,
    model,
    splits,
    device: torch.device,
    mlf,
    final_metrics: dict[str, float],
) -> None:
    if "test" not in cfg.run.mode:
        return

    m_test = evaluate_test(
        model=model,
        test_loader=splits.test_loader,
        device=device,
        pinn=cfg.model.pinn,
        block_diag=cfg.model.block_diag,
    )
    if mlf is not None:
        try:
            mlf.log_metric("test/loss", float(m_test.loss))
            mlf.log_metric("test/rmse", float(m_test.rmse))
            mlf.log_metric("test/rmse_mag", float(m_test.rmse_mag))
            mlf.log_metric("test/rmse_ang_deg", float(m_test.rmse_ang_deg))
        except Exception:
            pass

    final_metrics["test/loss"] = float(m_test.loss)
    final_metrics["test/rmse"] = float(m_test.rmse)
    final_metrics["test/rmse_mag"] = float(m_test.rmse_mag)
    final_metrics["test/rmse_ang_deg"] = float(m_test.rmse_ang_deg)


def _compare_with_baseline(mlf, compare: CompareCfg, final_metrics: dict[str, float]) -> None:
    if mlf is None or not compare.enabled or not compare.baseline_run_id:
        return

    baseline_run_id = compare.baseline_run_id
    metric_keys = list(compare.metrics)

    try:
        from mlflow.tracking import MlflowClient  # type: ignore

        client = MlflowClient()
        base_run = client.get_run(baseline_run_id)
        base_metrics = dict(getattr(base_run.data, "metrics", {}) or {})

        log.info("Comparing metrics vs baseline MLflow run_id=%s", baseline_run_id)
        try:
            mlf.set_tag("compare_baseline_run_id", baseline_run_id)
        except Exception:
            pass

        for key in metric_keys:
            if key not in final_metrics or key not in base_metrics:
                continue

            cur = float(final_metrics[key])
            base = float(base_metrics[key])
            # Skip if base/cur is 0 or NaN (NaN != NaN).
            if base == 0.0 or not (base == base) or not (cur == cur):
                continue

            # Lower-is-better RMSE convention. pct_change: negative => improved.
            # factor_x: negative magnitude => x times LOWER (base/cur);
            #           positive magnitude => x times HIGHER (cur/base).
            pct_change = 100.0 * ((cur - base) / base)
            if cur == 0.0:
                factor_x = float("-inf") if pct_change < 0 else float("inf")
            else:
                factor_x = -abs(base / cur) if cur < base else abs(cur / base)

            log.info(
                "Compare %s | base=%.6g cur=%.6g | rmse_change=%+.2f%% | rmse_factor=%+.3fx",
                key,
                base,
                cur,
                pct_change,
                factor_x,
            )

            safe_key = str(key).replace("/", "_")
            try:
                mlf.log_metric(f"compare/{safe_key}/base", base)
                mlf.log_metric(f"compare/{safe_key}/cur", cur)
                mlf.log_metric(f"compare/{safe_key}/rmse_change_pct", pct_change)
                mlf.log_metric(f"compare/{safe_key}/rmse_factor_x", factor_x)
            except Exception:
                pass

    except Exception as e:
        log.warning("Baseline comparison skipped (%s)", e)


def _write_history_csv(rows: dict[tuple[int, str], dict[str, float]], artifacts_dir: str) -> None:
    try:
        hist_csv = Path(artifacts_dir) / "history.csv"
        with hist_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["epoch", "split", "loss", "rmse", "rmse_mag", "rmse_ang_deg", "phys"],
            )
            w.writeheader()
            for _, row in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                w.writerow(
                    {
                        "epoch": int(row["epoch"]),
                        "split": row["split"],
                        "loss": row["loss"],
                        "rmse": row["rmse"],
                        "rmse_mag": row["rmse_mag"],
                        "rmse_ang_deg": row["rmse_ang_deg"],
                        "phys": row.get("phys", 0.0),
                    }
                )
    except Exception:
        pass


def _plot_history_safe(history, pinn: bool, plots_dir: str) -> None:
    if history is None:
        return
    try:
        from .plotting import plot_history

        plot_history(history=history, pinn=pinn, plots_dir=plots_dir)
    except Exception:
        pass
