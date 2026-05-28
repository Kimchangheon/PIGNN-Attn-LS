from __future__ import annotations

import logging
import os
import subprocess
import warnings
import zipfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import Iterable, Optional

log = logging.getLogger("simplegnn")


def _quiet_mlflow_deps() -> None:
    """Reduce verbosity of MLflow's dependencies (Alembic/SQLAlchemy).

    MLflow can emit a burst of INFO logs on startup (esp. when using
    sqlite:///...), which tends to drown out training logs.
    """

    if os.getenv("QUIET_THIRD_PARTY", "1") in {"0", "false", "False"}:
        return

    for noisy in (
        "alembic",
        "alembic.runtime.migration",
        "alembic.runtime.plugins",
        "sqlalchemy",
        "mlflow",
        "mlflow.store",
        "mlflow.store.db",
        "mlflow.store.db.utils",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _try_import_mlflow(*, strict: bool = False):
    try:
        import mlflow  # type: ignore

        return mlflow
    except Exception as e:
        if strict:
            raise RuntimeError(
                "MLflow was enabled, but could not be imported. "
                "Fix your environment or disable MLflow."
            ) from e
        log.warning("MLflow is not available (%s). Proceeding without MLflow.", e)
        return None


def _normalize_artifact_location(loc: Optional[str]) -> Optional[str]:
    """Normalize artifact locations for stable comparisons.

    MLflow stores file artifact locations as absolute file URIs (e.g. file:///abs/path).
    Users/configs often provide relative paths (e.g. file:./results/mlruns), which refer
    to the same location but differ as strings. Normalize those to avoid noisy warnings.

    Non-file schemes (s3://, gs://, etc.) are returned unchanged.
    """

    if not loc:
        return None
    s = str(loc).strip()
    if not s:
        return None

    # If it looks like a URI with a non-file scheme, keep as-is.
    if "://" in s and not s.startswith("file:"):
        return s

    if s.startswith("file:"):
        parsed = urlparse(s)
        # urlparse('file:./x') -> scheme='file', path='./x'
        raw_path = unquote(parsed.path or "")
        if not raw_path:
            return s
        try:
            return Path(raw_path).expanduser().resolve().as_uri()
        except Exception:
            return s

    # Treat as a filesystem path and normalize to file URI.
    try:
        return Path(s).expanduser().resolve().as_uri()
    except Exception:
        return s


def _git_sha_short(repo_root: Path) -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        sha = (res.stdout or "").strip()
        return sha or None
    except Exception:
        return None


def snapshot_code(*, repo_root: Path, out_zip: Path, include_globs: Iterable[str]) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()

    paths: list[Path] = []
    for pat in include_globs:
        paths.extend(repo_root.glob(pat))

    # Deduplicate + keep files only
    seen: set[Path] = set()
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            continue
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        files.append(p)

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(files, key=lambda x: str(x)):
            try:
                arc = str(p.relative_to(repo_root))
            except Exception:
                arc = p.name
            zf.write(p, arcname=arc)


@contextmanager
def mlflow_run(
    *,
    enabled: bool,
    strict: bool = False,
    tracking_uri: Optional[str],
    experiment: str,
    artifact_location: Optional[str] = None,
    run_name: str,
    tags: Optional[dict[str, str]] = None,
):
    mlflow = _try_import_mlflow(strict=strict)
    if not enabled or mlflow is None:
        yield None
        return

    # MLflow (or its deps) may configure logging; enforce our desired noise level.
    _quiet_mlflow_deps()

    file_backend_selected = bool(
        tracking_uri
        and (
            str(tracking_uri).startswith("file:")
            or ("://" not in str(tracking_uri))
        )
    )

    warn_ctx = warnings.catch_warnings() if file_backend_selected else nullcontext()
    with warn_ctx:
        if file_backend_selected:
            warnings.filterwarnings(
                "ignore",
                message=r"^The filesystem tracking backend .*",
                category=FutureWarning,
            )

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        if file_backend_selected:
            log.warning(
                "MLflow filesystem tracking backend is deprecated; consider using a database URI like "
                "sqlite:///mlflow.db (artifacts can still live on disk)."
            )

        # Ensure experiment exists with the desired artifact location.
        # Note: MLflow cannot change artifact_location of an existing experiment.
        artifact_location_norm = _normalize_artifact_location(artifact_location)
        try:
            exp = mlflow.get_experiment_by_name(experiment)
        except Exception:
            exp = None

        # If the experiment exists but was deleted, restore it so set_experiment works.
        try:
            if exp is not None and str(getattr(exp, "lifecycle_stage", "active")) == "deleted":
                try:
                    from mlflow.tracking import MlflowClient  # type: ignore

                    MlflowClient().restore_experiment(exp.experiment_id)
                    log.warning(
                        "MLflow experiment '%s' was in lifecycle_stage=deleted; restored it.",
                        experiment,
                    )
                    exp = mlflow.get_experiment_by_name(experiment)
                except Exception as e:
                    log.warning(
                        "MLflow experiment '%s' is deleted and could not be restored (%s). "
                        "Choose a new experiment name or restore it manually.",
                        experiment,
                        e,
                    )
        except Exception:
            pass

        if exp is None:
            if artifact_location_norm:
                try:
                    mlflow.create_experiment(experiment, artifact_location=artifact_location_norm)
                except Exception:
                    pass
            # Fall back to default creation behavior if needed.
        else:
            try:
                current_loc = getattr(exp, "artifact_location", None)
                current_norm = _normalize_artifact_location(str(current_loc) if current_loc else None)
                if artifact_location_norm and current_norm and str(current_norm) != str(artifact_location_norm):
                    log.warning(
                        "MLflow experiment '%s' already exists with artifact_location=%s; requested %s will be ignored. "
                        "Use a new experiment name or delete the existing experiment to change it.",
                        experiment,
                        current_loc,
                        artifact_location_norm,
                    )
            except Exception:
                pass

        mlflow.set_experiment(experiment)

        with mlflow.start_run(run_name=run_name):
            if tags:
                for k, v in tags.items():
                    try:
                        mlflow.set_tag(k, v)
                    except Exception:
                        pass
            yield mlflow


def log_run_artifacts(
    *,
    mlflow,
    run_dir: Path,
    artifact_path: str = "run",
) -> None:
    try:
        mlflow.log_artifacts(str(run_dir), artifact_path=artifact_path)
    except Exception as e:
        log.warning("Failed to log artifacts to MLflow: %s", e)


def log_params_safe(mlflow, params: dict) -> None:
    for k, v in params.items():
        try:
            mlflow.log_param(k, v)
        except Exception:
            # MLflow can reject long strings or unsupported types.
            try:
                mlflow.log_param(k, str(v))
            except Exception:
                pass


def log_metrics_history(mlflow, *, history: dict[str, list[float]]) -> None:
    # Expect 1-indexed epoch lists
    for name, series in history.items():
        for step, value in enumerate(series, start=1):
            try:
                mlflow.log_metric(name, float(value), step=step)
            except Exception:
                pass


def add_basic_tags(*, repo_root: Path) -> dict[str, str]:
    tags: dict[str, str] = {}
    sha = _git_sha_short(repo_root)
    if sha:
        tags["git_sha"] = sha
    tags["cwd"] = str(repo_root)
    tags["host"] = os.uname().nodename if hasattr(os, "uname") else "unknown"
    return tags
