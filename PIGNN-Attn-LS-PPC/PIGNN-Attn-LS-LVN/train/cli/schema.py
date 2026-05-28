"""Typed, frozen training configuration.

TrainConfig is composed of seven sub-configs, one per concern. Helpers and
consumers should accept only the sub-config they actually need (interface
segregation), e.g. `_apply_peft_and_freezing(model, cfg.peft)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class RunCfg:
    runname: str
    seed: int
    mode: str
    init_ckpt_path: str | None = None


@dataclass(frozen=True)
class DataCfg:
    parquet_paths: List[str]
    per_unit: bool
    normalize: bool
    split_mode: str  # "ratio" | "equal3"
    train_ratio: float
    valid_ratio: float
    train_subset_frac: float | None = None
    train_subset_min_n: int = 1


@dataclass(frozen=True)
class ModelCfg:
    name: str
    d: int
    d_hi: int
    num_attn_layers: int
    K: int
    gamma: float
    vlimit: bool
    use_armijo: bool
    dtheta_max: float
    dvm_frac: float
    pinn: bool
    block_diag: bool
    weight_init: str
    bias_init: float


@dataclass(frozen=True)
class OptimCfg:
    batch_size: int
    epochs: int
    lr: float
    val_every: int
    weight_decay: float
    lr_scheduler: str
    cosine_restart_epoch: int
    mag_ang_mse: bool = False


@dataclass(frozen=True)
class PeftCfg:
    enabled: bool = False
    method: str = "lora"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: ["q", "k", "v", "out"])
    train_base: bool = False
    base_ckpt_path: str | None = None
    unfreeze_modules: List[str] = field(default_factory=list)
    head_only_ft: bool = False
    head_only_modules: List[str] = field(default_factory=lambda: ["theta_head", "v_head", "m_head"])


@dataclass(frozen=True)
class MlflowCfg:
    enabled: bool = False
    tracking_uri: str | None = None
    experiment: str = "SimpleGNN"
    artifact_location: str | None = None
    artifact_path: str = "run"
    keep_local_run_dir: bool = False
    strict: bool = True


@dataclass(frozen=True)
class CompareCfg:
    enabled: bool = False
    baseline_run_id: str | None = None
    metrics: List[str] = field(default_factory=lambda: ["test/rmse", "test/rmse_mag", "test/rmse_ang_deg"])


@dataclass(frozen=True)
class TrainConfig:
    run: RunCfg
    data: DataCfg
    model: ModelCfg
    optim: OptimCfg
    peft: PeftCfg
    mlflow: MlflowCfg
    compare: CompareCfg
