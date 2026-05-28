"""Default hyperparameter table.

Single source of truth for CLI defaults and YAML-merge fallbacks. Referenced by
parser.py (to build argparse defaults) and loader.py (to seed the merged config
when YAML keys are missing).
"""

from __future__ import annotations


DEFAULTS = {
    "PINN": False,
    "BLOCK_DIAG": False,
    "NORMALIZE": False,
    "PER_UNIT": False,
    "float64": False,
    "mode": "train_test",
    "mag_ang_mse": False,
    "model": "GNSMsg_EdgeSelfAttn",
    "d": 4,
    "d_hi": 16,
    "num_attn_layers": 1,
    "K": 40,
    "gamma": 0.9,
    "use_armijo": False,
    "vlimit": False,
    "DthetaMax": 0.3,
    "DvmFrac": 0.1,
    "train_ratio": 0.8,
    "valid_ratio": 0.1,
    "split_mode": "ratio",  # ratio | equal3
    # Few-shot: optional subsample of the *training split only*.
    # Keeps val/test fixed across budgets when seed/split_mode are fixed.
    "train_subset_frac": None,
    "train_subset_min_n": 1,
    "ADJ_MODE": "cplx",
    "weight_init": "sd0.02",
    "bias_init": 0.0,
    "weight_decay": 1e-3,
    "lr_scheduler": "default",
    "cosineRestartEpoch": 20,
    "BATCH": 16,
    "EPOCHS": 20,
    "LR": 1e-4,
    "VAL_EVERY": 1,
    "PARQUET": ["./datasets/HVN_15000_NR_plain_4_to_32_buses.parquet"],
    "seed_value": 42,

    # Initialization
    # Optional checkpoint to load before training (works for both full fine-tuning and PEFT).
    "init_ckpt_path": None,

    # PEFT / LoRA
    "peft": False,
    "peft_method": "lora",
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "lora_target_modules": ["q", "k", "v", "out"],
    "peft_train_base": False,
    "peft_base_ckpt_path": None,
    # Optional: keep select modules trainable while the base is frozen for LoRA.
    # Example: ["theta_head", "v_head", "m_head"]
    "peft_unfreeze_modules": [],

    # Head-only fine-tuning (no LoRA): freeze all, unfreeze heads.
    "head_only_ft": False,
    # Optional: override which modules to unfreeze for head-only FT.
    # Default matches the three prediction heads in the model.
    "head_only_modules": ["theta_head", "v_head", "m_head"],

    # Compare metrics vs a baseline MLflow run (optional)
    "compare": False,
    "compare_baseline_run_id": None,
    # This project focuses on RMSE; compare overall RMSE plus mag/angle components.
    "compare_metrics": ["test/rmse", "test/rmse_mag", "test/rmse_ang_deg"],

    # MLflow
    "mlflow": False,
    # Keep MLflow tracking DB + artifacts under results/ by default.
    "mlflow_tracking_uri": "sqlite:///results/mlflow.db",
    "mlflow_experiment": "SimpleGNN",
    "mlflow_artifact_location": "file:./results/mlruns",
    # Upload local run folder under this subdirectory in MLflow's artifact tree.
    "mlflow_artifact_path": "run",
    # If false and MLflow is enabled, run artifacts are staged temporarily and deleted after upload.
    "mlflow_keep_local_run_dir": False,
    # If true, enabling MLflow fails loudly when MLflow isn't importable.
    "mlflow_strict": True,
}
