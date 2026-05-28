from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .metrics import mse_components
from .logger import console, log
from rich.progress import track


@dataclass
class EpochMetrics:
    loss: float
    mse: float
    mse_mag: float
    mse_ang: float
    phys: float = 0.0  # KCL residual; 0.0 when pinn=False (no phys term tracked)

    @property
    def rmse(self) -> float:
        return math.sqrt(self.mse)

    @property
    def rmse_mag(self) -> float:
        return math.sqrt(self.mse_mag)

    @property
    def rmse_ang_deg(self) -> float:
        return math.sqrt(self.mse_ang) * (180.0 / math.pi)


def run_epoch(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    train: bool,
    pinn: bool,
    block_diag: bool,
    optim: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    clip_grad_norm: float = 1.0,
    desc: str | None = None,
    show_progress: bool = True,
) -> EpochMetrics:
    model.train() if train else model.eval()

    sum_loss = 0.0
    sum_mse = 0.0
    sum_mse_mag = 0.0
    sum_mse_ang = 0.0
    sum_phys = 0.0
    n_samples = 0

    if desc is None:
        desc = "train" if train else "eval"

    try:
        total = len(loader)
    except Exception:
        total = None

    iterator = loader
    if show_progress:
        iterator = track(
            loader,
            total=total,
            description=desc,
            console=console,
        )

    with torch.set_grad_enabled(train):
        for batch in iterator:
            B = batch["bus_type"].size(0)
            n_samples += B

            n_nodes_per_graph = batch["sizes"].to(device) if block_diag else None

            bus_type = batch["bus_type"].to(device)
            Line = batch["Lines_connected"].to(device)
            Y_raw = batch.get("Ybus", None)
            Y = Y_raw.to(device, non_blocking=True) if isinstance(Y_raw, torch.Tensor) else None
            Ys = batch["Y_Lines"].to(device)
            Yc = batch["Y_C_Lines"].to(device)

            Sstart = batch["S_start"].to(device)
            Vstart = batch["V_start"].to(device)
            Vnewton = batch["V_newton"].to(device)

            # Optional per-bus voltage-class feature (LVN). Passed only if
            # the model accepts the kwarg AND the batch carries it -- HVN
            # data + HVN-trained models behave unchanged.
            vn_log_raw = batch.get("vn_log", None)
            extra_kw = {}
            if isinstance(vn_log_raw, torch.Tensor):
                extra_kw["vn_log"] = vn_log_raw.to(device)

            if pinn:
                Vpred, loss_phys = model(bus_type, Line, Y, Ys, Yc, Sstart, Vstart, n_nodes_per_graph, **extra_kw)
                mse, mse_mag, mse_ang = mse_components(Vpred, Vnewton)
                # Optional combined loss: phys_loss + GNN_MSE_WEIGHT * mse.
                # Env-var gated -- default 0.0 preserves the old phys-only behavior.
                # Setting >0 lets the optimizer drive both metrics simultaneously.
                _mse_weight = float(os.environ.get("GNN_MSE_WEIGHT", "0.0"))
                loss = loss_phys + _mse_weight * mse if _mse_weight > 0 else loss_phys
            else:
                Vpred = model(bus_type, Line, Y, Ys, Yc, Sstart, Vstart, n_nodes_per_graph, **extra_kw)
                mse, mse_mag, mse_ang = mse_components(Vpred, Vnewton)
                loss = mse

            if train:
                assert optim is not None
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                optim.step()
                if scheduler is not None:
                    scheduler.step()

            sum_loss += float(loss.item()) * B
            sum_mse += float(mse.item()) * B
            sum_mse_mag += float(mse_mag.item()) * B
            sum_mse_ang += float(mse_ang.item()) * B
            if pinn:
                sum_phys += float(loss_phys.item()) * B

    return EpochMetrics(
        loss=sum_loss / n_samples,
        mse=sum_mse / n_samples,
        mse_mag=sum_mse_mag / n_samples,
        mse_ang=sum_mse_ang / n_samples,
        phys=sum_phys / n_samples,
    )


@dataclass
class TrainHistory:
    train_loss: list[float]
    train_rmse: list[float]
    train_rmse_mag: list[float]
    train_rmse_ang_deg: list[float]
    val_loss: list[float]
    val_rmse: list[float]
    val_rmse_mag: list[float]
    val_rmse_ang_deg: list[float]
    best_epoch: int
    best_score: float
    best_val_rmse_mag: float
    best_val_rmse_ang_deg: float


def train_validate(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    pinn: bool,
    block_diag: bool,
    optim: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    val_every: int,
    best_ckpt_path: str | None = None,
    on_epoch_metrics: Callable[[int, str, EpochMetrics], None] | None = None,
    show_progress: bool = True,
) -> TrainHistory:
    train_loss_hist: list[float] = []
    train_rmse_hist: list[float] = []
    train_rmse_mag_hist: list[float] = []
    train_rmse_ang_hist_deg: list[float] = []

    val_loss_hist: list[float] = []
    val_rmse_hist: list[float] = []
    val_rmse_mag_hist: list[float] = []
    val_rmse_ang_hist_deg: list[float] = []

    best_epoch = 0
    best_score = float("inf")
    best_val_rmse_mag = float("inf")
    best_val_rmse_ang_deg = float("inf")

    log.info("Initial metrics before training:")
    m_train0 = run_epoch(
        model=model,
        loader=train_loader,
        device=device,
        train=False,
        pinn=pinn,
        block_diag=block_diag,
        desc="init/train",
        show_progress=show_progress,
    )
    m_val0 = run_epoch(
        model=model,
        loader=val_loader,
        device=device,
        train=False,
        pinn=pinn,
        block_diag=block_diag,
        desc="init/valid",
        show_progress=show_progress,
    )

    log.info(
        "Epoch %3d | train loss %.4e  rmse %.4e (mag %.4e, ang %.4e°) | valid loss %.4e  rmse %.4e (mag %.4e, ang %.4e°)",
        0,
        m_train0.loss,
        m_train0.rmse,
        m_train0.rmse_mag,
        m_train0.rmse_ang_deg,
        m_val0.loss,
        m_val0.rmse,
        m_val0.rmse_mag,
        m_val0.rmse_ang_deg,
    )

    if on_epoch_metrics is not None:
        on_epoch_metrics(0, "train", m_train0)
        on_epoch_metrics(0, "val", m_val0)

    # Treat epoch 0 as candidate best as well
    score0 = m_val0.rmse_mag + m_val0.rmse_ang_deg
    best_epoch = 0
    best_score = float(score0)
    best_val_rmse_mag = float(m_val0.rmse_mag)
    best_val_rmse_ang_deg = float(m_val0.rmse_ang_deg)
    if best_ckpt_path is not None:
        torch.save(model.state_dict(), best_ckpt_path)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Hook for models that need per-epoch state (e.g. PE_DEQ_PF curriculum
        # switches between unrolled-warmup and DEQ forward based on epoch).
        # Optional and additive: no-op for models without the method.
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)

        m_train = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            train=True,
            pinn=pinn,
            block_diag=block_diag,
            optim=optim,
            scheduler=scheduler,
            desc=f"epoch {epoch}/{epochs} train",
            show_progress=show_progress,
        )

        train_loss_hist.append(m_train.loss)
        train_rmse_hist.append(m_train.rmse)
        train_rmse_mag_hist.append(m_train.rmse_mag)
        train_rmse_ang_hist_deg.append(m_train.rmse_ang_deg)

        if on_epoch_metrics is not None:
            on_epoch_metrics(epoch, "train", m_train)

        if epoch % val_every == 0 or epoch == epochs:
            m_val = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                train=False,
                pinn=pinn,
                block_diag=block_diag,
                desc=f"epoch {epoch}/{epochs} valid",
                show_progress=show_progress,
            )

            val_loss_hist.append(m_val.loss)
            val_rmse_hist.append(m_val.rmse)
            val_rmse_mag_hist.append(m_val.rmse_mag)
            val_rmse_ang_hist_deg.append(m_val.rmse_ang_deg)

            if on_epoch_metrics is not None:
                on_epoch_metrics(epoch, "val", m_val)

            log.info(
                "Epoch %3d | train loss %.4e  rmse %.4e (mag %.4e, ang %.4e°) | valid loss %.4e  rmse %.4e (mag %.4e, ang %.4e°) | time %.2fs",
                epoch,
                m_train.loss,
                m_train.rmse,
                m_train.rmse_mag,
                m_train.rmse_ang_deg,
                m_val.loss,
                m_val.rmse,
                m_val.rmse_mag,
                m_val.rmse_ang_deg,
                time.time() - t0,
            )

            # Select best checkpoint by combined val RMSE components.
            score = m_val.rmse_mag + m_val.rmse_ang_deg
            if score < best_score:
                best_epoch = epoch
                best_score = float(score)
                best_val_rmse_mag = float(m_val.rmse_mag)
                best_val_rmse_ang_deg = float(m_val.rmse_ang_deg)
                if best_ckpt_path is not None:
                    torch.save(model.state_dict(), best_ckpt_path)
                    log.info("best checkpoint updated (%s) -> %s", best_epoch, best_ckpt_path)
        else:
            log.info(
                "Epoch %3d | train loss %.4e  rmse %.4e (mag %.4e, ang %.4e°) | time %.2fs",
                epoch,
                m_train.loss,
                m_train.rmse,
                m_train.rmse_mag,
                m_train.rmse_ang_deg,
                time.time() - t0,
            )

    return TrainHistory(
        train_loss=train_loss_hist,
        train_rmse=train_rmse_hist,
        train_rmse_mag=train_rmse_mag_hist,
        train_rmse_ang_deg=train_rmse_ang_hist_deg,
        val_loss=val_loss_hist,
        val_rmse=val_rmse_hist,
        val_rmse_mag=val_rmse_mag_hist,
        val_rmse_ang_deg=val_rmse_ang_hist_deg,
        best_epoch=int(best_epoch),
        best_score=float(best_score),
        best_val_rmse_mag=float(best_val_rmse_mag),
        best_val_rmse_ang_deg=float(best_val_rmse_ang_deg),
    )


def evaluate_test(
    *,
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    pinn: bool,
    block_diag: bool,
    show_progress: bool = True,
) -> EpochMetrics:
    return run_epoch(
        model=model,
        loader=test_loader,
        device=device,
        train=False,
        pinn=pinn,
        block_diag=block_diag,
        desc="test",
        show_progress=show_progress,
    )
