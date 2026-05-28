from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .loop import TrainHistory


def plot_history(*, history: TrainHistory, pinn: bool, plots_dir: str) -> None:
    os.makedirs(plots_dir, exist_ok=True)

    epochs = range(1, len(history.train_loss) + 1)

    if pinn:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history.train_loss, label="Train Physics Loss")
        plt.plot(epochs[: len(history.val_loss)], history.val_loss, label="Validation Physics Loss")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("PINN: Physics Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/physics_loss.png")
        plt.clf()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history.train_rmse, label="Train RMSE (phasor/all)")
    plt.plot(epochs[: len(history.val_rmse)], history.val_rmse, label="Val RMSE (phasor/all)")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Supervised RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/rmse_total.png")
    plt.clf()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(epochs, history.train_rmse_mag, label="Train |V|")
    ax[0].plot(epochs[: len(history.val_rmse_mag)], history.val_rmse_mag, label="Val |V|")
    ax[0].set_title("Magnitude RMSE")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(epochs, history.train_rmse_ang_deg, label="Train θ (deg)")
    ax[1].plot(epochs[: len(history.val_rmse_ang_deg)], history.val_rmse_ang_deg, label="Val θ (deg)")
    ax[1].set_title("Angle RMSE (degrees)")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    fig.suptitle("Magnitude vs Angle RMSE")
    fig.tight_layout()
    fig.savefig(f"{plots_dir}/rmse_components.png")
    plt.close(fig)
