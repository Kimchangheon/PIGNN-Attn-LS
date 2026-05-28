from __future__ import annotations

import torch


def mag_ang_mse(Vpred: torch.Tensor, Vref: torch.Tensor, w_mag: float = 1.0, w_ang: float = 1 / torch.pi) -> torch.Tensor:
    """Scale-balanced MSE for (magnitude, angle) representation."""
    dmag = (Vpred[..., 0] - Vref[..., 0]) / w_mag
    dang = torch.atan2(
        torch.sin(Vpred[..., 1] - Vref[..., 1]),
        torch.cos(Vpred[..., 1] - Vref[..., 1]),
    ) / w_ang
    return torch.mean(dmag**2 + dang**2)


def mse_components(Vpred: torch.Tensor, Vref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dmag = Vpred[..., 0] - Vref[..., 0]
    dang = torch.atan2(
        torch.sin(Vpred[..., 1] - Vref[..., 1]),
        torch.cos(Vpred[..., 1] - Vref[..., 1]),
    )
    mse_mag = torch.mean(dmag**2)
    mse_ang = torch.mean(dang**2)
    return mse_mag + mse_ang, mse_mag, mse_ang
