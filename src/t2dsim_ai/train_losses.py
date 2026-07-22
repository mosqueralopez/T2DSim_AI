"""Digital-twin training losses (aligned with T2D-simulator ``Population.model``)."""

from __future__ import annotations

import torch

from t2dsim_ai.options import states
from t2dsim_ai.preprocess import scale_single_state


def scaled_gc_limits_mgdl(
    lim_inferior_mgdl: float = 70.0,
    lim_superior_mgdl: float = 250.0,
) -> tuple[float, float]:
    """Return hypo/hyper CGM thresholds in scaled ``state_Gc`` units."""
    return (
        float(scale_single_state(lim_inferior_mgdl, "Gc")),
        float(scale_single_state(lim_superior_mgdl, "Gc")),
    )


def state_minimum_values() -> dict[int, float]:
    """Minimum allowed value per state index in scaled space."""
    mins: dict[int, float] = {}
    for idx, state in enumerate(states):
        mins[idx] = float(scale_single_state(0, state))
    return mins


def loss_fit(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    *,
    lim_inferior: float,
    lim_superior: float,
    hypo_penalization: float,
    hyper_penalization: float,
    dcgm_weight: float = 100.0,
) -> torch.Tensor:
    """Penalized CGM MSE plus derivative matching (``lossFit`` in research code)."""
    err_fit = y_pred[1:] - y_true[1:]
    err_df = torch.diff(y_pred, dim=0) - torch.diff(y_true, dim=0)

    penalty = torch.ones_like(y_true[1:])
    penalty[
        torch.logical_and(y_true[1:] <= lim_inferior, y_pred[1:] > y_true[1:])
    ] = hypo_penalization
    penalty[
        torch.logical_and(y_true[1:] >= lim_superior, y_pred[1:] < y_true[1:])
    ] = hyper_penalization

    mse_cgm = torch.mean((err_fit**2) * penalty)
    mse_dcgm = torch.mean(err_df**2)
    return mse_cgm + dcgm_weight * mse_dcgm


def loss_consistency(x_pred: torch.Tensor, state_mins: dict[int, float]) -> torch.Tensor:
    """Barrier loss keeping simulated states above physiological minima."""
    barrier = torch.zeros((), dtype=x_pred.dtype, device=x_pred.device)
    for state_idx, state_min in state_mins.items():
        barrier = barrier + torch.relu(-(x_pred[:, :, state_idx] - state_min)).sum()
    return barrier
