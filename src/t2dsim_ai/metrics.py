"""Glucose outcome metrics for digital-twin evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_rmse(error: np.ndarray) -> float:
    error_proc = error.copy()
    error_proc = error_proc[~np.isnan(error_proc)]
    if error_proc.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(error_proc**2)))


def get_tir(cgm, lim_inf: float = 70, lim_sup: float = 180) -> float:
    cgm_proc = cgm.copy()
    cgm_proc = cgm_proc[~np.isnan(cgm_proc)]
    if cgm_proc.size == 0:
        return 0.0
    return float(np.sum((cgm_proc >= lim_inf) & (cgm_proc <= lim_sup)) / cgm_proc.size)


def get_tbr70(cgm, lim_inf: float = 70) -> float:
    cgm_proc = cgm.copy()
    cgm_proc = cgm_proc[~np.isnan(cgm_proc)]
    if cgm_proc.size == 0:
        return 0.0
    return float(np.sum(cgm_proc < lim_inf) / cgm_proc.size)


def get_tar180(cgm, lim_sup: float = 180) -> float:
    cgm_proc = cgm.copy()
    cgm_proc = cgm_proc[~np.isnan(cgm_proc)]
    if cgm_proc.size == 0:
        return 0.0
    return float(np.sum(cgm_proc > lim_sup) / cgm_proc.size)


def get_glucose_variability(cgm) -> float:
    cgm = np.asarray(cgm, dtype=float)
    mean = np.nanmean(cgm)
    if np.isnan(mean) or mean == 0:
        return float("nan")
    return float(np.nanstd(cgm) / mean)


def glucose_values(data: dict[str, np.ndarray]) -> pd.DataFrame:
    """Summarize sequence-wise glucose metrics (research ``glucose_values`` helper)."""
    funcs = {
        "RMSE": get_rmse,
        "Glucose Variability": get_glucose_variability,
        "TITR": get_tir,
        "TIR": get_tir,
        "TAR": get_tar180,
        "TBR": get_tbr70,
    }
    rows: dict[str, list[float]] = {}
    for name, func in funcs.items():
        if name == "RMSE":
            rows[name] = [
                func(data["pred"][:, seq, 0] - data["true"][:, seq, 0])
                for seq in range(data["true"].shape[1])
            ]
        elif name == "TITR":
            rows[f"{name}_true"] = [
                100 * func(data["true"][:, seq, 0], lim_inf=70, lim_sup=140)
                for seq in range(data["true"].shape[1])
            ]
            rows[f"{name}_pred"] = [
                100 * func(data["pred"][:, seq, 0], lim_inf=70, lim_sup=140)
                for seq in range(data["pred"].shape[1])
            ]
        else:
            rows[f"{name}_true"] = [
                100 * func(data["true"][:, seq, 0]) for seq in range(data["true"].shape[1])
            ]
            rows[f"{name}_pred"] = [
                100 * func(data["pred"][:, seq, 0]) for seq in range(data["pred"].shape[1])
            ]
    return pd.DataFrame(rows)
