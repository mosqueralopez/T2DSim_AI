"""Raw CGM / lifestyle CSV processing for digital-twin training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from t2dsim_ai.medications import MEDICATION_SAMPLING_DT_MIN, discrete_oral_med_kernel
from t2dsim_ai.options import inputs, inputs_OGTT, inputs_Pop, rename_cols_dict, states

REQUIRED_COLUMNS = [
    "timestamp",
    "cgm_value",
    "heartRate_mean",
    "heartRate_min",
    "heartRate_max",
    "heartRate_std",
    "meals_mealSize",
    "meds_medicationDose$rapid_acting_insulin",
    "grouped_meds_medicationGroup$sulfonylurea",
    "grouped_meds_medicationGroup$sglt2",
    "grouped_meds_medicationGroup$glp1",
    "grouped_meds_medicationGroup$biguanide",
]


def calculate_insulin_availability_and_iob_single_delivery(
    insulin, ts_min, t_action_max_min
):
    tmax = 55
    ke = 0.138
    result_array_size = t_action_max_min // ts_min
    q1 = np.zeros(result_array_size)
    q2 = np.zeros(result_array_size)
    i_plasma = np.zeros(result_array_size)
    q4 = np.zeros(result_array_size)

    for tt in range(result_array_size - 1):
        if tt == 0:
            dq1 = -(q1[tt] / tmax) + (insulin / ts_min)
        else:
            dq1 = -(q1[tt] / tmax)
        dq2 = (q1[tt] / tmax) - (q2[tt] / tmax)
        di = (q2[tt] / tmax) - ke * i_plasma[tt]
        dq4 = ke * i_plasma[tt]
        q1[tt + 1] = q1[tt] + dq1 * ts_min
        q2[tt + 1] = q2[tt] + dq2 * ts_min
        i_plasma[tt + 1] = i_plasma[tt] + di * ts_min
        q4[tt + 1] = q4[tt] + dq4 * ts_min
    return i_plasma, insulin - q4


def calculate_insulin_availability_and_iob(insulin, ts_min=5, t_action_max_min=500):
    insulin_on_board = np.zeros_like(insulin, dtype=float)
    insulin_idx = np.where(insulin > 0)[0]
    window = t_action_max_min // ts_min
    for idx in insulin_idx:
        _, iob = calculate_insulin_availability_and_iob_single_delivery(
            insulin[idx], ts_min, t_action_max_min
        )
        end = min(idx + window, len(insulin))
        insulin_on_board[idx:end] += iob[: end - idx]
    return insulin_on_board


def impute_heart_rate(df: pd.DataFrame, hr_columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in hr_columns:
        if col not in df.columns:
            continue
        mean_hr = df[col].replace(0, np.nan).mean()
        if np.isnan(mean_hr):
            mean_hr = df["input_hr"].mean() if "input_hr" in df.columns else 80.0
        df[col] = df[col].fillna(mean_hr).replace(0, mean_hr)
    return df


def process_oral_medications(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "input_biguanide",
        "input_sulfonylurea",
        "input_sglt2",
        "input_glp1",
    ]:
        if col not in df.columns:
            df[col] = 0.0
        doses = df[col].fillna(0).to_numpy(dtype=float)
        acc = np.zeros(len(df), dtype=float)
        med_key = (
            "input_sulfonylurea_glimepiride"
            if col == "input_sulfonylurea"
            else col
        )
        for index in np.flatnonzero(doses != 0):
            kernel = discrete_oral_med_kernel(
                float(doses[index]), med_key, ts_min=MEDICATION_SAMPLING_DT_MIN
            )
            end = min(index + len(kernel), len(df))
            acc[index:end] += kernel[: end - index]
        df[col] = acc
    return df


def process_insulin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "input_insulin" not in df.columns:
        df["input_insulin"] = 0.0
    bolus = df["input_insulin"].fillna(0).to_numpy(dtype=float)
    df["input_insulin"] = calculate_insulin_availability_and_iob(bolus)
    df["input_insulin"] = df["input_insulin"] * 12  # U -> U/h
    return df


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns and "timestamp_local_offset" in df.columns:
        df = df.rename(columns={"timestamp_local_offset": "timestamp"})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def process_raw_csv(path: str | Path) -> pd.DataFrame:
    """Convert example/training CSV into model input columns."""
    df = load_raw_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    rename = {k: v for k, v in rename_cols_dict.items() if k in df.columns}
    df = df.rename(columns=rename)

    for col in inputs:
        if col not in df.columns:
            df[col] = 0.0

    if "sleep_efficiency" in df.columns:
        df["input_sleep"] = df["sleep_efficiency"].fillna(0)
    elif "input_sleep" not in df.columns:
        df["input_sleep"] = 0.0

    df = impute_heart_rate(
        df,
        ["input_hr", "input_hr_min", "input_hr_max", "input_hr_std"],
    )
    df = process_insulin(df)
    df = process_oral_medications(df)

    df["feat_hour_of_day_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["feat_hour_of_day_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["feat_is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(float)

    for col in inputs:
        df[col] = df[col].fillna(0)

    df["Gc"] = df["cgm_value"] if "cgm_value" in df.columns else df["Gc"]
    for st in states:
        if st not in df.columns:
            df[st] = 0.0
    df.loc[0, "state_Gc"] = df.loc[0, "Gc"]

    return df


def _frame_sequences(arr: np.ndarray, seq_len: int, overlap: float) -> np.ndarray:
    step = max(1, int((1 - overlap) * seq_len))
    if len(arr) < seq_len:
        raise ValueError(
            f"Need at least {seq_len} rows for seq_len={seq_len}, got {len(arr)}"
        )
    starts = np.arange(0, len(arr) - seq_len + 1, step)
    return np.stack([arr[s : s + seq_len] for s in starts], axis=0)


def prepare_training_data(
    path: str | Path,
    seq_len: int = 5 * 12,
    overlap: float = 0.98,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
):
    """Return train/validation/test sequence dictionaries for training."""
    df = process_raw_csv(path)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    splits = {
        "train": df.iloc[:train_end],
        "validation": df.iloc[train_end:val_end],
        "test": df.iloc[val_end:],
    }
    out = {}
    for name, part in splits.items():
        if len(part) < seq_len:
            continue
        x = _frame_sequences(part[states].to_numpy(dtype=float), seq_len, overlap)
        u_ogtt = _frame_sequences(
            part[inputs_OGTT].to_numpy(dtype=float), seq_len, overlap
        )
        u_pop = _frame_sequences(part[inputs_Pop].to_numpy(dtype=float), seq_len, overlap)
        y = _frame_sequences(part[["Gc"]].to_numpy(dtype=float), seq_len, overlap)
        out[name] = {
            "states": x,
            "inputs_OGTT": u_ogtt,
            "inputs_Pop": u_pop,
            "output": y,
        }
    return out
