"""Raw CGM / lifestyle CSV processing for digital-twin training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from t2dsim_ai.medications import MEDICATION_SAMPLING_DT_MIN, discrete_oral_med_kernel
from t2dsim_ai.options import inputs, inputs_OGTT, inputs_Pop, rename_cols_dict, states, default_seq_len
from t2dsim_ai.preprocess import scaler_Pop

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
    insulin,
    ts_min,
    t_action_max_min,
    *,
    return_last=False,
):
    """Insulin PK for one delivery [U] on a 5-min grid (Hovorka-style two-compartment IOB)."""
    ts_min = float(ts_min)
    tmax = 55.0
    ke = 0.138
    result_array_size = int(t_action_max_min // ts_min)
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

    ins_availability = i_plasma
    iob = insulin - q4
    if return_last:
        return ins_availability[-1], iob[-1], q1[-1], q2[-1], i_plasma[-1]
    return ins_availability, iob, q1, q2, i_plasma


def basal_micro_bolus_u(daily_u: float, ts_min: float = MEDICATION_SAMPLING_DT_MIN) -> float:
    """Split a daily basal dose into equal 5-min micro-boluses [U/step]."""
    steps_per_day = int(24 * 60 // ts_min)
    return float(daily_u) / steps_per_day


def basal_insulin_plot_uh(daily_u: float, ts_min: float = MEDICATION_SAMPLING_DT_MIN) -> float:
    """Prescribed basal rate for plotting [U/h] (= micro-bolus U/step × 12)."""
    return basal_micro_bolus_u(daily_u, ts_min) * (60.0 / ts_min)


def insulin_deliveries_u_to_uh(
    deliveries_u: np.ndarray,
    *,
    ts_min: float = MEDICATION_SAMPLING_DT_MIN,
    t_action_max_min: int = 500,
) -> np.ndarray:
    """Map insulin deliveries [U/step] to model input ``input_insulin`` [U/h] via IOB."""
    deliveries_u = np.asarray(deliveries_u, dtype=float)
    insulin_on_board = np.zeros_like(deliveries_u, dtype=float)
    for index in np.flatnonzero(deliveries_u > 0):
        bolus = deliveries_u[index]
        _, iob, _, _, _ = calculate_insulin_availability_and_iob_single_delivery(
            bolus, ts_min, t_action_max_min
        )
        end = min(index + len(iob), len(insulin_on_board))
        n_add = end - index
        if n_add > 0:
            insulin_on_board[index:end] += iob[:n_add]
    return insulin_on_board * (60.0 / ts_min)  # U/step IOB -> U/h


def calculate_insulin_availability_and_iob(
    insulin, ts_min=5, t_action_max_min=500, return_last=False
):
    insulin = np.asarray(insulin, dtype=float)
    insulin_availability = np.zeros_like(insulin)
    insulin_on_board = np.zeros_like(insulin)
    insulin_s1 = np.zeros_like(insulin)
    insulin_s2 = np.zeros_like(insulin)
    insulin_i = np.zeros_like(insulin)

    window = int(t_action_max_min // float(ts_min))
    for index in np.where(insulin > 0)[0]:
        iav, iob, s1, s2, i_val = calculate_insulin_availability_and_iob_single_delivery(
            insulin[index], ts_min, t_action_max_min
        )
        if index + window <= insulin.size:
            insulin_availability[index : index + window] += iav
            insulin_on_board[index : index + window] += iob
            insulin_s1[index : index + window] += s1
            insulin_s2[index : index + window] += s2
            insulin_i[index : index + window] += i_val
        else:
            n_tail = insulin.size - index
            insulin_availability[index:] += iav[-n_tail:]
            insulin_on_board[index:] += iob[-n_tail:]
            insulin_s1[index:] += s1[-n_tail:]
            insulin_s2[index:] += s2[-n_tail:]
            insulin_i[index:] += i_val[-n_tail:]

    if return_last:
        return (
            insulin_availability[-1],
            insulin_on_board[-1],
            insulin_s1[-1],
            insulin_s2[-1],
            insulin_i[-1],
        )
    return insulin_availability, insulin_on_board, insulin_s1, insulin_s2, insulin_i


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
    rapid = df["input_insulin"].fillna(0).to_numpy(dtype=float)
    long_acting = np.zeros(len(df), dtype=float)
    if "meds_medicationDose$long_acting_insulin_MOD" in df.columns:
        long_acting = (
            df["meds_medicationDose$long_acting_insulin_MOD"].fillna(0).to_numpy(dtype=float)
        )
    deliveries_u = rapid + long_acting
    df["input_insulin"] = insulin_deliveries_u_to_uh(deliveries_u)
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
    seq_len: int = default_seq_len,
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


def scale_sequence_splits(
    splits: dict,
    seq_len: int,
    scaler_dir: str | Path,
) -> None:
    """Apply a fitted population scaler to sequence splits in place."""
    scaler_dir = Path(scaler_dir)
    for split in splits.values():
        n_seq, _, n_x = split["states"].shape
        flat_x = split["states"].reshape(n_seq * seq_len, n_x)
        flat_u_ogtt = split["inputs_OGTT"].reshape(n_seq * seq_len, 2)
        flat_u_pop = split["inputs_Pop"].reshape(n_seq * seq_len, 12)
        flat_x, flat_u_ogtt, flat_u_pop = scaler_Pop(
            flat_x, flat_u_ogtt, flat_u_pop, str(scaler_dir), train=False
        )
        split["states"] = flat_x.reshape(n_seq, seq_len, n_x)
        split["inputs_OGTT"] = flat_u_ogtt.reshape(n_seq, seq_len, 2)
        split["inputs_Pop"] = flat_u_pop.reshape(n_seq, seq_len, 12)
        split["output"] = split["states"][:, :, [states.index("state_Gc")]]


def resolve_subject_id(df: pd.DataFrame, fallback: str) -> str:
    """Return subject ID in bundled twin format (e.g. ``021-005``)."""
    if "subjectID" in df.columns:
        subject_id = str(df["subjectID"].iloc[0])
        if subject_id.startswith("S") and len(subject_id) > 1:
            return subject_id[1:]
        return subject_id
    return fallback


def build_training_metadata(df: pd.DataFrame, subject_id: str) -> dict:
    """Build demographics and medication flags for ``info.csv``."""
    info: dict = {}
    for col in df.columns:
        if col.startswith("demog_"):
            value = df[col].iloc[0]
            if pd.notna(value):
                info[col] = value

    train_end = int(len(df) * 0.7)
    train_df = df.iloc[:train_end].copy()
    if "timestamp" in train_df.columns:
        train_df["date"] = train_df["timestamp"].dt.date

    med_columns = {
        "sulfonylurea": "input_sulfonylurea",
        "sglt2": "input_sglt2",
        "glp1": "input_glp1",
        "biguanide": "input_biguanide",
        "insulin": "input_insulin",
    }
    for med, col in med_columns.items():
        if col not in df.columns:
            info[f"med_{med}"] = False
            continue
        if med == "glp1":
            info[f"med_{med}"] = bool(df[col].fillna(0).sum() > 0)
            continue
        positive = train_df.loc[train_df[col].fillna(0) > 0]
        if "date" in train_df.columns:
            n_days = positive["date"].nunique()
        else:
            n_days = len(positive)
        info[f"med_{med}"] = n_days > 10

    info["subjectID"] = subject_id
    return info
