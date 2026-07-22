from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from t2dsim_ai.data_processing import (
    basal_insulin_plot_uh,
    basal_micro_bolus_u,
    insulin_deliveries_u_to_uh,
)
from t2dsim_ai.medications import MEDICATION_SAMPLING_DT_MIN, discrete_oral_med_kernel
from t2dsim_ai.options import inputs, inputs_OGTT, inputs_Pop, states, ts

_MODELS_DIR = Path(__file__).parent / "models"
_DEFAULT_MEALS = [(60, 75), (240, 90), (660, 30)]  # (minutes from start, carbs in g)
_DEFAULT_ORAL_DOSE = 1.0
_DEFAULT_ORAL_TIME_MIN = 8 * 60  # 08:00 from midnight; adjusted by initial_time
_DEFAULT_BASAL_INSULIN_DAILY_U = 54.0  # U/day
_SULFONYLUREA_GLIMEPIRIDE_IDS = {5, 28, 42, 47, 56}
_SULFONYLUREA_GLIPIZIDE_IDS = {31, 34, 45, 46}
_ORAL_MED_PLOT = {
    "input_glp1": ("X", "GLP-1"),
    "input_sulfonylurea": ("h", "Sulfonylurea"),
    "input_biguanide": ("p", "Biguanide"),
    "input_sglt2": ("s", "SGLT-2"),
}

def _subject_numeric_id(subject_id) -> int | None:
    if subject_id is None or (isinstance(subject_id, float) and np.isnan(subject_id)):
        return None
    try:
        return int(str(subject_id).split("-")[-1])
    except ValueError:
        return None


def _sulfonylurea_med_key(subject_id) -> str:
    sid = _subject_numeric_id(subject_id)
    if sid in _SULFONYLUREA_GLIPIZIDE_IDS:
        return "input_sulfonylurea_glipizide"
    return "input_sulfonylurea_glimepiride"


def _oral_dose_index(initial_time: str) -> int:
    h0, m0, _ = initial_time.split(":")
    return max(0, (_DEFAULT_ORAL_TIME_MIN - (int(h0) * 60 + int(m0))) // ts)


def active_medications(meta: dict) -> list[str]:
    """Return active medication input columns for a twin metadata dict."""
    active: list[str] = []
    if bool(meta.get("med_insulin", False)):
        active.append("input_insulin")
    if bool(meta.get("med_glp1", False)):
        active.append("input_glp1")
    if bool(meta.get("med_sulfonylurea", False)):
        active.append("input_sulfonylurea")
    if bool(meta.get("med_biguanide", False)):
        active.append("input_biguanide")
    if bool(meta.get("med_sglt2", False)):
        active.append("input_sglt2")
    return active


def oral_medication_dose_indices(
    df_scenario: pd.DataFrame,
    meta: dict,
    *,
    initial_time: str = "08:00:00",
) -> dict[str, int]:
    """Map active oral medication columns to their once-daily dose row index."""
    dose_index = _oral_dose_index(initial_time)
    indices: dict[str, int] = {}
    for col in active_medications(meta):
        if col == "input_insulin":
            continue
        if 0 <= dose_index < len(df_scenario):
            indices[col] = dose_index
    return indices


def _info_to_dict(info: pd.DataFrame | str | Path) -> dict:
    if not isinstance(info, pd.DataFrame):
        info = pd.read_csv(info)
    if "Unnamed: 0" in info.columns:
        keys = info["Unnamed: 0"].astype(str)
        values = info.iloc[:, 1]
    else:
        keys = info.iloc[:, 0].astype(str)
        values = info.iloc[:, 1]
    out = {}
    for key, value in zip(keys, values):
        if isinstance(value, str):
            if value.lower() == "true":
                out[key] = True
            elif value.lower() == "false":
                out[key] = False
            else:
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = value
        else:
            out[key] = value
    return out


def _apply_oral_med_trace(
    series: np.ndarray,
    dose: float,
    med_key: str,
    dose_index: int,
) -> None:
    kernel = discrete_oral_med_kernel(dose, med_key, ts_min=MEDICATION_SAMPLING_DT_MIN)
    end = min(dose_index + len(kernel), len(series))
    series[dose_index:end] += kernel[: end - dose_index]


def ogtt_scenario(init_cgm=110, meal_size=75, sim_time=5 * 60, t_meal_from_start=15):
    df_init = pd.read_csv(_MODELS_DIR / "initSteadyStates.csv").set_index("initCGM")

    df_scenario = pd.DataFrame()
    df_scenario["time"] = np.arange(0, sim_time, ts)

    df_scenario[states + inputs_OGTT] = 0.0
    df_scenario["cgm_G0"] = np.nan
    df_scenario.loc[0, "cgm_G0"] = init_cgm
    df_scenario.loc[t_meal_from_start // ts, "input_carbs"] = meal_size
    df_scenario.loc[0, states] = df_init.loc[int(init_cgm), states].values

    return df_scenario


def digitalTwin_scenario(
    meal_size_array=(75,),
    meal_time_fromStart_array=(60,),
    init_cgm=110,
    sim_time=5 * 60,
    hr=80,
    initial_time="08:00:00",
    bedtime=13 * 60,
    sleep_duration=8,
):
    np.random.seed(0)
    df_init = pd.read_csv(_MODELS_DIR / "initSteadyStates.csv").set_index("initCGM")

    base_date = datetime.datetime(2024, 8, 15)
    h, m, s = initial_time.split(":")
    start_delta = datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(s))

    df_scenario = pd.DataFrame()
    df_scenario["time"] = pd.date_range(
        pd.Timestamp(base_date + start_delta),
        pd.Timestamp(base_date + start_delta + datetime.timedelta(minutes=sim_time)),
        freq=f"{ts} min",
    )

    df_scenario[states + inputs_OGTT + inputs_Pop] = 0.0
    df_scenario["cgm_G0"] = np.nan
    df_scenario["feat_hour_of_day_cos"] = np.cos(
        2 * np.pi * df_scenario["time"].dt.hour / 24
    )
    df_scenario["feat_hour_of_day_sin"] = np.sin(
        2 * np.pi * df_scenario["time"].dt.hour / 24
    )
    df_scenario["feat_is_weekend"] = (df_scenario["time"].dt.dayofweek >= 5).astype(
        float
    )
    df_scenario.loc[0, "cgm_G0"] = init_cgm
    df_scenario.loc[np.array(meal_time_fromStart_array) // ts, "input_carbs"] = (
        meal_size_array
    )

    hr_signal = hr + np.random.normal(0, 5, len(df_scenario))
    df_scenario["input_hr"] = hr_signal
    sleep_start = bedtime // ts
    sleep_end = (bedtime + sleep_duration * 60) // ts
    df_scenario.loc[sleep_start:sleep_end, "input_sleep"] = 1.0
    df_scenario.loc[sleep_start:sleep_end, "input_hr"] = (
        hr - 10 + np.random.normal(0, 1, sleep_end - sleep_start + 1)
    )

    hr_series = pd.Series(df_scenario["input_hr"].values)
    df_scenario["input_hr_min"] = hr_series.rolling(3, min_periods=1).min().values
    df_scenario["input_hr_max"] = hr_series.rolling(3, min_periods=1).max().values
    df_scenario["input_hr_std"] = hr_series.rolling(3, min_periods=1).std().fillna(0).values

    df_scenario.loc[0, states] = df_init.loc[int(init_cgm), states].values
    return df_scenario


def scenario_from_twin_info(
    info: pd.DataFrame | str | Path,
    *,
    sim_time: int = 24 * 60,
    initial_time: str = "08:00:00",
    init_cgm: float | None = None,
    meal_schedule: list | None = None,
    bedtime: int = 13 * 60,
    sleep_duration: int = 8,
    basal_insulin_daily_u: float = _DEFAULT_BASAL_INSULIN_DAILY_U,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a one-day simulation scenario from a digital twin ``info.csv``."""
    meta = _info_to_dict(info)
    np.random.seed(seed)

    hr_baseline = float(meta.get("demog_heartRateBaseline", 80))
    if init_cgm is None:
        init_cgm = 110.0

    base_date = datetime.datetime(2024, 8, 15)
    h, m, s = initial_time.split(":")
    start_delta = datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(s))

    n_steps = sim_time // ts
    df_scenario = pd.DataFrame()
    df_scenario["time"] = pd.date_range(
        pd.Timestamp(base_date + start_delta),
        periods=n_steps,
        freq=f"{ts} min",
    )
    df_scenario[states + inputs] = 0.0
    df_scenario["cgm_G0"] = np.nan

    df_scenario["feat_hour_of_day_cos"] = np.cos(
        2 * np.pi * df_scenario["time"].dt.hour / 24
    )
    df_scenario["feat_hour_of_day_sin"] = np.sin(
        2 * np.pi * df_scenario["time"].dt.hour / 24
    )
    df_scenario["feat_is_weekend"] = (df_scenario["time"].dt.dayofweek >= 5).astype(
        float
    )

    df_init = pd.read_csv(_MODELS_DIR / "initSteadyStates.csv").set_index("initCGM")
    init_key = int(np.clip(init_cgm, 40, 400))
    df_scenario.loc[0, "cgm_G0"] = init_cgm
    df_scenario.loc[0, states] = df_init.loc[init_key, states].values

    meals = meal_schedule if meal_schedule is not None else _DEFAULT_MEALS
    for meal_time, meal_size in meals:
        idx = meal_time // ts
        if 0 <= idx < len(df_scenario):
            df_scenario.loc[idx, "input_carbs"] = meal_size

    hr_signal = hr_baseline + np.random.normal(0, 5, len(df_scenario))
    sleep_start = bedtime // ts
    sleep_end = min((bedtime + sleep_duration * 60) // ts, len(df_scenario) - 1)
    df_scenario.loc[sleep_start:sleep_end, "input_sleep"] = 1.0
    hr_signal[sleep_start : sleep_end + 1] = (
        hr_baseline - 10 + np.random.normal(0, 1, sleep_end - sleep_start + 1)
    )
    df_scenario["input_hr"] = hr_signal

    hr_series = pd.Series(df_scenario["input_hr"].values)
    df_scenario["input_hr_min"] = hr_series.rolling(3, min_periods=1).min().values
    df_scenario["input_hr_max"] = hr_series.rolling(3, min_periods=1).max().values
    df_scenario["input_hr_std"] = hr_series.rolling(3, min_periods=1).std().fillna(0).values

    oral_dose_index = _oral_dose_index(initial_time)
    subject_id = meta.get("subjectID")

    df_scenario["basal_insulin_plot_uh"] = 0.0
    if bool(meta.get("med_insulin", False)):
        micro_bolus_u = basal_micro_bolus_u(basal_insulin_daily_u, ts_min=ts)
        deliveries_u = np.full(n_steps, micro_bolus_u, dtype=float)
        df_scenario["input_insulin"] = insulin_deliveries_u_to_uh(
            deliveries_u, ts_min=ts
        )
        df_scenario["basal_insulin_plot_uh"] = basal_insulin_plot_uh(
            basal_insulin_daily_u, ts_min=ts
        )

    if bool(meta.get("med_biguanide", False)):
        _apply_oral_med_trace(
            df_scenario["input_biguanide"].values,
            _DEFAULT_ORAL_DOSE,
            "input_biguanide",
            oral_dose_index,
        )
    if bool(meta.get("med_glp1", False)):
        _apply_oral_med_trace(
            df_scenario["input_glp1"].values,
            _DEFAULT_ORAL_DOSE,
            "input_glp1",
            oral_dose_index,
        )
    if bool(meta.get("med_sglt2", False)):
        _apply_oral_med_trace(
            df_scenario["input_sglt2"].values,
            _DEFAULT_ORAL_DOSE,
            "input_sglt2",
            oral_dose_index,
        )
    if bool(meta.get("med_sulfonylurea", False)):
        _apply_oral_med_trace(
            df_scenario["input_sulfonylurea"].values,
            _DEFAULT_ORAL_DOSE,
            _sulfonylurea_med_key(subject_id),
            oral_dose_index,
        )

    return df_scenario
