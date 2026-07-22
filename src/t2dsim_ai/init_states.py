"""CGM-trend-aware initial state optimization for digital-twin simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.optimize import root

from t2dsim_ai.options import states_name, ts
from t2dsim_ai.preprocess import scale_inverse_state, scale_single_state

STATES_INFO = {name: idx for idx, name in enumerate(states_name)}
N_STATES = len(states_name)


def rate_of_change(cgm: pd.Series) -> float:
    """Estimate CGM trend [mg/dL per 5 min] from a short window."""
    cgm_dropna = cgm.dropna()
    if cgm_dropna.size <= 1:
        return 0.0
    time_min = (cgm_dropna.index - cgm_dropna.index[0]) * ts
    return float(np.polyfit(time_min, cgm_dropna, 1)[0])


def compute_cgm_trend(
    cgm_values: np.ndarray | pd.Series,
    *,
    window: int = 6,
) -> float:
    """Rolling CGM trend at the last sample of ``cgm_values``."""
    series = pd.Series(np.asarray(cgm_values, dtype=float))
    if series.isna().all():
        return 0.0
    trend_series = series.rolling(window=window, min_periods=1).apply(
        lambda x: rate_of_change(series.loc[x.index]),
        raw=False,
    )
    return float(trend_series.iloc[-1])


def build_state_bounds() -> list[tuple[float, float | None]]:
    bounds: list[tuple[float, float | None]] = []
    for state in states_name:
        if state == "I":
            bounds.append(
                (
                    scale_single_state(0, state),
                    scale_single_state(56.456678 * 2.5, state),
                )
            )
        else:
            bounds.append((scale_single_state(0, state), None))
    return bounds


def _steady_state_roots(
    x0_sim,
    nn_solution,
    u_ogtt,
    u_pop,
    fixed_states,
    states_info,
    gc_trend=0.0,
    bounds=None,
    sim_time=2,
):
    with torch.no_grad():
        x0 = torch.zeros((1, N_STATES), dtype=torch.float32)

        for i, key in enumerate(states_info.keys()):
            if key != "Gc":
                try:
                    x0[0, states_info[key]] = float(x0_sim[i])
                except RuntimeError:
                    x0[0, states_info[key]] = bounds[states_info[key]][0]

        if isinstance(fixed_states, dict):
            for key in fixed_states:
                x0[0, fixed_states[key][0]] = fixed_states[key][1]

        u_ogtt = torch.tile(u_ogtt, (sim_time, 1)).reshape(sim_time, 1, -1)
        x = nn_solution(x0, u_ogtt, u_pop, is_DT=False)
        x = x.detach().numpy()

        dx = []
        for key in states_info.keys():
            if key == "Gc":
                gc_pred = scale_inverse_state(
                    x[:, 0, [states_info[key]]], states_info[key]
                ).reshape(-1)
                dx.append(gc_trend - np.polyfit(ts * np.arange(sim_time), gc_pred, 1)[0])
            else:
                dx.append(
                    0
                    - np.polyfit(ts * np.arange(sim_time), x[:, 0, states_info[key]], 1)[0]
                )
        dx = np.array(dx, dtype=np.float16)
        for idx, val in enumerate(dx):
            if np.abs(val) < 1e-4:
                dx[idx] = 0
        return dx


def get_initial_c1c2(nn_solution, x0_sim, u_ogtt, u_pop, bounds):
    x0_sim = x0_sim.clone()
    x0_sim[:, STATES_INFO["C1"]] = 0
    x0_sim[:, STATES_INFO["C2"]] = 0

    x0_vec = x0_sim.reshape(-1).detach().numpy()
    subset = {key: STATES_INFO[key] for key in ["C1", "C2"]}
    result = root(
        _steady_state_roots,
        x0_vec[[STATES_INFO["C1"], STATES_INFO["C2"]]],
        args=(
            nn_solution,
            u_ogtt,
            u_pop,
            None,
            subset,
            0.0,
            bounds,
            12,
        ),
    )
    if bounds is not None:
        for i, key in enumerate(["C1", "C2"]):
            if result.x[i] < bounds[STATES_INFO[key]][0] or result.x[i] > 1e4:
                result.x[i] = bounds[STATES_INFO[key]][0]
    return {"C1_0": result.x[0], "C2_0": result.x[1]}


def get_initial_states(
    nn_solution,
    x0_sim,
    u_ogtt,
    u_pop,
    gc_trend=0.0,
    bounds=None,
):
    """Optimize initial hidden states given CGM level and trend."""
    fixed_states = {
        "C1": [STATES_INFO["C1"], x0_sim[STATES_INFO["C1"]].detach()],
        "C2": [STATES_INFO["C2"], x0_sim[STATES_INFO["C2"]].detach()],
        "Gc": [STATES_INFO["Gc"], x0_sim[STATES_INFO["Gc"]].detach()],
    }

    x0_vec = x0_sim.reshape(-1).detach().numpy().astype(np.float64)
    subset = {key: STATES_INFO[key] for key in ["Gc", "Ge", "Ie", "I"]}
    result = root(
        _steady_state_roots,
        x0_vec[[STATES_INFO["Gc"], STATES_INFO["Ge"], STATES_INFO["Ie"], STATES_INFO["I"]]],
        tol=1e-4,
        method="hybr",
        args=(
            nn_solution,
            u_ogtt,
            u_pop,
            fixed_states,
            subset,
            gc_trend,
            bounds,
        ),
    )
    for val in result.x:
        if val > 1e5:
            result = root(
                _steady_state_roots,
                x0_vec[
                    [
                        STATES_INFO["Gc"],
                        STATES_INFO["Ge"],
                        STATES_INFO["Ie"],
                        STATES_INFO["I"],
                    ]
                ],
                tol=1e-4,
                args=(
                    nn_solution,
                    u_ogtt,
                    u_pop,
                    fixed_states,
                    subset,
                    0.0,
                    bounds,
                ),
            )
            break

    if bounds is not None:
        if result.x[3] < bounds[STATES_INFO["I"]][0]:
            result.x[3] = bounds[STATES_INFO["I"]][0]
        elif result.x[3] > bounds[STATES_INFO["I"]][1]:
            result.x[3] = bounds[STATES_INFO["I"]][1]

    fixed_states["Ge"] = [STATES_INFO["Ge"], result.x[1]]
    fixed_states["Ie"] = [STATES_INFO["Ie"], result.x[2]]
    fixed_states["I"] = [STATES_INFO["I"], result.x[3]]

    x0 = torch.zeros(len(states_name), dtype=torch.float32)
    for key in fixed_states:
        x0[STATES_INFO[key]] = float(fixed_states[key][1])
    return x0


def resolve_cgm_g0(df_scenario, init_cgm: float | None = None) -> float:
    """Return initial CGM in mg/dL for trend-aware state initialization."""
    if init_cgm is not None:
        return float(init_cgm)
    if "cgm_G0" in df_scenario.columns and pd.notna(df_scenario.loc[0, "cgm_G0"]):
        return float(df_scenario.loc[0, "cgm_G0"])
    if "Gc" in df_scenario.columns and pd.notna(df_scenario.loc[0, "Gc"]):
        return float(df_scenario.loc[0, "Gc"])
    scaled_gc = float(df_scenario.loc[0, "state_Gc"])
    return float(
        scale_inverse_state(np.array([[scaled_gc]]), STATES_INFO["Gc"])[0, 0]
    )


def build_initial_state(
    nn_solution,
    init_cgm_mgdl: float,
    u_ogtt_step,
    gc_trend: float = 0.0,
    *,
    device=torch.device("cpu"),
):
    """Build trend-aware initial state from CGM [mg/dL] and first-step OGTT inputs."""
    bounds = build_state_bounds()
    x0_base = torch.tensor(
        [[scale_single_state(0, state) for state in states_name]],
        dtype=torch.float32,
        device=device,
    )
    u0 = u_ogtt_step.reshape(1, -1).to(device)

    with torch.no_grad():
        c1c2 = get_initial_c1c2(nn_solution, x0_base, u0, None, bounds)
        x0_base[:, STATES_INFO["C1"]] = c1c2["C1_0"]
        x0_base[:, STATES_INFO["C2"]] = c1c2["C2_0"]
        x0_base[:, STATES_INFO["Gc"]] = scale_single_state(init_cgm_mgdl, "Gc")
        x0 = get_initial_states(
            nn_solution, x0_base[0], u0, None, gc_trend, bounds
        )
    return x0.reshape(1, -1).to(device)
