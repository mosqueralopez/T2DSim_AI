from t2dsim_ai.create_scenarios import (
    _ORAL_MED_PLOT,
    _info_to_dict,
    oral_medication_dose_indices,
    scenario_from_twin_info,
)
from t2dsim_ai.init_states import compute_cgm_trend
from t2dsim_ai.model_DTNeuralOGTT import DigitalTwin
from t2dsim_ai.options import ts
import matplotlib.pyplot as plt
import numpy as np

n_digitalTwin = 0
myDigitalTwin = DigitalTwin(n_digitalTwin=n_digitalTwin)
twin_meta = _info_to_dict(myDigitalTwin.digital_twin_Info)
df_scenario = scenario_from_twin_info(myDigitalTwin.digital_twin_Info)
med_dose_idx = oral_medication_dose_indices(df_scenario, twin_meta)

# 30-min CGM history before t=0 used to estimate the initial trend [mg/dL per 5 min].
TREND_MIN = 30
trend_steps = TREND_MIN // ts
cgm_g0 = float(df_scenario.loc[0, "cgm_G0"])
prior_idx = np.arange(-trend_steps, 0)
prior_cgm = cgm_g0 + np.linspace(-4, 0, trend_steps)
cgm_trend = compute_cgm_trend(prior_cgm)

df_simulation = myDigitalTwin.simulate(
    df_scenario,
    use_trend_init=True,
    cgm_trend=cgm_trend,
)
color_AI = "#0072B2"
color_trend = "tab:red"
SMALL_SIZE = 15
plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=SMALL_SIZE)
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = False
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
time = np.arange(len(df_simulation))

ax1.set_title(
    f"DT-NeuralOGTT {myDigitalTwin.subject_id} (twin #{myDigitalTwin.n_digitalTwin})"
)

ax1.plot(
    np.append(prior_idx, 0),
    np.append(prior_cgm, cgm_g0),
    color=color_trend,
    lw=2,
    label="CGM trend (30 min)",
)
ax1.plot(time, df_simulation.Gc, ".", ms=5, c=color_AI, label="Simulated CGM")
ax1.axvline(0, color="k", alpha=0.25, lw=0.8)
ax1.axhspan(70, 180, facecolor="gray", alpha=0.1)
for y in [70, 180, 250]:
    ax1.axhline(y, color="k", alpha=0.2, lw=0.3)
ax1.set_ylim(40, 380)
ax1.set_ylabel("CGM [mg/dL]")
ax1.legend(loc="upper right", frameon=False)

uses_insulin = bool(twin_meta.get("med_insulin", False))
if uses_insulin:
    ax2.plot(
        time,
        df_simulation.basal_insulin_plot_uh,
        c="k",
        lw=1,
        label="Basal insulin (prescribed)",
    )
    insulin_max = float(df_simulation.basal_insulin_plot_uh.iloc[0])
    ax2.set_ylabel("Insulin [U/h]")
    ax2.set_ylim(-1, max(12, insulin_max * 1.2))
else:
    ax2.set_yticks([])

ax2_carbs = ax2.twinx()
color = "tab:red"
ax2_carbs.plot(
    df_simulation.loc[df_simulation.input_carbs != 0].index,
    df_simulation.loc[df_simulation.input_carbs != 0, "input_carbs"],
    "o",
    color=color,
    label="Meals",
)
for col, (marker, label) in _ORAL_MED_PLOT.items():
    if col not in med_dose_idx:
        continue
    idx = med_dose_idx[col]
    y_val = { "input_glp1": 100, "input_sulfonylurea": 100, "input_biguanide": 110, "input_sglt2": 110 }[col]
    ax2_carbs.plot(
        idx,
        y_val,
        marker=marker,
        color="k",
        ms=10,
        linestyle="None",
        label=label,
    )
ax2_carbs.set_ylim(-5, 140)
ax2_carbs.set_yticks([0, 30, 60, 90])
ax2_carbs.set_ylabel("Meal carbs [g]", color=color)
ax2_carbs.tick_params(axis="y", labelcolor=color)
ax2_carbs.legend(loc="upper center", ncol=4, frameon=False, fontsize=11)

ax2_hr = ax2.twinx()
ax2_hr.spines["right"].set_position(("axes", 1.08))
color = "tab:green"
ax2_hr.fill_between(
    time,
    df_simulation.input_hr_min,
    df_simulation.input_hr_max,
    color=color,
    alpha=0.2,
    label="HR min–max",
)
ax2_hr.plot(time, df_simulation.input_hr, lw=0.5, color=color, label="HR mean")
ax2_hr.set_ylabel("Heart rate [BPM]", color=color)
ax2_hr.tick_params(axis="y", labelcolor=color)

ax2.set_xticks(time[:: 3 * 12], df_simulation.time.dt.time.values[:: 3 * 12])
ax2.set_xlabel("Simulation time [hour]")
ax1.set_xlim(-trend_steps - 2, len(df_simulation))
ax2.set_xlim(-trend_steps - 2, len(df_simulation))
plt.tight_layout()

plt.savefig("img/example_DTneuralOGTT_digitalTwin"+str(n_digitalTwin)+".png")
plt.show()
