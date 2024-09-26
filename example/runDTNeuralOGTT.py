from t2dsim_ai.create_scenarios import digitalTwin_scenario
from t2dsim_ai.model_DTNeuralOGTT import DigitalTwin
import matplotlib.pyplot as plt
import numpy as np

myDigitalTwin = DigitalTwin()
# print(myDigitalTwin.digital_twin_Info)

df_simulation = myDigitalTwin.simulate(
    digitalTwin_scenario(
        meal_size_array=[75, 90, 30],  # g
        meal_time_fromStart_array=[60, 60 * 4, 60 * 11],  # min from start of simulation
        init_cgm=110,  # mg/dL
        sim_time=24 * 60,
        hr=80,  # int or array of len sim_time
        initial_time="08:00:00",
    )
)

color_AI = "#0072B2"
SMALL_SIZE = 15
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = False
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
time = np.arange(len(df_simulation))


ax1.set_title("DT-NeuralOGTT digital twin #" + str(myDigitalTwin.n_digitalTwin))

ax1.plot(time, df_simulation.Gc, ".", ms=5, c=color_AI)
ax1.axhspan(70, 180, facecolor="gray", alpha=0.1)

for y in [70, 180, 250]:
    ax1.axhline(y, color="k", alpha=0.2, lw=0.3)

ax1.set_ylim(40, 380)
ax1.set_ylabel("CGM [mg/dL]")

for location in ["left", "right", "top", "bottom"]:
    ax1.spines[location].set_linewidth(0.1)

ax2.plot(time, df_simulation.input_insulin, c="k", lw=1)
ax2.set_ylabel("Insulin [U/h]")
ax2.set_ylim(-1, 12)

for location in ["left", "right", "top", "bottom"]:
    ax2.spines[location].set_linewidth(0.1)
ax2.spines["top"].set_linewidth(0.5)

ax2_carbs = ax2.twinx()
color = "tab:red"
ax2_carbs.plot(
    df_simulation.loc[df_simulation.input_carbs != 0].index,
    df_simulation.loc[df_simulation.input_carbs != 0, "input_carbs"],
    "o",
    color=color,
)


if len(df_simulation.loc[df_simulation.input_biguanide > 0, "input_biguanide"]) > 0:
    ax2_carbs.plot(
        df_simulation.loc[df_simulation.input_biguanide > 0].index,
        90
        * np.ones(
            len(df_simulation.loc[df_simulation.input_biguanide > 0, "input_biguanide"])
        ),
        "kp",
        ms=10,
        label="Biguanide",
    )

if len(df_simulation.loc[df_simulation.input_sglt2 > 0, "input_sglt2"]) > 0:
    ax2_carbs.plot(
        df_simulation.loc[df_simulation.input_sglt2 != 0].index,
        90
        * np.ones(
            len(df_simulation.loc[df_simulation.input_sglt2 != 0, "input_sglt2"])
        ),
        "ks",
        markerfacecolor="white",
        label="SGLT-2",
    )

if len(df_simulation.loc[df_simulation.input_glp1 > 0, "input_glp1"]) > 0:
    ax2_carbs.plot(
        df_simulation.loc[df_simulation.input_glp1 > 0].index,
        100
        * np.ones(len(df_simulation.loc[df_simulation.input_glp1 > 0, "input_glp1"])),
        "kX",
        ms=10,
        label="GLP-1",
    )  # X all results

if (
    len(df_simulation.loc[df_simulation.input_sulfonylurea > 0, "input_sulfonylurea"])
    > 0
):
    ax2_carbs.plot(
        df_simulation.loc[df_simulation.input_sulfonylurea != 0].index,
        100
        * np.ones(
            len(
                df_simulation.loc[
                    df_simulation.input_sulfonylurea != 0, "input_sulfonylurea"
                ]
            )
        ),
        "kh",
        markerfacecolor="white",
        ms=8,
        label="Sulfonylurea",
    )

ax2_carbs.set_ylim(-5, 140)
ax2_carbs.set_yticks([0, 30, 60, 90], [0, 30, 60, 90])

ax2_carbs.legend(loc=9, ncol=4, frameon=False)
ax2_carbs.set_ylabel("Meal carbs [g]", color=color)
ax2_carbs.tick_params(axis="y", labelcolor=color)
ax2_carbs.spines["right"].set_position(("axes", 0))

for location in ["left", "right", "top", "bottom"]:
    ax2_carbs.spines[location].set_linewidth(0)
ax2_hr = ax2.twinx()
color = "tab:green"
ax2_hr.plot(time, df_simulation.input_hr, lw=0.5, color=color)
ax2_hr.set_ylabel("Heart rate [BPM]", color=color)
ax2_hr.set_ylim(df_simulation.input_hr.min() - 2, df_simulation.input_hr.max() + 20)

ax2_hr.tick_params(axis="y", labelcolor=color)
for location in ["left", "right", "top", "bottom"]:
    ax2_hr.spines[location].set_linewidth(0)

ax2_pa = ax2.twinx()
color = "tab:purple"
ax2_pa.plot(time, df_simulation.input_sleep, lw=1, color=color)

for location in ["left", "right", "top", "bottom"]:
    ax2_pa.spines[location].set_linewidth(0)

ax2_pa.set_ylabel("Sleep efficiency", color=color, labelpad=-35)
ax2_pa.tick_params(axis="y", labelcolor=color)
ax2_pa.set_yticks([0, 1], [0, 1])
ax2_pa.tick_params(axis="y", direction="in", pad=-15)
ax2_pa.set_ylim(-0.2, 1.5)

ax2.set_xticks(time[:: 3 * 12], df_simulation.time.dt.time.values[:: 3 * 12])
ax2.set_xlabel("Simulation time [hour]")
ax2.set_xlim(time[0] - 12 * 2, time[-1] + 12 * 2)

plt.subplots_adjust(hspace=0)
plt.tight_layout()
plt.show()
# plt.savefig('example_DTneuralOGTT_digitaltwin#'+str(myDigitalTwin.n_digitalTwin)+'.png',dpi=500, bbox_inches='tight')
