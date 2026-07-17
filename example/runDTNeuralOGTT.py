from t2dsim_ai.create_scenarios import scenario_from_twin_info
from t2dsim_ai.model_DTNeuralOGTT import DigitalTwin
import matplotlib.pyplot as plt
import numpy as np

n_digitalTwin = 10
myDigitalTwin = DigitalTwin(n_digitalTwin=n_digitalTwin)
df_scenario = scenario_from_twin_info(myDigitalTwin.digital_twin_Info)
df_simulation = myDigitalTwin.simulate(df_scenario)

color_AI = "#0072B2"
SMALL_SIZE = 15
plt.rc("font", size=SMALL_SIZE)
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

ax2.plot(time, df_simulation.input_insulin, c="k", lw=1)
ax2.set_ylabel("Insulin [U/h]")
ax2.set_ylim(-1, 12)

ax2_carbs = ax2.twinx()
color = "tab:red"
ax2_carbs.plot(
    df_simulation.loc[df_simulation.input_carbs != 0].index,
    df_simulation.loc[df_simulation.input_carbs != 0, "input_carbs"],
    "o",
    color=color,
)
ax2_carbs.set_ylabel("Meal carbs [g]", color=color)
ax2_carbs.tick_params(axis="y", labelcolor=color)

ax2_hr = ax2.twinx()
ax2_hr.spines["right"].set_position(("axes", 1.08))
color = "tab:green"
ax2_hr.plot(time, df_simulation.input_hr, lw=0.5, color=color)
ax2_hr.set_ylabel("Heart rate [BPM]", color=color)
ax2_hr.tick_params(axis="y", labelcolor=color)

ax2.set_xticks(time[:: 3 * 12], df_simulation.time.dt.time.values[:: 3 * 12])
ax2.set_xlabel("Simulation time [hour]")
plt.tight_layout()
plt.show()
