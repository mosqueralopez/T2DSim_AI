from t2dsim_ai.create_scenarios import ogtt_scenario
from t2dsim_ai.model_NeuralOGTT import NeuralOGTT
import matplotlib.pyplot as plt
import numpy as np

# Simulation
initCGM = 100

myOGTT = NeuralOGTT()
df_simulation = myOGTT.simulate(ogtt_scenario(initCGM))


# Visualization
state_col = ['state_Gc','state_I']
state_name = ['Glucose in plasma [mg/dL]','Insulin in plasma [mu/L]']

fig, ax = plt.subplots(figsize =(9, 7),ncols=1, nrows=2)

SMALL_SIZE = 15
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = False

plt.suptitle('NeuralOGTT - Initial CGM:'+str(initCGM)+' mg/dL')
for i in [0,1]:
    ax[i].plot(df_simulation['time'],df_simulation[state_col[i]])
    ax[i].set_ylabel(state_name[i])
    ax[i].axvline(15,ls='--',c='gray',alpha=0.5)

    ax[i].tick_params(axis='y', labelsize=15)

    for side in ['right','top']:
        ax[i].spines[side].set_visible(False)

    for side in ['left','bottom']:
        ax[i].spines[side].set(alpha=1)

ax[1].set_xlabel('Time [min]')
plt.show()