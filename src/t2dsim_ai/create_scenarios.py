import numpy as np
import pandas as pd
import datetime
from pathlib import Path
from t2dsim_ai.options import states,inputs_OGTT,inputs_Pop

def ogtt_scenario(init_cgm=110,meal_size=75,sim_time=5*60,t_meal_from_start=15):
    dfInitStates = pd.read_csv(Path(__file__).parent /"models/initSteadyStates.csv").set_index('initCGM')

    df_scenario = pd.DataFrame()
    df_scenario['time'] = np.arange(0,sim_time,5)

    df_scenario[states + inputs_OGTT] = 0.0
    df_scenario.loc[0,'Gc'] = init_cgm
    df_scenario.loc[t_meal_from_start//5,'input_carbs'] = meal_size
    df_scenario.loc[0,states] = dfInitStates.loc[int(init_cgm),states]

    return df_scenario
def meal_scenario(meal_size=75,init_cgm=110,sim_time=5*60, t_meal_from_start=60,hr=80,initial_time = '08:00:00'):
    np.random.seed(0)
    dfInitStates = pd.read_csv(Path(__file__).parent /"models/initSteadyStates.csv")
    base_date = datetime.datetime(2024, 8, 15)
    (h, m, s) = initial_time.split(':')
    initial_time = datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(s))

    df_scenario = pd.DataFrame()
    df_scenario['time'] = pd.date_range(pd.Timestamp(base_date + initial_time),pd.Timestamp(base_date + initial_time + datetime.timedelta(minutes=sim_time)), freq='5 min')

    df_scenario[states + inputs_OGTT +inputs_Pop] = 0.0
    df_scenario['feat_hour_of_day_cos'] = np.cos(2*np.pi*df_scenario['time'].dt.hour/24)
    df_scenario['feat_hour_of_day_cos'] = np.sin(2*np.pi*df_scenario['time'].dt.hour/24)
    df_scenario.loc[0,'Gc'] = init_cgm
    df_scenario.loc[t_meal_from_start//5,'input_carbs'] = meal_size
    df_scenario['input_hr'] = hr+ np.random.normal(0, 10, len(df_scenario))
    df_scenario.loc[0,states] = dfInitStates.loc[int(init_cgm),states]

    return df_scenario