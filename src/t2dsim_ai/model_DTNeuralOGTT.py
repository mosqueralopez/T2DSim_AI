import pandas as pd

from t2dsim_ai.options import (
    states,
    inputs_OGTT,
    inputs_Pop,
    n_neurons_ogtt,
    n_neuron_ind,
)
from t2dsim_ai.ss_simulator import ForwardEulerSimulator
from t2dsim_ai.model_NeuralOGTT import CGMOHSUSimStateSpaceModel_T2DOGTT, WeightClipper
from t2dsim_ai.preprocess import scaler_inverse, scaler_Pop
import torch
from pathlib import Path
import torch.nn as nn
import numpy as np
import os


class CGMOHSUSimStateSpaceModel_T2D(nn.Module):
    def __init__(self, hidden_compartments, init_small=True):
        super(CGMOHSUSimStateSpaceModel_T2D, self).__init__()

        # C1'
        layers_model = []
        for i in range(len(hidden_compartments["C1"]) - 2):
            layers_model.append(
                nn.Linear(
                    hidden_compartments["C1"][i], hidden_compartments["C1"][i + 1]
                )
            )
            layers_model.append(nn.ReLU())

        layers_model.append(nn.Linear(hidden_compartments["C1"][-2], 1))
        self.C1_pop = nn.Sequential(*layers_model)

        # C2'
        layers_model = []
        for i in range(len(hidden_compartments["C2"]) - 2):
            layers_model.append(
                nn.Linear(
                    hidden_compartments["C2"][i], hidden_compartments["C2"][i + 1]
                )
            )
            layers_model.append(nn.ReLU())

        layers_model.append(nn.Linear(hidden_compartments["C2"][-2], 1))
        self.C2_pop = nn.Sequential(*layers_model)

        # Gc'
        layers_model = []
        for i in range(len(hidden_compartments["Gc"]) - 2):
            layers_model.append(
                nn.Linear(
                    hidden_compartments["Gc"][i], hidden_compartments["Gc"][i + 1]
                )
            )
            layers_model.append(nn.ReLU())

        layers_model.append(nn.Linear(hidden_compartments["Gc"][-2], 1))
        self.Gc_pop = nn.Sequential(*layers_model)

        # Ge'
        layers_model = []
        for i in range(len(hidden_compartments["Ge"]) - 2):
            layers_model.append(
                nn.Linear(
                    hidden_compartments["Ge"][i], hidden_compartments["Ge"][i + 1]
                )
            )
            layers_model.append(nn.ReLU())

        layers_model.append(nn.Linear(hidden_compartments["Ge"][-2], 1))
        self.Ge_pop = nn.Sequential(*layers_model)

        # Ie'
        layers_model = []
        for i in range(len(hidden_compartments["Ie"]) - 2):
            layers_model.append(
                nn.Linear(
                    hidden_compartments["Ie"][i], hidden_compartments["Ie"][i + 1]
                )
            )
            layers_model.append(nn.ReLU())

        layers_model.append(nn.Linear(hidden_compartments["Ie"][-2], 1))
        self.Ie_pop = nn.Sequential(*layers_model)

        # I'
        layers_model = []
        for i in range(len(hidden_compartments["I"]) - 2):
            layers_model.append(
                nn.Linear(hidden_compartments["I"][i], hidden_compartments["I"][i + 1])
            )
            layers_model.append(nn.ReLU())

        layers_model.append(nn.Linear(hidden_compartments["I"][-2], 1))
        self.I_pop = nn.Sequential(*layers_model)

        clipper = WeightClipper()

        if init_small:
            networks = {
                "C1": self.C1_pop,
                "C2": self.C2_pop,
                "Gc": self.Gc_pop,
                "Ge": self.Ge_pop,
                "Ie": self.Ie_pop,
                "I": self.I_pop,
            }

            for key in networks.keys():
                net = networks[key]
                for m in net.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0, std=1e-4)
                        nn.init.constant_(m.bias, val=0)

                net.apply(clipper)

    def forward(self, in_x, u_ogtt, u_pop):
        C1, C2, Gc, Ge, Ie, I = (
            in_x[..., [0]],
            in_x[..., [1]],
            in_x[..., [2]],
            in_x[..., [3]],
            in_x[..., [4]],
            in_x[..., [5]],
        )

        hr, sleep, sulfo, sglt2, glp1, biguanide, is_weekend, hour_cos, hour_sin = (
            u_pop[..., [0]],
            u_pop[..., [1]],
            u_pop[..., [2]],
            u_pop[..., [3]],
            u_pop[..., [4]],
            u_pop[..., [5]],
            u_pop[..., [6]],
            u_pop[..., [7]],
            u_pop[..., [8]],
        )
        u_I, u_carbs = u_ogtt[..., [0]], u_ogtt[..., [1]]

        # C1'(C1, u_carbs, u_glp1)
        inp = torch.cat((C1, u_carbs, glp1), -1)
        dC1_pop = self.C1_pop(inp)

        # C2'(C1,C2)
        inp = torch.cat((C1, C2), -1)
        dC2_pop = self.C2_pop(inp)

        # Gc'(Gc,C2,Ie,HR, sleep, is_weekend, hour_cos, hour_sin, biguanide, sglt2)
        inp = torch.cat(
            (Gc, C2, Ie, hr, sleep, is_weekend, hour_cos, hour_sin, biguanide, sglt2),
            -1,
        )
        dGc_pop = self.Gc_pop(inp)

        # Ge'(Ge, Gc)
        inp = torch.cat((Ge, Gc), -1)
        dGe_pop = self.Ge_pop(inp)

        # Ie'(Ie,I)
        inp = torch.cat((Ie, I), -1)
        dIe_pop = self.Ie_pop(inp)

        # I'(I,Ge,C2,uI)
        inp = torch.cat((I, Ge, C2, u_I, glp1, sulfo), -1)
        dI_pop = self.I_pop(inp)

        return {
            "C1": dC1_pop,
            "C2": dC2_pop,
            "Gc": dGc_pop,
            "Ge": dGe_pop,
            "Ie": dIe_pop,
            "I": dI_pop,
        }


class DigitalTwin:
    def __init__(self, n_digitalTwin=0, device=torch.device("cpu"), ts=5):
        self.ts = ts
        self.device = device

        self.n_digitalTwin = n_digitalTwin
        digitalTwin_list = [
            f.path
            for f in os.scandir(Path(__file__).parent / "models/DigitalTwins/")
            if f.is_dir()
        ]
        digitalTwin_list.sort()
        self.digital_twin_folder = digitalTwin_list[self.n_digitalTwin]

        self.digital_twin_Info = pd.read_csv(self.digital_twin_folder + "/info.csv")
        self.setup_simulator()

    def setup_simulator(self):
        ### OGTT Model
        ss_ogtt_model = CGMOHSUSimStateSpaceModel_T2DOGTT(n_feat=n_neurons_ogtt)
        ss_ogtt_model.to(self.device)
        ss_ogtt_model.load_state_dict(
            torch.load(
                Path(__file__).parent / "models/OGTT_productionModel_6compartments.pt"
            )
        )

        ### Individual Model

        ss_individual_model = CGMOHSUSimStateSpaceModel_T2D(
            hidden_compartments=n_neuron_ind
        )
        ss_individual_model.to(self.device)
        ss_individual_model.load_state_dict(
            torch.load(self.digital_twin_folder + "/model.pt")
        )

        self.nn_solution = ForwardEulerSimulator(
            ss_model=ss_ogtt_model, ss_DT_model=ss_individual_model, ts=self.ts
        )

    def prepare_data(self, df_scenario):

        _, u_ogtt, u_pop = scaler_Pop(
            df_scenario[states].values,
            df_scenario[inputs_OGTT].values,
            df_scenario[inputs_Pop].values,
            self.digital_twin_folder,
            False,
        )

        u_ogtt = u_ogtt.reshape(-1, u_ogtt.shape[0], u_ogtt.shape[1])
        u_pop = u_pop.reshape(-1, u_pop.shape[0], u_pop.shape[1])

        sim_time_test = len(df_scenario)
        batch_start = np.array([0], dtype=np.int)
        batch_idx = batch_start[:, np.newaxis] + np.arange(sim_time_test)

        x0_est = torch.tensor(
            df_scenario.loc[0, states].astype(float).values.reshape(1, -1),
            dtype=torch.float32,
        ).to(self.device)
        u_ogtt = torch.tensor(u_ogtt[[0], batch_idx.T], dtype=torch.float32).to(
            self.device
        )
        u_pop = torch.tensor(u_pop[[0], batch_idx.T], dtype=torch.float32).to(
            self.device
        )

        return x0_est, u_ogtt[:, [0], :], u_pop[:, [0], :]

    def simulate(self, df_scenario_original, is_DT=True):
        # Prepare data
        df_scenario = df_scenario_original.copy()
        x0_est, u_ogtt, u_pop = self.prepare_data(df_scenario)
        with torch.no_grad():
            x_sim = self.nn_solution(x0_est, u_ogtt, u_pop, is_DT=is_DT)
            df_scenario[states] = scaler_inverse(
                x_sim[:, 0, :].to("cpu").detach().numpy()
            )

        df_scenario["Gc"] = df_scenario["state_Gc"]

        return df_scenario
