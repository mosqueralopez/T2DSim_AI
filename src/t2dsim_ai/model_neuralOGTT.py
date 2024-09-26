from t2dsim_ai.options import states, inputs_OGTT, n_neurons_ogtt
from t2dsim_ai.ss_simulator import ForwardEulerSimulator
from t2dsim_ai.preprocess import scaler_inverse, scaler_OGTT
import torch
from pathlib import Path
import torch.nn as nn
import numpy as np


class WeightClipper(object):
    def __init__(self, min=-1, max=1):

        self.min = min
        self.max = max

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, "weight"):
            w = module.weight.data
            w = w.clamp(self.min, self.max)


class CGMOHSUSimStateSpaceModel_T2DOGTT(nn.Module):
    def __init__(self, n_feat=None, scale_dx=1.0, init_small=True):
        super(CGMOHSUSimStateSpaceModel_T2DOGTT, self).__init__()
        self.n_feat = n_feat
        self.scale_dx = scale_dx

        # NN1(c1,u_carbs)
        self.net_dC1 = nn.Sequential(
            nn.Linear(2, self.n_feat["C1"]),
            nn.ReLU(),
            nn.Linear(self.n_feat["C1"], 1),
        )
        # NN2(c1,c2)
        self.net_dC2 = nn.Sequential(
            nn.Linear(2, self.n_feat["C2"]),
            nn.ReLU(),
            nn.Linear(self.n_feat["C2"], 1),
        )
        # NN3(Gc,C2,Ie)
        self.net_dGc = nn.Sequential(
            nn.Linear(3, self.n_feat["Gc"]),
            nn.ReLU(),
            nn.Linear(self.n_feat["Gc"], 1),
        )

        # NN4(Ge,Gc)
        self.net_dGe = nn.Sequential(
            nn.Linear(2, self.n_feat["Ge"]),
            nn.ReLU(),
            nn.Linear(self.n_feat["Ge"], 1),
        )

        # NN5(Ie,I)
        self.net_dIe = nn.Sequential(
            nn.Linear(2, self.n_feat["Ie"]),
            nn.ReLU(),
            nn.Linear(self.n_feat["Ie"], 1),
        )
        # NN6(I,Ge,C2,uI)
        self.net_dI = nn.Sequential(
            nn.Linear(4, self.n_feat["I"]),
            nn.ReLU(),
            nn.Linear(self.n_feat["I"], 1),
        )

        clipper = WeightClipper()

        # Small initialization is better for multi-step methods
        if init_small:

            networks = {
                "C1": self.net_dC1,
                "C2": self.net_dC2,
                "Gc": self.net_dGc,
                "Ge": self.net_dGe,
                "Ie": self.net_dIe,
                "I": self.net_dI,
            }

            for key in networks.keys():
                net = networks[key]
                for m in net.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0, std=1e-4)
                        nn.init.constant_(m.bias, val=0)
                net.apply(clipper)

    def forward(self, in_x, in_u):

        C1, C2, Gc, Ge, Ie, I = (
            in_x[..., [0]],
            in_x[..., [1]],
            in_x[..., [2]],
            in_x[..., [3]],
            in_x[..., [4]],
            in_x[..., [5]],
        )
        u_I, u_carbs = in_u[..., [0]], in_u[..., [1]]

        # NN1(C1,u_carbs)
        in_1 = torch.cat((C1, u_carbs), -1)
        dC1 = self.net_dC1(in_1)

        ## NN2 (c1,c2)
        in_2 = torch.cat((C1, C2), -1)
        dC2 = self.net_dC2(in_2)

        # NN3(Gc,C2,Ie,)
        in_3 = torch.cat((Gc, C2, Ie), -1)
        dGc = self.net_dGc(in_3)

        # NN4(Ge,Gc)
        in_4 = torch.cat((Ge, Gc), -1)
        dGe = self.net_dGe(in_4)

        # NN5(Ie,I)
        in_5 = torch.cat((Ie, I), -1)
        dIe = self.net_dIe(in_5)

        # NN6(I,Ge,C2,uI)
        in_6 = torch.cat((I, Ge, C2, u_I), -1)
        dI = self.net_dI(in_6)

        return {"C1": dC1, "C2": dC2, "Gc": dGc, "Ge": dGe, "Ie": dIe, "I": dI}


class NeuralOGTT:
    def __init__(self, device=torch.device("cpu"), ts=5):
        self.ts = ts
        self.device = device

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

        self.nn_solution = ForwardEulerSimulator(
            ss_model=ss_ogtt_model, ss_DT_model=None, ts=self.ts
        )

    def prepare_data(self, df_scenario):
        _, u_ogtt = scaler_OGTT(
            df_scenario[states].values, df_scenario[inputs_OGTT].values
        )

        u_ogtt = u_ogtt.reshape(-1, u_ogtt.shape[0], u_ogtt.shape[1])

        batch_start = np.array([0], dtype=np.int)
        batch_idx = batch_start[:, np.newaxis] + np.arange(len(df_scenario))

        x0_est = torch.tensor(
            df_scenario.loc[0, states].astype(float).values.reshape(1, -1),
            dtype=torch.float32,
        ).to(self.device)
        u_ogtt = torch.tensor(u_ogtt[[0], batch_idx.T], dtype=torch.float32).to(
            self.device
        )

        return x0_est, u_ogtt[:, [0], :]

    def simulate(self, df_scenario_original):
        # Prepare data
        df_scenario = df_scenario_original.copy()
        x0_est, u_ogtt = self.prepare_data(df_scenario)
        with torch.no_grad():
            x_sim = self.nn_solution(x0_est, u_ogtt)
            df_scenario[states] = scaler_inverse(
                x_sim[:, 0, :].to("cpu").detach().numpy()
            )
        df_scenario["Gc"] = df_scenario["state_Gc"]
        return df_scenario
