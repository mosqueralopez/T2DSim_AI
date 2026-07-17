import torch
import torch.nn as nn
from typing import List


class ForwardEulerSimulator(nn.Module):

    """This class implements prediction/simulation methods for the SS model structure

    Attributes
    ----------
    ss_model: nn.Module
              The neural SS model to be fitted
    ts: float
        model sampling time

    """

    def __init__(
        self,
        ss_model,
        ss_DT_model=None,
        ts=1.0,
        states_info={"C1": 0, "C2": 1, "Gc": 2, "Ge": 3, "Ie": 4, "I": 5},
    ):
        super(ForwardEulerSimulator, self).__init__()
        self.ss_model = ss_model
        self.ss_DT_model = ss_DT_model
        self.ts = ts
        self.states_info = states_info

    def forward(
        self,
        x0_batch: torch.Tensor,
        u_batch_ogtt: torch.Tensor,
        u_batch_pop=None,
        is_DT=False,
    ) -> torch.Tensor:
        """Multi-step simulation over (mini)batches

        Parameters
        ----------
        x0_batch: Tensor. Size: (q, n_x)
             Initial state for each subsequence in the minibatch

        u_batch: Tensor. Size: (m, q, n_u)
            Input sequence for each subsequence in the minibatch

        Returns
        -------
        Tensor. Size: (m, q, n_x)
            Simulated state for all subsequences in the minibatch

        """

        X_sim_list: List[torch.Tensor] = []

        x_step = x0_batch

        for step in range(u_batch_ogtt.shape[0]):
            X_sim_list += [x_step]
            dx = torch.zeros_like(x_step)

            u_step = u_batch_ogtt[step]
            dx_OGTT = self.ss_model(x_step, u_step)
            dx[:, self.states_info["C1"]] += dx_OGTT["C1"][:, 0]
            dx[:, self.states_info["C2"]] += dx_OGTT["C2"][:, 0]
            dx[:, self.states_info["Gc"]] += dx_OGTT["Gc"][:, 0]
            dx[:, self.states_info["Ge"]] += dx_OGTT["Ge"][:, 0]
            dx[:, self.states_info["I"]] += dx_OGTT["I"][:, 0]
            dx[:, self.states_info["Ie"]] += dx_OGTT["Ie"][:, 0]

            if is_DT:
                u_pop_step = u_batch_pop[step]
                dx_DT = self.ss_DT_model(x_step, u_step, u_pop_step)
                dx[:, self.states_info["C1"]] += dx_DT["C1"][:, 0]
                dx[:, self.states_info["C2"]] += dx_DT["C2"][:, 0]
                dx[:, self.states_info["Gc"]] += dx_DT["Gc"][:, 0]
                dx[:, self.states_info["Ge"]] += dx_DT["Ge"][:, 0]
                dx[:, self.states_info["I"]] += dx_DT["I"][:, 0]
                dx[:, self.states_info["Ie"]] += dx_DT["Ie"][:, 0]

            x_step = x_step + self.ts * dx

        X_sim = torch.stack(X_sim_list, 0)

        return X_sim

    def forward_sobol(
        self, x0_batch: torch.Tensor, u_batch: torch.Tensor, weights: list
    ) -> torch.Tensor:
        """Multi-step simulation over (mini)batches

        Parameters
        ----------
        x0_batch: Tensor. Size: (q, n_x)
             Initial state for each subsequence in the minibatch

        u_batch: Tensor. Size: (m, q, n_u)
            Input sequence for each subsequence in the minibatch

        Returns
        -------
        Tensor. Size: (m, q, n_x)
            Simulated state for all subsequences in the minibatch

        """

        X_sim_list: List[torch.Tensor] = []
        dX_sim_list: List[torch.Tensor] = []

        x_step = x0_batch
        dx = torch.zeros_like(x0_batch)
        for u_step in u_batch.split(1):  # i in range(seq_len):
            u_step = u_step.squeeze(0)

            X_sim_list += [x_step]
            dX_sim_list += [dx]

            dx = self.ss_model(x_step, u_step) * torch.tensor(
                weights, dtype=torch.float32
            )
            x_step = x_step + self.ts * dx

        X_sim = torch.stack(X_sim_list, 0)
        dX_sim = torch.stack(dX_sim_list, 0)

        return X_sim, dX_sim
