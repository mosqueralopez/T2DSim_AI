"""Digital-twin training utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from t2dsim_ai.data_processing import prepare_training_data, process_raw_csv
from t2dsim_ai.model_DTNeuralOGTT import CGMOHSUSimStateSpaceModel_T2D
from t2dsim_ai.model_neuralOGTT import CGMOHSUSimStateSpaceModel_T2DOGTT
from t2dsim_ai.options import n_neuron_ind, n_neurons_ogtt, states
from t2dsim_ai.preprocess import scaler_Pop
from t2dsim_ai.ss_simulator import ForwardEulerSimulator

_PACKAGE_ROOT = Path(__file__).parent
_OUTPUT_Gc = 2


class _SequenceBatch:
    def __init__(self, data_dict, batch_size, device):
        self.states = data_dict["states"]
        self.u_ogtt = data_dict["inputs_OGTT"]
        self.u_pop = data_dict["inputs_Pop"]
        self.y = data_dict["output"]
        self.n_seq = self.states.shape[0]
        self.batch_size = batch_size
        self.device = device
        self.idx = np.arange(self.n_seq)

    def get_batch(self):
        if self.n_seq <= self.batch_size:
            chosen = self.idx
        else:
            chosen = np.random.choice(self.idx, self.batch_size, replace=False)
        x0 = torch.tensor(self.states[chosen, 0, :], dtype=torch.float32).to(
            self.device
        )
        u_ogtt = torch.tensor(self.u_ogtt[chosen].transpose(1, 0, 2), dtype=torch.float32).to(
            self.device
        )
        u_pop = torch.tensor(self.u_pop[chosen].transpose(1, 0, 2), dtype=torch.float32).to(
            self.device
        )
        y = torch.tensor(self.y[chosen], dtype=torch.float32).to(self.device)
        return x0, u_ogtt, u_pop, y


def _rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2))


def train_digital_twin(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    n_epochs: int = 5,
    lr: float = 1e-5,
    batch_size: int = 32,
    seq_len: int = 5 * 12,
    overlap: float = 0.98,
    alpha: float = 1e-4,
    device: str | torch.device | None = None,
) -> Path:
    """Train a digital twin from a subject CSV and save artifacts to ``output_dir``."""
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    df = process_raw_csv(data_path)
    train_part = df.iloc[: int(len(df) * 0.7)]
    x_fit = train_part[states].to_numpy(dtype=float)
    u_ogtt_fit = train_part[["input_insulin", "input_carbs"]].to_numpy(dtype=float)
    u_pop_fit = train_part[
        [
            "input_hr",
            "input_hr_min",
            "input_hr_max",
            "input_hr_std",
            "input_sleep",
            "input_sulfonylurea",
            "input_sglt2",
            "input_glp1",
            "input_biguanide",
            "feat_is_weekend",
            "feat_hour_of_day_cos",
            "feat_hour_of_day_sin",
        ]
    ].to_numpy(dtype=float)
    scaler_Pop(x_fit, u_ogtt_fit, u_pop_fit, str(output_dir), train=True)

    data = prepare_training_data(
        data_path, seq_len=seq_len, overlap=overlap, train_frac=0.7, val_frac=0.15
    )
    if "train" not in data:
        raise ValueError("Training split is empty; provide a longer CSV.")

    for split in data.values():
        n_seq, _, n_x = split["states"].shape
        flat_x = split["states"].reshape(n_seq * seq_len, n_x)
        flat_u_ogtt = split["inputs_OGTT"].reshape(n_seq * seq_len, 2)
        flat_u_pop = split["inputs_Pop"].reshape(n_seq * seq_len, 12)
        flat_x, flat_u_ogtt, flat_u_pop = scaler_Pop(
            flat_x, flat_u_ogtt, flat_u_pop, str(output_dir), train=False
        )
        split["states"] = flat_x.reshape(n_seq, seq_len, n_x)
        split["inputs_OGTT"] = flat_u_ogtt.reshape(n_seq, seq_len, 2)
        split["inputs_Pop"] = flat_u_pop.reshape(n_seq, seq_len, 12)
        split["output"] = split["states"][:, :, [_OUTPUT_Gc]]

    ss_ogtt = CGMOHSUSimStateSpaceModel_T2DOGTT(n_feat=n_neurons_ogtt).to(device)
    ss_ogtt.load_state_dict(
        torch.load(_PACKAGE_ROOT / "models/OGTT_productionModel_6compartments.pt")
    )
    for param in ss_ogtt.parameters():
        param.requires_grad = False

    ss_dt = CGMOHSUSimStateSpaceModel_T2D(hidden_compartments=n_neuron_ind).to(device)
    simulator = ForwardEulerSimulator(ss_model=ss_ogtt, ss_DT_model=ss_dt, ts=5)

    optimizer = optim.Adam(ss_dt.parameters(), lr=lr, weight_decay=1e-3)
    batch = _SequenceBatch(data["train"], batch_size, device)
    val_batch = (
        _SequenceBatch(data["validation"], 1, device) if "validation" in data else None
    )

    best_val = float("inf")
    best_state = None
    loss_history = []

    n_iter = max(1, n_epochs * max(1, batch.n_seq // batch_size))
    for itr in range(n_iter):
        optimizer.zero_grad()
        x0, u_ogtt, u_pop, y = batch.get_batch()
        x_sim = simulator(x0, u_ogtt, u_pop, is_DT=True)
        pred = x_sim[:, :, [_OUTPUT_Gc]]
        target = y.permute(1, 0, 2)
        fit_loss = torch.mean((pred - target) ** 2)
        consistency = torch.mean(x_sim**2) * 0.0 + torch.mean(torch.relu(-x_sim)) * 0.0
        loss = fit_loss + alpha * consistency
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

        if val_batch is not None and (itr + 1) % max(1, n_iter // n_epochs) == 0:
            with torch.no_grad():
                x0_v, u_ogtt_v, u_pop_v, y_v = val_batch.get_batch()
                x_sim_v = simulator(x0_v, u_ogtt_v, u_pop_v, is_DT=True)
                val_rmse = _rmse(
                    x_sim_v[:, :, [_OUTPUT_Gc]], y_v.permute(1, 0, 2)
                ).item()
            if val_rmse < best_val:
                best_val = val_rmse
                best_state = {k: v.cpu().clone() for k, v in ss_dt.state_dict().items()}

    if best_state is None:
        best_state = ss_dt.state_dict()
    torch.save(best_state, output_dir / "model.pt")

    info = pd.DataFrame(
        {
            "metric": ["best_val_rmse_scaled", "n_epochs", "n_iter", "subjectID"],
            "value": [best_val, n_epochs, n_iter, data_path.stem],
        }
    )
    info.to_csv(output_dir / "info.csv", index=False)
    pd.DataFrame({"loss": loss_history}).to_csv(output_dir / "loss.csv", index=False)

    return output_dir
