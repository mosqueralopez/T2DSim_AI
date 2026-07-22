"""Digital-twin training utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from t2dsim_ai.data_processing import (
    build_training_metadata,
    prepare_training_data,
    process_raw_csv,
    resolve_subject_id,
    scale_sequence_splits,
)
from t2dsim_ai.metrics import glucose_values
from t2dsim_ai.model_DTNeuralOGTT import CGMOHSUSimStateSpaceModel_T2D
from t2dsim_ai.model_neuralOGTT import CGMOHSUSimStateSpaceModel_T2DOGTT
from t2dsim_ai.options import default_seq_len, n_neuron_ind, n_neurons_ogtt, states, ts
from t2dsim_ai.preprocess import scaler_Pop, scale_inverse_state
from t2dsim_ai.ss_simulator import ForwardEulerSimulator
from t2dsim_ai.train_losses import (
    loss_consistency,
    loss_fit,
    scaled_gc_limits_mgdl,
    state_minimum_values,
)

_PACKAGE_ROOT = Path(__file__).parent
_OUTPUT_Gc = 2
_METRICS_OVERLAP = 0.0  # research validation/test batches use no overlap


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



def _split_dt_rmse(
    simulator: ForwardEulerSimulator,
    split: dict,
    device: torch.device,
) -> float:
    """Validation RMSE (mg/dL) for the digital twin on a full data split."""
    n_seq, _, _ = split["states"].shape
    x0 = torch.tensor(split["states"][:, 0, :], dtype=torch.float32).to(device)
    u_ogtt = torch.tensor(
        split["inputs_OGTT"].transpose(1, 0, 2), dtype=torch.float32
    ).to(device)
    u_pop = torch.tensor(
        split["inputs_Pop"].transpose(1, 0, 2), dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        x_dt = simulator(x0, u_ogtt, u_pop, is_DT=True)

    true_gc = split["states"][:, :, _OUTPUT_Gc].transpose(1, 0)[:, :, np.newaxis]
    true_mgdl = scale_inverse_state(true_gc[1:], _OUTPUT_Gc)
    pred_dt_mgdl = scale_inverse_state(
        x_dt[1:, :, _OUTPUT_Gc : _OUTPUT_Gc + 1].cpu().numpy(), _OUTPUT_Gc
    )
    return float(np.sqrt(np.mean((pred_dt_mgdl - true_mgdl) ** 2)))


def _evaluate_split(
    simulator: ForwardEulerSimulator,
    split: dict,
    device: torch.device,
    group: str,
) -> dict[str, float]:
    """Run full-sequence glucose metrics for a data split (``train``, ``validation``, etc.)."""
    n_seq, _, _ = split["states"].shape
    x0 = torch.tensor(split["states"][:, 0, :], dtype=torch.float32).to(device)
    u_ogtt = torch.tensor(
        split["inputs_OGTT"].transpose(1, 0, 2), dtype=torch.float32
    ).to(device)
    u_pop = torch.tensor(
        split["inputs_Pop"].transpose(1, 0, 2), dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        x_ogtt = simulator(x0, u_ogtt, u_pop, is_DT=False).cpu().numpy()
        x_dt = simulator(x0, u_ogtt, u_pop, is_DT=True).cpu().numpy()

    true_gc = split["states"][:, :, _OUTPUT_Gc].transpose(1, 0)[:, :, np.newaxis]
    true_mgdl = scale_inverse_state(true_gc[1:], _OUTPUT_Gc)
    pred_ogtt_mgdl = scale_inverse_state(
        x_ogtt[1:, :, _OUTPUT_Gc : _OUTPUT_Gc + 1], _OUTPUT_Gc
    )
    pred_dt_mgdl = scale_inverse_state(
        x_dt[1:, :, _OUTPUT_Gc : _OUTPUT_Gc + 1], _OUTPUT_Gc
    )

    ogtt_results = glucose_values({"pred": pred_ogtt_mgdl, "true": true_mgdl}).describe().loc[
        "mean"
    ]
    dt_results = glucose_values({"pred": pred_dt_mgdl, "true": true_mgdl}).describe().loc["mean"]

    metrics = {
        f"RMSE_NeuralOGTT_{group}": float(ogtt_results["RMSE"]),
        f"RMSE_DigitalTwin_{group}": float(dt_results["RMSE"]),
        f"n_seqs_{group}": float(n_seq),
    }
    for name, value in dt_results.items():
        if name == "RMSE":
            continue
        metrics[f"{name}_{group}"] = float(value)
    return metrics


def train_digital_twin(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    n_epochs: int = 5,
    lr: float = 1e-5,
    batch_size: int = 32,
    seq_len: int = default_seq_len,
    overlap: float = 0.98,
    alpha: float = 1e-4,
    hypo_penalization: float = 9.0,
    hyper_penalization: float = 90.0,
    dcgm_weight: float = 100.0,
    device: str | torch.device | None = None,
) -> Path:
    """Train a digital twin from a subject CSV and save artifacts to ``output_dir``.

    Parameters
    ----------
    seq_len
        Number of 5-minute samples per training sequence. Default is 12 hours
        (``12 * 60 // ts`` = 144 steps).
    """
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    lim_inferior, lim_superior = scaled_gc_limits_mgdl()
    state_mins = state_minimum_values()

    df = process_raw_csv(data_path)
    subject_id = resolve_subject_id(df, data_path.stem)
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

    scale_sequence_splits(data, seq_len, output_dir)

    eval_data = prepare_training_data(
        data_path,
        seq_len=seq_len,
        overlap=_METRICS_OVERLAP,
        train_frac=0.7,
        val_frac=0.15,
    )
    scale_sequence_splits(eval_data, seq_len, output_dir)

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

    n_train_seq = batch.n_seq
    iters_per_epoch = max(1, n_train_seq // batch_size)
    has_validation = "validation" in eval_data

    print(f"Sequence length: {seq_len} steps ({seq_len * ts / 60:.0f} h)")
    print(f"Training sequences: {n_train_seq}")
    print(f"Iterations per epoch: {iters_per_epoch}")
    if "validation" in eval_data:
        print(
            f"Validation sequences (no overlap, for metrics): "
            f"{eval_data['validation']['states'].shape[0]}"
        )

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    epoch_records: list[dict[str, float | int]] = []

    for epoch in range(1, n_epochs + 1):
        epoch_losses: list[float] = []

        for _ in range(iters_per_epoch):
            optimizer.zero_grad()
            x0, u_ogtt, u_pop, y = batch.get_batch()
            x_sim = simulator(x0, u_ogtt, u_pop, is_DT=True)

            if torch.isnan(x_sim).any() or torch.isinf(x_sim).any():
                raise RuntimeError(f"Non-finite simulation in epoch {epoch}")

            pred = x_sim[:, :, [_OUTPUT_Gc]]
            target = y.permute(1, 0, 2)

            fit_loss = loss_fit(
                pred,
                target,
                lim_inferior=lim_inferior,
                lim_superior=lim_superior,
                hypo_penalization=hypo_penalization,
                hyper_penalization=hyper_penalization,
                dcgm_weight=dcgm_weight,
            )
            consistency_loss = loss_consistency(x_sim, state_mins)
            loss = fit_loss + alpha * consistency_loss
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        avg_loss = float(np.mean(epoch_losses))
        record: dict[str, float | int] = {"epoch": epoch, "loss": avg_loss}

        if has_validation:
            val_rmse = _split_dt_rmse(simulator, eval_data["validation"], device)
            record["val_rmse_mgdl"] = val_rmse
            if val_rmse < best_val:
                best_val = val_rmse
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in ss_dt.state_dict().items()}
            print(
                f"Epoch {epoch}/{n_epochs} | loss {avg_loss:.6f} | "
                f"val_rmse {val_rmse:.4f} mg/dL"
            )
        else:
            print(f"Epoch {epoch}/{n_epochs} | loss {avg_loss:.6f}")

        epoch_records.append(record)

    if best_state is None:
        best_state = ss_dt.state_dict()
    ss_dt.load_state_dict(best_state)
    torch.save(best_state, output_dir / "model.pt")

    ogtt_model_path = _PACKAGE_ROOT / "models/OGTT_productionModel_6compartments.pt"
    info = build_training_metadata(df, subject_id)
    info["best_epoch"] = best_epoch
    info["best_valLoss"] = best_val
    info["NeuralOGTT_path"] = str(ogtt_model_path)
    info["hypo_penalization"] = hypo_penalization
    info["hyper_penalization"] = hyper_penalization
    info["seq_len"] = seq_len
    info["seq_len_hours"] = seq_len * ts / 60

    for group in ("train", "validation"):
        if group in eval_data:
            info.update(_evaluate_split(simulator, eval_data[group], device, group))

    if "validation" in data:
        info["best_valLoss"] = info["RMSE_DigitalTwin_validation"]

    info_series = pd.Series(info)
    info_series.to_csv(output_dir / "info.csv")
    pd.DataFrame(epoch_records).to_csv(output_dir / "loss.csv", index=False)

    return output_dir
