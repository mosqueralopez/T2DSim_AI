import pytest
import numpy as np
import pandas as pd

from t2dsim_ai import DigitalTwin, NeuralOGTT, scenario_from_twin_info, train_digital_twin
from t2dsim_ai.create_scenarios import _info_to_dict, ogtt_scenario
from t2dsim_ai.options import inputs_Pop
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CSV = ROOT / "example/example_model/data_example.csv"


def test_neural_ogtt_simulate():
    result = NeuralOGTT().simulate(ogtt_scenario(100))
    assert len(result) == 60
    assert "state_Gc" in result.columns
    assert result["state_Gc"].notna().all()


def test_scenario_from_twin_info():
    twin = DigitalTwin(0)
    df = scenario_from_twin_info(twin.digital_twin_Info)
    assert len(df) == 24 * 12
    for col in inputs_Pop:
        assert col in df.columns


def test_digital_twin_simulate_from_info():
    twin = DigitalTwin(0)
    df = scenario_from_twin_info(twin.digital_twin_Info)
    result = twin.simulate(df)
    assert "Gc" in result.columns
    assert result["Gc"].notna().all()


@pytest.mark.slow
def test_train_digital_twin_smoke(tmp_path):
    out = train_digital_twin(
        EXAMPLE_CSV,
        tmp_path / "output",
        n_epochs=1,
        batch_size=8,
        hypo_penalization=9.0,
        hyper_penalization=90.0,
    )
    assert (out / "model.pt").exists()
    assert (out / "scaler_inputsPop.pkl").exists()
    info = pd.read_csv(out / "info.csv")
    meta = _info_to_dict(info)
    assert meta["subjectID"] == "example-001"
    assert "best_epoch" in meta
    assert "best_valLoss" in meta
    assert meta["hypo_penalization"] == 9.0
    assert meta["hyper_penalization"] == 90.0
    assert "RMSE_DigitalTwin_validation" in meta
    assert "RMSE_DigitalTwin_train" in meta
    assert "TIR_true_validation" in meta
    assert "TIR_true_train" in meta


def test_basal_insulin_micro_bolus_and_plot_rate():
    from t2dsim_ai.data_processing import basal_insulin_plot_uh, basal_micro_bolus_u

    daily_u = 54.0
    assert np.isclose(basal_micro_bolus_u(daily_u), daily_u / (12 * 24))
    assert np.isclose(basal_insulin_plot_uh(daily_u), daily_u / 24.0)


def test_insulin_deliveries_use_iob_pipeline():
    from t2dsim_ai.data_processing import basal_micro_bolus_u, insulin_deliveries_u_to_uh

    deliveries = np.zeros(50, dtype=float)
    deliveries[0] = 5.0
    iob_profile = insulin_deliveries_u_to_uh(deliveries)
    assert iob_profile[0] > 0
    assert iob_profile[1] > 0
    assert iob_profile[10] < iob_profile[0]

    # Full-day constant basal: tail indices must not overflow IOB window.
    daily_u = 54.0
    n_steps = 24 * 12
    basal = np.full(n_steps, basal_micro_bolus_u(daily_u), dtype=float)
    full_day = insulin_deliveries_u_to_uh(basal)
    assert len(full_day) == n_steps
    assert np.all(full_day >= 0)
    assert full_day[-1] > 0


def test_scenario_without_insulin_has_plot_column():
    info_path = ROOT / "src/t2dsim_ai/models/DigitalTwins/S021-002/info.csv"
    df = scenario_from_twin_info(info_path)
    assert "basal_insulin_plot_uh" in df.columns
    assert df["basal_insulin_plot_uh"].max() == 0.0


def test_scenario_from_twin_info_includes_medications():
    info_path = (
        ROOT / "src/t2dsim_ai/models/DigitalTwins/S021-005/info.csv"
    )
    df = scenario_from_twin_info(info_path)
    assert df["input_insulin"].max() > 0
    assert np.isclose(df["basal_insulin_plot_uh"].iloc[0], 54.0 / 24.0)
    assert df["input_glp1"].max() > 0
    assert df["input_sulfonylurea"].max() > 0


def test_no_mit_license_headers():
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "SPDX-License-Identifier: MIT" not in text
        assert "AllInputs" not in text
