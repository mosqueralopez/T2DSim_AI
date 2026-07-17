import pytest

from t2dsim_ai import DigitalTwin, NeuralOGTT, scenario_from_twin_info, train_digital_twin
from t2dsim_ai.create_scenarios import ogtt_scenario
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
    )
    assert (out / "model.pt").exists()
    assert (out / "scaler_inputsPop.pkl").exists()
    assert (out / "info.csv").exists()


def test_no_mit_license_headers():
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "SPDX-License-Identifier: MIT" not in text
        assert "AllInputs" not in text
