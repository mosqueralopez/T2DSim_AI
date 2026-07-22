import numpy as np

from t2dsim_ai.metrics import get_rmse, get_tir, glucose_values


def test_glucose_values_matches_research_shape():
    true = np.array([[[100.0], [110.0]], [[90.0], [120.0]]])
    pred = np.array([[[102.0], [108.0]], [[95.0], [115.0]]])
    results = glucose_values({"true": true, "pred": pred}).describe().loc["mean"]
    assert "RMSE" in results.index
    assert "TIR_true" in results.index
    assert "TIR_pred" in results.index
    assert 0 <= results["TIR_true"] <= 100


def test_get_tir_respects_limits():
    cgm = np.array([60.0, 100.0, 200.0])
    assert get_tir(cgm, lim_inf=70, lim_sup=180) == 1 / 3


def test_get_rmse_ignores_nan():
    err = np.array([3.0, np.nan, 4.0])
    assert np.isclose(get_rmse(err), np.sqrt((3.0**2 + 4.0**2) / 2))
