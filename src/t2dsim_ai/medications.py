import numpy as np
from scipy.special import gamma, gammainc

MEDICATION_SAMPLING_DT_MIN = 5.0

meds_dict = {
    "input_biguanide": {"t_peak": 3 * 60, "h": 6 * 60, "T_max": 24 * 60},
    "input_glp1": {"t_peak": 3 * 60 * 24, "h": 7 * 60 * 24, "T_max": 5 * 7 * 60 * 24},
    "input_sglt2": {"t_peak": 1.5 * 60, "h": 13 * 60, "T_max": 24 * 60},
    "input_sulfonylurea_glimepiride": {
        "t_peak": 3 * 60,
        "h": 8 * 60,
        "T_max": 24 * 60,
    },
    "input_sulfonylurea_glipizide": {
        "t_peak": 2 * 60,
        "h": 3 * 60,
        "T_max": 12 * 60,
    },
}


def gamma_variate_truncated(t, t_peak, h, T_max):
    t = np.asarray(t, dtype=float)
    beta = np.log(2) / h
    alpha = 1 + beta * t_peak
    norm = gamma(alpha) * gammainc(alpha, beta * T_max)
    c = np.zeros_like(t)
    mask = (t >= 0) & (t <= T_max)
    c[mask] = (
        (beta**alpha) * (t[mask] ** (alpha - 1)) * np.exp(-beta * t[mask]) / norm
    )
    return c


def medication_time_samples(end_min, dt_min=MEDICATION_SAMPLING_DT_MIN):
    dt_min = float(dt_min)
    end_min = float(end_min)
    n = int(np.floor(end_min / dt_min + 1e-12))
    return np.arange(0, n + 1, dtype=float) * dt_min


def discrete_oral_med_kernel(dose, med_key, ts_min=MEDICATION_SAMPLING_DT_MIN):
    p = meds_dict[med_key]
    t = medication_time_samples(p["T_max"], dt_min=ts_min)
    return float(dose) * gamma_variate_truncated(t, p["t_peak"], p["h"], p["T_max"])
