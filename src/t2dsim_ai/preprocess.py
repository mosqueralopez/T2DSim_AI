from pathlib import Path
from pickle import dump, load

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from t2dsim_ai.options import inputs_Pop, inputs_to_scale, states_name

_PACKAGE_ROOT = Path(__file__).parent
_SCALER_DIR = _PACKAGE_ROOT / "models" / "scaler"


def pop_scale_column_indices(pop_input_names, scale_input_names=None):
    name_to_idx = {n: i for i, n in enumerate(pop_input_names)}
    if scale_input_names is None:
        to_scale = [n for n in pop_input_names if n in inputs_to_scale]
    else:
        to_scale = list(scale_input_names)
        missing = [n for n in to_scale if n not in name_to_idx]
        if missing:
            raise ValueError(
                "scale_input_names contains names not in pop_input_names: "
                + str(missing)
            )
    return sorted(name_to_idx[n] for n in to_scale)


def scaler_Pop(
    x_est,
    u_ogtt,
    u_pop,
    path_scaler,
    train=True,
    is_robust_scaler=True,
    pop_input_names=None,
    scale_input_names=None,
):
    path_ogtt_scaler = str(_SCALER_DIR)
    path_scaler = str(path_scaler)

    if u_pop is None:
        if len(x_est.shape) == 2:
            x_est, u_ogtt = scaler_OGTT(x_est, u_ogtt, train=False)
        elif len(x_est.shape) == 3:
            for idx in range(x_est.shape[0]):
                x_est[idx], u_ogtt[idx] = scaler_OGTT(x_est[idx], u_ogtt[idx], train=False)
        else:
            raise ValueError(f"Unsupported x_est shape: {x_est.shape}")
        return x_est, u_ogtt, None

    if pop_input_names is None:
        pop_input_names = list(inputs_Pop)
    else:
        pop_input_names = list(pop_input_names)

    scale_idx = pop_scale_column_indices(pop_input_names, scale_input_names)
    n_pop = int(u_pop.shape[-1])
    if n_pop != len(pop_input_names):
        raise ValueError(
            f"u_pop last dimension ({n_pop}) must match len(pop_input_names) "
            f"({len(pop_input_names)})"
        )

    if len(scale_idx) == 0:
        if len(x_est.shape) == 2:
            x_est, u_ogtt = scaler_OGTT(x_est, u_ogtt, train=False)
        elif len(x_est.shape) == 3:
            for idx in range(x_est.shape[0]):
                x_est[idx], u_ogtt[idx] = scaler_OGTT(x_est[idx], u_ogtt[idx], train=False)
        else:
            raise ValueError(f"Unsupported x_est shape: {x_est.shape}")
        return x_est, u_ogtt, u_pop

    if len(x_est.shape) == 2:
        x_est, u_ogtt = scaler_OGTT(x_est, u_ogtt, train=False)

        if train:
            scaler_inputs = (
                RobustScaler() if is_robust_scaler else MinMaxScaler()
            )
            scaler_inputs.fit(u_pop[:, scale_idx].reshape(-1, len(scale_idx)))
            dump(
                scaler_inputs,
                open(Path(path_scaler) / "scaler_inputsPop.pkl", "wb"),
            )
        else:
            scaler_inputs = load(
                open(Path(path_scaler) / "scaler_inputsPop.pkl", "rb")
            )

        u_pop[:, scale_idx] = scaler_inputs.transform(
            u_pop[:, scale_idx].reshape(-1, len(scale_idx))
        )
        return x_est, u_ogtt, u_pop

    if len(x_est.shape) == 3:
        if train:
            scaler_inputs = (
                RobustScaler() if is_robust_scaler else MinMaxScaler()
            )
            scaler_inputs.fit(u_pop[:, :, scale_idx].reshape(-1, len(scale_idx)))
            dump(
                scaler_inputs,
                open(Path(path_scaler) / "scaler_inputsPop.pkl", "wb"),
            )
        else:
            scaler_inputs = load(
                open(Path(path_scaler) / "scaler_inputsPop.pkl", "rb")
            )

        for idx in range(x_est.shape[0]):
            x_est[idx], u_ogtt[idx] = scaler_OGTT(x_est[idx], u_ogtt[idx], train=False)
            u_pop[idx, :, scale_idx] = scaler_inputs.transform(
                u_pop[idx, :, scale_idx].reshape(-1, len(scale_idx))
            )
        return x_est, u_ogtt, u_pop

    raise ValueError(f"Unsupported x_est shape: {x_est.shape}")


def scaler_OGTT(x_est, u_id, train=False):
    scaler_inputs = load(open(_SCALER_DIR / "scaler_inputs_T1D.pkl", "rb"))

    if train:
        scaler_states = RobustScaler()
        scaler_states.fit(x_est)
        dump(scaler_states, open(_SCALER_DIR / "scaler_states_OGTT.pkl", "wb"))
    else:
        scaler_states = load(open(_SCALER_DIR / "scaler_states_OGTT.pkl", "rb"))

    x_est = scaler_states.transform(x_est[:, [0, 1, 2, 2, 3, 4, 5]])[
        :, [0, 1, 3, 4, 5, 6]
    ]
    u_id[:, [0, 1]] = scaler_inputs.transform(u_id[:, [0, 1]])
    return x_est, u_id


def scaler_inverse(x_est):
    scaler_states = load(open(_SCALER_DIR / "scaler_states_OGTT.pkl", "rb"))
    x_est = scaler_states.inverse_transform(x_est[:, [0, 1, 2, 2, 3, 4, 5]])[
        :, [0, 1, 3, 4, 5, 6]
    ]
    return x_est


def scale_inverse_state(value_array, index):
    scaler_states = load(open(_SCALER_DIR / "scaler_states_OGTT.pkl", "rb"))
    scale = scaler_states.scale_[index]
    center = scaler_states.center_[index]
    value_array = value_array * scale + center
    return value_array


def scale_single_state(value, state, is_array=False):
    if isinstance(state, str):
        pos = np.where(states_name == state.split("_")[-1])[0][0]
    else:
        pos = state

    scaler_states = load(open(_SCALER_DIR / "scaler_states_OGTT.pkl", "rb"))

    if is_array:
        x_est = np.zeros((len(value), len(states_name)), dtype=float)
        x_est[:, pos] = value
        x_est = scaler_states.transform(
            x_est[:, [0, 1, 2, 2, 3, 4, 5]]
        )[:, [0, 1, 3, 4, 5, 6]]
        return x_est[:, pos]

    x_est = np.zeros((1, len(states_name)), dtype=float)
    x_est[0, pos] = value
    x_est = scaler_states.transform(x_est[:, [0, 1, 2, 2, 3, 4, 5]])[
        :, [0, 1, 3, 4, 5, 6]
    ]
    return x_est[0, pos]


def return_scaler_gc(index):
    scaler_states = load(open(_SCALER_DIR / "scaler_states_OGTT.pkl", "rb"))
    return scaler_states.scale_[index], scaler_states.center_[index]
