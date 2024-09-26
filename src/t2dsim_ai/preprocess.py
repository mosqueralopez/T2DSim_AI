from pathlib import Path
from pickle import dump, load
from sklearn.preprocessing import RobustScaler


def scaler_Pop(x_est, u_ogtt, u_pop, path_scaler, train=True):

    scale_input_pos = [0, 2, 3, 4, 5]

    if len(x_est.shape) == 2:
        x_est, u_ogtt = scaler_OGTT(x_est, u_ogtt, train=False)

        if train:
            scaler_inputs = RobustScaler()

            scaler_inputs.fit(
                u_pop[:, scale_input_pos].reshape(-1, len(scale_input_pos))
            )

            # Save the scaler
            dump(scaler_inputs, open(path_scaler + "/scaler_inputsPop.pkl", "wb"))

        else:
            scaler_inputs = load(open(path_scaler + "/scaler_inputsPop.pkl", "rb"))

        u_pop[:, scale_input_pos] = scaler_inputs.transform(
            u_pop[:, scale_input_pos].reshape(-1, len(scale_input_pos))
        )  # .reshape(-1) # Only scales the heart rate

        return x_est, u_ogtt, u_pop

    elif len(x_est.shape) == 3:
        for idx in range(x_est.shape[0]):
            x_est[idx], u_ogtt[idx] = scaler_OGTT(x_est[idx], u_ogtt[idx], train=False)

            scaler_inputs = load(open(path_scaler + "/scaler_inputsPop.pkl", "rb"))

            u_pop[idx, :, scale_input_pos] = scaler_inputs.transform(
                u_pop[idx, :, scale_input_pos].reshape(-1, len(scale_input_pos))
            )  # .reshape(-1) # Only scales the heart rate

        return x_est, u_ogtt, u_pop
    else:
        print("ErrorDimension")
        exit()


def scaler_OGTT(x_est, u_id, train=False):

    scaler_inputs = load(
        open(Path(__file__).parent / "models/scaler/scaler_inputs_T1D.pkl", "rb")
    )

    if train:
        scaler_states = RobustScaler()
        scaler_states.fit(x_est)

        # Save the scaler
        dump(
            scaler_states,
            open(Path(__file__).parent / "models/scaler/scaler_states_OGTT.pkl", "wb"),
        )

    else:
        scaler_states = load(
            open(Path(__file__).parent / "models/scaler/scaler_states_OGTT.pkl", "rb")
        )

    x_est = scaler_states.transform(x_est[:, [0, 1, 2, 2, 3, 4, 5]])[
        :, [0, 1, 3, 4, 5, 6]
    ]  # A trick to get rid of Gp
    u_id[:, [0, 1]] = scaler_inputs.transform(u_id[:, [0, 1]])

    return x_est, u_id


def scaler_inverse(x_est):
    scaler_states = load(
        open(Path(__file__).parent / "models/scaler/scaler_states_OGTT.pkl", "rb")
    )
    x_est = scaler_states.inverse_transform(x_est[:, [0, 1, 2, 2, 3, 4, 5]])[
        :, [0, 1, 3, 4, 5, 6]
    ]
    return x_est
