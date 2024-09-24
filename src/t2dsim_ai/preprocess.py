from pathlib import Path
from pickle import dump, load
from sklearn.preprocessing import RobustScaler
def scaler_OGTT(x_est,u_id, train=False):

    scaler_inputs = load(open(Path(__file__).parent /'models/scaler/scaler_inputs_T1D.pkl', 'rb'))

    if train:
        scaler_states = RobustScaler()
        scaler_states.fit(x_est)

        # Save the scaler
        dump(scaler_states, open(Path(__file__).parent /'models/scaler/scaler_states_OGTT.pkl', 'wb'))

    else:
        scaler_states = load(open(Path(__file__).parent /'models/scaler/scaler_states_OGTT.pkl', 'rb'))

    x_est = scaler_states.transform(x_est[:,[0,1,2,2,3,4,5]])[:,[0,1,3,4,5,6]] # A trick to get rid of Gp
    u_id[:,[0,1]]= scaler_inputs.transform(u_id[:,[0,1]])

    return x_est, u_id
def scaler_inverse(x_est):
    scaler_states = load(open(Path(__file__).parent /'models/scaler/scaler_states_OGTT.pkl', 'rb'))
    x_est = scaler_states.inverse_transform(x_est[:,[0,1,2,2,3,4,5]])[:,[0,1,3,4,5,6]]
    return x_est