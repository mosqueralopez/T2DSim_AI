import numpy as np

scale = True
states_name = np.array(["C1", "C2", "Gc", "Ge", "Ie", "I"])
states = ["state_" + ii for ii in states_name]


inputs_OGTT = ["input_insulin", "input_carbs"]
inputs_Pop = [
    "input_hr",
    "input_sleep",
    "input_sulfonylurea",
    "input_sglt2",
    "input_glp1",
    "input_biguanide",
    "feat_is_weekend",
    "feat_hour_of_day_cos",
    "feat_hour_of_day_sin",
]
inputs = inputs_OGTT + inputs_Pop

nn = 64
n_neurons_ogtt = {"C1": 149, "C2": 140, "Gc": nn, "Ge": nn, "Ie": nn, "I": nn}

nn_aux = 256
n_neuron_ind = {
    "C1": [3, nn_aux, 1],
    "C2": [2, nn_aux, 1],
    "Gc": [10, nn_aux, 1],
    "Ge": [2, nn_aux, 1],
    "Ie": [2, nn_aux, 1],
    "I": [6, nn_aux, 1],
}
