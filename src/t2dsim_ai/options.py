import numpy as np

scale = True
ts = 5  # min

states_name = np.array(["C1", "C2", "Gc", "Ge", "Ie", "I"])
states = ["state_" + s for s in states_name]

inputs_OGTT = ["input_insulin", "input_carbs"]

inputs_Pop = [
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

inputs = inputs_OGTT + inputs_Pop

inputs_to_scale = [
    "input_hr",
    "input_hr_min",
    "input_hr_max",
    "input_hr_std",
    "input_sulfonylurea",
    "input_sglt2",
    "input_glp1",
    "input_biguanide",
]

rename_cols_dict = {
    "cgm_value": "Gc",
    "meals_mealSize": "input_carbs",
    "heartRate_mean": "input_hr",
    "heartRate_min": "input_hr_min",
    "heartRate_max": "input_hr_max",
    "heartRate_std": "input_hr_std",
    "sleep_efficiency": "input_sleep",
    "meds_medicationDose$rapid_acting_insulin": "input_insulin",
    "grouped_meds_medicationGroup$sulfonylurea": "input_sulfonylurea",
    "grouped_meds_medicationGroup$sglt2": "input_sglt2",
    "grouped_meds_medicationGroup$glp1": "input_glp1",
    "grouped_meds_medicationGroup$biguanide": "input_biguanide",
}

nn = 64
n_neurons_ogtt = {"C1": 149, "C2": 140, "Gc": nn, "Ge": nn, "Ie": nn, "I": nn}

nn_aux = 256
n_neuron_ind = {
    "C1": [3, nn_aux, 1],
    "C2": [2, nn_aux, 1],
    "Gc": [13, nn_aux, 1],
    "Ge": [2, nn_aux, 1],
    "Ie": [2, nn_aux, 1],
    "I": [6, nn_aux, 1],
}
