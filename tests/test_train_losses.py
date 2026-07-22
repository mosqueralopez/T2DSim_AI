import torch

from t2dsim_ai.train_losses import loss_consistency, loss_fit, scaled_gc_limits_mgdl


def test_loss_fit_applies_hypo_hyper_penalties():
    lim_inf, lim_sup = scaled_gc_limits_mgdl()
    y_true = torch.tensor([[[lim_inf - 0.1]], [[lim_inf - 0.1]]])
    y_pred = torch.tensor([[[lim_inf + 0.2]], [[lim_inf + 0.2]]])

    base = loss_fit(
        y_true,
        y_true,
        lim_inferior=lim_inf,
        lim_superior=lim_sup,
        hypo_penalization=1.0,
        hyper_penalization=1.0,
        dcgm_weight=0.0,
    )
    penalized = loss_fit(
        y_pred,
        y_true,
        lim_inferior=lim_inf,
        lim_superior=lim_sup,
        hypo_penalization=10.0,
        hyper_penalization=10.0,
        dcgm_weight=0.0,
    )
    assert penalized.item() > base.item()


def test_loss_consistency_penalizes_negative_states():
    x = torch.tensor([[[-1.0, 0.5, 0.0, 0.0, 0.0, 0.0]]])
    state_mins = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    assert loss_consistency(x, state_mins).item() > 0.0
