import numpy as np
import pandas as pd

from crta.report import identity_jumps, label_statistics


def test_label_statistics_and_identity_jump():
    labels = ["bull"] * 10 + ["crisis"] * 10
    frame = pd.DataFrame({"log_ret": np.linspace(-0.01, 0.01, 20), "vol_20": np.full(20, 0.02)})
    diag = {
        "component_mu": [np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[0.2, 0.0], [1.0, 1.0]])],
        "component_sd": [np.ones((2, 2)), np.ones((2, 2))],
        "assign": [np.array([0, 3]), np.array([0, 3])],
        "responsibility": [np.array([1.0, 1.0]), np.array([1.0, 1.0])],
        "matched_costs": [0.1, 0.3],
    }
    stats = label_statistics(labels, frame, periods=10, diag=diag)
    assert stats["per_regime"]["bull"]["share"] == 0.5
    assert stats["switches_per_year"] == 0.5
    assert stats["mean_matched_w2"] == 0.2
    jumps = identity_jumps(diag)
    assert jumps["bull"] == np.float64(0.04) or abs(jumps["bull"] - 0.04) < 1e-9
    assert jumps["crisis"] == 0.0
