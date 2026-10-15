import numpy as np

from crta.regime import REGIMES
from crta.verify import (
    audit_bar_v2,
    audit_many_v2,
    empirical_banks,
    ignore_action,
    inverted_action,
    label_agreement,
    probe_action,
    verifier_report,
)


def _bank_and_bars(n=240, seed=0):
    rng = np.random.default_rng(seed)
    bases = {
        "bull": np.array([0.86, 0.04, 0.07, 0.03]),
        "bear": np.array([0.05, 0.82, 0.08, 0.05]),
        "calm": np.array([0.08, 0.07, 0.78, 0.07]),
        "crisis": np.array([0.03, 0.06, 0.05, 0.86]),
    }
    rows = []
    for name in REGIMES:
        for _ in range(40):
            draw = bases[name] + rng.normal(0.0, 0.01, 4)
            draw = np.clip(draw, 1e-3, None)
            rows.append(draw / draw.sum())
    bank_rows = np.stack(rows)
    bars = []
    for _ in range(n):
        name = REGIMES[int(rng.integers(0, 4))]
        draw = 0.8 * bases[name] + 0.2 * rng.dirichlet(np.ones(4))
        bars.append(draw / draw.sum())
    return empirical_banks(bank_rows), np.stack(bars)


def _rate(action, probs, banks):
    def predict(batch):
        return np.array([action(row) for row in batch])

    result = audit_many_v2(predict, probs, banks)
    return float(result["passed"].mean()), result


def test_reference_policies_on_verifier_v2():
    banks, probs = _bank_and_bars()
    faithful, faithful_result = _rate(probe_action, probs, banks)
    ignoring, _ = _rate(ignore_action, probs, banks)
    inverted, _ = _rate(inverted_action, probs, banks)
    assert faithful > 0.90
    assert ignoring < 0.02
    assert inverted < 0.02
    report = verifier_report(faithful_result, regimes=np.array(REGIMES)[np.argmax(probs, axis=1)])
    assert report["median_ablation_delta"] > 0.0
    assert set(report["per_regime"]) == set(REGIMES)
    one = audit_bar_v2(probe_action, probs[0], banks)
    assert one["sensitivity_span"] >= 0.05


def test_a_flat_sensitivity_curve_fails_and_a_starved_regime_is_untestable():
    banks, probs = _bank_and_bars(n=30, seed=1)
    # Drop bull so bars that are not already bull cannot be pushed toward it.
    banks = dict(banks)
    banks["bull"] = np.zeros((0, 4))
    bear = np.array([0.05, 0.8, 0.1, 0.05])
    result = audit_bar_v2(probe_action, bear, banks)
    assert result["untestable"] == ["bull"]
    assert result["counterfactual"] == "untestable"
    flat = audit_bar_v2(ignore_action, bear, empirical_banks(np.stack([bear, np.array([0.8, 0.05, 0.1, 0.05])])))
    assert flat["sensitivity"] is False
    assert flat["sensitivity_span"] < 0.05


def test_label_agreement_and_many_matches_one_bar():
    banks, probs = _bank_and_bars(n=25, seed=2)
    dwell = [REGIMES[int(np.argmax(row))] for row in probs]
    assert label_agreement(probs, dwell) == 1.0
    assert label_agreement(probs, ["calm"] * len(probs)) < 1.0

    def predict(batch):
        return np.array([probe_action(row) for row in batch])

    many = audit_many_v2(predict, probs[:1], banks)
    one = audit_bar_v2(probe_action, probs[0], banks)
    assert bool(many["passed"][0]) is one["passed"]
    assert many["ablation_delta"][0] == np.float64(one["ablation_delta"]) or np.isclose(
        many["ablation_delta"][0], one["ablation_delta"]
    )
