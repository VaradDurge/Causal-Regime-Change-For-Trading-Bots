"""Validation selection, fixed before any holdout or live number is read.

A tape passes when both twins have position standard deviation above 0.05,
both spend less than 80% of bars at |position| > 0.95, and the conditioned
twin's ablation pass rate is at least 10%. Returns are not an input.
The deficit is the sum of relative shortfalls, averaged across tapes by the caller.
"""

from __future__ import annotations

CLEAR_STD = 0.05
CLEAR_EXTREME = 0.80
CLEAR_ABLATION = 0.10


def meets_criterion(std_c, std_b, extreme_c, extreme_b, ablation) -> bool:
    return (
        float(std_c) > CLEAR_STD
        and float(std_b) > CLEAR_STD
        and float(extreme_c) < CLEAR_EXTREME
        and float(extreme_b) < CLEAR_EXTREME
        and float(ablation) >= CLEAR_ABLATION
    )


def criterion_deficit(std_c, std_b, extreme_c, extreme_b, ablation) -> float:
    """Zero when the tape passes. Otherwise the sum of relative shortfalls."""
    deficit = 0.0
    deficit += max(0.0, CLEAR_STD - float(std_c)) / CLEAR_STD
    deficit += max(0.0, CLEAR_STD - float(std_b)) / CLEAR_STD
    deficit += max(0.0, float(extreme_c) - CLEAR_EXTREME) / CLEAR_EXTREME
    deficit += max(0.0, float(extreme_b) - CLEAR_EXTREME) / CLEAR_EXTREME
    deficit += max(0.0, CLEAR_ABLATION - float(ablation)) / CLEAR_ABLATION
    return float(deficit)


# Pre-registered before the action-mode screen. Returns are not an input.
# A remaining tie prefers the continuous squash, then the discrete grid, then the clipped Gaussian.
_MODE_TIE = {"tanh_env": 2, "discrete5": 1, "gaussian_clip": 0}


def choose_action_mode(rows: list[dict]) -> dict:
    """Choose one action mode from validation criterion rows, or none.

    Each row is one tape of one mode and must carry std_c, std_b, extreme_c,
    extreme_b and ablation. A mode is eligible only when every one of its tapes
    passes ``meets_criterion``. The winner has the highest minimum ablation
    across its tapes. Ties break on the minimum twin position standard deviation,
    then on tanh_env, discrete5, gaussian_clip.
    """
    by_mode: dict[str, list[dict]] = {}
    for row in rows:
        by_mode.setdefault(str(row["mode"]), []).append(row)
    passing = []
    rejected = []
    for mode, tapes in by_mode.items():
        ok = all(
            meets_criterion(tape["std_c"], tape["std_b"], tape["extreme_c"], tape["extreme_b"], tape["ablation"])
            for tape in tapes
        )
        min_ablation = min(float(tape["ablation"]) for tape in tapes)
        min_std = min(min(float(tape["std_c"]), float(tape["std_b"])) for tape in tapes)
        record = {
            "mode": mode,
            "n_tapes": len(tapes),
            "passed": ok,
            "min_ablation": min_ablation,
            "min_position_std": min_std,
        }
        if ok:
            passing.append(record)
        else:
            rejected.append(record)
    if not passing:
        return {"mode": "none", "passed": [], "rejected": rejected}
    passing.sort(
        key=lambda item: (item["min_ablation"], item["min_position_std"], _MODE_TIE.get(item["mode"], -1)),
        reverse=True,
    )
    return {"mode": passing[0]["mode"], "passed": passing, "rejected": rejected}
