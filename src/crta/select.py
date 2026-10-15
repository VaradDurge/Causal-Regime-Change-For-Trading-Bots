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
