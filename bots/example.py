"""Copy this file, edit act(), and pick it in the console.

features: log_ret, vol_20, trend, rsi, macd. Rows at t use prices up to t.
regime: bull, bear, calm, crisis. The auditor replaces this vector.
position: the bet you returned on the previous bar, in [-1, 1]. The first bar is 0.
Return the next bet in [-1, 1]. +1 is fully long. -1 is fully short.

A Stable-Baselines3 zip trained on a different observation belongs in act(),
not in the zip loader. Load it here and build the vector that model expects.
"""

import numpy as np

# bull, bear, calm, crisis. This copy is the hand rule, so the checks have a known answer.
_WEIGHTS = np.array([1.0, -0.4, 0.5, -1.0])


def act(features, regime, position):
    del features, position
    bet = float(np.dot(np.asarray(regime, dtype=float), _WEIGHTS))
    return float(np.clip(bet, -1.0, 1.0))
