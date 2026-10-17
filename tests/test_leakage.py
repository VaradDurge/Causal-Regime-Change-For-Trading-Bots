import numpy as np
import pytest

from crta.leakage import lag_rows


def test_lag_rows_hands_bar_t_the_row_from_t_minus_k():
    values = np.arange(10.0).reshape(5, 2)
    lagged = lag_rows(values, 2)
    assert lagged[2:].tolist() == values[:3].tolist()
    assert lagged[:2].tolist() == [values[0].tolist()] * 2


def test_lag_zero_is_a_copy():
    values = np.ones((3, 4))
    lagged = lag_rows(values, 0)
    assert np.array_equal(lagged, values)
    assert lagged is not values


def test_lag_must_not_be_negative():
    with pytest.raises(ValueError):
        lag_rows(np.ones((3, 1)), -1)
