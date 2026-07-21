from __future__ import annotations

import numpy as np
import pandas as pd

from regression import run_csad_regression


def test_baseline_regression_defaults_to_hac_and_records_metadata() -> None:
    index = pd.date_range("2025-01-01", periods=100, freq="min", tz="UTC")
    market_return = pd.Series(np.linspace(-0.01, 0.01, len(index)), index=index)
    csad = pd.Series(0.002 + 0.3 * market_return.abs() + 2.0 * market_return.pow(2), index=index)
    _, _, _, model, summary = run_csad_regression(csad, market_return)
    assert model.cov_type == "HAC"
    assert summary["cov_type"] == "HAC"
