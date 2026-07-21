from __future__ import annotations

import numpy as np
import pandas as pd

from cmc_structural_breaks import (
    DESIGN_COLUMNS,
    build_break_date_table,
    build_paper_regime_comparison,
    fit_scsad_regimes,
    run_no_break_stability_diagnostics,
    search_structural_breaks,
)


def test_exact_dynamic_programming_recovers_two_known_breaks() -> None:
    frame = _piecewise_linear_frame()
    result = search_structural_breaks(
        frame,
        {
            "trimming_fraction": 0.20,
            "maximum_breaks": 2,
            "primary_criterion": "bic",
            "strong_evidence_delta_bic": 10.0,
            "report_criteria": ["aic", "bic", "hqic"],
        },
    )

    assert result.selected_break_count == 2
    assert result.structural_change_supported
    assert result.minimum_segment_size == 72
    assert abs(result.selected_break_indices[0] - 120) <= 2
    assert abs(result.selected_break_indices[1] - 240) <= 2
    assert all(
        right - left >= result.minimum_segment_size
        for left, right in zip(
            [0, *result.selected_break_indices],
            [*result.selected_break_indices, len(frame)],
        )
    )


def test_regime_hac_results_and_break_date_matching_are_complete() -> None:
    rng = np.random.default_rng(13)
    index = pd.date_range("2020-01-01", periods=240, freq="D", tz="UTC")
    market = pd.Series(rng.normal(0.0, 0.04, len(index)), index=index)
    csad = 0.03 + 0.55 * market.abs() - 2.0 * market.pow(2)
    csad += pd.Series(rng.normal(0.0, 0.001, len(index)), index=index)
    frame = pd.DataFrame({"csad": csad, "market_return": market})
    frame["scsad"] = np.where(market.ge(0), csad, -csad)
    frame["const"] = 1.0
    frame["market_return_sq"] = market.pow(2)
    frame["market_return_cu"] = market.pow(3)

    targets, coefficients, diagnostics, fitted = fit_scsad_regimes(
        frame,
        [120],
        {"cov_type": "HAC", "hac_maxlags": "auto", "fdr_alpha": 0.05},
        solution_name="test",
    )
    dates = build_break_date_table(
        index,
        [120],
        ["2020-04-29"],
        solution_name="test",
    )

    assert len(targets) == 2
    assert targets["gamma3_q_value_bh_fdr"].between(0, 1).all()
    assert targets["standardized_gamma3"].notna().all()
    assert targets["mean_csad"].gt(0).all()
    assert set(coefficients["term"]) == set(DESIGN_COLUMNS)
    assert not diagnostics.empty
    assert fitted.index.equals(index)
    assert dates.loc[0, "next_regime_start"] == index[120]
    assert dates.loc[0, "signed_calendar_day_difference"] == 1

    comparison = build_paper_regime_comparison(
        targets,
        {
            "regime_gamma3": [
                {"regime": 1, "gamma3": -2.0, "t_stat": -2.5, "nobs": 120},
                {"regime": 2, "gamma3": -2.0, "t_stat": -2.5, "nobs": 120},
            ]
        },
    )
    assert len(comparison) == 2
    assert comparison["coefficient_sign_matches"].all()


def test_no_break_stability_diagnostics_return_both_declared_tests() -> None:
    frame = _piecewise_linear_frame()
    diagnostics = run_no_break_stability_diagnostics(frame)

    assert set(diagnostics["test"]) == {
        "hansen_parameter_instability",
        "cusum_ols_residuals",
    }
    assert diagnostics["statistic"].notna().all()
    assert diagnostics["rejects_stability_5pct"].dtype == bool


def _piecewise_linear_frame() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    nobs = 360
    index = pd.date_range("2018-01-01", periods=nobs, freq="D", tz="UTC")
    market = rng.normal(0.0, 0.04, nobs)
    x = np.column_stack([np.ones(nobs), market, market**2, market**3])
    coefficients = np.vstack(
        [
            np.tile([0.02, 0.5, -0.5, -2.0], (120, 1)),
            np.tile([-0.01, 1.2, 4.0, 35.0], (120, 1)),
            np.tile([0.04, -0.8, -5.0, -45.0], (120, 1)),
        ]
    )
    scsad = np.einsum("ni,ni->n", x, coefficients)
    scsad += rng.normal(0.0, 0.0002, nobs)
    frame = pd.DataFrame(
        {
            "scsad": scsad,
            "const": 1.0,
            "market_return": market,
            "market_return_sq": market**2,
            "market_return_cu": market**3,
        },
        index=index,
    )
    return frame
