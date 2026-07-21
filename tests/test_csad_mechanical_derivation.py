from __future__ import annotations

import numpy as np
import pandas as pd

from csad_mechanical_derivation import (
    gaussian_closed_form_coefficients,
    gaussian_expected_csad,
    gaussian_moment_projection_coefficients,
    half_normal_moment,
    verify_gaussian_equations,
)
from csad_mechanical_reporting import build_master_report_update
from csad_mechanical_simulation import (
    simulate_mechanical_null_returns,
    simulate_research_weights,
)
from csad_null_simulation import construct_market_and_csad, fit_fast_hac_model


def test_half_normal_moments_match_known_gaussian_values() -> None:
    scale = 0.7
    assert np.isclose(half_normal_moment(1, scale), scale * np.sqrt(2.0 / np.pi))
    assert np.isclose(half_normal_moment(2, scale), scale**2)
    assert np.isclose(half_normal_moment(4, scale), 3.0 * scale**4)
    assert np.isclose(half_normal_moment(6, scale), 15.0 * scale**6)


def test_closed_form_matches_population_moment_normal_equations() -> None:
    verification = verify_gaussian_equations([14, 50, 62], 0.025, 1e-10)
    assert verification["equation_gate_pass"].all()
    assert verification["maximum_absolute_difference"].max() <= 1e-10


def test_gaussian_pseudo_true_targets_have_preregistered_signs() -> None:
    values = gaussian_closed_form_coefficients(0.025, 14)
    solved = gaussian_moment_projection_coefficients(0.025, 14)
    assert values["no_intercept_target"] < 0.0
    assert values["scsad_target"] < 0.0
    assert values["standard_target"] == 0.0
    assert values["restored_target"] == 0.0
    assert np.isclose(values["no_intercept_target"], solved["no_intercept_target"])
    assert np.isclose(values["scsad_target"], solved["scsad_target"])


def test_expected_csad_matches_iid_gaussian_simulation() -> None:
    rng = np.random.default_rng(101)
    returns = rng.normal(0.0, 0.025, size=(60_000, 14))
    market = returns.mean(axis=1)
    observed = np.mean(np.abs(returns - market[:, None]))
    expected = gaussian_expected_csad(0.025, 14)
    assert np.isclose(observed, expected, rtol=0.005)


def test_large_iid_gaussian_fit_approaches_mechanical_targets() -> None:
    rng = np.random.default_rng(2026)
    returns = simulate_mechanical_null_returns(
        "independent_gaussian",
        {"idiosyncratic_scale": 0.025},
        observations=40_000,
        assets=14,
        rng=rng,
    )
    weights = np.ones_like(returns)
    market, csad = construct_market_and_csad(returns, weights)
    theory = gaussian_closed_form_coefficients(0.025, 14)
    no_intercept = fit_fast_hac_model(csad, market, "no_intercept_csad")
    scsad = fit_fast_hac_model(csad, market, "scsad")
    assert np.isclose(
        no_intercept["target_coefficient"],
        theory["no_intercept_target"],
        rtol=0.08,
    )
    assert np.isclose(scsad["target_coefficient"], theory["scsad_target"], rtol=0.08)


def test_all_preregistered_dgps_are_finite_and_reproducible() -> None:
    configs = {
        "independent_gaussian": {"idiosyncratic_scale": 0.025},
        "common_factor": _factor_config(),
        "stochastic_volatility_factor": {
            **_factor_config(),
            "log_volatility_phi": 0.95,
            "log_volatility_innovation": 0.16,
        },
        "student_t_factor": {**_factor_config(), "degrees_of_freedom": 5},
        "skewed_factor": {**_factor_config(), "lognormal_shape": 0.65},
        "jump_diffusion_factor": {
            **_factor_config(),
            "common_jump_probability": 0.015,
            "common_jump_scale": 0.055,
            "idiosyncratic_jump_probability": 0.006,
            "idiosyncratic_jump_scale": 0.045,
        },
        "time_varying_correlation": {
            "total_scale": 0.03,
            "correlation_mean": 0.35,
            "correlation_phi": 0.97,
            "correlation_innovation": 0.16,
            "correlation_minimum": 0.05,
            "correlation_maximum": 0.85,
        },
        "asymmetric_common_shock": {
            **_factor_config(),
            "negative_volatility_multiplier": 1.8,
        },
    }
    for dgp, config in configs.items():
        first = simulate_mechanical_null_returns(
            dgp, config, 500, 8, np.random.default_rng(7)
        )
        second = simulate_mechanical_null_returns(
            dgp, config, 500, 8, np.random.default_rng(7)
        )
        assert first.shape == (500, 8)
        assert np.isfinite(first).all()
        assert np.allclose(first, second)


def test_weight_processes_are_normalized_and_lagged_process_ignores_returns() -> None:
    config = {
        "lognormal_sigma": 1.25,
        "liquidity_ar1": 0.97,
        "liquidity_innovation": 0.12,
        "concentrated_top_weight": 0.45,
        "minimum_weight": 1e-12,
    }
    for weighting in ("equal", "lagged_lognormal_liquidity", "concentrated_static"):
        first = simulate_research_weights(
            100, 14, weighting, config, np.random.default_rng(17)
        )
        second = simulate_research_weights(
            100, 14, weighting, config, np.random.default_rng(17)
        )
        assert np.all(first > 0.0)
        assert np.allclose(first.sum(axis=1), 1.0)
        assert np.allclose(first, second)
    concentrated = simulate_research_weights(
        10, 14, "concentrated_static", config, np.random.default_rng(18)
    )
    assert np.isclose(concentrated.max(), 0.45)


def test_master_report_update_is_additive_and_does_not_remove_prior_text() -> None:
    source = "# report\n\n최종 갱신: 2026-07-20\n\nprior evidence\n\n## 8. Tick 연구와 미래수익률\n\ntick evidence\n"
    decision = pd.DataFrame(
        [
            {
                "mechanical_convergence_cells_passed": 6,
                "mechanical_convergence_cells_total": 6,
                "nominal_control_cells_passed": 6,
                "nominal_control_cells_total": 6,
                "classification": "mechanical_null_confirmed",
            }
        ]
    )
    convergence = pd.DataFrame(
        {
            "model": ["no_intercept_csad", "scsad"],
            "raw_false_positive_rate": [1.0, 0.99],
        }
    )
    robustness = pd.DataFrame([{"passing_cells": 45, "evaluated_cells": 45}])
    updated = build_master_report_update(source, decision, convergence, robustness)
    assert "최종 갱신: 2026-07-21" in updated
    assert "## 7-1. 음의 계수가 생기는 수학적 원인" in updated
    assert "prior evidence" in updated
    assert "tick evidence" in updated


def _factor_config() -> dict[str, float]:
    return {
        "factor_scale": 0.018,
        "idiosyncratic_scale": 0.025,
        "factor_loading_mean": 1.0,
        "factor_loading_std": 0.2,
    }

