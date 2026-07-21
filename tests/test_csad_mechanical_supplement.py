from __future__ import annotations

import pandas as pd

from csad_mechanical_supplement import (
    build_master_supplement_update,
    build_supplement_decision,
    build_supplement_gates,
)


def test_supplement_gates_only_use_longest_sample_as_primary() -> None:
    rows = []
    for observations in (3000, 12000, 48000):
        for assets in (14, 50, 62):
            for model in (
                "standard_csad",
                "no_intercept_csad",
                "intercept_restored",
                "scsad",
            ):
                mechanical = model in {"no_intercept_csad", "scsad"}
                rows.append(
                    {
                        "observations": observations,
                        "assets": assets,
                        "model": model,
                        "theoretical_target_coefficient": -1.0 if mechanical else 0.0,
                        "mean_target_coefficient": -1.01 if mechanical else 0.0,
                        "absolute_relative_error": 0.01 if mechanical else float("nan"),
                        "theory_inside_mean_ci": True,
                        "negative_coefficient_rate": 1.0 if mechanical else 0.5,
                        "raw_false_positive_rate": 1.0 if mechanical else 0.05,
                        "bh3_false_positive_rate": 1.0 if mechanical else 0.03,
                    }
                )
    config = _config()
    gates = build_supplement_gates(pd.DataFrame(rows), config)
    primary = gates.loc[gates["primary_supplement_cell"]]
    assert len(primary) == 12
    assert primary["gate_pass"].all()
    descriptive = gates.loc[~gates["primary_supplement_cell"]]
    assert descriptive["gate_pass"].isna().all()


def test_supplement_decision_preserves_parent_failure() -> None:
    parent = pd.DataFrame(
        [
            {
                "classification": "mechanical_null_not_confirmed",
                "mechanical_convergence_cells_passed": 5,
                "mechanical_convergence_cells_total": 6,
            }
        ]
    )
    equations = pd.DataFrame({"equation_gate_pass": [True, True, True]})
    summary = _passing_summary()
    gates = build_supplement_gates(summary, _config())
    decision = build_supplement_decision(parent, equations, gates).iloc[0]
    assert decision["parent_classification_preserved"]
    assert decision["parent_mechanical_convergence_passed"] == 5
    assert decision["finite_sample_convergence_supported"]
    assert decision["analytic_mechanism_established"]


def test_master_supplement_update_is_transparent_about_parent_failure() -> None:
    marker = "상세 보고서: `outputs/v2/csad_mechanical_derivation_v1/csad_mechanical_derivation_report.md`"
    source = f"before\n\n{marker}\n\nafter\n"
    decision = pd.DataFrame(
        [
            {
                "supplement_mechanical_cells_passed": 6,
                "supplement_mechanical_cells_total": 6,
                "supplement_control_cells_passed": 6,
                "supplement_control_cells_total": 6,
                "supplement_classification": "finite_sample_convergence_supported",
            }
        ]
    )
    updated = build_master_supplement_update(source, decision)
    assert "원 v1은" in updated
    assert "5/6으로 실패" in updated
    assert "v1.1 유한표본 수렴 보충" in updated
    assert "after" in updated


def _config() -> dict:
    return {
        "simulation": {"observations": [3000, 12000, 48000]},
        "theory": {
            "relative_error_maximum": 0.025,
            "negative_sign_rate_minimum": 0.95,
            "mechanical_raw_fpr_minimum": 0.95,
            "mechanical_bh_fpr_minimum": 0.95,
            "nominal_raw_fpr_maximum": 0.075,
        },
    }


def _passing_summary() -> pd.DataFrame:
    rows = []
    for assets in (14, 50, 62):
        for model in (
            "standard_csad",
            "no_intercept_csad",
            "intercept_restored",
            "scsad",
        ):
            mechanical = model in {"no_intercept_csad", "scsad"}
            rows.append(
                {
                    "observations": 48000,
                    "assets": assets,
                    "model": model,
                    "theoretical_target_coefficient": -1.0 if mechanical else 0.0,
                    "mean_target_coefficient": -1.01 if mechanical else 0.0,
                    "absolute_relative_error": 0.01 if mechanical else float("nan"),
                    "theory_inside_mean_ci": True,
                    "negative_coefficient_rate": 1.0 if mechanical else 0.5,
                    "raw_false_positive_rate": 1.0 if mechanical else 0.05,
                    "bh3_false_positive_rate": 1.0 if mechanical else 0.03,
                }
            )
    return pd.DataFrame(rows)

