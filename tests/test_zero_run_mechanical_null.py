from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zero_run_mechanical_null import (
    MECHANISM_OUTCOMES,
    analyze_group_null_calibration,
    build_audit_decisions,
    conditional_run_z,
    draw_stratified_audit_sample,
    exact_run_count_pmf,
    validate_frozen_config,
    verify_preregistration_seal,
)


def frozen_config() -> dict:
    return {
        "data": {
            "symbols": [
                "BTCUSDT",
                "ETHUSDT",
                "XRPUSDT",
                "SOLUSDT",
                "DOGEUSDT",
                "ADAUSDT",
                "AVAXUSDT",
            ],
            "expected_rows": 490_560,
        },
        "analysis": {
            "monte_carlo_repetitions": 999,
            "stratification_quantiles": 5,
            "clustering_z_cutoff": -1.96,
            "mechanism_outcomes": MECHANISM_OUTCOMES,
            "minimum_group_rows": 1,
            "fdr_alpha": 0.05,
            "empirical_percentile_threshold": 0.95,
            "null_fpr_lower_bound": 0.015,
            "null_fpr_upper_bound": 0.035,
            "null_fpr_group_maximum": 0.05,
            "minimum_calibrated_group_share": 0.90,
            "minimum_supporting_assets": 5,
            "mechanism_minimum_count": 3,
            "mechanism_independent_outcomes": [
                "log_amihud_illiquidity",
                "abs_aggressor_imbalance",
            ],
        },
        "output": {"base_dir": "outputs/v2/final_research_completion_v1"},
    }


def count_success_runs(bits: tuple[int, ...]) -> int:
    return sum(value == 1 and (index == 0 or bits[index - 1] == 0) for index, value in enumerate(bits))


@pytest.mark.parametrize("n,k", [(5, 2), (7, 3), (8, 5)])
def test_exact_run_count_pmf_matches_binary_enumeration(n: int, k: int) -> None:
    observed: dict[int, int] = {}
    for positions in combinations(range(n), k):
        bits = tuple(int(index in positions) for index in range(n))
        runs = count_success_runs(bits)
        observed[runs] = observed.get(runs, 0) + 1
    expected_runs, probabilities = exact_run_count_pmf(n, k)
    expected = np.asarray([observed[int(run)] / len(list(combinations(range(n), k))) for run in expected_runs])
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-14)
    assert probabilities == pytest.approx(expected, abs=1e-14)


def test_conditional_run_z_vector_and_scalar_broadcast() -> None:
    result = conditional_run_z([2, 3], 4, [10, 12])
    assert result.shape == (2,)
    assert np.isfinite(result).all()
    expected_first = (2 - 4 * 7 / 10) / np.sqrt(4 * 6 * 3 * 7 / (10**2 * 9))
    assert result[0] == pytest.approx(expected_first)


def test_stratified_sample_is_deterministic_and_population_weighted() -> None:
    rows = []
    for symbol in ["A", "B"]:
        for tx_bin in range(1, 6):
            for zero_bin in range(1, 6):
                for index in range(6):
                    rows.append(
                        {
                            "symbol": symbol,
                            "transaction_count_quintile": tx_bin,
                            "zero_tick_share_quintile": zero_bin,
                            "bucket_start": pd.Timestamp("2025-01-01", tz="UTC")
                            + pd.Timedelta(minutes=len(rows)),
                            "value": index,
                        }
                    )
    frame = pd.DataFrame(rows)
    first, summary = draw_stratified_audit_sample(frame, 3, 17)
    second, _ = draw_stratified_audit_sample(frame, 3, 17)
    assert first[["symbol", "bucket_start"]].equals(second[["symbol", "bucket_start"]])
    assert len(first) == 2 * 5 * 5 * 3
    assert summary["sampling_weight"].eq(2.0).all()
    assert first["sampling_weight"].sum() == pytest.approx(len(frame))


def test_group_null_analysis_builds_frozen_23_test_families() -> None:
    rows = []
    for symbol_index, symbol in enumerate(frozen_config()["data"]["symbols"]):
        for quintile in range(1, 6):
            rows.append(
                {
                    "symbol": symbol,
                    "transaction_count_quintile": quintile,
                    "zero_tick_share_quintile": quintile,
                    "liquidity_quintile": quintile,
                    "zero_run_intensity": 3.0,
                    "run_z_zero": -3.0,
                    "sampling_weight": 1.0,
                }
            )
    sample = pd.DataFrame(rows)
    null = np.zeros((len(sample), 99), dtype=float)
    summary, replicates = analyze_group_null_calibration(sample, null, frozen_config())
    assert len(summary) == 23
    assert len(replicates) == 23 * 99
    assert summary["supports_excess_intensity"].all()
    assert summary["supports_excess_tail"].all()


def test_decision_requires_calibration_excess_and_independent_mechanism() -> None:
    groups = []
    for dimension, values in [
        ("pooled", ["all"]),
        ("asset", frozen_config()["data"]["symbols"]),
        ("transaction_count_quintile", [f"Q{i}" for i in range(1, 6)]),
        ("zero_tick_share_quintile", [f"Q{i}" for i in range(1, 6)]),
        ("liquidity_quintile", [f"Q{i}" for i in range(1, 6)]),
    ]:
        for value in values:
            groups.append(
                {
                    "dimension": dimension,
                    "group": value,
                    "null_fpr_mean": 0.025,
                    "empirical_clustering_share": 0.10,
                    "supports_excess_intensity": True,
                    "supports_excess_tail": True,
                }
            )
    mechanisms = pd.DataFrame(
        {
            "outcome": MECHANISM_OUTCOMES,
            "exceeds_count_conditioned_null": [True, True, True, False, False],
        }
    )
    decisions = build_audit_decisions(pd.DataFrame(groups), mechanisms, frozen_config())
    assert decisions["passed"].all()

    mechanisms["exceeds_count_conditioned_null"] = [False, True, True, True, False]
    failed = build_audit_decisions(pd.DataFrame(groups), mechanisms, frozen_config())
    assert not bool(failed.loc[failed["decision"].eq("mechanism_beyond_conditional_null"), "passed"].iloc[0])


def test_config_and_preregistration_seal_reject_drift(tmp_path: Path) -> None:
    config = frozen_config()
    validate_frozen_config(config)
    drifted = deepcopy(config)
    drifted["analysis"]["monte_carlo_repetitions"] = 998
    with pytest.raises(ValueError, match="999"):
        validate_frozen_config(drifted)

    protocol = tmp_path / "protocol.md"
    config_path = tmp_path / "config.yaml"
    seal_path = tmp_path / "seal.json"
    protocol.write_text("frozen", encoding="utf-8")
    config_path.write_text("frozen: true", encoding="utf-8")
    import hashlib
    import json

    seal_path.write_text(
        json.dumps(
            {
                "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "results_observed_at_seal": False,
            }
        ),
        encoding="utf-8",
    )
    assert verify_preregistration_seal(protocol, config_path, seal_path)["verified"]
    protocol.write_text("drifted", encoding="utf-8")
    with pytest.raises(ValueError, match="seal mismatch"):
        verify_preregistration_seal(protocol, config_path, seal_path)
