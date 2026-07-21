from __future__ import annotations

import pandas as pd

from corrected_diagnostics import build_same_state_event_control_masks
from tick_lead_lag import build_lead_lag_matrix_report


def test_conditional_controls_use_the_same_state_bucket() -> None:
    frame = pd.DataFrame(
        {
            "is_primary_event": [True, False, False, True],
            "state_bucket": ["low", "low", "high", "high"],
        }
    )
    event, control = build_same_state_event_control_masks(frame, "state_bucket", "low")
    assert event.tolist() == [True, False, False, False]
    assert control.tolist() == [False, True, False, False]


def test_lead_lag_report_counts_fdr_rejections_not_raw_p_values() -> None:
    summary = pd.DataFrame(
        [
            {
                "leader": "AUSDT",
                "target": "BUSDT",
                "event_filter": "run_clustering_side=down",
                "delta_mean_return": 0.01,
                "delta_t_stat": 3.0,
                "event_count": 50,
                "n_raw_events": 50,
                "n_nonoverlapping_events": 30,
                "p_value_block": 0.01,
                "p_value_block_bh_fdr": 0.10,
                "reject_block_bh_fdr_05": False,
            },
            {
                "leader": "BUSDT",
                "target": "AUSDT",
                "event_filter": "run_clustering_side=down",
                "delta_mean_return": -0.02,
                "delta_t_stat": -4.0,
                "event_count": 60,
                "n_raw_events": 60,
                "n_nonoverlapping_events": 40,
                "p_value_block": 0.001,
                "p_value_block_bh_fdr": 0.04,
                "reject_block_bh_fdr_05": True,
            },
        ]
    )
    config = {
        "analysis": {
            "symbols": ["AUSDT", "BUSDT"],
            "interval_minutes": [15],
            "forward_horizons_minutes": [30],
        },
        "data": {"start": "2025-01-01", "end": "2025-02-01"},
    }
    report = build_lead_lag_matrix_report(config, pd.DataFrame(), summary, [])
    assert "BH-FDR 5%를 적용해 통과한 directed edge 수: 1건" in report
    assert "최강 음(-) edge: B" in report
    assert "최강 양(+) edge" not in report
