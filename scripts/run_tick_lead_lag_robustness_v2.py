from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tick_event_schema import require_tick_schema_v2
from tick_lead_lag import run_lead_lag_matrix
from utils import load_config, save_config_snapshot, save_dataframe, save_input_manifest, save_provenance_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _event_filters(event_filter: dict) -> list[dict]:
    keys = list(event_filter)
    values = [value if isinstance(value, list) else [value] for value in event_filter.values()]
    return [{key: [value] for key, value in zip(keys, combination)} for combination in itertools.product(*values)]


def _run_matrix(frame: pd.DataFrame, analysis: dict) -> pd.DataFrame:
    return run_lead_lag_matrix(
        micro_frame=frame,
        symbols=list(analysis["symbols"]),
        interval_minutes=int(analysis["interval_minutes"]),
        forward_horizon_minutes=int(analysis["forward_horizon_minutes"]),
        event_filters=_event_filters(analysis["event_filter"]),
        min_inference_events=int(analysis.get("min_inference_events", 30)),
        min_inference_unique_days=int(analysis.get("min_inference_unique_days", 20)),
    )


def _annotate_family(matrix: pd.DataFrame, analysis: dict, scope: str) -> pd.DataFrame:
    family = {
        "scope": scope,
        "symbols": list(analysis["symbols"]),
        "event_filter": analysis["event_filter"],
        "horizon_minutes": int(analysis["forward_horizon_minutes"]),
    }
    payload = json.dumps(family, sort_keys=True, separators=(",", ":"))
    result = matrix.copy()
    result["multiple_testing_family_scope"] = scope
    result["multiple_testing_family_size"] = len(result)
    result["multiple_testing_family_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Schema-v2 lead-lag stability and regime analysis.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    analysis = config["analysis"]
    micro_path = PROJECT_ROOT / config["input"]["micro_frame_path"]
    micro = pd.read_csv(micro_path)
    for column in ["bucket_start", "bucket_end", "signal_timestamp"]:
        micro[column] = pd.to_datetime(micro[column], utc=True)
    require_tick_schema_v2(micro)
    output = PROJECT_ROOT / config["output"]["base_dir"]
    output.mkdir(parents=True, exist_ok=True)

    period_frames = []
    for period in analysis["periods"]:
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"])
        subset = micro.loc[(micro["bucket_start"] >= start) & (micro["bucket_start"] <= end)].copy()
        matrix = _annotate_family(_run_matrix(subset, analysis), analysis, f"period:{period['name']}")
        matrix["period"] = period["name"]
        matrix["period_start"] = start
        matrix["period_end"] = end
        period_frames.append(matrix)
    period_results = pd.concat(period_frames, ignore_index=True)
    save_dataframe(period_results, output / "period_stability_matrix.csv", index=False)

    pivot = period_results.pivot_table(
        index=["leader", "target", "event_filter"],
        columns="period",
        values="delta_mean_return",
    )
    stability = pivot.reset_index()
    period_names = [period["name"] for period in analysis["periods"]]
    available = [name for name in period_names if name in stability.columns]
    if available:
        signs = np.sign(stability[available])
        stability["same_sign_all_periods"] = signs.nunique(axis=1, dropna=True) <= 1
        stability["periods_available"] = stability[available].notna().sum(axis=1)
    save_dataframe(stability, output / "period_stability_summary.csv", index=False)

    regime_cfg = analysis["volatility_regime"]
    btc = micro.loc[micro["symbol"] == regime_cfg.get("reference_symbol", "BTCUSDT")].copy()
    btc = btc.sort_values("bucket_start")
    btc["rolling_volatility"] = btc["bucket_return"].rolling(int(regime_cfg["window_buckets"])).std()
    fit_end = pd.Timestamp(regime_cfg["fit_end"])
    oos_start = pd.Timestamp(regime_cfg["oos_start"])
    if fit_end >= oos_start:
        raise ValueError("Volatility regime fit and OOS windows overlap")
    threshold = float(
        btc.loc[btc["bucket_start"] <= fit_end, "rolling_volatility"].dropna().quantile(
            float(regime_cfg.get("quantile", 0.5))
        )
    )
    regime = btc[["bucket_start", "rolling_volatility"]].copy()
    regime["volatility_regime"] = np.where(regime["rolling_volatility"] >= threshold, "high", "low")
    oos = micro.loc[micro["bucket_start"] >= oos_start].merge(
        regime[["bucket_start", "volatility_regime"]], on="bucket_start", how="left"
    )
    regime_frames = []
    for regime_name in ["low", "high"]:
        matrix = _annotate_family(
            _run_matrix(oos.loc[oos["volatility_regime"] == regime_name].copy(), analysis),
            analysis,
            f"volatility_regime:{regime_name}",
        )
        matrix["volatility_regime"] = regime_name
        matrix["volatility_threshold"] = threshold
        matrix["threshold_fit_end"] = fit_end
        matrix["oos_start"] = oos_start
        regime_frames.append(matrix)
    save_dataframe(pd.concat(regime_frames, ignore_index=True), output / "volatility_regime_matrix.csv", index=False)
    save_config_snapshot(config, output / "config_snapshot.yaml")
    input_manifest = save_input_manifest([micro_path], output / "input_manifest.json")
    save_provenance_manifest(
        config,
        output / "provenance.json",
        schema_version=2,
        pipeline_version="tick-semantics-v2",
        statistical_method="period stability; train-fitted volatility regime; UTC-day cluster bootstrap; BH-FDR",
        input_manifest_path=input_manifest,
        random_seed=20260715,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
