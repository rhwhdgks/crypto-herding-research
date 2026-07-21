from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from research_validation import fit_quantile_thresholds, save_threshold_artifact
from tick_event_schema import require_tick_schema_v2


def build_same_state_event_control_masks(
    frame: pd.DataFrame,
    state_bucket_column: str,
    bucket: str,
) -> tuple[pd.Series, pd.Series]:
    primary_event = frame["is_primary_event"].fillna(False).astype(bool)
    same_state = frame[state_bucket_column].eq(bucket)
    return primary_event & same_state, (~primary_event) & same_state


def _cluster_bootstrap_delta(frame: pd.DataFrame, event_mask: pd.Series, seed: int, draws: int) -> dict:
    working = frame[["bucket_start", "forward_return"]].copy()
    working["event"] = event_mask.reindex(frame.index, fill_value=False).to_numpy(dtype=bool)
    working["day"] = pd.to_datetime(working["bucket_start"], utc=True).dt.floor("D")
    daily = []
    for _, group in working.groupby("day"):
        event = group.loc[group["event"], "forward_return"].dropna()
        control = group.loc[~group["event"], "forward_return"].dropna()
        daily.append((event.sum(), len(event), control.sum(), len(control)))
    values = np.asarray(daily, dtype=float)
    if len(values) < 2 or values[:, 1].sum() < 2 or values[:, 3].sum() < 2:
        return {"bootstrap_se": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_value_block": np.nan}
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(values), size=(int(draws), len(values)))
    totals = values[selected].sum(axis=1)
    valid = (totals[:, 1] > 0) & (totals[:, 3] > 0)
    bootstrap = totals[valid, 0] / totals[valid, 1] - totals[valid, 2] / totals[valid, 3]
    event = working.loc[working["event"], "forward_return"].dropna()
    control = working.loc[~working["event"], "forward_return"].dropna()
    observed = float(event.mean() - control.mean())
    centered = bootstrap - bootstrap.mean()
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "bootstrap_se": float(bootstrap.std(ddof=1)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value_block": float((np.sum(np.abs(centered) >= abs(observed)) + 1) / (len(bootstrap) + 1)),
    }


def run_corrected_state_diagnostics(config: dict, output_dir: str | Path) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_cfg = config["input"]
    micro = pd.read_csv(input_cfg["micro_frame_path"])
    for column in ["bucket_start", "bucket_end", "signal_timestamp"]:
        micro[column] = pd.to_datetime(micro[column], utc=True)
    require_tick_schema_v2(micro)
    state = pd.read_csv(input_cfg["futures_state_path"])
    state["bucket_start"] = pd.to_datetime(state["bucket_start"], utc=True)
    flow = pd.read_csv(input_cfg["flow_state_path"])
    flow["bucket_start"] = pd.to_datetime(flow["bucket_start"], utc=True)

    analysis = config["analysis"]
    leader_symbol = analysis["leader"]
    horizon = int(analysis["horizon_minutes"])
    forward_column = f"forward_return_{horizon}m"
    leader = micro.loc[micro["symbol"] == leader_symbol].copy()
    leader["is_primary_event"] = (
        leader["is_micro_run_clustering_event"].fillna(False).astype(bool)
        & leader["run_clustering_side"].eq(analysis["run_clustering_side"])
        & leader["price_direction"].eq(analysis["price_direction"])
    )
    leader = leader.merge(flow, on="bucket_start", how="left", validate="one_to_one")
    state_columns = [column for column in analysis["state_columns"] if column in state.columns]
    leader = leader.merge(state[["bucket_start", *state_columns]], on="bucket_start", how="left", validate="one_to_one")
    btc = micro.loc[micro["symbol"] == analysis.get("volatility_reference", "BTCUSDT")].sort_values("bucket_start")
    vol = btc[["bucket_start", "bucket_return"]].copy()
    vol["vol_24h"] = vol["bucket_return"].rolling(int(analysis.get("volatility_window_buckets", 96))).std()
    leader = leader.merge(vol[["bucket_start", "vol_24h"]], on="bucket_start", how="left", validate="one_to_one")

    targets = micro.loc[micro["symbol"].isin(analysis["targets"]), ["symbol", "bucket_start", forward_column]].rename(
        columns={"symbol": "target", forward_column: "forward_return"}
    )
    frame = leader.merge(targets, on="bucket_start", how="inner", validate="one_to_many")
    fit_end = pd.Timestamp(config["sample_split"]["fit_end"])
    oos_start = pd.Timestamp(config["sample_split"]["oos_start"])
    oos_end = pd.Timestamp(config["sample_split"]["oos_end"])
    if fit_end >= oos_start:
        raise ValueError("Diagnostic threshold fit and OOS windows overlap")
    train_events = frame.loc[(frame["bucket_start"] <= fit_end) & frame["is_primary_event"]].drop_duplicates(
        ["bucket_start"]
    )
    threshold_columns = list(analysis["threshold_columns"])
    fit_frame = train_events[["bucket_start", *threshold_columns]].copy()
    quantile_map = {}
    for column in threshold_columns:
        fit_frame[f"{column}_q33"] = fit_frame[column]
        fit_frame[f"{column}_q67"] = fit_frame[column]
        quantile_map[f"{column}_q33"] = 1 / 3
        quantile_map[f"{column}_q67"] = 2 / 3
    artifact = fit_quantile_thresholds(fit_frame, "bucket_start", quantile_map)
    save_threshold_artifact(artifact, output / "diagnostic_thresholds.json")
    oos = frame.loc[(frame["bucket_start"] >= oos_start) & (frame["bucket_start"] <= oos_end)].copy()

    rows = []
    for column in threshold_columns:
        low = artifact.thresholds[f"{column}_q33"]
        high = artifact.thresholds[f"{column}_q67"]
        oos[f"{column}_bucket"] = np.select(
            [oos[column] <= low, oos[column] <= high], ["low", "mid"], default="high"
        )
        oos.loc[oos[column].isna(), f"{column}_bucket"] = "unavailable"
        for target, target_frame in oos.groupby("target"):
            for bucket in ["low", "mid", "high"]:
                event_mask, control_mask = build_same_state_event_control_masks(
                    target_frame,
                    state_bucket_column=f"{column}_bucket",
                    bucket=bucket,
                )
                event = target_frame.loc[event_mask, "forward_return"].dropna()
                control = target_frame.loc[control_mask, "forward_return"].dropna()
                inference = _cluster_bootstrap_delta(
                    target_frame.loc[event_mask | control_mask],
                    event_mask.loc[event_mask | control_mask],
                    seed=int(analysis.get("seed", 20260715)),
                    draws=int(analysis.get("bootstrap_draws", 1000)),
                )
                rows.append(
                    {
                        "target": target,
                        "state_variable": column,
                        "state_bucket": bucket,
                        "threshold_low": low,
                        "threshold_high": high,
                        "event_count": int(len(event)),
                        "control_count": int(len(control)),
                        "event_mean_return": float(event.mean()) if len(event) else np.nan,
                        "control_mean_return": float(control.mean()) if len(control) else np.nan,
                        "delta_mean_return": float(event.mean() - control.mean()) if len(event) and len(control) else np.nan,
                        "fit_end": fit_end,
                        "oos_start": oos_start,
                        **inference,
                    }
                )
    summary = pd.DataFrame(rows)
    valid_p = summary["p_value_block"].dropna().sort_values()
    summary["p_value_block_bh_fdr"] = np.nan
    if not valid_p.empty:
        adjusted = valid_p.to_numpy() * len(valid_p) / np.arange(1, len(valid_p) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        summary.loc[valid_p.index, "p_value_block_bh_fdr"] = np.clip(adjusted, 0.0, 1.0)
    summary["reject_block_bh_fdr_05"] = summary["p_value_block_bh_fdr"] <= 0.05
    summary.to_csv(output / "corrected_state_diagnostics.csv", index=False)
    (output / "diagnostic_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "corrected_oos_diagnostics",
                "threshold_artifact": asdict(artifact),
                "rows": int(len(summary)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary
