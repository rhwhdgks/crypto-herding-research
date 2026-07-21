from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from research_validation import (
    CandidateSpec,
    apply_thresholds,
    evaluate_candidate_grid,
    fit_quantile_thresholds,
    run_selection_aware_permutation,
    save_threshold_artifact,
    simulate_complete_basket,
)
from tick_event_schema import require_tick_schema_v2


def load_corrected_candidate_frame(config: dict) -> pd.DataFrame:
    input_cfg = config["input"]
    micro = pd.read_csv(input_cfg["micro_frame_path"])
    for column in ["bucket_start", "bucket_end", "signal_timestamp"]:
        if column in micro.columns:
            micro[column] = pd.to_datetime(micro[column], utc=True)
    require_tick_schema_v2(micro)

    state = pd.read_csv(input_cfg["futures_state_path"])
    state_timestamp = str(input_cfg.get("state_timestamp_column", "bucket_start"))
    state[state_timestamp] = pd.to_datetime(state[state_timestamp], utc=True)
    state = state.rename(columns={state_timestamp: "bucket_start"})

    family = config["candidate_family"]
    leaders = sorted({item["leader"] for item in family})
    targets = sorted({target for item in family for target in item["targets"]})
    leader_columns = [
        "symbol",
        "bucket_start",
        "signal_timestamp",
        "hour_utc",
        "is_micro_run_clustering_event",
        "run_clustering_side",
        "price_direction",
        "aggressor_direction",
    ]
    leader = micro.loc[micro["symbol"].isin(leaders), leader_columns].rename(
        columns={
            "symbol": "leader",
            "is_micro_run_clustering_event": "is_event",
        }
    )
    state_columns = [column for column in input_cfg.get("state_columns", []) if column in state.columns]
    leader = leader.merge(state[["bucket_start", *state_columns]], on="bucket_start", how="left", validate="many_to_one")

    horizon_values = sorted({int(item["horizon_minutes"]) for item in family})
    rows = []
    for horizon in horizon_values:
        return_column = f"forward_return_{horizon}m"
        target = micro.loc[
            micro["symbol"].isin(targets), ["symbol", "bucket_start", return_column]
        ].rename(columns={"symbol": "target", return_column: "forward_return"})
        merged = leader.merge(target, on="bucket_start", how="inner", validate="many_to_many")
        merged["horizon_minutes"] = horizon
        rows.append(merged)
    return pd.concat(rows, ignore_index=True).sort_values(["bucket_start", "leader", "target"]).reset_index(drop=True)


def candidate_specs_from_config(config: dict, threshold_column: str) -> tuple[CandidateSpec, ...]:
    specs = []
    for item in config["candidate_family"]:
        oi_rule = item.get("oi_rule")
        if oi_rule == "above_fitted_threshold":
            oi_rule = f"{threshold_column} > {threshold_column}_fitted_threshold"
        elif oi_rule == "below_or_equal_fitted_threshold":
            oi_rule = f"{threshold_column} <= {threshold_column}_fitted_threshold"
        specs.append(
            CandidateSpec(
                leader=str(item["leader"]),
                targets=tuple(item["targets"]),
                price_direction=str(item["price_direction"]),
                run_clustering_side=item.get("run_clustering_side"),
                horizon_minutes=int(item["horizon_minutes"]),
                session_hours_utc=tuple(int(value) for value in item["session_hours_utc"]),
                funding_threshold=item.get("funding_threshold"),
                oi_rule=oi_rule,
            )
        )
    return tuple(specs)


def run_corrected_candidate_validation(config: dict, output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = load_corrected_candidate_frame(config)
    split = config["sample_split"]
    train_start = pd.Timestamp(split["train_start"])
    train_end = pd.Timestamp(split["train_end"])
    oos_start = pd.Timestamp(split["oos_start"])
    oos_end = pd.Timestamp(split["oos_end"])
    for name, timestamp in {
        "train_start": train_start,
        "train_end": train_end,
        "oos_start": oos_start,
        "oos_end": oos_end,
    }.items():
        if timestamp.tzinfo is None:
            raise ValueError(f"{name} must include an explicit timezone")
    if train_end >= oos_start:
        raise ValueError("Training and OOS windows overlap")

    train = frame.loc[(frame["bucket_start"] >= train_start) & (frame["bucket_start"] <= train_end)].copy()
    oos = frame.loc[(frame["bucket_start"] >= oos_start) & (frame["bucket_start"] <= oos_end)].copy()
    threshold_cfg = config["threshold_fit"]
    threshold_column = str(threshold_cfg["column"])
    fit_source = train.drop_duplicates(["leader", "bucket_start"])
    artifact = fit_quantile_thresholds(
        fit_source,
        timestamp_column="bucket_start",
        quantiles={threshold_column: float(threshold_cfg["quantile"])},
    )
    save_threshold_artifact(artifact, output / "threshold_artifact.json")
    oos = apply_thresholds(oos, artifact, "bucket_start")
    specs = candidate_specs_from_config(config, threshold_column)

    permutation_cfg = config["permutation"]
    observed, permutation = run_selection_aware_permutation(
        oos,
        specs,
        n_draws=int(permutation_cfg["n_draws"]),
        seed=int(permutation_cfg["seed"]),
        min_shift=int(permutation_cfg.get("min_shift", 1)),
        statistic=str(permutation_cfg.get("statistic", "t_stat")),
    )
    observed.to_csv(output / "candidate_grid_observed.csv", index=False)
    (output / "selection_aware_permutation.json").write_text(
        json.dumps(permutation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    statistic_column = str(permutation_cfg.get("statistic", "t_stat"))
    evaluable = observed.dropna(subset=[statistic_column]).sort_values(statistic_column, ascending=False)
    if evaluable.empty:
        empty_trades = pd.DataFrame(
            columns=[
                "signal_timestamp",
                "entry_timestamp",
                "exit_timestamp",
                "gross_return",
                "fee_return",
                "slippage_return",
                "funding_return",
                "net_return",
                "equity_after",
            ]
        )
        empty_trades.to_csv(output / "execution_trade_log.csv", index=False)
        result = {
            "schema_version": 2,
            "status": "no_evaluable_oos_candidate",
            "selected_candidate": None,
            "selected_candidate_id": None,
            "selection_aware_p_value": permutation["p_value_add_one"],
            "execution": {"status": "not_run_no_evaluable_candidate", "trade_count": 0},
            "threshold_artifact": asdict(artifact),
        }
        (output / "corrected_candidate_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result
    best_id = evaluable.iloc[0]["candidate_id"]
    best_spec = next(spec for spec in specs if spec.candidate_id == best_id)
    event_mask = (
        oos["leader"].eq(best_spec.leader)
        & oos["target"].isin(best_spec.targets)
        & oos["is_event"].fillna(False).astype(bool)
        & oos["price_direction"].eq(best_spec.price_direction)
        & oos["hour_utc"].isin(best_spec.session_hours_utc)
    )
    if best_spec.run_clustering_side is not None:
        event_mask &= oos["run_clustering_side"].eq(best_spec.run_clustering_side)
    if best_spec.funding_threshold is not None:
        event_mask &= oos["funding_pre"] > best_spec.funding_threshold
    if best_spec.oi_rule:
        event_mask &= oos.eval(best_spec.oi_rule)
    signal_returns = oos.loc[event_mask, ["signal_timestamp", "target", "forward_return"]].rename(
        columns={"forward_return": "gross_return"}
    )

    execution_cfg = config["execution"]
    trades, execution_summary = simulate_complete_basket(
        signal_returns,
        basket_targets=best_spec.targets,
        holding_minutes=best_spec.horizon_minutes,
        overlap_policy=str(execution_cfg.get("overlap_policy", "skip_while_position_open")),
        round_trip_fee=float(execution_cfg["round_trip_fee"]),
        slippage=float(execution_cfg.get("slippage", 0.0)),
        funding=float(execution_cfg.get("funding", 0.0)),
        execution_latency_minutes=float(execution_cfg.get("latency_minutes", 0.0)),
        execution_mode=str(execution_cfg.get("mode", "taker")),
        maker_fill_probability=execution_cfg.get("maker_fill_probability"),
        execution_price_proxy=str(
            execution_cfg.get("price_proxy", "bucket_close_return_with_explicit_slippage")
        ),
        maker_adverse_selection=float(execution_cfg.get("maker_adverse_selection", 0.0)),
        fill_seed=int(execution_cfg.get("fill_seed", 20260715)),
    )
    trades.to_csv(output / "execution_trade_log.csv", index=False)
    result = {
        "schema_version": 2,
        "status": "corrected_oos_validation",
        "selected_candidate": asdict(best_spec),
        "selected_candidate_id": best_spec.candidate_id,
        "selection_aware_p_value": permutation["p_value_add_one"],
        "execution": execution_summary,
        "threshold_artifact": asdict(artifact),
    }
    (output / "corrected_candidate_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
