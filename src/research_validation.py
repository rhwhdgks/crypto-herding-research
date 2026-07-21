from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateSpec:
    leader: str
    targets: tuple[str, ...]
    price_direction: str
    run_clustering_side: str | None
    horizon_minutes: int
    session_hours_utc: tuple[int, ...]
    funding_threshold: float | None = None
    oi_rule: str | None = None

    @property
    def candidate_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ThresholdArtifact:
    schema_version: int
    fit_start: str
    fit_end: str
    created_at: str
    source_hash: str
    thresholds: dict[str, float]


def fit_quantile_thresholds(
    train_frame: pd.DataFrame,
    timestamp_column: str,
    quantiles: dict[str, float],
) -> ThresholdArtifact:
    if train_frame.empty:
        raise ValueError("Training frame is empty")
    timestamps = pd.to_datetime(train_frame[timestamp_column], utc=True)
    ordered = train_frame.assign(**{timestamp_column: timestamps}).sort_values(timestamp_column)
    values = {
        name: float(pd.to_numeric(ordered[name], errors="coerce").dropna().quantile(quantile))
        for name, quantile in quantiles.items()
    }
    source_bytes = ordered.to_csv(index=False).encode("utf-8")
    return ThresholdArtifact(
        schema_version=2,
        fit_start=timestamps.min().isoformat(),
        fit_end=timestamps.max().isoformat(),
        created_at=datetime.now(timezone.utc).isoformat(),
        source_hash=hashlib.sha256(source_bytes).hexdigest(),
        thresholds=values,
    )


def save_threshold_artifact(artifact: ThresholdArtifact, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(artifact), ensure_ascii=False, indent=2), encoding="utf-8")


def load_threshold_artifact(path: str | Path) -> ThresholdArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    artifact = ThresholdArtifact(**payload)
    if artifact.schema_version != 2:
        raise ValueError(f"Unsupported threshold schema: {artifact.schema_version}")
    return artifact


def validate_oos_against_artifact(
    oos_frame: pd.DataFrame,
    artifact: ThresholdArtifact,
    timestamp_column: str,
) -> None:
    if oos_frame.empty:
        raise ValueError("OOS frame is empty")
    oos_start = pd.to_datetime(oos_frame[timestamp_column], utc=True).min()
    fit_end = pd.Timestamp(artifact.fit_end)
    if oos_start <= fit_end:
        raise ValueError(f"Train/OOS overlap: OOS starts {oos_start}, fit ends {fit_end}")


def apply_thresholds(
    oos_frame: pd.DataFrame,
    artifact: ThresholdArtifact,
    timestamp_column: str,
) -> pd.DataFrame:
    validate_oos_against_artifact(oos_frame, artifact, timestamp_column)
    result = oos_frame.copy()
    for name, value in artifact.thresholds.items():
        result[f"{name}_fitted_threshold"] = float(value)
    result["threshold_fit_start"] = artifact.fit_start
    result["threshold_fit_end"] = artifact.fit_end
    result["threshold_source_hash"] = artifact.source_hash
    return result


def evaluate_candidate_grid(
    frame: pd.DataFrame,
    candidates: tuple[CandidateSpec, ...],
    event_masks: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        mask = (
            frame["leader"].eq(candidate.leader)
            & frame["target"].isin(candidate.targets)
            & frame["hour_utc"].isin(candidate.session_hours_utc)
        )
        if event_masks is None:
            mask &= frame["is_event"].fillna(False).astype(bool)
            mask &= frame["price_direction"].eq(candidate.price_direction)
            if candidate.run_clustering_side is not None:
                mask &= frame["run_clustering_side"].eq(candidate.run_clustering_side)
        if candidate.funding_threshold is not None:
            mask &= frame["funding_pre"] > candidate.funding_threshold
        if candidate.oi_rule:
            mask &= frame.eval(candidate.oi_rule)
        if event_masks is not None:
            mask &= event_masks[candidate.candidate_id].reindex(frame.index, fill_value=False)
        sample = pd.to_numeric(frame.loc[mask, "forward_return"], errors="coerce").dropna()
        standard_error = sample.std(ddof=1) / np.sqrt(len(sample)) if len(sample) >= 2 else np.nan
        t_stat = float(sample.mean() / standard_error) if standard_error and np.isfinite(standard_error) else np.nan
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "n": int(len(sample)),
                "mean_return": float(sample.mean()) if not sample.empty else np.nan,
                "t_stat": t_stat,
            }
        )
    return pd.DataFrame(rows)


def run_selection_aware_permutation(
    frame: pd.DataFrame,
    candidates: tuple[CandidateSpec, ...],
    n_draws: int,
    seed: int,
    min_shift: int = 1,
    statistic: str = "t_stat",
) -> tuple[pd.DataFrame, dict]:
    if not candidates:
        raise ValueError("Candidate family must be fixed and non-empty")
    observed = evaluate_candidate_grid(frame, candidates)
    finite_observed = pd.to_numeric(observed[statistic], errors="coerce").dropna()
    if finite_observed.empty:
        return observed, {
            "method": "selection_aware_circular_shift_max_stat",
            "status": "no_evaluable_candidate",
            "statistic": statistic,
            "observed_best": np.nan,
            "p_value_add_one": np.nan,
            "n_draws_requested": int(n_draws),
            "n_draws_valid": 0,
            "seed": int(seed),
            "min_shift": int(min_shift),
            "candidate_family": [asdict(candidate) for candidate in candidates],
            "draw_shifts": [],
            "null_best_statistics": [],
        }
    observed_best = float(finite_observed.max())
    if "bucket_start" not in frame.columns:
        raise ValueError("Selection-aware permutation requires bucket_start")
    # Shift the complete candidate-selection state against future returns.  The
    # event semantics are always part of the state, even when a candidate does
    # not use optional funding/OI filters.
    state_columns = {
        column
        for column in ("is_event", "price_direction", "run_clustering_side")
        if column in frame.columns
    }
    if any(candidate.funding_threshold is not None for candidate in candidates):
        state_columns.add("funding_pre")
    for candidate in candidates:
        if candidate.oi_rule:
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", candidate.oi_rule):
                if token in frame.columns and not token.endswith("_fitted_threshold"):
                    state_columns.add(token)
    if not state_columns:
        raise ValueError("Selection-aware permutation requires event-state columns")
    leaders = sorted({candidate.leader for candidate in candidates})
    base_states = {}
    for leader in leaders:
        leader_state = frame.loc[frame["leader"].eq(leader)].sort_values("bucket_start").drop_duplicates("bucket_start")
        base_states[leader] = leader_state[["bucket_start", *sorted(state_columns)]].copy()
    shortest_state = min(len(state) for state in base_states.values())
    if shortest_state <= min_shift + 1:
        raise ValueError("Not enough rows for the requested circular shifts")
    rng = np.random.default_rng(seed)
    null_best = []
    shifts = []
    for _ in range(int(n_draws)):
        shift = int(rng.integers(min_shift, shortest_state))
        shifts.append(shift)
        draw_frame = frame.copy()
        frame_timestamps = pd.to_datetime(draw_frame["bucket_start"], utc=True)
        for leader, state in base_states.items():
            leader_rows = draw_frame["leader"].eq(leader)
            timestamps = pd.to_datetime(state["bucket_start"], utc=True).to_numpy()
            for column in state_columns:
                shifted_by_timestamp = dict(zip(timestamps, np.roll(state[column].to_numpy(), shift)))
                draw_frame.loc[leader_rows, column] = frame_timestamps.loc[leader_rows].map(shifted_by_timestamp).to_numpy()
        draw = evaluate_candidate_grid(draw_frame, candidates)
        null_best.append(float(draw[statistic].max()))
    null_values = np.asarray(null_best, dtype=float)
    finite = null_values[np.isfinite(null_values)]
    exceedances = int(np.sum(finite >= observed_best))
    p_value = float((exceedances + 1) / (len(finite) + 1))
    metadata = {
        "method": "selection_aware_circular_shift_max_stat",
        "status": "ok",
        "statistic": statistic,
        "observed_best": observed_best,
        "p_value_add_one": p_value,
        "n_draws_requested": int(n_draws),
        "n_draws_valid": int(len(finite)),
        "seed": int(seed),
        "min_shift": int(min_shift),
        "shifted_state_columns": sorted(state_columns),
        "candidate_family": [asdict(candidate) for candidate in candidates],
        "draw_shifts": shifts,
        "null_best_statistics": null_best,
    }
    return observed, metadata


def simulate_complete_basket(
    signal_returns: pd.DataFrame,
    basket_targets: tuple[str, ...],
    holding_minutes: int,
    overlap_policy: str,
    round_trip_fee: float,
    slippage: float = 0.0,
    funding: float = 0.0,
    execution_latency_minutes: float = 0.0,
    execution_mode: str = "taker",
    maker_fill_probability: float | None = None,
    execution_price_proxy: str = "bucket_close_return_with_explicit_slippage",
    maker_adverse_selection: float = 0.0,
    fill_seed: int = 20260715,
) -> tuple[pd.DataFrame, dict]:
    required = {"signal_timestamp", "target", "gross_return"}
    missing = required.difference(signal_returns.columns)
    if missing:
        raise ValueError(f"Missing execution columns: {', '.join(sorted(missing))}")
    frame = signal_returns.copy()
    frame["signal_timestamp"] = pd.to_datetime(frame["signal_timestamp"], utc=True)
    wide = frame.pivot_table(index="signal_timestamp", columns="target", values="gross_return", aggfunc="first")
    basket_wide = wide.reindex(columns=list(basket_targets))
    complete_mask = basket_wide.notna().all(axis=1)
    incomplete_count = int((~complete_mask).sum())
    complete = basket_wide.loc[complete_mask]
    trades = complete.mean(axis=1).rename("gross_return").reset_index()
    trades["entry_timestamp"] = trades["signal_timestamp"] + pd.to_timedelta(execution_latency_minutes, unit="m")
    trades["exit_timestamp"] = trades["entry_timestamp"] + pd.to_timedelta(holding_minutes, unit="m")
    trades = trades.sort_values("entry_timestamp").reset_index(drop=True)
    skipped_overlap = 0
    if overlap_policy in {"skip", "skip_while_position_open"}:
        keep = []
        next_available = None
        for index, row in trades.iterrows():
            if next_available is None or row["entry_timestamp"] >= next_available:
                keep.append(index)
                next_available = row["exit_timestamp"]
            else:
                skipped_overlap += 1
        trades = trades.loc[keep].reset_index(drop=True)
    elif overlap_policy != "allow":
        raise ValueError("overlap_policy must be 'skip' or 'allow'")
    if execution_mode == "maker" and maker_fill_probability is None:
        raise ValueError("Maker execution is sensitivity-only unless maker_fill_probability is specified")
    if execution_mode not in {"taker", "hybrid", "maker"}:
        raise ValueError("execution_mode must be taker, hybrid, or maker")
    unfilled_orders = 0
    if execution_mode == "maker":
        probability = float(maker_fill_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("maker_fill_probability must be between 0 and 1")
        filled = np.random.default_rng(fill_seed).random(len(trades)) < probability
        unfilled_orders = int((~filled).sum())
        trades = trades.loc[filled].reset_index(drop=True)
    trades["entry_price"] = np.nan
    trades["exit_price"] = np.nan
    trades["fee_return"] = float(round_trip_fee)
    trades["slippage_return"] = float(slippage) + (
        float(maker_adverse_selection) if execution_mode == "maker" else 0.0
    )
    funding_boundaries = (
        (trades["exit_timestamp"].dt.floor("8h") - trades["entry_timestamp"].dt.floor("8h"))
        / pd.Timedelta(hours=8)
    ).clip(lower=0)
    trades["funding_intervals_crossed"] = funding_boundaries.astype(int)
    trades["funding_return"] = trades["funding_intervals_crossed"] * float(funding)
    trades["capital_fraction"] = 1.0
    trades["concurrent_positions"] = 1
    trades["skipped_reason"] = ""
    trades["execution_mode"] = execution_mode
    trades["execution_price_proxy"] = execution_price_proxy
    trades["maker_fill_probability"] = maker_fill_probability
    trades["net_return"] = (
        trades["gross_return"]
        - trades["fee_return"]
        - trades["slippage_return"]
        - trades["funding_return"]
    )
    trades["equity_after"] = (1.0 + trades["net_return"]).cumprod()
    trades["drawdown"] = trades["equity_after"] / trades["equity_after"].cummax() - 1.0
    summary = {
        "complete_basket_only": True,
        "basket_targets": list(basket_targets),
        "overlap_policy": "skip_while_position_open" if overlap_policy == "skip" else overlap_policy,
        "trade_count": int(len(trades)),
        "signals_total": int(len(basket_wide)),
        "skipped_incomplete_basket": incomplete_count,
        "skipped_overlap": int(skipped_overlap),
        "terminal_return": float(trades["equity_after"].iloc[-1] - 1.0) if not trades.empty else np.nan,
        "arithmetic_return_sum": float(trades["net_return"].sum()) if not trades.empty else 0.0,
        "max_drawdown": float(trades["drawdown"].min()) if not trades.empty else np.nan,
        "round_trip_fee": float(round_trip_fee),
        "slippage": float(slippage),
        "funding_rate_per_8h": float(funding),
        "execution_latency_minutes": float(execution_latency_minutes),
        "execution_mode": execution_mode,
        "maker_fill_probability": maker_fill_probability,
        "maker_adverse_selection": float(maker_adverse_selection),
        "unfilled_orders": int(unfilled_orders),
        "fill_seed": int(fill_seed),
        "execution_price_proxy": execution_price_proxy,
        "skipped_reasons": {
            "incomplete_basket": incomplete_count,
            "position_already_open": int(skipped_overlap),
            "maker_unfilled": int(unfilled_orders),
        },
    }
    return trades, summary
