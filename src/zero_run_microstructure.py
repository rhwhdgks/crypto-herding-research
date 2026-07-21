from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm

from frequency_sensitivity import benjamini_hochberg
from tick_event_schema import require_tick_schema_v2


MECHANISM_OUTCOMES = [
    "log_amihud_illiquidity",
    "zero_tick_share",
    "log_mean_intertrade_ms",
    "log_quote_volume",
    "abs_aggressor_imbalance",
]
MECHANISM_CONTROLS = ["abs_bucket_return", "current_market_abs_return_loo"]
FUTURE_CONTROLS = [
    "abs_bucket_return",
    "current_market_abs_return_loo",
    "trailing_volatility",
    "log_transaction_count",
    "log_quote_volume",
    "zero_tick_share",
    "abs_aggressor_imbalance",
]
FIXED_EFFECTS = ["symbol", "hour_utc", "weekday_utc"]
TICK_COLUMNS = [
    "bucket_start",
    "signal_timestamp",
    "symbol",
    "schema_version",
    "interval_minutes",
    "transaction_count",
    "last_price",
    "total_quote_quantity",
    "bucket_return",
    "up_ticks",
    "down_ticks",
    "zero_ticks",
    "run_z_zero",
    "run_clustering_side",
    "aggressor_imbalance",
    "aggressor_direction",
    "price_direction",
    "is_micro_run_clustering_event",
    "is_control_bucket",
]


@dataclass(frozen=True)
class ScalingArtifact:
    schema_version: int
    fit_start: str
    fit_end: str
    created_at_utc: str
    lower_quantile: float
    upper_quantile: float
    columns: dict[str, dict[str, float]]


def validate_frozen_config(config: Mapping) -> None:
    symbols = list(config["data"]["symbols"])
    if symbols != [
        "BTCUSDT",
        "ETHUSDT",
        "XRPUSDT",
        "SOLUSDT",
        "DOGEUSDT",
        "ADAUSDT",
        "AVAXUSDT",
    ]:
        raise ValueError("Frozen zero-run universe must contain the declared seven assets")
    analysis = config["analysis"]
    if [int(value) for value in analysis["horizons_minutes"]] != [5, 15, 30]:
        raise ValueError("Frozen zero-run horizons must be [5, 15, 30]")
    if int(analysis["primary_horizon_minutes"]) != 15:
        raise ValueError("Frozen primary horizon must be 15 minutes")
    if int(analysis["mechanism_family_size"]) != len(MECHANISM_OUTCOMES):
        raise ValueError("Mechanism family size does not match the frozen outcomes")
    if int(analysis["future_family_size"]) != 3:
        raise ValueError("Future family size must be three horizons")
    if int(analysis["permutation_repetitions"]) != 199:
        raise ValueError("Frozen permutation repetition count must remain 199")
    if int(config["data"]["expected_rows"]) != 490_560:
        raise ValueError("Frozen tick panel must contain 490,560 rows")
    if config["output"]["base_dir"] != "outputs/v2/zero_run_microstructure_v1":
        raise ValueError("Zero-run output directory is frozen")


def scaled_column(column: str) -> str:
    return f"scaled__{column}"


def as_utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def load_tick_frame(path: str | Path, expected_symbols: Sequence[str]) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() != ".parquet":
        raise ValueError("Zero-run research requires a Parquet schema v2 tick frame")
    available = set(pq.ParquetFile(source).schema.names)
    missing = sorted(set(TICK_COLUMNS).difference(available))
    if missing:
        raise ValueError(f"Tick frame is missing columns: {', '.join(missing)}")
    frame = pd.read_parquet(source, columns=TICK_COLUMNS)
    frame["bucket_start"] = pd.to_datetime(frame["bucket_start"], utc=True)
    frame["signal_timestamp"] = pd.to_datetime(frame["signal_timestamp"], utc=True)
    frame = frame.sort_values(["symbol", "bucket_start"]).reset_index(drop=True)
    require_tick_schema_v2(frame)
    actual_symbols = set(frame["symbol"].dropna().astype(str).unique())
    if actual_symbols != set(expected_symbols):
        raise ValueError("Tick symbol universe does not match the frozen config")
    return frame


def validate_tick_frame(frame: pd.DataFrame, data_cfg: Mapping) -> pd.DataFrame:
    expected_rows = int(data_cfg["expected_rows"])
    symbols = list(data_cfg["symbols"])
    expected_start = as_utc_timestamp(data_cfg["expected_start"])
    expected_end = as_utc_timestamp(data_cfg["expected_end"])
    interval = int(data_cfg["expected_interval_minutes"])
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} tick rows, found {len(frame):,}")
    if frame.duplicated(["symbol", "bucket_start"]).any():
        raise ValueError("Tick frame contains duplicate symbol/timestamp rows")
    if frame["bucket_start"].min() != expected_start:
        raise ValueError("Tick frame start does not match the frozen config")
    if frame["bucket_start"].max() != expected_end:
        raise ValueError("Tick frame end does not match the frozen config")
    if not frame["signal_timestamp"].eq(
        frame["bucket_start"] + pd.Timedelta(minutes=interval)
    ).all():
        raise ValueError("Tick signal timestamps are not exact bucket ends")
    expected_per_symbol = expected_rows // len(symbols)
    rows = []
    for symbol, subset in frame.groupby("symbol", sort=True):
        timestamps = subset["bucket_start"].sort_values()
        differences = timestamps.diff().dropna()
        complete_grid = bool(
            len(subset) == expected_per_symbol
            and (differences == pd.Timedelta(minutes=interval)).all()
        )
        rows.append(
            {
                "symbol": symbol,
                "rows": int(len(subset)),
                "start": timestamps.min(),
                "end": timestamps.max(),
                "complete_15m_grid": complete_grid,
                "run_z_zero_available_share": float(subset["run_z_zero"].notna().mean()),
                "aggressor_available_share": float(
                    subset["aggressor_imbalance"].notna().mean()
                ),
            }
        )
    coverage = pd.DataFrame(rows)
    if not coverage["complete_15m_grid"].all():
        raise ValueError("Tick frame does not contain a complete 15-minute grid")
    if coverage["run_z_zero_available_share"].min() < 1.0:
        raise ValueError("run_z_zero must be available for every frozen tick row")
    if coverage["aggressor_available_share"].min() < 1.0:
        raise ValueError("Aggressor imbalance must be available for every frozen tick row")
    return coverage


def build_analysis_frame(
    tick_frame: pd.DataFrame,
    ohlcv_paths: Mapping[str, str | Path],
    config: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_cfg = config["analysis"]
    horizons = [int(value) for value in analysis_cfg["horizons_minutes"]]
    interval_minutes = int(config["data"]["expected_interval_minutes"])
    frame = tick_frame.copy()
    frame["zero_run_intensity"] = -pd.to_numeric(frame["run_z_zero"], errors="coerce")
    tick_count = frame[["up_ticks", "down_ticks", "zero_ticks"]].sum(axis=1)
    frame["zero_tick_share"] = pd.to_numeric(
        frame["zero_ticks"], errors="coerce"
    ).div(tick_count.where(tick_count > 0))
    frame["abs_bucket_return"] = pd.to_numeric(
        frame["bucket_return"], errors="coerce"
    ).abs()
    quote_volume = pd.to_numeric(frame["total_quote_quantity"], errors="coerce")
    transaction_count = pd.to_numeric(frame["transaction_count"], errors="coerce")
    frame["log_quote_volume"] = np.log1p(quote_volume.clip(lower=0.0))
    frame["log_transaction_count"] = np.log1p(transaction_count.clip(lower=0.0))
    duration_ms = interval_minutes * 60.0 * 1000.0
    frame["log_mean_intertrade_ms"] = np.log(
        duration_ms / transaction_count.where(transaction_count > 0)
    )
    frame["log_amihud_illiquidity"] = np.log1p(
        frame["abs_bucket_return"] * 1_000_000.0 / quote_volume.where(quote_volume > 0)
    )
    frame["abs_aggressor_imbalance"] = pd.to_numeric(
        frame["aggressor_imbalance"], errors="coerce"
    ).abs()
    frame["current_market_abs_return_loo"] = leave_one_out_mean(
        frame, "abs_bucket_return"
    )
    frame["trailing_volatility"] = frame.groupby("symbol", sort=False)[
        "bucket_return"
    ].transform(
        lambda values: pd.to_numeric(values, errors="coerce").rolling(
            int(analysis_cfg["trailing_volatility_buckets"]),
            min_periods=int(analysis_cfg["trailing_volatility_buckets"]),
        ).std(ddof=1)
    )
    frame["hour_utc"] = frame["signal_timestamp"].dt.hour.astype(str)
    frame["weekday_utc"] = frame["signal_timestamp"].dt.dayofweek.astype(str)
    frame["utc_day"] = frame["signal_timestamp"].dt.floor("D")
    frame["meets_minimum_trade_count"] = transaction_count.ge(
        int(analysis_cfg["minimum_transaction_count"])
    )
    frame["sample_split"] = assign_sample_split(frame["bucket_start"], config["split"])

    quality_rows = []
    outcome_parts = []
    for symbol in config["data"]["symbols"]:
        local = frame.loc[frame["symbol"].eq(symbol)].copy()
        outcomes, quality = build_symbol_forward_outcomes(
            local,
            ohlcv_paths[symbol],
            horizons,
            config["split"],
        )
        outcome_parts.append(outcomes)
        quality_rows.append({"symbol": symbol, **quality})
    outcomes = pd.concat(outcome_parts, ignore_index=True)
    frame = frame.merge(
        outcomes,
        on=["symbol", "bucket_start"],
        how="left",
        validate="one_to_one",
    )
    frame = frame.sort_values(["bucket_start", "symbol"]).reset_index(drop=True)
    return frame, pd.DataFrame(quality_rows)


def build_symbol_forward_outcomes(
    tick_symbol_frame: pd.DataFrame,
    ohlcv_path: str | Path,
    horizons: Sequence[int],
    split_cfg: Mapping,
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = Path(ohlcv_path)
    ohlcv = pd.read_parquet(path, columns=["timestamp", "close"])
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)
    ohlcv = ohlcv.sort_values("timestamp")
    if ohlcv["timestamp"].duplicated().any():
        raise ValueError(f"Duplicate 1-minute timestamps in {path}")
    differences = ohlcv["timestamp"].diff().dropna()
    if not (differences == pd.Timedelta(minutes=1)).all():
        raise ValueError(f"Incomplete 1-minute OHLCV grid in {path}")
    close = pd.Series(
        pd.to_numeric(ohlcv["close"], errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(ohlcv["timestamp"]),
    )
    if close.isna().any() or (close <= 0).any():
        raise ValueError(f"Invalid close prices in {path}")
    minute_returns = np.log(close).diff()
    signal = pd.DatetimeIndex(tick_symbol_frame["signal_timestamp"])
    start_timestamps = signal - pd.Timedelta(minutes=1)
    start_close = close.reindex(start_timestamps).to_numpy(dtype=float)
    result = tick_symbol_frame[["symbol", "bucket_start"]].copy()
    max_price_diff_bps = np.nanmax(
        np.abs(np.log(pd.to_numeric(tick_symbol_frame["last_price"]).to_numpy() / start_close))
        * 10_000.0
    )
    for horizon in horizons:
        end_timestamps = signal + pd.Timedelta(minutes=horizon - 1)
        end_close = close.reindex(end_timestamps).to_numpy(dtype=float)
        absolute_return = np.abs(np.log(end_close / start_close)) * 10_000.0
        rolling_count = minute_returns.rolling(horizon, min_periods=horizon).count()
        rolling_variance = minute_returns.pow(2).rolling(
            horizon, min_periods=horizon
        ).sum()
        exact = rolling_count.reindex(end_timestamps).to_numpy(dtype=float) == horizon
        realized_volatility = (
            np.sqrt(rolling_variance.reindex(end_timestamps).to_numpy(dtype=float))
            * 10_000.0
        )
        result[f"abs_forward_return_{horizon}m_bps"] = absolute_return
        result[f"realized_volatility_{horizon}m_bps"] = realized_volatility
        result[f"horizon_is_exact_{horizon}m"] = exact
        available_at = tick_symbol_frame["signal_timestamp"] + pd.Timedelta(minutes=horizon)
        result[f"outcome_available_at_{horizon}m"] = available_at
        result[f"outcome_within_split_{horizon}m"] = outcome_within_split(
            tick_symbol_frame["sample_split"], available_at, split_cfg
        )
    quality = {
        "ohlcv_path": path.as_posix(),
        "ohlcv_rows": int(len(ohlcv)),
        "ohlcv_start": ohlcv["timestamp"].min(),
        "ohlcv_end": ohlcv["timestamp"].max(),
        "maximum_tick_vs_1m_close_difference_bps": float(max_price_diff_bps),
        "all_horizons_exact_share": float(
            result[[f"horizon_is_exact_{h}m" for h in horizons]].all(axis=1).mean()
        ),
    }
    return result, quality


def assign_sample_split(timestamps: pd.Series, split_cfg: Mapping) -> pd.Series:
    values = pd.to_datetime(timestamps, utc=True)
    development_start = as_utc_timestamp(split_cfg["development_start"])
    oos_start = as_utc_timestamp(split_cfg["oos_start"])
    oos_end = as_utc_timestamp(split_cfg["oos_end_exclusive"])
    if not development_start < oos_start < oos_end:
        raise ValueError("Frozen split boundaries are invalid")
    return pd.Series(
        np.select(
            [
                values.ge(development_start) & values.lt(oos_start),
                values.ge(oos_start) & values.lt(oos_end),
            ],
            ["development", "oos"],
            default="outside",
        ),
        index=timestamps.index,
        dtype="object",
    )


def outcome_within_split(
    split: pd.Series,
    available_at: pd.Series,
    split_cfg: Mapping,
) -> pd.Series:
    available = pd.to_datetime(available_at, utc=True)
    oos_start = as_utc_timestamp(split_cfg["oos_start"])
    oos_end = as_utc_timestamp(split_cfg["oos_end_exclusive"])
    return (
        split.eq("development") & available.lt(oos_start)
    ) | (split.eq("oos") & available.lt(oos_end))


def leave_one_out_mean(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("bucket_start", sort=False)[column]
    total = grouped.transform("sum")
    count = grouped.transform("count")
    own = pd.to_numeric(frame[column], errors="coerce")
    denominator = count - own.notna().astype(int)
    return (total - own.fillna(0.0)).where(denominator > 0) / denominator.where(
        denominator > 0
    )


def fit_scaling_artifact(frame: pd.DataFrame, config: Mapping) -> ScalingArtifact:
    columns = sorted(
        {
            "zero_run_intensity",
            *MECHANISM_OUTCOMES,
            *MECHANISM_CONTROLS,
            *FUTURE_CONTROLS,
        }
    )
    development = frame.loc[
        frame["sample_split"].eq("development")
        & frame["meets_minimum_trade_count"]
    ]
    lower_quantile = float(config["analysis"]["winsor_lower_quantile"])
    upper_quantile = float(config["analysis"]["winsor_upper_quantile"])
    parameters: dict[str, dict[str, float]] = {}
    for column in columns:
        values = pd.to_numeric(development[column], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"No development values available for scaler: {column}")
        lower = float(values.quantile(lower_quantile))
        upper = float(values.quantile(upper_quantile))
        clipped = values.clip(lower=lower, upper=upper)
        mean = float(clipped.mean())
        std = float(clipped.std(ddof=1))
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"Degenerate scaler column: {column}")
        parameters[column] = {
            "winsor_lower": lower,
            "winsor_upper": upper,
            "mean": mean,
            "std": std,
        }
    timestamps = development["bucket_start"]
    return ScalingArtifact(
        schema_version=1,
        fit_start=timestamps.min().isoformat(),
        fit_end=timestamps.max().isoformat(),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        columns=parameters,
    )


def apply_scaling_artifact(
    frame: pd.DataFrame, artifact: ScalingArtifact
) -> pd.DataFrame:
    result = frame.copy()
    for column, parameters in artifact.columns.items():
        values = pd.to_numeric(result[column], errors="coerce").clip(
            lower=float(parameters["winsor_lower"]),
            upper=float(parameters["winsor_upper"]),
        )
        result[scaled_column(column)] = (
            values - float(parameters["mean"])
        ) / float(parameters["std"])
    return result


def save_scaling_artifact(artifact: ScalingArtifact, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(artifact), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_primary_models(
    frame: pd.DataFrame, config: Mapping
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficient_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    for split_name in ("development", "oos"):
        for outcome in MECHANISM_OUTCOMES:
            subset = select_model_frame(frame, split_name)
            row, diagnostic = fit_clustered_focal_model(
                subset,
                outcome=scaled_column(outcome),
                controls=[scaled_column(value) for value in MECHANISM_CONTROLS],
                focal=scaled_column("zero_run_intensity"),
            )
            coefficient_rows.append(
                {
                    **row,
                    "split": split_name,
                    "family": "mechanism",
                    "outcome": outcome,
                    "horizon_minutes": np.nan,
                    "unit": "standard_deviation",
                }
            )
            diagnostic_rows.append(
                {**diagnostic, "split": split_name, "family": "mechanism", "outcome": outcome}
            )
        for family, prefix in (
            ("future_absolute_return", "abs_forward_return"),
            ("future_realized_volatility", "realized_volatility"),
        ):
            for horizon in config["analysis"]["horizons_minutes"]:
                horizon = int(horizon)
                outcome = f"{prefix}_{horizon}m_bps"
                subset = select_model_frame(frame, split_name, horizon)
                row, diagnostic = fit_clustered_focal_model(
                    subset,
                    outcome=outcome,
                    controls=[scaled_column(value) for value in FUTURE_CONTROLS],
                    focal=scaled_column("zero_run_intensity"),
                )
                coefficient_rows.append(
                    {
                        **row,
                        "split": split_name,
                        "family": family,
                        "outcome": outcome,
                        "horizon_minutes": horizon,
                        "unit": "bps",
                    }
                )
                diagnostic_rows.append(
                    {**diagnostic, "split": split_name, "family": family, "outcome": outcome}
                )
    coefficients = pd.DataFrame(coefficient_rows)
    coefficients["q_value_bh_fdr"] = coefficients.groupby(
        ["split", "family"], sort=False
    )["cluster_p_value"].transform(benjamini_hochberg)
    return coefficients, pd.DataFrame(diagnostic_rows)


def select_model_frame(
    frame: pd.DataFrame,
    split_name: str,
    horizon_minutes: int | None = None,
) -> pd.DataFrame:
    mask = frame["sample_split"].eq(split_name) & frame[
        "meets_minimum_trade_count"
    ]
    if horizon_minutes is not None:
        mask &= frame[f"horizon_is_exact_{horizon_minutes}m"].fillna(False)
        mask &= frame[f"outcome_within_split_{horizon_minutes}m"].fillna(False)
    return frame.loc[mask].copy()


def fit_clustered_focal_model(
    frame: pd.DataFrame,
    outcome: str,
    controls: Sequence[str],
    focal: str,
) -> tuple[dict[str, float], dict[str, float]]:
    required = [outcome, focal, *controls, *FIXED_EFFECTS, "utc_day"]
    model_frame = frame.dropna(subset=required).copy()
    design = build_design_matrix(model_frame, controls, focal)
    y = pd.to_numeric(model_frame[outcome], errors="coerce").to_numpy(dtype=float)
    groups = pd.factorize(model_frame["utc_day"], sort=True)[0]
    result = sm.OLS(y, design.to_numpy(dtype=float)).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups, "use_correction": True},
    )
    term_index = design.columns.get_loc(focal)
    ci = result.conf_int()[term_index]
    row = {
        "coefficient": float(result.params[term_index]),
        "cluster_std_error": float(result.bse[term_index]),
        "cluster_t_stat": float(result.tvalues[term_index]),
        "cluster_p_value": float(result.pvalues[term_index]),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "outcome_mean": float(np.mean(y)),
        "outcome_std": float(np.std(y, ddof=1)),
    }
    diagnostic = {
        "observations": int(len(model_frame)),
        "utc_day_clusters": int(np.unique(groups).size),
        "rsquared": float(result.rsquared),
        "condition_number": float(
            np.sqrt(np.linalg.cond(design.to_numpy(dtype=float).T @ design.to_numpy(dtype=float)))
        ),
    }
    return row, diagnostic


def build_design_matrix(
    frame: pd.DataFrame,
    controls: Sequence[str],
    focal: str | None,
    expected_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    columns = list(controls)
    if focal is not None:
        columns = [focal, *columns]
    continuous = frame[columns].apply(pd.to_numeric, errors="coerce").astype(float)
    fixed = pd.get_dummies(
        frame[FIXED_EFFECTS].astype(str),
        columns=FIXED_EFFECTS,
        drop_first=True,
        dtype=float,
    )
    design = pd.concat(
        [pd.Series(1.0, index=frame.index, name="const"), continuous, fixed], axis=1
    )
    if expected_columns is not None:
        design = design.reindex(columns=list(expected_columns), fill_value=0.0)
    return design


def run_oos_prediction_metrics(frame: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    rows = []
    controls = [scaled_column(value) for value in FUTURE_CONTROLS]
    focal = scaled_column("zero_run_intensity")
    for family, prefix in (
        ("future_absolute_return", "abs_forward_return"),
        ("future_realized_volatility", "realized_volatility"),
    ):
        for horizon in config["analysis"]["horizons_minutes"]:
            horizon = int(horizon)
            outcome = f"{prefix}_{horizon}m_bps"
            development = select_model_frame(frame, "development", horizon).dropna(
                subset=[outcome, *controls, focal, *FIXED_EFFECTS]
            )
            oos = select_model_frame(frame, "oos", horizon).dropna(
                subset=[outcome, *controls, focal, *FIXED_EFFECTS]
            )
            baseline_train = build_design_matrix(development, controls, None)
            baseline_test = build_design_matrix(
                oos, controls, None, baseline_train.columns
            )
            augmented_train = build_design_matrix(development, controls, focal)
            augmented_test = build_design_matrix(
                oos, controls, focal, augmented_train.columns
            )
            y_train = development[outcome].to_numpy(dtype=float)
            y_test = oos[outcome].to_numpy(dtype=float)
            baseline_params = np.linalg.lstsq(
                baseline_train.to_numpy(dtype=float), y_train, rcond=None
            )[0]
            augmented_params = np.linalg.lstsq(
                augmented_train.to_numpy(dtype=float), y_train, rcond=None
            )[0]
            baseline_prediction = baseline_test.to_numpy(dtype=float) @ baseline_params
            augmented_prediction = augmented_test.to_numpy(dtype=float) @ augmented_params
            baseline_error = y_test - baseline_prediction
            augmented_error = y_test - augmented_prediction
            baseline_rmse = float(np.sqrt(np.mean(baseline_error**2)))
            augmented_rmse = float(np.sqrt(np.mean(augmented_error**2)))
            baseline_mae = float(np.mean(np.abs(baseline_error)))
            augmented_mae = float(np.mean(np.abs(augmented_error)))
            denominator = float(np.sum((y_test - np.mean(y_train)) ** 2))
            rows.append(
                {
                    "family": family,
                    "outcome": outcome,
                    "horizon_minutes": horizon,
                    "development_observations": int(len(development)),
                    "oos_observations": int(len(oos)),
                    "baseline_rmse": baseline_rmse,
                    "augmented_rmse": augmented_rmse,
                    "rmse_improvement_percent": 100.0
                    * (baseline_rmse - augmented_rmse)
                    / baseline_rmse,
                    "baseline_mae": baseline_mae,
                    "augmented_mae": augmented_mae,
                    "mae_improvement_percent": 100.0
                    * (baseline_mae - augmented_mae)
                    / baseline_mae,
                    "baseline_oos_r_squared": 1.0
                    - float(np.sum(baseline_error**2)) / denominator,
                    "augmented_oos_r_squared": 1.0
                    - float(np.sum(augmented_error**2)) / denominator,
                    "development_zero_coefficient": float(
                        augmented_params[augmented_train.columns.get_loc(focal)]
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_leave_one_asset_out(
    frame: pd.DataFrame,
    config: Mapping,
    primary_coefficients: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    details = []
    controls = [scaled_column(value) for value in FUTURE_CONTROLS]
    focal = scaled_column("zero_run_intensity")
    symbols = list(config["data"]["symbols"])
    for family, prefix in (
        ("future_absolute_return", "abs_forward_return"),
        ("future_realized_volatility", "realized_volatility"),
    ):
        for horizon in config["analysis"]["horizons_minutes"]:
            horizon = int(horizon)
            outcome = f"{prefix}_{horizon}m_bps"
            base = select_model_frame(frame, "oos", horizon)
            for omitted_symbol in symbols:
                subset = base.loc[~base["symbol"].eq(omitted_symbol)]
                row, diagnostic = fit_clustered_focal_model(
                    subset, outcome, controls, focal
                )
                details.append(
                    {
                        **row,
                        **diagnostic,
                        "family": family,
                        "outcome": outcome,
                        "horizon_minutes": horizon,
                        "omitted_symbol": omitted_symbol,
                    }
                )
    detail = pd.DataFrame(details)
    detail["q_value_bh_fdr_21"] = detail.groupby("family", sort=False)[
        "cluster_p_value"
    ].transform(benjamini_hochberg)
    primary = primary_coefficients.loc[
        primary_coefficients["split"].eq("oos")
    ].copy()
    summaries = []
    for (family, horizon), subset in detail.groupby(
        ["family", "horizon_minutes"], sort=False
    ):
        matching_primary = primary.loc[
            primary["family"].eq(family)
            & primary["horizon_minutes"].eq(float(horizon)),
            "coefficient",
        ]
        if len(matching_primary) != 1:
            raise ValueError(
                f"Expected one primary OOS coefficient for {family}/{horizon}, "
                f"found {len(matching_primary)}"
            )
        primary_coefficient = float(matching_primary.iloc[0])
        same_sign = np.sign(subset["coefficient"]) == np.sign(primary_coefficient)
        median_abs = float(subset["coefficient"].abs().median())
        ratio = median_abs / abs(primary_coefficient) if primary_coefficient != 0 else np.nan
        summaries.append(
            {
                "family": family,
                "horizon_minutes": int(horizon),
                "primary_oos_coefficient": primary_coefficient,
                "same_sign_count": int(same_sign.sum()),
                "loao_runs": int(len(subset)),
                "median_absolute_coefficient": median_abs,
                "median_magnitude_ratio": ratio,
                "minimum_coefficient": float(subset["coefficient"].min()),
                "maximum_coefficient": float(subset["coefficient"].max()),
            }
        )
    return detail, pd.DataFrame(summaries)


def run_permutation_tests(
    frame: pd.DataFrame, config: Mapping
) -> tuple[pd.DataFrame, pd.DataFrame]:
    repetitions = int(config["analysis"]["permutation_repetitions"])
    minimum_shift_days = int(config["analysis"]["permutation_minimum_shift_days"])
    seed = int(config["analysis"]["random_seed"])
    interval = int(config["data"]["expected_interval_minutes"])
    controls = [scaled_column(value) for value in FUTURE_CONTROLS]
    focal = scaled_column("zero_run_intensity")
    details = []
    summaries = []
    for family_index, (family, prefix) in enumerate(
        (
            ("future_absolute_return", "abs_forward_return"),
            ("future_realized_volatility", "realized_volatility"),
        )
    ):
        for horizon in config["analysis"]["horizons_minutes"]:
            horizon = int(horizon)
            outcome = f"{prefix}_{horizon}m_bps"
            subset = select_model_frame(frame, "oos", horizon).dropna(
                subset=[outcome, focal, *controls, *FIXED_EFFECTS]
            )
            subset = subset.sort_values(["bucket_start", "symbol"])
            design = build_design_matrix(subset, controls, None)
            z = design.to_numpy(dtype=float)
            x = subset[focal].to_numpy(dtype=float)
            y = subset[outcome].to_numpy(dtype=float)
            x_residual = residualize(x, z)
            y_residual = residualize(y, z)
            observed = float(np.dot(x_residual, y_residual) / np.dot(x_residual, x_residual))
            grid = frame.loc[
                frame["sample_split"].eq("oos"), ["bucket_start", "symbol"]
            ].drop_duplicates().sort_values(["bucket_start", "symbol"])
            timestamps = pd.Index(grid["bucket_start"].unique())
            symbols = list(config["data"]["symbols"])
            timestamp_codes = pd.Categorical(
                subset["bucket_start"], categories=timestamps, ordered=True
            ).codes
            symbol_codes = pd.Categorical(
                subset["symbol"], categories=symbols, ordered=True
            ).codes
            x_grid = np.full((len(timestamps), len(symbols)), np.nan, dtype=float)
            y_grid = np.full_like(x_grid, np.nan)
            x_grid[timestamp_codes, symbol_codes] = x_residual
            y_grid[timestamp_codes, symbol_codes] = y_residual
            minimum_shift = minimum_shift_days * 24 * 60 // interval
            maximum_shift = len(timestamps) - minimum_shift
            if minimum_shift >= maximum_shift:
                raise ValueError("Permutation minimum shift leaves no valid circular offsets")
            rng = np.random.default_rng(seed + family_index * 100 + horizon)
            shifts = rng.integers(minimum_shift, maximum_shift + 1, size=repetitions)
            null_coefficients = []
            for draw_index, shift in enumerate(shifts):
                shifted = np.roll(x_grid, int(shift), axis=0)
                valid = np.isfinite(shifted) & np.isfinite(y_grid)
                numerator = float(np.sum(shifted[valid] * y_grid[valid]))
                denominator = float(np.sum(shifted[valid] ** 2))
                coefficient = numerator / denominator
                null_coefficients.append(coefficient)
                details.append(
                    {
                        "family": family,
                        "outcome": outcome,
                        "horizon_minutes": horizon,
                        "draw": draw_index + 1,
                        "shift_buckets": int(shift),
                        "shift_days": float(shift * interval / 1440.0),
                        "permuted_coefficient": coefficient,
                        "valid_observations": int(valid.sum()),
                    }
                )
            null_values = np.asarray(null_coefficients)
            empirical_p = (1.0 + np.sum(np.abs(null_values) >= abs(observed))) / (
                repetitions + 1.0
            )
            summaries.append(
                {
                    "family": family,
                    "outcome": outcome,
                    "horizon_minutes": horizon,
                    "observed_residualized_coefficient": observed,
                    "permutation_repetitions": repetitions,
                    "empirical_p_value": float(empirical_p),
                    "null_mean": float(np.mean(null_values)),
                    "null_std": float(np.std(null_values, ddof=1)),
                    "null_q025": float(np.quantile(null_values, 0.025)),
                    "null_q975": float(np.quantile(null_values, 0.975)),
                }
            )
    summary = pd.DataFrame(summaries)
    summary["q_value_bh_fdr"] = summary.groupby("family", sort=False)[
        "empirical_p_value"
    ].transform(benjamini_hochberg)
    return pd.DataFrame(details), summary


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    parameters = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ parameters


def run_future_lead_placebo(frame: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    lead_days = int(config["analysis"]["placebo_lead_days"])
    interval = int(config["data"]["expected_interval_minutes"])
    lead_buckets = lead_days * 24 * 60 // interval
    placebo_column = "placebo_future_zero_intensity"
    frame = frame.copy().sort_values(["symbol", "bucket_start"])
    frame[placebo_column] = frame.groupby("symbol", sort=False)[
        scaled_column("zero_run_intensity")
    ].shift(-lead_buckets)
    rows = []
    controls = [scaled_column(value) for value in FUTURE_CONTROLS]
    for family, prefix in (
        ("future_absolute_return", "abs_forward_return"),
        ("future_realized_volatility", "realized_volatility"),
    ):
        for horizon in config["analysis"]["horizons_minutes"]:
            horizon = int(horizon)
            outcome = f"{prefix}_{horizon}m_bps"
            subset = select_model_frame(frame, "oos", horizon)
            row, diagnostic = fit_clustered_focal_model(
                subset, outcome, controls, placebo_column
            )
            rows.append(
                {
                    **row,
                    **diagnostic,
                    "family": family,
                    "outcome": outcome,
                    "horizon_minutes": horizon,
                    "placebo_lead_days": lead_days,
                }
            )
    result = pd.DataFrame(rows)
    result["q_value_bh_fdr"] = result.groupby("family", sort=False)[
        "cluster_p_value"
    ].transform(benjamini_hochberg)
    return result


def build_decision_tables(
    coefficients: pd.DataFrame,
    prediction: pd.DataFrame,
    loao_summary: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    placebo: pd.DataFrame,
    config: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis = config["analysis"]
    fdr_alpha = float(analysis["fdr_alpha"])
    mechanism = coefficients.loc[
        coefficients["split"].eq("oos") & coefficients["family"].eq("mechanism")
    ].copy()
    mechanism["effect_threshold"] = float(
        analysis["mechanism_minimum_standardized_effect"]
    )
    mechanism["passes_q_value"] = mechanism["q_value_bh_fdr"].le(fdr_alpha)
    mechanism["passes_effect_size"] = mechanism["coefficient"].abs().ge(
        mechanism["effect_threshold"]
    )
    mechanism["passes_predeclared_criteria"] = (
        mechanism["passes_q_value"] & mechanism["passes_effect_size"]
    )

    future = coefficients.loc[
        coefficients["split"].eq("oos")
        & coefficients["family"].isin(
            ["future_absolute_return", "future_realized_volatility"]
        )
    ].copy()
    future = future.merge(
        prediction,
        on=["family", "outcome", "horizon_minutes"],
        how="left",
        validate="one_to_one",
    ).merge(
        loao_summary,
        on=["family", "horizon_minutes"],
        how="left",
        validate="one_to_one",
    ).merge(
        permutation_summary,
        on=["family", "outcome", "horizon_minutes"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_permutation"),
    ).merge(
        placebo[
            [
                "family",
                "outcome",
                "horizon_minutes",
                "coefficient",
                "q_value_bh_fdr",
            ]
        ].rename(
            columns={
                "coefficient": "placebo_coefficient",
                "q_value_bh_fdr": "placebo_q_value_bh_fdr",
            }
        ),
        on=["family", "outcome", "horizon_minutes"],
        how="left",
        validate="one_to_one",
    )
    future["effect_threshold_bps"] = np.maximum(
        float(analysis["future_minimum_effect_bps"]),
        future["outcome_mean"].abs()
        * float(analysis["future_minimum_relative_effect"]),
    )
    future["passes_cluster_q"] = future["q_value_bh_fdr"].le(fdr_alpha)
    future["passes_permutation_q"] = future["q_value_bh_fdr_permutation"].le(
        fdr_alpha
    )
    future["passes_effect_size"] = future["coefficient"].abs().ge(
        future["effect_threshold_bps"]
    )
    future["passes_loao"] = future["same_sign_count"].ge(
        int(analysis["loao_minimum_same_sign_count"])
    ) & future["median_magnitude_ratio"].ge(
        float(analysis["loao_minimum_median_magnitude_ratio"])
    )
    future["passes_oos_prediction"] = future["rmse_improvement_percent"].ge(
        float(analysis["oos_rmse_improvement_percent"])
    )
    future["placebo_veto"] = (
        future["placebo_q_value_bh_fdr"].le(fdr_alpha)
        & (np.sign(future["placebo_coefficient"]) == np.sign(future["coefficient"]))
        & future["placebo_coefficient"].abs().ge(0.5 * future["coefficient"].abs())
    )
    future["passes_predeclared_criteria"] = (
        future["passes_cluster_q"]
        & future["passes_permutation_q"]
        & future["passes_effect_size"]
        & future["passes_loao"]
        & future["passes_oos_prediction"]
        & ~future["placebo_veto"]
    )

    family_rows = []
    primary_horizon = int(analysis["primary_horizon_minutes"])
    for family, subset in future.groupby("family", sort=False):
        primary_pass = bool(
            subset.loc[
                subset["horizon_minutes"].eq(primary_horizon),
                "passes_predeclared_criteria",
            ].iloc[0]
        )
        passed_horizons = subset.loc[
            subset["passes_predeclared_criteria"], "horizon_minutes"
        ].astype(int).tolist()
        family_rows.append(
            {
                "family": family,
                "primary_horizon_minutes": primary_horizon,
                "primary_horizon_pass": primary_pass,
                "passed_horizon_count": len(passed_horizons),
                "passed_horizons": ",".join(map(str, passed_horizons)),
                "family_success": primary_pass and len(passed_horizons) >= 2,
            }
        )
    mechanism_passes = mechanism.loc[mechanism["passes_predeclared_criteria"]]
    broad_mechanism = len(mechanism_passes) >= 3 and bool(
        mechanism_passes["outcome"].isin(
            ["log_amihud_illiquidity", "zero_tick_share"]
        ).any()
    )
    family_rows.append(
        {
            "family": "mechanism",
            "primary_horizon_minutes": np.nan,
            "primary_horizon_pass": np.nan,
            "passed_horizon_count": int(len(mechanism_passes)),
            "passed_horizons": "",
            "family_success": broad_mechanism,
        }
    )
    return mechanism, future, pd.DataFrame(family_rows)


def build_sample_coverage(frame: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    rows = []
    for split_name in ("development", "oos"):
        subset = frame.loc[frame["sample_split"].eq(split_name)]
        row = {
            "split": split_name,
            "rows": int(len(subset)),
            "start": subset["bucket_start"].min(),
            "end": subset["bucket_start"].max(),
            "symbols": int(subset["symbol"].nunique()),
            "utc_days": int(subset["utc_day"].nunique()),
            "minimum_trade_count_rows": int(
                subset["meets_minimum_trade_count"].sum()
            ),
            "minimum_trade_count_share": float(
                subset["meets_minimum_trade_count"].mean()
            ),
        }
        for horizon in config["analysis"]["horizons_minutes"]:
            horizon = int(horizon)
            row[f"exact_{horizon}m_rows"] = int(
                (
                    subset[f"horizon_is_exact_{horizon}m"]
                    & subset[f"outcome_within_split_{horizon}m"]
                ).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def validate_inference_gates(
    coverage: pd.DataFrame,
    diagnostics: pd.DataFrame,
    config: Mapping,
) -> None:
    analysis = config["analysis"]
    minimum_rows = int(analysis["minimum_complete_observations_per_split"])
    minimum_clusters = int(analysis["minimum_utc_day_clusters_per_split"])
    if coverage["minimum_trade_count_rows"].min() < minimum_rows:
        raise ValueError("A split does not meet the preregistered observation gate")
    if coverage["utc_days"].min() < minimum_clusters:
        raise ValueError("A split does not meet the preregistered UTC-day gate")
    if diagnostics["observations"].min() < minimum_rows:
        raise ValueError("A fitted model does not meet the observation gate")
    if diagnostics["utc_day_clusters"].min() < minimum_clusters:
        raise ValueError("A fitted model does not meet the cluster gate")


def build_artifact_manifest(output_dir: str | Path) -> pd.DataFrame:
    base = Path(output_dir)
    rows = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.csv":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "path": path.relative_to(base).as_posix(),
                "size": path.stat().st_size,
                "sha256": digest,
            }
        )
    return pd.DataFrame(rows)
