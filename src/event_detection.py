from __future__ import annotations

import numpy as np
import pandas as pd


def detect_events(csad: pd.Series, market_return: pd.Series, config: dict) -> pd.DataFrame:
    frame = pd.concat(
        [csad.rename("csad"), market_return.rename("market_return")],
        axis=1,
        join="inner",
    ).dropna()

    lookback_window = int(config["lookback_window"])
    min_history = int(config.get("min_history", lookback_window))
    volatility_window = int(config["volatility_window"])
    cooldown_periods = int(config.get("cooldown_periods", 0))

    herding_cfg = config.get("herding", {})
    shock_cfg = config.get("shock", {})
    herding_volatility_gate_cfg = herding_cfg.get("volatility_gate", {})

    frame["abs_market_return"] = frame["market_return"].abs()
    frame["rolling_volatility"] = frame["market_return"].rolling(
        window=volatility_window,
        min_periods=volatility_window,
    ).std()

    frame["csad_low_threshold"] = _trailing_quantile(
        frame["csad"],
        lookback_window,
        min_history,
        float(herding_cfg.get("csad_low_percentile", config.get("csad_low_percentile", 0.20))),
    )
    frame["market_abs_upper_threshold"] = _trailing_quantile(
        frame["abs_market_return"],
        lookback_window,
        min_history,
        float(herding_cfg.get("market_abs_upper_percentile", config.get("market_abs_upper_percentile", 0.60))),
    )
    frame["shock_abs_return_threshold"] = _trailing_quantile(
        frame["abs_market_return"],
        lookback_window,
        min_history,
        float(shock_cfg.get("abs_return_percentile", config.get("shock_abs_return_percentile", 0.90))),
    )
    frame["shock_vol_threshold"] = _trailing_quantile(
        frame["rolling_volatility"],
        lookback_window,
        min_history,
        float(shock_cfg.get("volatility_percentile", config.get("shock_vol_percentile", 0.90))),
    )
    herding_volatility_percentile = float(
        herding_volatility_gate_cfg.get(
            "percentile",
            herding_volatility_gate_cfg.get("volatility_percentile", 0.70),
        )
    )
    frame["herding_volatility_threshold"] = _trailing_quantile(
        frame["rolling_volatility"],
        lookback_window,
        min_history,
        herding_volatility_percentile,
    )

    frame["csad_zscore"] = _trailing_zscore(frame["csad"], lookback_window, min_history)
    frame["abs_market_return_zscore"] = _trailing_zscore(frame["abs_market_return"], lookback_window, min_history)
    frame["rolling_volatility_zscore"] = _trailing_zscore(frame["rolling_volatility"], lookback_window, min_history)

    frame["is_low_csad_condition"] = _build_condition(
        series=frame["csad"],
        threshold_series=frame["csad_low_threshold"],
        zscore_series=frame["csad_zscore"],
        method=str(herding_cfg.get("csad_method", "percentile")).lower(),
        direction="low",
        zscore_threshold=float(herding_cfg.get("csad_zscore_threshold", -1.0)),
    )
    frame["is_moderate_market_condition"] = _build_condition(
        series=frame["abs_market_return"],
        threshold_series=frame["market_abs_upper_threshold"],
        zscore_series=frame["abs_market_return_zscore"],
        method=str(herding_cfg.get("market_abs_method", "percentile")).lower(),
        direction="low",
        zscore_threshold=float(herding_cfg.get("market_abs_zscore_threshold", 0.5)),
    )
    frame["is_high_abs_return_condition"] = _build_condition(
        series=frame["abs_market_return"],
        threshold_series=frame["shock_abs_return_threshold"],
        zscore_series=frame["abs_market_return_zscore"],
        method=str(shock_cfg.get("abs_return_method", "percentile")).lower(),
        direction="high",
        zscore_threshold=float(shock_cfg.get("abs_return_zscore_threshold", 1.5)),
    )
    frame["is_high_vol_condition"] = _build_condition(
        series=frame["rolling_volatility"],
        threshold_series=frame["shock_vol_threshold"],
        zscore_series=frame["rolling_volatility_zscore"],
        method=str(shock_cfg.get("volatility_method", "percentile")).lower(),
        direction="high",
        zscore_threshold=float(shock_cfg.get("volatility_zscore_threshold", 1.5)),
    )

    herding_volatility_gate_enabled = bool(herding_volatility_gate_cfg.get("enabled", False))
    if herding_volatility_gate_enabled:
        frame["is_herding_volatility_condition"] = _build_condition(
            series=frame["rolling_volatility"],
            threshold_series=frame["herding_volatility_threshold"],
            zscore_series=frame["rolling_volatility_zscore"],
            method=str(herding_volatility_gate_cfg.get("method", "percentile")).lower(),
            direction=str(herding_volatility_gate_cfg.get("direction", "high")).lower(),
            zscore_threshold=float(herding_volatility_gate_cfg.get("zscore_threshold", 0.5)),
        )
    else:
        frame["is_herding_volatility_condition"] = True

    herding_raw = (
        frame["is_low_csad_condition"]
        & frame["is_moderate_market_condition"]
        & frame["is_herding_volatility_condition"]
    )
    shock_raw = frame["is_high_abs_return_condition"] | frame["is_high_vol_condition"]
    herding_raw = herding_raw & ~shock_raw

    frame["is_herding_event_raw"] = herding_raw
    frame["is_shock_event_raw"] = shock_raw
    frame["is_herding_event"] = _apply_cooldown(herding_raw, cooldown_periods)
    frame["is_shock_event"] = _apply_cooldown(shock_raw & ~frame["is_herding_event"], cooldown_periods)

    frame["event_type"] = "none"
    frame.loc[frame["is_shock_event"], "event_type"] = "shock"
    frame.loc[frame["is_herding_event"], "event_type"] = "herding"

    return frame


def summarize_event_counts(
    event_frame: pd.DataFrame,
    label_column: str = "event_type",
    labels: list[str] | None = None,
) -> pd.DataFrame:
    if labels is None:
        labels = [label for label in event_frame[label_column].dropna().unique().tolist()]

    records = []
    for label in labels:
        subset = event_frame[event_frame[label_column] == label]
        records.append(
            {
                label_column: label,
                "count": int(len(subset)),
                "first_timestamp": subset.index.min() if not subset.empty else pd.NaT,
                "last_timestamp": subset.index.max() if not subset.empty else pd.NaT,
            }
        )
    return pd.DataFrame(records)


def extract_event_timestamps(
    event_frame: pd.DataFrame,
    label_column: str = "event_type",
) -> pd.DataFrame:
    subset = event_frame[event_frame[label_column].fillna("none") != "none"][[label_column]].copy()
    subset = subset.reset_index().rename(columns={"timestamp": "event_timestamp"})
    return subset


def _trailing_quantile(series: pd.Series, window: int, min_periods: int, quantile: float) -> pd.Series:
    return series.rolling(window=window, min_periods=min_periods).quantile(quantile).shift(1)


def _trailing_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean().shift(1)
    rolling_std = series.rolling(window=window, min_periods=min_periods).std().shift(1)
    rolling_std = rolling_std.replace(0.0, np.nan)
    return (series - rolling_mean) / rolling_std


def _build_condition(
    series: pd.Series,
    threshold_series: pd.Series,
    zscore_series: pd.Series,
    method: str,
    direction: str,
    zscore_threshold: float,
) -> pd.Series:
    if direction == "low":
        percentile_condition = series <= threshold_series
        zscore_condition = zscore_series <= zscore_threshold
    else:
        percentile_condition = series >= threshold_series
        zscore_condition = zscore_series >= zscore_threshold

    if method == "percentile":
        return percentile_condition.fillna(False)
    if method == "zscore":
        return zscore_condition.fillna(False)
    if method == "either":
        return (percentile_condition | zscore_condition).fillna(False)
    if method == "both":
        return (percentile_condition & zscore_condition).fillna(False)

    raise ValueError(f"Unsupported threshold method: {method}")


def _apply_cooldown(flags: pd.Series, cooldown_periods: int) -> pd.Series:
    if cooldown_periods <= 0:
        return flags.fillna(False).astype(bool)

    accepted = []
    last_accepted_position = -cooldown_periods - 1

    for position, flag in enumerate(flags.fillna(False).tolist()):
        is_accepted = bool(flag) and (position - last_accepted_position > cooldown_periods)
        accepted.append(is_accepted)
        if is_accepted:
            last_accepted_position = position

    return pd.Series(accepted, index=flags.index, dtype=bool)
