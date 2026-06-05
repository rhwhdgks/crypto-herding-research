from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

TOKEN_PATTERN = re.compile(r"[A-Za-z$#][A-Za-z0-9_.$#-]*")

DEFAULT_POSITIVE_TERMS = {
    "bullish",
    "breakout",
    "pump",
    "moon",
    "mooning",
    "rip",
    "ripping",
    "squeeze",
    "strong",
    "strength",
    "bounce",
    "rebound",
    "support",
    "buy",
    "long",
    "accumulate",
    "send",
    "green",
    "uptrend",
}

DEFAULT_NEGATIVE_TERMS = {
    "bearish",
    "dump",
    "crash",
    "rug",
    "rugged",
    "weak",
    "weakness",
    "rejection",
    "sell",
    "short",
    "breakdown",
    "downtrend",
    "liquidation",
    "red",
    "fade",
    "panic",
    "capitulation",
}


def _normalize_terms(terms: Iterable[str]) -> set[str]:
    return {str(term).strip().lower() for term in terms if str(term).strip()}


def score_x_posts(posts: pd.DataFrame, config: dict) -> pd.DataFrame:
    if posts.empty:
        return posts.copy()

    scoring_cfg = dict(config.get("scoring", {}))
    positive_terms = _normalize_terms(scoring_cfg.get("positive_terms", DEFAULT_POSITIVE_TERMS))
    negative_terms = _normalize_terms(scoring_cfg.get("negative_terms", DEFAULT_NEGATIVE_TERMS))
    weights = dict(scoring_cfg.get("engagement_weights", {}))
    neutral_band = float(scoring_cfg.get("neutral_band", 0.0))

    frame = posts.copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    frame["token_list"] = frame["text"].str.lower().apply(lambda text: TOKEN_PATTERN.findall(text))
    frame["positive_hits"] = frame["token_list"].apply(lambda tokens: sum(token in positive_terms for token in tokens))
    frame["negative_hits"] = frame["token_list"].apply(lambda tokens: sum(token in negative_terms for token in tokens))
    frame["raw_sentiment_score"] = frame["positive_hits"] - frame["negative_hits"]
    frame["engagement_total"] = (
        frame["like_count"].fillna(0).astype(float) * float(weights.get("like", 1.0))
        + frame["reply_count"].fillna(0).astype(float) * float(weights.get("reply", 1.0))
        + frame["repost_count"].fillna(0).astype(float) * float(weights.get("repost", 2.0))
        + frame["quote_count"].fillna(0).astype(float) * float(weights.get("quote", 2.0))
        + frame["bookmark_count"].fillna(0).astype(float) * float(weights.get("bookmark", 1.0))
    )
    frame["engagement_weight"] = 1.0 + np.log1p(frame["engagement_total"].clip(lower=0.0))
    frame["weighted_sentiment_score"] = frame["raw_sentiment_score"] * frame["engagement_weight"]
    frame["sentiment_label"] = np.where(
        frame["raw_sentiment_score"] > neutral_band,
        "positive",
        np.where(frame["raw_sentiment_score"] < -neutral_band, "negative", "neutral"),
    )
    frame["token_count"] = frame["token_list"].apply(len)
    frame["token_list"] = frame["token_list"].apply(lambda tokens: " ".join(tokens))
    return frame


def attach_trailing_sentiment_to_trades(
    trades: pd.DataFrame,
    scored_posts: pd.DataFrame,
    lookback_minutes: list[int],
    timestamp_column: str,
    symbol_column: str,
    sentiment_abs_threshold: float,
) -> pd.DataFrame:
    frame = trades.copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    frame[symbol_column] = frame[symbol_column].astype(str).str.upper()

    if frame.empty:
        return frame

    posts = scored_posts.copy()
    if posts.empty:
        for lookback in lookback_minutes:
            prefix = f"x_{int(lookback)}m"
            frame[f"{prefix}_post_count"] = 0
            frame[f"{prefix}_positive_count"] = 0
            frame[f"{prefix}_negative_count"] = 0
            frame[f"{prefix}_neutral_count"] = 0
            frame[f"{prefix}_raw_sentiment_sum"] = 0.0
            frame[f"{prefix}_weighted_sentiment_sum"] = 0.0
            frame[f"{prefix}_engagement_sum"] = 0.0
            frame[f"{prefix}_dominant_sentiment"] = "no_posts"
        return frame

    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True, errors="coerce")
    posts["symbol"] = posts["symbol"].astype(str).str.upper()
    posts = posts.sort_values(["symbol", "created_at"]).reset_index(drop=True)

    grouped_cache: dict[str, dict[str, np.ndarray]] = {}
    for symbol, group in posts.groupby("symbol", sort=False):
        metric_map = {
            "post_count": np.ones(len(group), dtype=float),
            "positive_count": (group["sentiment_label"] == "positive").astype(float).to_numpy(),
            "negative_count": (group["sentiment_label"] == "negative").astype(float).to_numpy(),
            "neutral_count": (group["sentiment_label"] == "neutral").astype(float).to_numpy(),
            "raw_sentiment_sum": group["raw_sentiment_score"].astype(float).to_numpy(),
            "weighted_sentiment_sum": group["weighted_sentiment_score"].astype(float).to_numpy(),
            "engagement_sum": group["engagement_total"].astype(float).to_numpy(),
        }
        arrays = {
            "timestamps": group["created_at"].astype("int64").to_numpy(),
        }
        for name, values in metric_map.items():
            arrays[name] = np.concatenate([[0.0], np.cumsum(values)])
        grouped_cache[symbol] = arrays

    frame = frame.sort_values(timestamp_column).reset_index(drop=True)
    trade_times = frame[timestamp_column].astype("int64").to_numpy()

    metric_names = [
        "post_count",
        "positive_count",
        "negative_count",
        "neutral_count",
        "raw_sentiment_sum",
        "weighted_sentiment_sum",
        "engagement_sum",
    ]

    for lookback in lookback_minutes:
        prefix = f"x_{int(lookback)}m"
        window_ns = int(pd.Timedelta(minutes=int(lookback)).value)
        for metric_name in metric_names:
            frame[f"{prefix}_{metric_name}"] = 0.0

        for symbol, index_values in frame.groupby(symbol_column, sort=False).groups.items():
            arrays = grouped_cache.get(symbol)
            if arrays is None:
                continue
            index_array = np.array(list(index_values))
            symbol_trade_times = trade_times[index_array]
            right = np.searchsorted(arrays["timestamps"], symbol_trade_times, side="left")
            left = np.searchsorted(arrays["timestamps"], symbol_trade_times - window_ns, side="left")
            for metric_name in metric_names:
                values = arrays[metric_name][right] - arrays[metric_name][left]
                frame.loc[index_array, f"{prefix}_{metric_name}"] = values

        frame[f"{prefix}_dominant_sentiment"] = np.where(
            frame[f"{prefix}_post_count"] <= 0,
            "no_posts",
            np.where(
                frame[f"{prefix}_weighted_sentiment_sum"] >= sentiment_abs_threshold,
                "bullish",
                np.where(
                    frame[f"{prefix}_weighted_sentiment_sum"] <= -sentiment_abs_threshold,
                    "bearish",
                    "neutral",
                ),
            ),
        )

    return frame
