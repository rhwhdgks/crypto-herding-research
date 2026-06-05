from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TOKEN_PATTERN = re.compile(r"[A-Za-z$#][A-Za-z0-9_.$#-]*")

DEFAULT_POSITIVE_TERMS = {
    "bullish",
    "breakout",
    "beats",
    "beat",
    "surge",
    "surges",
    "soar",
    "soars",
    "jump",
    "jumps",
    "rally",
    "rallies",
    "gain",
    "gains",
    "strong",
    "strength",
    "upgrade",
    "upgrades",
    "buy",
    "buying",
    "adoption",
    "partnership",
    "approval",
    "approved",
    "record",
    "high",
    "positive",
    "optimism",
    "optimistic",
}

DEFAULT_NEGATIVE_TERMS = {
    "bearish",
    "selloff",
    "sell-off",
    "drop",
    "drops",
    "fall",
    "falls",
    "plunge",
    "plunges",
    "crash",
    "crashes",
    "weak",
    "weakness",
    "downgrade",
    "downgrades",
    "lawsuit",
    "hack",
    "hacked",
    "ban",
    "bans",
    "risk",
    "risks",
    "negative",
    "panic",
    "fear",
    "fraud",
    "liquidation",
    "investigation",
}


def _normalize_terms(terms: Iterable[str]) -> set[str]:
    return {str(term).strip().lower() for term in terms if str(term).strip()}


def _normalize_asset_value(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    return text.replace("/", "").replace("-", "").replace("_", "")


def load_news_headlines(path: str | Path, config: dict) -> pd.DataFrame:
    resolved = Path(path)
    timestamp_column = str(config.get("timestamp_column", "timestamp"))
    source_column = str(config.get("source_column", "source"))
    asset_column = str(config.get("asset_column", "asset"))
    headline_column = str(config.get("headline_column", "headline"))

    if not resolved.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "source",
                "asset",
                "headline",
            ]
        )

    frame = pd.read_csv(resolved)
    rename_map = {}
    if timestamp_column in frame.columns and timestamp_column != "timestamp":
        rename_map[timestamp_column] = "timestamp"
    if source_column in frame.columns and source_column != "source":
        rename_map[source_column] = "source"
    if asset_column in frame.columns and asset_column != "asset":
        rename_map[asset_column] = "asset"
    if headline_column in frame.columns and headline_column != "headline":
        rename_map[headline_column] = "headline"
    frame = frame.rename(columns=rename_map)

    if "timestamp" not in frame.columns or "headline" not in frame.columns:
        raise ValueError("뉴스 파일에는 최소한 timestamp, headline 컬럼이 필요합니다.")

    if "source" not in frame.columns:
        frame["source"] = "unknown"
    if "asset" not in frame.columns:
        frame["asset"] = None

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["headline"] = frame["headline"].fillna("").astype(str).str.strip()
    frame["source"] = frame["source"].fillna("unknown").astype(str).str.strip()
    frame["asset"] = frame["asset"].apply(_normalize_asset_value)
    frame = frame.dropna(subset=["timestamp"])
    frame = frame.loc[frame["headline"] != ""].copy()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame[["timestamp", "source", "asset", "headline"]]


def score_news_headlines(news_frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    if news_frame.empty:
        return news_frame.copy()

    scoring_cfg = dict(config.get("scoring", {}))
    positive_terms = _normalize_terms(scoring_cfg.get("positive_terms", DEFAULT_POSITIVE_TERMS))
    negative_terms = _normalize_terms(scoring_cfg.get("negative_terms", DEFAULT_NEGATIVE_TERMS))
    positive_threshold = float(scoring_cfg.get("headline_positive_threshold", 0.0))
    negative_threshold = float(scoring_cfg.get("headline_negative_threshold", 0.0))

    frame = news_frame.copy()
    frame["headline_lower"] = frame["headline"].str.lower()
    frame["token_list"] = frame["headline_lower"].apply(TOKEN_PATTERN.findall)
    frame["token_count"] = frame["token_list"].apply(len)
    frame["positive_hits"] = frame["token_list"].apply(lambda tokens: sum(token in positive_terms for token in tokens))
    frame["negative_hits"] = frame["token_list"].apply(lambda tokens: sum(token in negative_terms for token in tokens))
    total_hits = frame["positive_hits"] + frame["negative_hits"]

    frame["sentiment_score"] = np.where(
        total_hits > 0,
        (frame["positive_hits"] - frame["negative_hits"]) / total_hits,
        0.0,
    )
    frame["sentiment_label"] = np.where(
        frame["sentiment_score"] > positive_threshold,
        "positive",
        np.where(frame["sentiment_score"] < negative_threshold, "negative", "neutral"),
    )
    frame["is_market_level"] = frame["asset"].isna()
    frame["matched_term_count"] = total_hits
    frame["token_list"] = frame["token_list"].apply(lambda tokens: " ".join(tokens))
    return frame
