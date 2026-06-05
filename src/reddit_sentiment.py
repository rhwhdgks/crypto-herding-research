from __future__ import annotations

from pathlib import Path

import pandas as pd

from event_sentiment import build_sentiment_extension_report
from news_sentiment import load_news_headlines, score_news_headlines


def load_reddit_posts(path: str | Path, config: dict) -> pd.DataFrame:
    return load_news_headlines(path, config)


def score_reddit_posts(reddit_frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    return score_news_headlines(reddit_frame, config)


def build_reddit_extension_report(
    posts_scored: pd.DataFrame,
    event_features: pd.DataFrame,
    summary: pd.DataFrame,
    feature_thresholds: pd.DataFrame,
    feature_overview: pd.DataFrame,
    feature_summary: pd.DataFrame,
    label_window_minutes: int,
    output_paths: list[str],
) -> str:
    base = build_sentiment_extension_report(
        news_scored=posts_scored,
        event_features=event_features,
        summary=summary,
        feature_thresholds=feature_thresholds,
        feature_overview=feature_overview,
        feature_summary=feature_summary,
        label_window_minutes=label_window_minutes,
        output_paths=output_paths,
    )
    return (
        base
        .replace("# 뉴스 Sentiment 확장 리포트", "# Reddit Sentiment 확장 리포트")
        .replace("## 뉴스 데이터", "## Reddit 데이터")
        .replace("scored headline 수", "scored Reddit 제목 수")
        .replace("뉴스 제목 데이터", "Reddit 제목 데이터")
        .replace("news feature group", "reddit feature group")
    )
