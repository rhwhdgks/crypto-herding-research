from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from event_sentiment import (
    attach_event_sentiment_features,
    build_sentiment_event_groups,
    build_sentiment_extension_report,
    engineer_sentiment_feature_layer,
    plot_feature_group_summary,
    plot_sentiment_split_summary,
    summarize_feature_layer,
    summarize_sentiment_event_study,
)
from news_sentiment import load_news_headlines, score_news_headlines
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="뉴스 sentiment 확장 레이어를 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "baseline" / "config.yaml"),
        help="baseline 설정 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    if "sentiment_extension" not in config:
        raise ValueError("config에 sentiment_extension 섹션이 없습니다.")

    sentiment_cfg = dict(config["sentiment_extension"])
    output_cfg = {"output": sentiment_cfg.get("output", {"base_dir": "outputs/baseline/sentiment_extension"})}
    output_dirs = prepare_output_dirs(PROJECT_ROOT, output_cfg)
    save_config_snapshot(config, output_dirs["base"] / "config_snapshot.yaml")

    news_cfg = dict(sentiment_cfg.get("news", {}))
    news_path = PROJECT_ROOT / news_cfg.get("input_path", "data/news/news_headlines.csv")
    news_frame = load_news_headlines(news_path, news_cfg)
    scored_news = score_news_headlines(news_frame, news_cfg)
    save_dataframe(scored_news, output_dirs["base"] / "news_sentiment_scored.csv", index=False)

    baseline_input_cfg = dict(sentiment_cfg.get("baseline_input", {}))
    event_path = PROJECT_ROOT / baseline_input_cfg.get("event_path", "outputs/baseline/events.csv")
    if not event_path.exists():
        raise FileNotFoundError(f"baseline event 파일을 찾을 수 없습니다: {event_path}")

    events = pd.read_csv(event_path)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    events = events.dropna(subset=["timestamp"]).copy()

    lookback_minutes = [int(value) for value in sentiment_cfg.get("lookback_minutes", [5, 15, 30])]
    event_features = attach_event_sentiment_features(
        events=events,
        scored_news=scored_news,
        lookback_minutes=lookback_minutes,
        timestamp_column="timestamp",
        positive_threshold=float(sentiment_cfg.get("event_positive_threshold", 0.05)),
        negative_threshold=float(sentiment_cfg.get("event_negative_threshold", -0.05)),
    )
    label_window = int(sentiment_cfg.get("label_window_minutes", 15))
    event_features = build_sentiment_event_groups(
        events_with_sentiment=event_features,
        label_window_minutes=label_window,
        event_type_column="event_type",
    )
    feature_cfg = dict(sentiment_cfg.get("feature_layer", {}))
    event_features, feature_thresholds = engineer_sentiment_feature_layer(
        events_with_sentiment=event_features,
        focus_window_minutes=int(feature_cfg.get("focus_window_minutes", label_window)),
        feature_cfg=feature_cfg,
        event_type_column="event_type",
    )
    save_dataframe(event_features, output_dirs["base"] / "event_sentiment_features.csv", index=False)

    summary = summarize_sentiment_event_study(
        events_with_groups=event_features,
        holding_periods=[int(value) for value in config["event_study"]["holding_periods"]],
        group_column=f"event_sentiment_group_{label_window}m",
    )
    feature_group_column = f"news_feature_group_{int(feature_cfg.get('focus_window_minutes', label_window))}m"
    feature_summary = summarize_sentiment_event_study(
        events_with_groups=event_features,
        holding_periods=[int(value) for value in config["event_study"]["holding_periods"]],
        group_column=feature_group_column,
    )
    feature_overview = summarize_feature_layer(
        events_with_groups=event_features,
        focus_window_minutes=int(feature_cfg.get("focus_window_minutes", label_window)),
        feature_group_column=feature_group_column,
    )
    save_dataframe(summary, output_dirs["base"] / "sentiment_event_study_summary.csv", index=False)
    save_dataframe(feature_summary, output_dirs["base"] / "sentiment_feature_event_study_summary.csv", index=False)
    save_dataframe(feature_overview, output_dirs["base"] / "sentiment_feature_overview.csv", index=False)
    save_dataframe(feature_thresholds, output_dirs["base"] / "sentiment_feature_thresholds.csv", index=False)

    split_plot_path = output_dirs["plots"] / "sentiment_split_plots.png"
    feature_plot_path = output_dirs["plots"] / "sentiment_feature_group_plots.png"
    plot_sentiment_split_summary(summary, split_plot_path)
    plot_feature_group_summary(feature_summary, feature_plot_path)

    report = build_sentiment_extension_report(
        news_scored=scored_news,
        event_features=event_features,
        summary=summary,
        feature_thresholds=feature_thresholds,
        feature_overview=feature_overview,
        feature_summary=feature_summary,
        label_window_minutes=label_window,
        output_paths=[
            str(output_dirs["base"] / "news_sentiment_scored.csv"),
            str(output_dirs["base"] / "event_sentiment_features.csv"),
            str(output_dirs["base"] / "sentiment_event_study_summary.csv"),
            str(output_dirs["base"] / "sentiment_feature_event_study_summary.csv"),
            str(output_dirs["base"] / "sentiment_feature_overview.csv"),
            str(output_dirs["base"] / "sentiment_feature_thresholds.csv"),
            str(split_plot_path),
            str(feature_plot_path),
        ],
    )
    save_text(report, output_dirs["base"] / "sentiment_extension_report.md")

    LOGGER.info(
        "뉴스 sentiment 확장 완료. scored_headlines=%d, events_with_features=%d, summary_rows=%d",
        len(scored_news),
        len(event_features),
        len(summary),
    )


if __name__ == "__main__":
    raise SystemExit(main())
