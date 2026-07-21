from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from event_sentiment import (
    attach_event_sentiment_features,
    build_sentiment_event_groups,
    engineer_sentiment_feature_layer,
    plot_feature_group_summary,
    plot_sentiment_split_summary,
    summarize_feature_layer,
    summarize_sentiment_event_study,
)
from reddit_sentiment import (
    build_reddit_extension_report,
    load_reddit_posts,
    score_reddit_posts,
)
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reddit sentiment 확장 레이어를 실행합니다.")
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

    if "reddit_sentiment_extension" not in config:
        raise ValueError("config에 reddit_sentiment_extension 섹션이 없습니다.")

    sentiment_cfg = dict(config["reddit_sentiment_extension"])
    output_cfg = {"output": sentiment_cfg.get("output", {"base_dir": "outputs/baseline/reddit_sentiment_extension"})}
    output_dirs = prepare_output_dirs(PROJECT_ROOT, output_cfg)
    save_config_snapshot(config, output_dirs["base"] / "config_snapshot.yaml")

    reddit_cfg = dict(sentiment_cfg.get("reddit", {}))
    reddit_path = PROJECT_ROOT / reddit_cfg.get("input_path", "data/reddit/reddit_posts.csv")
    reddit_frame = load_reddit_posts(reddit_path, reddit_cfg)
    scored_reddit = score_reddit_posts(reddit_frame, reddit_cfg)
    save_dataframe(scored_reddit, output_dirs["base"] / "reddit_sentiment_scored.csv", index=False)

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
        scored_news=scored_reddit,
        lookback_minutes=lookback_minutes,
        timestamp_column="timestamp",
        positive_threshold=float(sentiment_cfg.get("event_positive_threshold", 0.05)),
        negative_threshold=float(sentiment_cfg.get("event_negative_threshold", -0.05)),
        availability_timestamp_column=str(sentiment_cfg.get("availability_timestamp_column", "first_seen_at_utc")),
        require_point_in_time=bool(sentiment_cfg.get("require_point_in_time", True)),
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
    save_dataframe(event_features, output_dirs["base"] / "event_reddit_sentiment_features.csv", index=False)

    oos_start = pd.Timestamp(feature_cfg["oos_start"])
    oos_features = event_features.loc[event_features["timestamp"] >= oos_start].copy()
    summary = summarize_sentiment_event_study(
        events_with_groups=oos_features,
        holding_periods=[int(value) for value in config["event_study"]["holding_periods"]],
        group_column=f"event_sentiment_group_{label_window}m",
    )
    feature_group_column = f"news_feature_group_{int(feature_cfg.get('focus_window_minutes', label_window))}m"
    feature_summary = summarize_sentiment_event_study(
        events_with_groups=oos_features,
        holding_periods=[int(value) for value in config["event_study"]["holding_periods"]],
        group_column=feature_group_column,
    )
    feature_overview = summarize_feature_layer(
        events_with_groups=oos_features,
        focus_window_minutes=int(feature_cfg.get("focus_window_minutes", label_window)),
        feature_group_column=feature_group_column,
    )
    save_dataframe(summary, output_dirs["base"] / "reddit_sentiment_event_study_summary.csv", index=False)
    save_dataframe(feature_summary, output_dirs["base"] / "reddit_sentiment_feature_event_study_summary.csv", index=False)
    save_dataframe(feature_overview, output_dirs["base"] / "reddit_sentiment_feature_overview.csv", index=False)
    save_dataframe(feature_thresholds, output_dirs["base"] / "reddit_sentiment_feature_thresholds.csv", index=False)

    split_plot_path = output_dirs["plots"] / "reddit_sentiment_split_plots.png"
    feature_plot_path = output_dirs["plots"] / "reddit_sentiment_feature_group_plots.png"
    plot_sentiment_split_summary(summary, split_plot_path)
    plot_feature_group_summary(feature_summary, feature_plot_path)

    report = build_reddit_extension_report(
        posts_scored=scored_reddit,
        event_features=event_features,
        summary=summary,
        feature_thresholds=feature_thresholds,
        feature_overview=feature_overview,
        feature_summary=feature_summary,
        label_window_minutes=label_window,
        output_paths=[
            str(output_dirs["base"] / "reddit_sentiment_scored.csv"),
            str(output_dirs["base"] / "event_reddit_sentiment_features.csv"),
            str(output_dirs["base"] / "reddit_sentiment_event_study_summary.csv"),
            str(output_dirs["base"] / "reddit_sentiment_feature_event_study_summary.csv"),
            str(output_dirs["base"] / "reddit_sentiment_feature_overview.csv"),
            str(output_dirs["base"] / "reddit_sentiment_feature_thresholds.csv"),
            str(split_plot_path),
            str(feature_plot_path),
        ],
    )
    save_text(report, output_dirs["base"] / "reddit_sentiment_extension_report.md")

    LOGGER.info(
        "Reddit sentiment 확장 완료. scored_posts=%d, events_with_features=%d, summary_rows=%d",
        len(scored_reddit),
        len(event_features),
        len(summary),
    )


if __name__ == "__main__":
    raise SystemExit(main())
