from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging
from x_api import collect_recent_search_posts
from x_bridge import (
    build_x_sentiment_report,
    filter_trade_log_to_post_coverage,
    load_trade_log,
    summarize_post_collection,
    summarize_sentiment_bridge,
)
from x_sentiment import attach_trailing_sentiment_to_trades, score_x_posts


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="X sentiment 별도 연구 파이프라인을 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "x" / "candidate_basket_sentiment.yaml"),
        help="X sentiment 설정 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    output_dirs = prepare_output_dirs(PROJECT_ROOT, config)
    save_config_snapshot(config, output_dirs["base"] / "config_snapshot.yaml")

    posts, collection_log, _ = collect_recent_search_posts(config)
    scored_posts = score_x_posts(posts, config)
    post_summary = summarize_post_collection(scored_posts)

    save_dataframe(collection_log, output_dirs["base"] / "collection_log.csv", index=False)
    save_dataframe(post_summary, output_dirs["base"] / "post_summary.csv", index=False)
    save_dataframe(scored_posts, output_dirs["intermediate"] / "x_posts_scored.csv", index=False)

    trade_cfg = dict(config["bridge"]["trade_log"])
    trade_log = load_trade_log(
        PROJECT_ROOT / trade_cfg["path"],
        timestamp_column=str(trade_cfg.get("timestamp_column", "entry_timestamp")),
    )
    covered_trade_log = filter_trade_log_to_post_coverage(
        trade_log=trade_log,
        posts=scored_posts,
        timestamp_column=str(trade_cfg.get("timestamp_column", "entry_timestamp")),
    )
    bridged_trades = attach_trailing_sentiment_to_trades(
        trades=covered_trade_log,
        scored_posts=scored_posts,
        lookback_minutes=[int(value) for value in config["bridge"]["lookback_minutes"]],
        timestamp_column=str(trade_cfg.get("timestamp_column", "entry_timestamp")),
        symbol_column=str(trade_cfg.get("symbol_column", "symbol")),
        sentiment_abs_threshold=float(config["bridge"].get("sentiment_abs_threshold", 0.5)),
    )
    bridge_summary = summarize_sentiment_bridge(
        bridged_trades=bridged_trades,
        variant_column=str(trade_cfg.get("variant_column", "variant_name")),
        return_column=str(trade_cfg.get("return_column", "net_return")),
        focus_lookback_minutes=int(config["bridge"].get("focus_lookback_minutes", 60)),
    )

    save_dataframe(bridged_trades, output_dirs["base"] / "bridged_trades.csv", index=False)
    save_dataframe(bridge_summary, output_dirs["base"] / "bridge_summary.csv", index=False)

    report = build_x_sentiment_report(
        collection_log=collection_log,
        post_summary=post_summary,
        bridged_trades=bridged_trades,
        bridge_summary=bridge_summary,
        output_paths=[
            str(output_dirs["base"] / "collection_log.csv"),
            str(output_dirs["base"] / "post_summary.csv"),
            str(output_dirs["base"] / "bridged_trades.csv"),
            str(output_dirs["base"] / "bridge_summary.csv"),
        ],
        token_present=bool(os.environ.get(str(config["collection"].get("bearer_token_env", "X_BEARER_TOKEN")))),
    )
    save_text(report, output_dirs["base"] / "x_sentiment_bridge_report.md")
    LOGGER.info("X sentiment pipeline complete. posts=%s, bridged_trades=%s", len(scored_posts), len(bridged_trades))


if __name__ == "__main__":
    raise SystemExit(main())
