from __future__ import annotations

import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from news_collection import build_news_collection_report, collect_news_headlines, load_existing_news, merge_and_save_news
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="뉴스 헤드라인을 수집해 data/news/news_headlines.csv에 누적 저장합니다.")
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

    if "sentiment_extension" not in config or "news_collection" not in config["sentiment_extension"]:
        raise ValueError("config에 sentiment_extension.news_collection 섹션이 없습니다.")

    collection_cfg = dict(config["sentiment_extension"]["news_collection"])
    output_cfg = {"output": collection_cfg.get("output", {"base_dir": "outputs/baseline/news_collection"})}
    output_dirs = prepare_output_dirs(PROJECT_ROOT, output_cfg)
    save_config_snapshot(config, output_dirs["base"] / "config_snapshot.yaml")

    output_path = PROJECT_ROOT / collection_cfg.get("output_path", "data/news/news_headlines.csv")
    existing = load_existing_news(output_path)
    fresh, collection_log = collect_news_headlines(config)
    merged = merge_and_save_news(existing=existing, fresh=fresh, output_path=output_path)

    save_dataframe(collection_log, output_dirs["base"] / "news_collection_log.csv", index=False)
    save_dataframe(merged, output_dirs["base"] / "news_headlines_snapshot.csv", index=False)

    report = build_news_collection_report(
        collection_log=collection_log,
        merged_news=merged,
        output_paths=[
            str(output_path),
            str(output_dirs["base"] / "news_collection_log.csv"),
            str(output_dirs["base"] / "news_headlines_snapshot.csv"),
        ],
    )
    save_text(report, output_dirs["base"] / "news_collection_report.md")

    LOGGER.info(
        "뉴스 헤드라인 수집 완료. fresh=%d, merged=%d, output=%s",
        len(fresh),
        len(merged),
        output_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
