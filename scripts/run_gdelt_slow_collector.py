from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from news_collection import load_existing_news, merge_and_save_news
from news_collection_slow import build_gdelt_slow_report, collect_gdelt_slow_batch
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GDELT 저속 수집 모드로 뉴스 headline을 천천히 누적합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "baseline" / "config.yaml"),
        help="baseline 설정 파일 경로",
    )
    parser.add_argument(
        "--query-name",
        default=None,
        help="특정 query 하나만 강제로 실행할 때 사용",
    )
    parser.add_argument(
        "--queries-per-run",
        type=int,
        default=None,
        help="한 번에 처리할 query 개수 override",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    if "sentiment_extension" not in config or "news_collection" not in config["sentiment_extension"]:
        raise ValueError("config에 sentiment_extension.news_collection 섹션이 없습니다.")

    collection_cfg = dict(config["sentiment_extension"]["news_collection"])
    slow_cfg = dict(collection_cfg.get("gdelt_slow", {}))
    output_cfg = {"output": slow_cfg.get("output", {"base_dir": "outputs/baseline/news_collection_gdelt_slow"})}
    output_dirs = prepare_output_dirs(PROJECT_ROOT, output_cfg)
    save_config_snapshot(config, output_dirs["base"] / "config_snapshot.yaml")

    output_path = PROJECT_ROOT / slow_cfg.get("output_path", collection_cfg.get("output_path", "data/news/news_headlines.csv"))
    existing = load_existing_news(output_path)
    fresh, collection_log, state = collect_gdelt_slow_batch(
        config,
        query_name_override=args.query_name,
        queries_per_run_override=args.queries_per_run,
    )
    merged = merge_and_save_news(existing=existing, fresh=fresh, output_path=output_path)

    save_dataframe(collection_log, output_dirs["base"] / "gdelt_slow_collection_log.csv", index=False)
    save_dataframe(fresh, output_dirs["base"] / "gdelt_slow_fresh_batch.csv", index=False)
    save_dataframe(merged, output_dirs["base"] / "news_headlines_snapshot.csv", index=False)

    report = build_gdelt_slow_report(
        state=state,
        collection_log=collection_log,
        merged_news=merged,
        output_paths=[
            str(output_path),
            str(output_dirs["base"] / "gdelt_slow_collection_log.csv"),
            str(output_dirs["base"] / "gdelt_slow_fresh_batch.csv"),
            str(output_dirs["base"] / "news_headlines_snapshot.csv"),
        ],
    )
    save_text(report, output_dirs["base"] / "gdelt_slow_report.md")

    LOGGER.info(
        "GDELT 저속 수집 완료. fresh=%d, merged=%d, next_index=%s",
        len(fresh),
        len(merged),
        state.get("next_index"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
