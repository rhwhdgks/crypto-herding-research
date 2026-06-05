from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reddit_collection import (
    build_reddit_collection_report,
    collect_reddit_posts,
    load_existing_reddit,
    merge_and_save_reddit,
)
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reddit 공개 search JSON으로 post title을 수집해 로컬 CSV에 누적 저장합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "baseline" / "config.yaml"),
        help="설정 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    if "reddit_collection" not in config:
        raise ValueError("config에 reddit_collection 섹션이 없습니다.")

    collection_cfg = dict(config["reddit_collection"])
    output_cfg = {"output": collection_cfg.get("output", {"base_dir": "outputs/baseline/reddit_collection"})}
    output_dirs = prepare_output_dirs(PROJECT_ROOT, output_cfg)
    save_config_snapshot(config, output_dirs["base"] / "config_snapshot.yaml")

    output_path = PROJECT_ROOT / collection_cfg.get("output_path", "data/reddit/reddit_posts.csv")
    existing = load_existing_reddit(output_path)
    fresh, collection_log = collect_reddit_posts(config)
    merged = merge_and_save_reddit(existing=existing, fresh=fresh, output_path=output_path)

    save_dataframe(collection_log, output_dirs["base"] / "reddit_collection_log.csv", index=False)
    save_dataframe(merged, output_dirs["base"] / "reddit_posts_snapshot.csv", index=False)

    report = build_reddit_collection_report(
        collection_log=collection_log,
        merged_posts=merged,
        output_paths=[
            str(output_path),
            str(output_dirs["base"] / "reddit_collection_log.csv"),
            str(output_dirs["base"] / "reddit_posts_snapshot.csv"),
        ],
    )
    save_text(report, output_dirs["base"] / "reddit_collection_report.md")

    LOGGER.info(
        "Reddit 데이터 수집 완료. fresh=%d, merged=%d, output=%s",
        len(fresh),
        len(merged),
        output_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
