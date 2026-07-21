from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from tick_archive_backfill import build_backfill_plan, build_backfill_report, execute_backfill_plan
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binance public tick archive backfill을 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "xrp_5y" / "backfill.yaml"),
        help="tick archive backfill 설정 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    output_dirs = prepare_output_dirs(PROJECT_ROOT, config)
    save_config_snapshot(config, output_dirs["base"] / "config_snapshot.yaml")

    resolved_cfg, plan = build_backfill_plan(config["data"])
    plan_frame = save_plan_frame(plan)
    save_dataframe(plan_frame, output_dirs["base"] / "download_plan.csv", index=False)

    summary = execute_backfill_plan(
        data_cfg=config["data"],
        max_workers=int(config.get("download", {}).get("max_workers", 4)),
    )
    save_dataframe(summary, output_dirs["base"] / "download_summary.csv", index=False)

    report = build_backfill_report(
        resolved_cfg=resolved_cfg,
        summary=summary,
        local_data_dir=str(config["data"].get("local_data_dir", "data/tick_archive")),
    )
    save_text(report, output_dirs["base"] / "tick_archive_backfill_report.md")


def save_plan_frame(plan: list[dict]) -> object:
    import pandas as pd

    if not plan:
        return pd.DataFrame(columns=["symbol", "granularity", "target"])
    rows = []
    for task in plan:
        rows.append(
            {
                "symbol": task["symbol"],
                "granularity": task["granularity"],
                "target": task["target"].strftime("%Y-%m" if task["granularity"] == "monthly" else "%Y-%m-%d"),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())
