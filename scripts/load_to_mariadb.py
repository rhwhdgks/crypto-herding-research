from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from data_loader import load_multi_asset_ohlcv
from database import export_raw_ohlcv_to_database, export_research_outputs_to_database
from utils import load_config, resolve_data_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="baseline 연구 데이터와 출력물을 MariaDB에 적재합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "outputs" / "baseline" / "config_snapshot.yaml"),
        help="DB 적재에 사용할 설정 또는 스냅샷 파일",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "baseline"),
        help="내보낼 CSV가 들어 있는 출력 디렉터리",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if "database" not in config:
        fallback_config = load_config(PROJECT_ROOT / "configs" / "baseline" / "config.yaml")
        if "database" in fallback_config:
            config["database"] = fallback_config["database"]
    config["data"] = resolve_data_window(config["data"])
    database_cfg = dict(config.get("database", {}))
    database_cfg["enabled"] = True
    config["database"] = database_cfg

    output_dir = Path(args.output_dir)
    asset_frames, _ = load_multi_asset_ohlcv(config)

    export_raw_ohlcv_to_mariadb(
        config=config,
        asset_frames=asset_frames,
        timeframe=config["data"].get("timeframe", "1m"),
    )

    frame_map = load_output_frames(output_dir)
    export_research_outputs_to_database(config, frame_map)
    print("MariaDB 적재가 완료됐습니다.")


def export_raw_ohlcv_to_mariadb(config: dict, asset_frames: dict[str, pd.DataFrame], timeframe: str) -> None:
    export_raw_ohlcv_to_database(
        config=config,
        asset_frames=asset_frames,
        timeframe=timeframe,
    )


def load_output_frames(output_dir: Path) -> dict[str, pd.DataFrame]:
    frame_paths = {
        "data_load_summary": output_dir / "data_load_summary.csv",
        "data_quality_summary": output_dir / "data_quality_summary.csv",
        "universe_coverage_summary": output_dir / "universe_coverage_summary.csv",
        "csad_series": output_dir / "csad_series.csv",
        "regression_results": output_dir / "regression_results.csv",
        "regression_diagnostics": output_dir / "regression_diagnostics.csv",
        "event_labels": output_dir / "event_labels.csv",
        "event_count_summary": output_dir / "event_count_summary.csv",
        "event_timestamps": output_dir / "event_timestamps.csv",
        "events": output_dir / "events.csv",
        "event_study_summary": output_dir / "event_study_summary.csv",
        "holding_period_comparison": output_dir / "holding_period_comparison.csv",
        "event_time_average_returns": output_dir / "event_time_average_returns.csv",
        "aligned_return_panel": output_dir / "intermediate" / "aligned_return_panel.csv",
        "market_return_series": output_dir / "intermediate" / "market_return_series.csv",
        "market_index": output_dir / "intermediate" / "market_index.csv",
        "regression_frame": output_dir / "intermediate" / "regression_frame.csv",
        "analysis_frame": output_dir / "intermediate" / "analysis_frame.csv",
    }

    frames: dict[str, pd.DataFrame] = {}
    for logical_name, path in frame_paths.items():
        if path.exists() and path.stat().st_size > 1:
            frames[logical_name] = pd.read_csv(path)
    return frames


if __name__ == "__main__":
    main()
