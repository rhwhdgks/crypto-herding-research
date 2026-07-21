from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from csad import compute_csad
from data_loader import load_multi_asset_ohlcv
from database import export_raw_ohlcv_to_database, export_research_outputs_to_database
from event_detection import detect_events, extract_event_timestamps, summarize_event_counts
from event_study import run_event_study
from market import compute_active_asset_count, compute_equal_weighted_market_return, compute_market_index
from preprocessing import build_price_and_return_panels
from regression import run_csad_regression, run_rolling_csad_regression
from reporting import generate_report_summary
from utils import (
    load_config,
    plot_csad_vs_market,
    plot_event_occurrences,
    plot_event_paths,
    plot_event_returns,
    prepare_output_dirs,
    resolve_data_window,
    save_config_snapshot,
    save_dataframe,
    save_json,
    save_input_manifest,
    save_provenance_manifest,
    save_text,
    setup_logging,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="암호화폐 허딩 baseline 연구 파이프라인을 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "baseline" / "config.yaml"),
        help="사용할 YAML 설정 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["data"] = resolve_data_window(config["data"])
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    output_dirs = prepare_output_dirs(PROJECT_ROOT, config)

    panel_cfg = config.get("panel", {})
    panel_mode = str(panel_cfg.get("mode", "intersection")).lower()
    total_assets = max(len(config["data"]["symbols"]), 1)
    default_min_assets = total_assets if panel_mode == "intersection" else min(2, total_assets)
    min_active_assets = int(panel_cfg.get("min_active_assets", default_min_assets))

    LOGGER.info(
        "baseline 허딩 파이프라인을 시작합니다. 종목 수=%d, 구간=%s ~ %s UTC.",
        total_assets,
        config["data"]["start"],
        config["data"]["end"],
    )
    save_config_snapshot(config, output_dirs["base"] / "config_snapshot.yaml")

    asset_frames, data_load_summary = load_multi_asset_ohlcv(config)
    (
        raw_close_prices,
        aligned_close_prices,
        log_returns,
        data_quality_summary,
        universe_coverage_summary,
        universe_transition_points,
    ) = build_price_and_return_panels(
        asset_frames=asset_frames,
        data_cfg=config["data"],
        panel_cfg=panel_cfg,
    )

    active_asset_count = compute_active_asset_count(log_returns)
    active_price_asset_count = aligned_close_prices.notna().sum(axis=1).rename("active_price_asset_count")
    market_return = compute_equal_weighted_market_return(log_returns, min_active_assets=min_active_assets)
    market_index = compute_market_index(market_return)
    csad = compute_csad(log_returns, market_return, min_active_assets=min_active_assets)

    regression_cfg = config.get("regression", {})
    regression_results, regression_diagnostics, regression_frame, model, regression_json = run_csad_regression(
        csad,
        market_return,
        cov_type=regression_cfg.get("cov_type", "HAC"),
        hac_maxlags=regression_cfg.get("hac_maxlags", "auto"),
    )
    rolling_cfg = config.get("regression", {}).get("rolling", {})
    rolling_regression = pd.DataFrame()
    if rolling_cfg.get("enabled", False):
        rolling_regression = run_rolling_csad_regression(
            csad=csad,
            market_return=market_return,
            window=int(rolling_cfg.get("window", 720)),
            min_periods=int(rolling_cfg.get("min_periods", 720)),
            cov_type=regression_cfg.get("cov_type", "HAC"),
            hac_maxlags=regression_cfg.get("hac_maxlags", "auto"),
        )

    event_frame = detect_events(csad, market_return, config["event_detection"])
    event_columns = [
        "rolling_volatility",
        "csad_low_threshold",
        "market_abs_upper_threshold",
        "shock_abs_return_threshold",
        "shock_vol_threshold",
        "herding_volatility_threshold",
        "csad_zscore",
        "abs_market_return_zscore",
        "rolling_volatility_zscore",
        "is_low_csad_condition",
        "is_moderate_market_condition",
        "is_high_abs_return_condition",
        "is_high_vol_condition",
        "is_herding_volatility_condition",
        "is_low_dispersion_event_raw",
        "is_shock_event_raw",
        "is_low_dispersion_event",
        "is_shock_event",
        "event_type",
    ]
    analysis_frame = regression_frame.join(active_asset_count, how="left")
    analysis_frame = analysis_frame.join(active_price_asset_count, how="left")
    analysis_frame["active_asset_share"] = analysis_frame["active_asset_count"] / total_assets
    analysis_frame["panel_mode"] = panel_mode
    analysis_frame["min_active_assets"] = min_active_assets
    analysis_frame = analysis_frame.join(event_frame[event_columns], how="left")

    event_labels = event_frame[event_columns].reset_index()
    event_count_summary = summarize_event_counts(
        analysis_frame,
        label_column="event_type",
        labels=["low_dispersion", "shock"],
    )
    event_timestamps = extract_event_timestamps(analysis_frame, label_column="event_type")

    events, event_study_summary, holding_period_comparison, event_paths = run_event_study(
        analysis_frame=analysis_frame,
        market_return=market_return,
        holding_periods=config["event_study"]["holding_periods"],
        event_label_column="event_type",
        event_types=["low_dispersion", "shock"],
        max_path_horizon=int(config["event_study"].get("max_path_horizon", max(config["event_study"]["holding_periods"]))),
    )

    save_dataframe(data_load_summary, output_dirs["base"] / "data_load_summary.csv", index=False)
    save_dataframe(data_quality_summary, output_dirs["base"] / "data_quality_summary.csv", index=False)
    save_dataframe(universe_coverage_summary, output_dirs["base"] / "universe_coverage_summary.csv", index=False)
    save_dataframe(universe_transition_points, output_dirs["base"] / "universe_transition_points.csv", index=False)
    save_dataframe(csad, output_dirs["base"] / "csad_series.csv")

    save_dataframe(raw_close_prices, output_dirs["intermediate"] / "raw_close_prices.csv")
    save_dataframe(aligned_close_prices, output_dirs["intermediate"] / "aligned_close_prices.csv")
    save_dataframe(log_returns, output_dirs["intermediate"] / "aligned_return_panel.csv")
    save_dataframe(log_returns, output_dirs["intermediate"] / "log_returns.csv")
    save_dataframe(market_return, output_dirs["intermediate"] / "market_return_series.csv")
    save_dataframe(market_index, output_dirs["intermediate"] / "market_index.csv")
    save_dataframe(csad, output_dirs["intermediate"] / "csad_series.csv")
    save_dataframe(regression_frame, output_dirs["intermediate"] / "regression_frame.csv")
    save_dataframe(analysis_frame, output_dirs["intermediate"] / "analysis_frame.csv")

    save_dataframe(regression_results, output_dirs["base"] / "regression_results.csv")
    save_dataframe(regression_diagnostics, output_dirs["base"] / "regression_diagnostics.csv", index=False)
    save_json(regression_json, output_dirs["base"] / "regression_results.json")
    save_text(model.summary().as_text(), output_dirs["base"] / "regression_summary.txt")
    if not rolling_regression.empty:
        save_dataframe(rolling_regression, output_dirs["base"] / "rolling_regression.csv")

    save_dataframe(event_labels, output_dirs["base"] / "event_labels.csv", index=False)
    save_dataframe(event_count_summary, output_dirs["base"] / "event_count_summary.csv", index=False)
    save_dataframe(event_timestamps, output_dirs["base"] / "event_timestamps.csv", index=False)
    save_dataframe(events, output_dirs["base"] / "events.csv")
    save_dataframe(event_study_summary, output_dirs["base"] / "event_study_summary.csv", index=False)
    save_dataframe(holding_period_comparison, output_dirs["base"] / "holding_period_comparison.csv", index=False)
    save_dataframe(event_paths, output_dirs["base"] / "event_time_average_returns.csv", index=False)

    export_raw_ohlcv_to_database(
        config=config,
        asset_frames=asset_frames,
        timeframe=config["data"].get("timeframe", "1m"),
    )
    export_research_outputs_to_database(
        config=config,
        frame_map={
            "data_load_summary": data_load_summary,
            "data_quality_summary": data_quality_summary,
            "universe_coverage_summary": universe_coverage_summary,
            "raw_close_prices": raw_close_prices,
            "aligned_close_prices": aligned_close_prices,
            "aligned_return_panel": log_returns,
            "market_return_series": market_return,
            "market_index": market_index,
            "csad_series": csad,
            "regression_frame": regression_frame,
            "analysis_frame": analysis_frame,
            "regression_results": regression_results,
            "regression_diagnostics": regression_diagnostics,
            "event_labels": event_labels,
            "event_count_summary": event_count_summary,
            "event_timestamps": event_timestamps,
            "events": events,
            "event_study_summary": event_study_summary,
            "holding_period_comparison": holding_period_comparison,
            "event_time_average_returns": event_paths,
        },
    )

    plot_csad_vs_market(analysis_frame, output_dirs["plots"] / "csad_vs_market_return.png")
    plot_event_occurrences(analysis_frame, market_index, output_dirs["plots"] / "event_occurrences.png", label_column="event_type")
    plot_event_returns(event_study_summary, output_dirs["plots"] / "event_forward_returns.png", label_column="event_type")
    plot_event_paths(event_paths, output_dirs["plots"] / "event_paths.png", label_column="event_type")

    plot_paths = [
        (output_dirs["plots"] / "csad_vs_market_return.png").relative_to(PROJECT_ROOT).as_posix(),
        (output_dirs["plots"] / "event_occurrences.png").relative_to(PROJECT_ROOT).as_posix(),
        (output_dirs["plots"] / "event_forward_returns.png").relative_to(PROJECT_ROOT).as_posix(),
        (output_dirs["plots"] / "event_paths.png").relative_to(PROJECT_ROOT).as_posix(),
    ]
    generate_report_summary(
        output_path=output_dirs["base"] / "report_summary.md",
        regression_json=regression_json,
        event_count_summary=event_count_summary,
        event_study_summary=event_study_summary,
        plot_paths=plot_paths,
        universe_coverage_summary=universe_coverage_summary,
    )

    input_paths = [
        PROJECT_ROOT / str(path)
        for path in data_load_summary.get("source_path", pd.Series(dtype=str)).dropna().unique()
        if (PROJECT_ROOT / str(path)).is_file()
    ]
    input_manifest = save_input_manifest(input_paths, output_dirs["base"] / "input_manifest.json")
    save_provenance_manifest(
        config,
        output_dirs["base"] / "provenance.json",
        schema_version=2,
        pipeline_version="baseline-csad-v2",
        train_start=config["data"]["start"],
        train_end=config["data"]["end"],
        statistical_method="classical CSAD OLS with Newey-West HAC covariance; block event-study inference",
        input_manifest_path=input_manifest,
        random_seed=20260715,
    )

    low_dispersion_count = int((analysis_frame["event_type"] == "low_dispersion").sum())
    shock_count = int((analysis_frame["event_type"] == "shock").sum())
    LOGGER.info(
        "파이프라인이 완료됐습니다. 관측치=%d, low-dispersion 이벤트=%d, shock 이벤트=%d.",
        len(analysis_frame),
        low_dispersion_count,
        shock_count,
    )


if __name__ == "__main__":
    main()
