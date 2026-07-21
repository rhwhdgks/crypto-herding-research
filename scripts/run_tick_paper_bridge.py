from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from csad import compute_csad
from data_loader import load_multi_asset_ohlcv
from event_detection import detect_events
from market import compute_equal_weighted_market_return
from preprocessing import build_price_and_return_panels
from regression import prepare_regression_frame, run_csad_regression
from tick_paper_bridge import (
    build_paper_bridge_report,
    build_shifted_daily_context,
    merge_tick_trades_with_daily_context,
    plot_paper_bridge_summary,
    summarize_paper_bridge,
)
from utils import load_config, prepare_output_dirs, resolve_data_window, save_config_snapshot, save_dataframe, save_text, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="paper-like 일봉 상태와 tick 규칙 연결 연구를 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "multi_asset_365d" / "paper_bridge.yaml"),
        help="설정 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["data"] = resolve_data_window(config["data"])
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    output_dirs = prepare_output_dirs(PROJECT_ROOT, config)
    save_config_snapshot(config, output_dirs["base"] / "config_snapshot.yaml")

    asset_frames, load_summary = load_multi_asset_ohlcv(config)
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
        panel_cfg=config.get("panel", {}),
    )
    market_return = compute_equal_weighted_market_return(
        log_returns,
        min_active_assets=int(config.get("panel", {}).get("min_active_assets", len(config["data"]["symbols"]))),
    )
    csad = compute_csad(
        log_returns,
        market_return,
        min_active_assets=int(config.get("panel", {}).get("min_active_assets", len(config["data"]["symbols"]))),
    )
    regression_input = prepare_regression_frame(csad, market_return)
    coeffs, diagnostics, _, model, json_summary = run_csad_regression(
        csad,
        market_return,
        cov_type=str(config.get("regression", {}).get("cov_type", "HAC")),
        hac_maxlags=config.get("regression", {}).get("hac_maxlags", "auto"),
    )
    daily_event_frame = detect_events(csad, market_return, config["event_detection"])
    daily_context = build_shifted_daily_context(daily_event_frame)

    trade_log = pd.read_csv(config["tick"]["trade_log_path"])
    bridge_frame = merge_tick_trades_with_daily_context(
        trade_log=trade_log,
        daily_context=daily_context,
        candidate_names=list(config["tick"]["candidate_names"]),
    )
    bridge_summary = summarize_paper_bridge(bridge_frame)

    save_dataframe(load_summary, output_dirs["base"] / "data_load_summary.csv", index=False)
    save_dataframe(data_quality_summary, output_dirs["base"] / "data_quality_summary.csv", index=False)
    save_dataframe(universe_coverage_summary, output_dirs["base"] / "universe_coverage_summary.csv", index=False)
    save_dataframe(universe_transition_points, output_dirs["base"] / "universe_transition_points.csv", index=False)
    save_dataframe(raw_close_prices, output_dirs["intermediate"] / "raw_close_prices.csv")
    save_dataframe(aligned_close_prices, output_dirs["intermediate"] / "aligned_close_prices.csv")
    save_dataframe(log_returns, output_dirs["intermediate"] / "log_returns.csv")
    save_dataframe(market_return, output_dirs["intermediate"] / "market_return.csv")
    save_dataframe(csad, output_dirs["intermediate"] / "csad.csv")
    save_dataframe(regression_input, output_dirs["intermediate"] / "regression_input.csv")
    save_dataframe(daily_event_frame, output_dirs["base"] / "daily_event_frame.csv", index=True)
    save_dataframe(daily_context, output_dirs["base"] / "daily_context.csv", index=False)
    save_dataframe(bridge_frame, output_dirs["base"] / "paper_bridge_trade_log.csv", index=False)
    save_dataframe(bridge_summary, output_dirs["base"] / "paper_bridge_summary.csv", index=False)
    save_dataframe(coeffs, output_dirs["base"] / "regression_coefficients.csv")
    save_dataframe(diagnostics, output_dirs["base"] / "regression_diagnostics.csv", index=False)
    save_text(model.summary().as_text(), output_dirs["base"] / "regression_summary.txt")

    plot_path = output_dirs["plots"] / "paper_bridge_summary.png"
    plot_paper_bridge_summary(bridge_summary, plot_path)
    report = build_paper_bridge_report(
        config=config,
        regression_summary={
            "beta2": float(model.params.get("market_return_sq", float("nan"))),
            "beta2_t_stat": float(model.tvalues.get("market_return_sq", float("nan"))),
            "interpretation": json_summary.get("interpretation", ""),
        },
        bridge_summary=bridge_summary,
        plot_paths=[str(plot_path)],
    )
    save_text(report, output_dirs["base"] / "tick_paper_bridge_report.md")


if __name__ == "__main__":
    raise SystemExit(main())
