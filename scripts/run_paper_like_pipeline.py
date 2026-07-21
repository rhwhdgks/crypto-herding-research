from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from csad import compute_csad
from data_loader import load_multi_asset_ohlcv
from market import compute_equal_weighted_market_return, compute_market_cap_weighted_market_return
from preprocessing import build_price_and_return_panels
from regression import (
    prepare_regression_frame,
    run_csad_regression,
    run_no_intercept_csad_regression,
    run_scsad_regression,
)
from utils import (
    load_config,
    plot_csad_vs_market,
    prepare_output_dirs,
    resolve_data_window,
    save_config_snapshot,
    save_dataframe,
    save_json,
    save_text,
    setup_logging,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="논문 유사 저빈도 암호화폐 허딩 파이프라인을 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "paper_like" / "daily.yaml"),
        help="사용할 YAML 설정 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["data"] = resolve_data_window(config["data"])
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    output_dirs = prepare_output_dirs(PROJECT_ROOT, config)

    LOGGER.info(
        "논문 유사 허딩 파이프라인을 시작합니다. 종목 수=%d, 구간=%s ~ %s UTC.",
        len(config["data"]["symbols"]),
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
        panel_cfg=config.get("panel", {}),
    )

    min_active_assets = int(config.get("panel", {}).get("min_active_assets", 1))
    market_return_cfg = config.get("paper_like", {}).get("market_return", {})
    market_return_method = str(market_return_cfg.get("method", "equal_weighted")).lower()
    market_cap_panel = _build_optional_panel(asset_frames, raw_close_prices.index, "market_cap")
    if not market_cap_panel.empty:
        save_dataframe(market_cap_panel, output_dirs["intermediate"] / "market_cap_panel.csv")

    if market_return_method == "market_cap_weighted":
        if market_cap_panel.empty:
            raise ValueError(
                "paper_like.market_return.method is market_cap_weighted, but no market_cap column was found in the loaded source files."
            )
        market_return = compute_market_cap_weighted_market_return(
            return_frame=log_returns,
            market_cap_frame=market_cap_panel.reindex(log_returns.index),
            min_active_assets=min_active_assets,
            weight_lag_periods=int(market_return_cfg.get("weight_lag_periods", 0)),
        )
    else:
        market_return = compute_equal_weighted_market_return(log_returns, min_active_assets=min_active_assets)
    csad = compute_csad(log_returns, market_return, min_active_assets=min_active_assets)
    regression_input = prepare_regression_frame(csad, market_return)

    save_dataframe(data_load_summary, output_dirs["base"] / "data_load_summary.csv", index=False)
    save_dataframe(data_quality_summary, output_dirs["base"] / "data_quality_summary.csv", index=False)
    save_dataframe(universe_coverage_summary, output_dirs["base"] / "universe_coverage_summary.csv", index=False)
    save_dataframe(universe_transition_points, output_dirs["base"] / "universe_transition_points.csv", index=False)
    save_dataframe(raw_close_prices, output_dirs["intermediate"] / "raw_close_prices.csv")
    save_dataframe(aligned_close_prices, output_dirs["intermediate"] / "aligned_close_prices.csv")
    save_dataframe(log_returns, output_dirs["intermediate"] / "log_returns.csv")
    save_dataframe(market_return, output_dirs["intermediate"] / "market_return_series.csv")
    save_dataframe(csad, output_dirs["intermediate"] / "csad_series.csv")
    save_dataframe(regression_input, output_dirs["intermediate"] / "regression_input.csv")

    regression_cfg = config.get("paper_like", {}).get("regression", {})
    cov_type = regression_cfg.get("cov_type", "HAC")
    hac_maxlags = regression_cfg.get("hac_maxlags", "auto")
    periods = _resolve_subperiods(config)

    model_specs = [
        ("standard_csad", run_csad_regression, "market_return_sq"),
        ("no_intercept_csad", run_no_intercept_csad_regression, "market_return_sq"),
        ("scsad", run_scsad_regression, "market_return_cu"),
    ]

    summary_rows = []
    coefficient_frames = []
    diagnostic_frames = []

    for model_name, model_fn, target_term in model_specs:
        for period in periods:
            period_label = period["name"]
            period_start = pd.Timestamp(period["start"], tz="UTC")
            period_end = pd.Timestamp(period["end"], tz="UTC")
            period_csad = csad.loc[(csad.index >= period_start) & (csad.index <= period_end)]
            period_market = market_return.loc[(market_return.index >= period_start) & (market_return.index <= period_end)]
            if period_csad.dropna().empty or period_market.dropna().empty:
                LOGGER.warning("Skipping %s for %s because the sample slice is empty.", model_name, period_label)
                continue

            coeffs, diagnostics, regression_frame, model, json_summary = model_fn(
                period_csad,
                period_market,
                cov_type=cov_type,
                hac_maxlags=hac_maxlags,
            )
            coeff_frame = coeffs.reset_index()
            coeff_frame["model_name"] = model_name
            coeff_frame["period_name"] = period_label
            coeff_frame["period_start"] = period_start
            coeff_frame["period_end"] = period_end
            coefficient_frames.append(coeff_frame)

            diag_frame = diagnostics.copy()
            diag_frame["model_name"] = model_name
            diag_frame["period_name"] = period_label
            diagnostic_frames.append(diag_frame)

            target_value = float(model.params.get(target_term))
            target_t_stat = float(model.tvalues.get(target_term))
            target_p_value = float(model.pvalues.get(target_term))
            summary_rows.append(
                {
                    "model_name": model_name,
                    "period_name": period_label,
                    "period_start": period_start,
                    "period_end": period_end,
                    "target_term": target_term,
                    "target_value": target_value,
                    "target_t_stat": target_t_stat,
                    "target_p_value": target_p_value,
                    "rsquared": float(model.rsquared),
                    "adj_rsquared": float(model.rsquared_adj),
                    "nobs": float(model.nobs),
                    "interpretation": json_summary["interpretation"],
                }
            )

            if period_label == "full_sample":
                save_dataframe(regression_frame, output_dirs["intermediate"] / f"{model_name}_frame.csv")
                save_dataframe(coeffs, output_dirs["base"] / f"{model_name}_coefficients.csv")
                save_dataframe(diagnostics, output_dirs["base"] / f"{model_name}_diagnostics.csv", index=False)
                save_json(json_summary, output_dirs["base"] / f"{model_name}_summary.json")
                save_text(model.summary().as_text(), output_dirs["base"] / f"{model_name}_summary.txt")

    regression_summary = pd.DataFrame(summary_rows)
    coefficient_summary = pd.concat(coefficient_frames, ignore_index=True) if coefficient_frames else pd.DataFrame()
    diagnostic_summary = pd.concat(diagnostic_frames, ignore_index=True) if diagnostic_frames else pd.DataFrame()

    save_dataframe(regression_summary, output_dirs["base"] / "paper_regression_summary.csv", index=False)
    save_dataframe(coefficient_summary, output_dirs["base"] / "paper_regression_coefficients.csv", index=False)
    save_dataframe(diagnostic_summary, output_dirs["base"] / "paper_regression_diagnostics.csv", index=False)

    plot_csad_vs_market(regression_input, output_dirs["plots"] / "csad_vs_market_return.png")
    save_text(
        _build_paper_like_report(
            config=config,
            universe_coverage_summary=universe_coverage_summary,
            regression_summary=regression_summary,
            market_return_method=market_return_method,
        ),
        output_dirs["base"] / "paper_like_summary.md",
    )

    LOGGER.info(
        "논문 유사 파이프라인이 완료됐습니다. 주기=%s, 관측치=%d, 출력 경로=%s.",
        config["data"].get("timeframe", "unknown"),
        len(regression_input),
        output_dirs["base"],
    )


def _resolve_subperiods(config: dict) -> list[dict]:
    data_cfg = config["data"]
    default_periods = [
        {
            "name": "full_sample",
            "start": data_cfg["start"],
            "end": (pd.Timestamp(data_cfg["end"], tz="UTC") - pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
        }
    ]
    configured_periods = config.get("paper_like", {}).get("subperiods", [])
    return default_periods + configured_periods


def _build_paper_like_report(
    config: dict,
    universe_coverage_summary: pd.DataFrame,
    regression_summary: pd.DataFrame,
    market_return_method: str,
) -> str:
    lines = ["# 논문 유사 환경 허딩 요약", ""]
    lines.append("## 설정")
    lines.append(
        "- 이 결과는 논문을 완전히 복제한 것이 아니라, 논문에 가깝게 맞춘 근사 환경입니다. "
        "CoinMarketCap 일봉이나 Bitstamp tick data 대신 Binance OHLCV와 현재 설정 종목을 사용합니다."
    )
    lines.append(f"- 주기: {config['data'].get('timeframe', 'n/a')}")
    lines.append(f"- 패널 모드: {config.get('panel', {}).get('mode', 'n/a')}")
    lines.append(f"- 최소 활성 자산 수: {int(config.get('panel', {}).get('min_active_assets', 0))}")
    lines.append(f"- 시장수익률 계산 방식: {market_return_method}")
    lines.append("- 공분산 추정량: Newey-West HAC")
    if str(config["data"].get("timeframe", "")).lower() == "1w":
        lines.append("- 비고: 주간 자료는 마지막 `2024-04-08` 라벨 바를 포함하기 위해 내부 요청 종료 시각을 다음 주 경계로 둡니다.")
    lines.append("")

    lines.append("## 패널")
    if not universe_coverage_summary.empty:
        row = universe_coverage_summary.iloc[0]
        lines.append(f"- 요청 구간: {row['requested_start']} ~ {row['requested_end']}")
        lines.append(f"- 실제 분석 구간: {row['analysis_candidate_start']} ~ {row['analysis_candidate_end']}")
        lines.append(f"- 최대 활성 자산 수: {int(row['max_active_return_assets'])}")
        lines.append(f"- 활성 자산 수 중앙값: {row['median_active_return_assets']}")
    else:
        lines.append("- 패널 요약을 만들 수 없었습니다.")
    lines.append("")

    lines.append("## 전체 표본")
    if regression_summary.empty:
        lines.append("- 생성된 회귀 결과가 없습니다.")
    else:
        full_sample = regression_summary[regression_summary["period_name"] == "full_sample"]
        for _, row in full_sample.iterrows():
            lines.append(
                f"- {row['model_name']}: {row['target_term']}={row['target_value']:.6f}, "
                f"t-통계량={row['target_t_stat']:.3f}, p-값={row['target_p_value']:.6g}, "
                f"R-squared={row['rsquared']:.4f}"
            )
    lines.append("")

    lines.append("## 하위 구간")
    subperiods = regression_summary[regression_summary["period_name"] != "full_sample"] if not regression_summary.empty else pd.DataFrame()
    if subperiods.empty:
        lines.append("- 설정된 하위 구간 회귀가 없습니다.")
    else:
        for _, row in subperiods.iterrows():
            lines.append(
                f"- {row['period_name']} / {row['model_name']}: {row['target_term']}={row['target_value']:.6f}, "
                f"t-통계량={row['target_t_stat']:.3f}, p-값={row['target_p_value']:.6g}"
            )
    lines.append("")

    return "\n".join(lines)


def _build_optional_panel(
    asset_frames: dict[str, pd.DataFrame],
    expected_index: pd.Index,
    column_name: str,
) -> pd.DataFrame:
    available = {
        symbol: frame[column_name].reindex(expected_index)
        for symbol, frame in asset_frames.items()
        if column_name in frame.columns
    }
    if not available:
        return pd.DataFrame(index=expected_index)

    panel = pd.concat(available, axis=1).sort_index()
    panel.index.name = "timestamp"
    return panel


if __name__ == "__main__":
    main()
