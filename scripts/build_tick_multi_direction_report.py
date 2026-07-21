from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from utils import save_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="multi-direction tick 연구 종합 리포트를 생성합니다.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "tick" / "multi_asset_365d" / "tick_multi_direction_report.md"),
        help="저장할 Markdown 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generalization_path = PROJECT_ROOT / "outputs" / "tick" / "multi_asset_365d" / "generalization" / "generalization_symbol_focus.csv"
    lead_lag_path = PROJECT_ROOT / "outputs" / "tick" / "multi_asset_365d" / "lead_lag" / "lead_lag_summary.csv"
    paper_bridge_path = PROJECT_ROOT / "outputs" / "tick" / "multi_asset_365d" / "paper_bridge" / "paper_bridge_summary.csv"

    lines = ["# Tick 연구 확장 종합 보고서", ""]

    if generalization_path.exists():
        generalization = pd.read_csv(generalization_path)
        lines.append("## 1. 다심볼 일반화")
        up = generalization.loc[generalization["event_label"] == "up"].sort_values(
            ["delta_mean_return", "delta_t_stat"], ascending=False
        )
        for _, row in up.iterrows():
            lines.append(
                f"- {row['symbol']}: up / 30m 차이 {row['delta_mean_return']:.4%}, "
                f"t={row['delta_t_stat']:.2f}, 판정 {row['signal_quality']}"
            )
        lines.append("")

    if lead_lag_path.exists():
        lead_lag = pd.read_csv(lead_lag_path)
        lines.append("## 2. Lead-Lag")
        for _, row in lead_lag.head(5).iterrows():
            lines.append(
                f"- {row['scenario_name']}: 차이 {row['delta_mean_return']:.4%}, "
                f"t={row['delta_t_stat']:.2f}, 이벤트 {int(row['event_count'])}건"
            )
        lines.append("")

    if paper_bridge_path.exists():
        bridge = pd.read_csv(paper_bridge_path)
        lines.append("## 3. Paper-like 연결")
        for variant_name, subset in bridge.groupby("variant_name"):
            best = subset.sort_values(["delta_vs_none", "t_stat"], ascending=False).iloc[0]
            lines.append(
                f"- {variant_name}: 전일 {best['prior_daily_event']} 상태에서 가장 좋았고, "
                f"none 대비 {best['delta_vs_none']:.4%}"
            )
        lines.append("")

    lines.append("## 해석")
    lines.append("- 이 세 방향은 각각 일반화, 선행성, 상위 시장 상태 연결을 점검합니다.")
    lines.append("- 앞으로는 가장 잘 나온 축을 메인라인으로 가져가고, 약한 축은 보조 설명 변수로 두는 전략이 좋습니다.")
    lines.append("")

    save_text("\n".join(lines), args.output)


if __name__ == "__main__":
    raise SystemExit(main())
