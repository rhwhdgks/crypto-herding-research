from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import save_dataframe, save_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="확장 일반화 리포트를 생성합니다.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "tick" / "multi_asset_365d" / "generalization_extended"),
        help="출력 디렉터리",
    )
    return parser.parse_args()


def load_focus_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        PROJECT_ROOT / "outputs" / "tick" / "multi_asset_365d" / "generalization" / "generalization_symbol_focus.csv",
        PROJECT_ROOT / "outputs" / "tick" / "multi_asset_365d" / "singles" / "btc" / "generalization_symbol_focus.csv",
        PROJECT_ROOT / "outputs" / "tick" / "multi_asset_365d" / "singles" / "doge" / "generalization_symbol_focus.csv",
        PROJECT_ROOT / "outputs" / "tick" / "multi_asset_365d" / "singles" / "ada" / "generalization_symbol_focus.csv",
        PROJECT_ROOT / "outputs" / "tick" / "multi_asset_365d" / "singles" / "avax" / "generalization_symbol_focus.csv",
    ]

    frames = [load_focus_table(path) for path in paths]
    combined = pd.concat([frame for frame in frames if not frame.empty], axis=0, ignore_index=True)
    if combined.empty:
        save_text("# 확장 일반화 연구\n\n- 결합할 결과가 없습니다.\n", output_dir / "tick_generalization_extended_report.md")
        return

    combined = combined.drop_duplicates(subset=["symbol", "event_label", "horizon_minutes"]).reset_index(drop=True)
    up = combined.loc[combined["event_label"] == "up"].sort_values(
        ["delta_mean_return", "delta_t_stat"], ascending=False
    )
    all_events = combined.loc[combined["event_label"] == "all"].sort_values(
        ["delta_mean_return", "delta_t_stat"], ascending=False
    )
    down = combined.loc[combined["event_label"] == "down"].sort_values(
        ["delta_mean_return", "delta_t_stat"], ascending=False
    )

    save_dataframe(combined, output_dir / "generalization_symbol_focus_combined.csv", index=False)

    lines = ["# 확장 Tick 일반화 연구", ""]
    lines.append("## 대상 심볼")
    lines.append("- BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT")
    lines.append("- 구간: 2025-04-09 ~ 2026-04-09")
    lines.append("- 이벤트 정의: 15분 up/down micro-herding")
    lines.append("- 반응 horizon: 다음 30분")
    lines.append("")

    lines.append("## up 이벤트 비교")
    for _, row in up.iterrows():
        lines.append(
            f"- {row['symbol']}: 차이 {row['delta_mean_return']:.4%}, "
            f"t={row['delta_t_stat']:.2f}, 이벤트 {int(row['event_count'])}건, 판정 {row['signal_quality']}"
        )
    lines.append("")

    lines.append("## all 이벤트 비교")
    for _, row in all_events.iterrows():
        lines.append(
            f"- {row['symbol']}: 차이 {row['delta_mean_return']:.4%}, "
            f"t={row['delta_t_stat']:.2f}, 이벤트 {int(row['event_count'])}건"
        )
    lines.append("")

    lines.append("## down 이벤트 비교")
    for _, row in down.iterrows():
        lines.append(
            f"- {row['symbol']}: 차이 {row['delta_mean_return']:.4%}, "
            f"t={row['delta_t_stat']:.2f}, 이벤트 {int(row['event_count'])}건"
        )
    lines.append("")

    lines.append("## 해석")
    if not up.empty:
        top = up.iloc[0]
        lines.append(
            f"- up micro-herding 기준 가장 강한 심볼은 {top['symbol']}이며, 차이는 {top['delta_mean_return']:.4%}입니다."
        )
    positive_up = up.loc[up["delta_mean_return"] > 0, "symbol"].nunique()
    negative_up = up.loc[up["delta_mean_return"] <= 0, "symbol"].nunique()
    lines.append(f"- up 기준 양(+) 반응 심볼은 {int(positive_up)}개, 음(-) 반응 심볼은 {int(negative_up)}개입니다.")
    lines.append("- 이 결과는 XRP 신호가 시장 전체 공통 구조인지, 특정 알트군에 더 가까운 구조인지 확인하는 확장 일반화 실험입니다.")
    lines.append("")

    save_text("\n".join(lines), output_dir / "tick_generalization_extended_report.md")


if __name__ == "__main__":
    raise SystemExit(main())
