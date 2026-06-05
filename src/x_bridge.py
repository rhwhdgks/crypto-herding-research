from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_table_file


def load_trade_log(path: str | Path, timestamp_column: str = "entry_timestamp") -> pd.DataFrame:
    frame = load_table_file(path)
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
    return frame


def filter_trade_log_to_post_coverage(
    trade_log: pd.DataFrame,
    posts: pd.DataFrame,
    timestamp_column: str = "entry_timestamp",
) -> pd.DataFrame:
    if trade_log.empty or posts.empty:
        return trade_log.iloc[0:0].copy()

    start_utc = pd.to_datetime(posts["created_at"], utc=True, errors="coerce").min()
    end_utc = pd.to_datetime(posts["created_at"], utc=True, errors="coerce").max()
    if pd.isna(start_utc) or pd.isna(end_utc):
        return trade_log.iloc[0:0].copy()

    return trade_log.loc[
        (trade_log[timestamp_column] >= start_utc)
        & (trade_log[timestamp_column] <= end_utc + pd.Timedelta(minutes=1))
    ].copy()


def summarize_post_collection(posts: pd.DataFrame) -> pd.DataFrame:
    if posts.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for symbol, group in posts.groupby("symbol", sort=True):
        created_at = pd.to_datetime(group["created_at"], utc=True, errors="coerce")
        rows.append(
            {
                "symbol": symbol,
                "post_count": int(len(group)),
                "start_utc": created_at.min(),
                "end_utc": created_at.max(),
                "positive_share": float((group["sentiment_label"] == "positive").mean()),
                "negative_share": float((group["sentiment_label"] == "negative").mean()),
                "neutral_share": float((group["sentiment_label"] == "neutral").mean()),
                "avg_weighted_sentiment_score": float(group["weighted_sentiment_score"].mean()),
                "avg_engagement_total": float(group["engagement_total"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def summarize_sentiment_bridge(
    bridged_trades: pd.DataFrame,
    variant_column: str,
    return_column: str,
    focus_lookback_minutes: int,
) -> pd.DataFrame:
    if bridged_trades.empty:
        return pd.DataFrame()

    prefix = f"x_{int(focus_lookback_minutes)}m"
    sentiment_column = f"{prefix}_dominant_sentiment"
    post_count_column = f"{prefix}_post_count"
    weighted_column = f"{prefix}_weighted_sentiment_sum"
    rows: list[dict] = []

    for (variant_name, sentiment_label), group in bridged_trades.groupby(
        [variant_column, sentiment_column],
        sort=True,
        dropna=False,
    ):
        returns = group[return_column].astype(float)
        count = int(len(group))
        std = float(returns.std(ddof=1)) if count > 1 else np.nan
        t_stat = float(returns.mean() / (std / np.sqrt(count))) if count > 1 and std and not np.isnan(std) else np.nan
        rows.append(
            {
                "variant_name": variant_name,
                "variant_label": group["variant_label"].iloc[0] if "variant_label" in group.columns else variant_name,
                "sentiment_label": sentiment_label,
                "trade_count": count,
                "coverage_ratio": float((group[post_count_column] > 0).mean()),
                "mean_return": float(returns.mean()),
                "median_return": float(returns.median()),
                "win_rate": float((returns > 0).mean()),
                "t_stat": t_stat,
                "avg_post_count": float(group[post_count_column].mean()),
                "avg_weighted_sentiment_sum": float(group[weighted_column].mean()),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(["variant_name", "mean_return"], ascending=[True, False]).reset_index(drop=True)


def build_x_sentiment_report(
    collection_log: pd.DataFrame,
    post_summary: pd.DataFrame,
    bridged_trades: pd.DataFrame,
    bridge_summary: pd.DataFrame,
    output_paths: list[str],
    token_present: bool,
) -> str:
    lines = ["# X Sentiment 연구 브리지", ""]

    lines.append("## 개요")
    lines.append("- 이 경로는 baseline/tick과 분리된 별도 X 데이터 프로젝트입니다.")
    lines.append("- 최근 X 게시물을 수집한 뒤, 15분/60분/240분 직전 sentiment를 기존 tick 후보 신호에 붙여봅니다.")
    lines.append("- 현재 감성 점수는 키워드 기반이므로 `보조 설명 변수` 수준으로 해석하는 것이 맞습니다.")
    lines.append("")

    lines.append("## 수집 상태")
    lines.append(f"- bearer token present: {'yes' if token_present else 'no'}")
    if collection_log.empty:
        lines.append("- 이번 실행에서 수집 로그가 없습니다.")
    else:
        for _, row in collection_log.iterrows():
            lines.append(
                f"- {row['query_name']} / {row['symbol']}: 요청 {int(row.get('request_count', 0))}회, "
                f"신규 게시물 {int(row.get('new_posts', 0))}건, 상태 {row.get('status', 'unknown')}"
            )
    lines.append("")

    lines.append("## 게시물 커버리지")
    if post_summary.empty:
        lines.append("- 로컬 캐시에 X 게시물이 없습니다.")
        lines.append("- X recent search는 최근 7일만 직접 가져올 수 있으므로, 토큰 설정 후 매일 쌓아야 장기 연구가 됩니다.")
    else:
        for _, row in post_summary.iterrows():
            lines.append(
                f"- {row['symbol']}: {int(row['post_count'])}건, {row['start_utc']} ~ {row['end_utc']}, "
                f"positive {row['positive_share']:.2%}, negative {row['negative_share']:.2%}, "
                f"평균 weighted score {row['avg_weighted_sentiment_score']:.2f}"
            )
    lines.append("")

    lines.append("## 결합 표본")
    if bridged_trades.empty:
        lines.append("- X 게시물 커버리지와 겹치는 trade log가 아직 없습니다.")
        lines.append("- 토큰을 넣고 며칠 수집을 쌓은 뒤 다시 실행하면, 최근 구간부터 sentiment bridge가 채워집니다.")
    else:
        lines.append(f"- 결합된 trade 수: {len(bridged_trades)}")
        lines.append(
            f"- 결합 구간: {bridged_trades['entry_timestamp'].min()} ~ {bridged_trades['entry_timestamp'].max()}"
        )
        lines.append("")
        lines.append("## sentiment별 반응")
        if bridge_summary.empty:
            lines.append("- sentiment별 요약 결과가 없습니다.")
        else:
            for variant_name, group in bridge_summary.groupby("variant_name", sort=True):
                label = group["variant_label"].iloc[0]
                lines.append(f"- {label}")
                for _, row in group.iterrows():
                    t_text = "nan" if pd.isna(row["t_stat"]) else f"{row['t_stat']:.2f}"
                    lines.append(
                        f"  - {row['sentiment_label']}: 거래 {int(row['trade_count'])}건, 평균 {row['mean_return']:.4%}, "
                        f"승률 {row['win_rate']:.2%}, t {t_text}, 평균 post {row['avg_post_count']:.2f}, "
                        f"평균 score {row['avg_weighted_sentiment_sum']:.2f}"
                    )
    lines.append("")

    lines.append("## 해석 가이드")
    lines.append("- bullish 구간의 평균 수익률이 계속 높게 유지되면 `기존 tick 후보를 X sentiment가 강화한다`고 볼 수 있습니다.")
    lines.append("- 반대로 bearish/no_posts가 더 좋으면, 현재 X 데이터는 순방향 필터가 아니라 역지표일 수도 있습니다.")
    lines.append("- 아직은 필터로 고정하지 말고, tracker처럼 일정 기간 누적 관찰하는 게 안전합니다.")
    lines.append("")

    lines.append("## 출력물")
    for path in output_paths:
        lines.append(f"- {path}")
    return "\n".join(lines) + "\n"
