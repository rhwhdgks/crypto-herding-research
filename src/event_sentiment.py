from __future__ import annotations

from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import horizon_to_label


def attach_event_sentiment_features(
    events: pd.DataFrame,
    scored_news: pd.DataFrame,
    lookback_minutes: list[int],
    timestamp_column: str = "timestamp",
    positive_threshold: float = 0.05,
    negative_threshold: float = -0.05,
) -> pd.DataFrame:
    frame = events.copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    frame = frame.sort_values(timestamp_column).reset_index(drop=True)

    if frame.empty:
        return frame

    if scored_news.empty:
        for lookback in lookback_minutes:
            prefix = _prefix(lookback)
            frame[f"sentiment_mean_{prefix}"] = np.nan
            frame[f"pos_count_{prefix}"] = 0
            frame[f"neg_count_{prefix}"] = 0
            frame[f"news_count_{prefix}"] = 0
            frame[f"sentiment_label_{prefix}"] = "neutral"
        return frame

    news = scored_news.copy()
    news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True, errors="coerce")
    news = news.sort_values("timestamp").reset_index(drop=True)
    news_times = news["timestamp"].astype("int64").to_numpy()
    event_times = frame[timestamp_column].astype("int64").to_numpy()

    metric_map = {
        "sentiment_sum": news["sentiment_score"].astype(float).to_numpy(),
        "pos_count": (news["sentiment_label"] == "positive").astype(float).to_numpy(),
        "neg_count": (news["sentiment_label"] == "negative").astype(float).to_numpy(),
        "news_count": np.ones(len(news), dtype=float),
    }
    cumsums = {
        metric: np.concatenate([[0.0], np.cumsum(values)])
        for metric, values in metric_map.items()
    }

    for lookback in lookback_minutes:
        prefix = _prefix(lookback)
        window_ns = int(pd.Timedelta(minutes=int(lookback)).value)
        right = np.searchsorted(news_times, event_times, side="left")
        left = np.searchsorted(news_times, event_times - window_ns, side="left")

        news_count = cumsums["news_count"][right] - cumsums["news_count"][left]
        sentiment_sum = cumsums["sentiment_sum"][right] - cumsums["sentiment_sum"][left]
        pos_count = cumsums["pos_count"][right] - cumsums["pos_count"][left]
        neg_count = cumsums["neg_count"][right] - cumsums["neg_count"][left]

        sentiment_mean = np.full(len(news_count), np.nan, dtype=float)
        np.divide(
            sentiment_sum,
            news_count,
            out=sentiment_mean,
            where=news_count > 0,
        )

        frame[f"sentiment_mean_{prefix}"] = sentiment_mean
        frame[f"pos_count_{prefix}"] = pos_count.astype(int)
        frame[f"neg_count_{prefix}"] = neg_count.astype(int)
        frame[f"news_count_{prefix}"] = news_count.astype(int)
        frame[f"sentiment_label_{prefix}"] = np.where(
            news_count <= 0,
            "neutral",
            np.where(
                sentiment_mean >= positive_threshold,
                "positive",
                np.where(sentiment_mean <= negative_threshold, "negative", "neutral"),
            ),
        )

    return frame


def build_sentiment_event_groups(
    events_with_sentiment: pd.DataFrame,
    label_window_minutes: int,
    event_type_column: str = "event_type",
) -> pd.DataFrame:
    frame = events_with_sentiment.copy()
    prefix = _prefix(label_window_minutes)
    sentiment_column = f"sentiment_label_{prefix}"
    group_column = f"event_sentiment_group_{prefix}"

    if sentiment_column not in frame.columns:
        raise ValueError(f"Sentiment label column not found: {sentiment_column}")

    def _map_group(row: pd.Series) -> str:
        event_type = str(row.get(event_type_column, "none"))
        sentiment_label = str(row.get(sentiment_column, "neutral"))
        if event_type == "herding" and sentiment_label == "positive":
            return "bullish_herd"
        if event_type == "herding" and sentiment_label == "negative":
            return "panic_herd"
        if event_type == "herding":
            return "neutral_herd"
        if event_type == "shock" and sentiment_label == "positive":
            return "positive_shock"
        if event_type == "shock" and sentiment_label == "negative":
            return "negative_shock"
        if event_type == "shock":
            return "neutral_shock"
        return "none"

    frame[group_column] = frame.apply(_map_group, axis=1)
    return frame


def engineer_sentiment_feature_layer(
    events_with_sentiment: pd.DataFrame,
    focus_window_minutes: int,
    feature_cfg: dict,
    event_type_column: str = "event_type",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = events_with_sentiment.copy()
    prefix = _prefix(focus_window_minutes)

    sentiment_mean_col = f"sentiment_mean_{prefix}"
    pos_count_col = f"pos_count_{prefix}"
    neg_count_col = f"neg_count_{prefix}"
    news_count_col = f"news_count_{prefix}"
    sentiment_label_col = f"sentiment_label_{prefix}"
    feature_group_col = f"news_feature_group_{prefix}"

    if sentiment_mean_col not in frame.columns:
        raise ValueError(f"Sentiment mean column not found: {sentiment_mean_col}")

    frame[f"news_attention_{prefix}"] = np.log1p(frame[news_count_col].fillna(0).astype(float))
    frame[f"sentiment_strength_{prefix}"] = frame[sentiment_mean_col].abs()
    frame[f"sentiment_pressure_{prefix}"] = frame[sentiment_mean_col].fillna(0.0) * frame[f"news_attention_{prefix}"]
    frame[f"sentiment_imbalance_{prefix}"] = np.where(
        frame[news_count_col].fillna(0).astype(float) > 0,
        (frame[pos_count_col].fillna(0).astype(float) - frame[neg_count_col].fillna(0).astype(float))
        / frame[news_count_col].fillna(0).astype(float),
        0.0,
    )
    frame[f"has_news_{prefix}"] = frame[news_count_col].fillna(0).astype(int) > 0

    news_with_coverage = frame.loc[frame[news_count_col].fillna(0).astype(int) > 0].copy()
    dense_news_threshold = _resolve_threshold(
        news_with_coverage[news_count_col].astype(float),
        feature_cfg.get("dense_news_threshold"),
        float(feature_cfg.get("dense_news_quantile", 0.75)),
        default_value=1.0,
    )
    strong_sentiment_threshold = _resolve_threshold(
        news_with_coverage[f"sentiment_strength_{prefix}"].astype(float),
        feature_cfg.get("strong_sentiment_threshold"),
        float(feature_cfg.get("strong_sentiment_quantile", 0.75)),
        default_value=0.15,
    )

    frame[f"is_dense_news_{prefix}"] = frame[news_count_col].fillna(0).astype(float) >= dense_news_threshold
    frame[f"is_strong_sentiment_{prefix}"] = frame[f"sentiment_strength_{prefix}"].fillna(0.0) >= strong_sentiment_threshold
    frame[f"is_news_confirmed_{prefix}"] = frame[f"is_dense_news_{prefix}"] & frame[f"is_strong_sentiment_{prefix}"]

    def _map_feature_group(row: pd.Series) -> str:
        event_type = str(row.get(event_type_column, "none"))
        sentiment_label = str(row.get(sentiment_label_col, "neutral"))
        is_confirmed = bool(row.get(f"is_news_confirmed_{prefix}", False))
        if event_type == "herding" and sentiment_label == "positive" and is_confirmed:
            return "bullish_herd_strong"
        if event_type == "herding" and sentiment_label == "negative" and is_confirmed:
            return "panic_herd_strong"
        if event_type == "shock" and sentiment_label == "positive" and is_confirmed:
            return "positive_shock_strong"
        if event_type == "shock" and sentiment_label == "negative" and is_confirmed:
            return "negative_shock_strong"
        if event_type == "herding":
            return "herding_other"
        if event_type == "shock":
            return "shock_other"
        return "none"

    frame[feature_group_col] = frame.apply(_map_feature_group, axis=1)

    thresholds = pd.DataFrame(
        [
            {
                "focus_window_minutes": focus_window_minutes,
                "dense_news_threshold": dense_news_threshold,
                "strong_sentiment_threshold": strong_sentiment_threshold,
                "dense_news_quantile": float(feature_cfg.get("dense_news_quantile", 0.75)),
                "strong_sentiment_quantile": float(feature_cfg.get("strong_sentiment_quantile", 0.75)),
            }
        ]
    )
    return frame, thresholds


def summarize_sentiment_event_study(
    events_with_groups: pd.DataFrame,
    holding_periods: list[int],
    group_column: str,
) -> pd.DataFrame:
    if events_with_groups.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    valid_groups = [label for label in events_with_groups[group_column].dropna().unique().tolist() if label != "none"]

    for group_name in sorted(valid_groups):
        subset = events_with_groups.loc[events_with_groups[group_column] == group_name]
        for horizon in holding_periods:
            column = f"forward_return_{int(horizon)}m"
            if column not in subset.columns:
                continue
            sample = subset[column].dropna().astype(float)
            rows.append(
                {
                    "event_sentiment_group": group_name,
                    "horizon_minutes": int(horizon),
                    "horizon_label": horizon_to_label(int(horizon)),
                    "count": int(sample.shape[0]),
                    "mean_return": sample.mean() if not sample.empty else np.nan,
                    "median_return": sample.median() if not sample.empty else np.nan,
                    "std_return": sample.std(ddof=1) if sample.shape[0] >= 2 else np.nan,
                    "t_stat": _compute_t_stat(sample),
                    "win_rate": (sample > 0).mean() if not sample.empty else np.nan,
                }
            )

    return pd.DataFrame(rows).sort_values(["event_sentiment_group", "horizon_minutes"]).reset_index(drop=True)


def summarize_feature_layer(
    events_with_groups: pd.DataFrame,
    focus_window_minutes: int,
    feature_group_column: str,
) -> pd.DataFrame:
    if events_with_groups.empty:
        return pd.DataFrame()

    prefix = _prefix(focus_window_minutes)
    rows: list[dict] = []
    for group_name, subset in events_with_groups.groupby(feature_group_column, sort=True):
        if group_name == "none" or subset.empty:
            continue
        rows.append(
            {
                "news_feature_group": group_name,
                "count": int(len(subset)),
                "mean_news_count": float(subset[f"news_count_{prefix}"].mean()),
                "mean_sentiment_mean": float(subset[f"sentiment_mean_{prefix}"].mean()),
                "mean_news_attention": float(subset[f"news_attention_{prefix}"].mean()),
                "mean_sentiment_strength": float(subset[f"sentiment_strength_{prefix}"].mean()),
                "mean_sentiment_pressure": float(subset[f"sentiment_pressure_{prefix}"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def plot_sentiment_split_summary(summary: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    if summary.empty:
        ax.text(0.5, 0.5, "표시할 sentiment split 결과가 없습니다.", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    ordered_groups = [
        "bullish_herd",
        "panic_herd",
        "neutral_herd",
        "positive_shock",
        "negative_shock",
        "neutral_shock",
    ]
    plotted = summary.copy()
    plotted["event_sentiment_group"] = pd.Categorical(
        plotted["event_sentiment_group"],
        categories=ordered_groups,
        ordered=True,
    )
    plotted = plotted.sort_values(["event_sentiment_group", "horizon_minutes"])

    palette = {
        "bullish_herd": "#2E8B57",
        "panic_herd": "#C0392B",
        "neutral_herd": "#7F8C8D",
        "positive_shock": "#1F77B4",
        "negative_shock": "#8E44AD",
        "neutral_shock": "#95A5A6",
    }

    for group_name, group in plotted.groupby("event_sentiment_group", sort=False):
        if group.empty:
            continue
        ax.plot(
            group["horizon_minutes"],
            group["mean_return"] * 100.0,
            marker="o",
            linewidth=2.0,
            label=str(group_name),
            color=palette.get(str(group_name), None),
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_title("Sentiment Split Event Study Mean Returns")
    ax.set_xlabel("Holding Period (minutes)")
    ax.set_ylabel("Mean Forward Return (%)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_group_summary(summary: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    if summary.empty:
        ax.text(0.5, 0.5, "표시할 feature group 결과가 없습니다.", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    ordered_groups = [
        "bullish_herd_strong",
        "panic_herd_strong",
        "herding_other",
        "positive_shock_strong",
        "negative_shock_strong",
        "shock_other",
    ]
    plotted = summary.copy()
    plotted["event_sentiment_group"] = pd.Categorical(
        plotted["event_sentiment_group"],
        categories=ordered_groups,
        ordered=True,
    )
    plotted = plotted.sort_values(["event_sentiment_group", "horizon_minutes"])

    for group_name, group in plotted.groupby("event_sentiment_group", sort=False):
        if group.empty:
            continue
        ax.plot(
            group["horizon_minutes"],
            group["mean_return"] * 100.0,
            marker="o",
            linewidth=2.0,
            label=str(group_name),
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_title("News Feature Group Event Study Mean Returns")
    ax.set_xlabel("Holding Period (minutes)")
    ax.set_ylabel("Mean Forward Return (%)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_sentiment_extension_report(
    news_scored: pd.DataFrame,
    event_features: pd.DataFrame,
    summary: pd.DataFrame,
    feature_thresholds: pd.DataFrame,
    feature_overview: pd.DataFrame,
    feature_summary: pd.DataFrame,
    label_window_minutes: int,
    output_paths: list[str],
) -> str:
    prefix = _prefix(label_window_minutes)
    group_column = f"event_sentiment_group_{prefix}"
    lines = ["# 뉴스 Sentiment 확장 리포트", ""]

    lines.append("## 뉴스 데이터")
    lines.append(f"- scored headline 수: {len(news_scored)}")
    if not news_scored.empty:
        lines.append(f"- 구간: {news_scored['timestamp'].min()} ~ {news_scored['timestamp'].max()}")
        lines.append(
            f"- positive {((news_scored['sentiment_label'] == 'positive').mean()):.2%}, "
            f"negative {((news_scored['sentiment_label'] == 'negative').mean()):.2%}, "
            f"neutral {((news_scored['sentiment_label'] == 'neutral').mean()):.2%}"
        )
    lines.append("")

    lines.append("## 이벤트 결합")
    lines.append(f"- sentiment feature가 붙은 event 수: {len(event_features)}")
    if not event_features.empty:
        label_col = f"sentiment_label_{prefix}"
        label_counts = event_features[label_col].value_counts(dropna=False).to_dict()
        lines.append(f"- {label_window_minutes}분 기준 sentiment 분포: {label_counts}")
    lines.append("")

    lines.append("## 허딩 feature 레이어")
    lines.append(
        f"- 뉴스 제목 데이터는 `sentiment_mean_{prefix}`, `news_count_{prefix}`, `news_attention_{prefix}`, "
        f"`sentiment_strength_{prefix}`, `sentiment_pressure_{prefix}` 형태로 event feature에 직접 들어갑니다."
    )
    if not feature_thresholds.empty:
        threshold_row = feature_thresholds.iloc[0]
        lines.append(
            f"- dense news 기준: news_count_{prefix} >= {threshold_row['dense_news_threshold']:.2f}"
        )
        lines.append(
            f"- strong sentiment 기준: |sentiment_mean_{prefix}| >= {threshold_row['strong_sentiment_threshold']:.4f}"
        )
    if not feature_overview.empty:
        lines.append("- feature group count:")
        for _, row in feature_overview.iterrows():
            lines.append(
                f"  - {row['news_feature_group']}: {int(row['count'])}건, "
                f"평균 news {row['mean_news_count']:.2f}, 평균 pressure {row['mean_sentiment_pressure']:.4f}"
            )
    lines.append("")

    lines.append("## sentiment split event study")
    if summary.empty:
        lines.append("- 요약 결과가 없습니다.")
    else:
        for group_name, group in summary.groupby("event_sentiment_group", sort=False):
            best_row = group.sort_values("mean_return", ascending=False).iloc[0]
            t_text = "nan" if pd.isna(best_row["t_stat"]) else f"{best_row['t_stat']:.2f}"
            lines.append(
                f"- {group_name}: 최고 horizon {best_row['horizon_label']}, 평균 {best_row['mean_return']:.4%}, "
                f"t {t_text}, "
                f"count {int(best_row['count'])}"
            )
    lines.append("")

    lines.append("## feature group event study")
    if feature_summary.empty:
        lines.append("- feature group 요약 결과가 없습니다.")
    else:
        for group_name, group in feature_summary.groupby("event_sentiment_group", sort=False):
            best_row = group.sort_values("mean_return", ascending=False).iloc[0]
            t_text = "nan" if pd.isna(best_row["t_stat"]) else f"{best_row['t_stat']:.2f}"
            lines.append(
                f"- {group_name}: 최고 horizon {best_row['horizon_label']}, 평균 {best_row['mean_return']:.4%}, "
                f"t {t_text}, count {int(best_row['count'])}"
            )
    lines.append("")

    lines.append("## 출력물")
    for path in output_paths:
        lines.append(f"- {path}")
    return "\n".join(lines) + "\n"


def _compute_t_stat(sample: pd.Series) -> float:
    if sample.shape[0] < 2:
        return np.nan
    std = sample.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(sample.mean() / (std / sqrt(sample.shape[0])))


def _prefix(minutes: int) -> str:
    return f"{int(minutes)}m"


def _resolve_threshold(
    series: pd.Series,
    explicit_value: float | None,
    quantile_value: float,
    default_value: float,
) -> float:
    clean = series.dropna().astype(float)
    if explicit_value is not None:
        return float(explicit_value)
    if clean.empty:
        return float(default_value)
    return float(clean.quantile(quantile_value))
