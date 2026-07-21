from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tick_event_schema import TICK_EVENT_SCHEMA_VERSION, TICK_PIPELINE_VERSION


RUN_TYPE_MAP = {
    "up": "p",
    "down": "n",
    "zero": "z",
}


def classify_tick_runs(trade_frame: pd.DataFrame, previous_price: float | None = None) -> pd.DataFrame:
    if trade_frame.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "price",
                "quantity",
                "quote_quantity",
                "is_buyer_maker",
                "is_best_match",
                "tick_type",
                "run_id",
                "run_length",
            ]
        )

    frame = trade_frame.sort_values("timestamp").reset_index(drop=True).copy()
    previous_prices = frame["price"].shift(1)
    if previous_price is not None:
        previous_prices.iloc[0] = previous_price

    price_delta = frame["price"] - previous_prices
    tick_type = np.select(
        [price_delta > 0, price_delta < 0],
        ["up", "down"],
        default="zero",
    )
    frame["tick_type"] = tick_type

    if previous_price is None and not frame.empty:
        frame.loc[0, "tick_type"] = "zero"

    run_start = frame["tick_type"].ne(frame["tick_type"].shift())
    run_start.iloc[0] = True
    frame["run_id"] = run_start.cumsum()
    frame["run_length"] = frame.groupby("run_id")["run_id"].transform("size")
    return frame


def compute_daily_tick_herding_statistics(
    classified_ticks: pd.DataFrame,
    symbol: str,
    date: str,
    herding_threshold: float = -1.96,
) -> dict:
    if classified_ticks.empty:
        return {
            "schema_version": TICK_EVENT_SCHEMA_VERSION,
            "pipeline_version": TICK_PIPELINE_VERSION,
            "symbol": symbol,
            "date": date,
            "transaction_count": 0,
            "first_timestamp": pd.NaT,
            "last_timestamp": pd.NaT,
            "last_price": np.nan,
        }

    n_transactions = int(len(classified_ticks))
    run_frame = classified_ticks[["tick_type", "run_id", "run_length"]].drop_duplicates()
    tick_counts = classified_ticks["tick_type"].value_counts().to_dict()
    run_counts = run_frame["tick_type"].value_counts().to_dict()
    mean_run_lengths = run_frame.groupby("tick_type")["run_length"].mean().to_dict()
    max_run_lengths = run_frame.groupby("tick_type")["run_length"].max().to_dict()

    stats = {
        "schema_version": TICK_EVENT_SCHEMA_VERSION,
        "pipeline_version": TICK_PIPELINE_VERSION,
        "symbol": symbol,
        "date": date,
        "transaction_count": n_transactions,
        "first_timestamp": classified_ticks["timestamp"].min(),
        "last_timestamp": classified_ticks["timestamp"].max(),
        "last_price": float(classified_ticks["price"].iloc[-1]),
        "total_quantity": float(classified_ticks["quantity"].sum()),
        "total_quote_quantity": float(classified_ticks["quote_quantity"].sum()),
    }

    for tick_type in ["up", "down", "zero"]:
        runs = int(run_counts.get(tick_type, 0))
        category_count = int(tick_counts.get(tick_type, 0))
        run_z = compute_conditional_run_z(
            runs=runs,
            category_count=category_count,
            n_transactions=n_transactions,
        )
        stats[f"{tick_type}_ticks"] = int(tick_counts.get(tick_type, 0))
        stats[f"{tick_type}_runs"] = runs
        stats[f"mean_{tick_type}_run_length"] = float(mean_run_lengths.get(tick_type, np.nan))
        stats[f"max_{tick_type}_run_length"] = float(max_run_lengths.get(tick_type, np.nan))
        stats[f"run_z_{tick_type}"] = run_z
        stats[f"is_run_clustering_{tick_type}"] = bool(pd.notna(run_z) and run_z <= herding_threshold)

    return stats


def summarize_tick_herding_by_symbol(daily_stats: pd.DataFrame) -> pd.DataFrame:
    if daily_stats.empty:
        return pd.DataFrame()

    summary_records = []
    for symbol, group in daily_stats.groupby("symbol"):
        record = {
            "symbol": symbol,
            "trading_days": int(group.shape[0]),
            "total_transactions": int(group["transaction_count"].sum()),
            "mean_transactions_per_day": float(group["transaction_count"].mean()),
        }
        for label in ["up", "down", "zero"]:
            z_column = f"run_z_{label}"
            flag_column = f"is_run_clustering_{label}"
            record[f"mean_{z_column}"] = float(group[z_column].mean())
            record[f"median_{z_column}"] = float(group[z_column].median())
            record[f"min_{z_column}"] = float(group[z_column].min())
            record[f"max_{z_column}"] = float(group[z_column].max())
            record[f"clustering_days_{label}"] = int(group[flag_column].sum())
            record[f"clustering_share_{label}"] = float(group[flag_column].mean())
        summary_records.append(record)

    return pd.DataFrame(summary_records).sort_values("symbol")


def build_market_level_tick_summary(daily_stats: pd.DataFrame) -> pd.DataFrame:
    if daily_stats.empty:
        return pd.DataFrame()

    grouped = daily_stats.groupby("date").agg(
        symbol_count=("symbol", "nunique"),
        total_transactions=("transaction_count", "sum"),
        mean_run_z_up=("run_z_up", "mean"),
        mean_run_z_down=("run_z_down", "mean"),
        mean_run_z_zero=("run_z_zero", "mean"),
        clustering_symbol_count_up=("is_run_clustering_up", "sum"),
        clustering_symbol_count_down=("is_run_clustering_down", "sum"),
        clustering_symbol_count_zero=("is_run_clustering_zero", "sum"),
    )
    grouped = grouped.reset_index()
    grouped["date"] = pd.to_datetime(grouped["date"], utc=True)
    return grouped.sort_values("date")


def build_tick_sample(classified_ticks: pd.DataFrame, symbol: str, date: str, sample_rows: int) -> pd.DataFrame:
    if classified_ticks.empty or sample_rows <= 0:
        return pd.DataFrame()

    sample = classified_ticks.head(sample_rows).copy()
    sample.insert(0, "symbol", symbol)
    sample.insert(1, "date", date)
    return sample


def plot_tick_herding_series(daily_stats: pd.DataFrame, path: str | Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    if daily_stats.empty:
        for axis, title in zip(axes, ["Up-Run Herding", "Down-Run Herding", "Zero-Run Herding"]):
            axis.axis("off")
            axis.text(0.5, 0.5, "No tick herding data available.", ha="center", va="center", fontsize=11)
            axis.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    plot_frame = daily_stats.copy()
    plot_frame["date"] = pd.to_datetime(plot_frame["date"], utc=True)
    axis_meta = [
        ("run_z_up", "Up-Run Conditional Z"),
        ("run_z_down", "Down-Run Conditional Z"),
        ("run_z_zero", "Zero-Run Conditional Z"),
    ]
    palette = ["#4C78A8", "#E45756", "#54A24B", "#F58518"]

    for axis, (column, title) in zip(axes, axis_meta):
        for idx, (symbol, subset) in enumerate(plot_frame.groupby("symbol")):
            axis.plot(
                subset["date"],
                subset[column],
                marker="o",
                linewidth=1.2,
                markersize=3.5,
                color=palette[idx % len(palette)],
                label=symbol,
            )
        axis.axhline(-1.96, color="#222222", linestyle="--", linewidth=1.0, alpha=0.8)
        axis.set_title(title)
        axis.set_ylabel("H")
        axis.grid(alpha=0.2)

    axes[-1].set_xlabel("Date")
    axes[0].legend(frameon=False, ncol=max(plot_frame["symbol"].nunique(), 1))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_conditional_run_z(runs: int, category_count: int, n_transactions: int) -> float:
    """Conditional category-run z score; negative values indicate clustering."""
    n = int(n_transactions)
    k = int(category_count)
    r = int(runs)
    if n <= 1 or k <= 0 or k >= n or r < 0:
        return np.nan

    m = n - k
    mean = k * (m + 1) / n
    variance = k * m * (k - 1) * (m + 1) / ((n**2) * (n - 1))
    if variance <= 0 or not np.isfinite(variance):
        return np.nan
    return float((r - mean) / np.sqrt(variance))


def compute_legacy_equal_probability_run_score(runs: int, n_transactions: int) -> float:
    """Legacy fixed-p score retained only for explicit historical reproduction."""
    if n_transactions <= 0:
        return np.nan

    p_value = 1.0 / 3.0
    variance = p_value * (1.0 - p_value) - 3.0 * (p_value**2) * ((1.0 - p_value) ** 2)
    if variance <= 0:
        return np.nan

    chi = ((runs + 0.5) - (n_transactions * p_value * (1.0 - p_value))) / np.sqrt(n_transactions)
    return float(chi / np.sqrt(variance))
