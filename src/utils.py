from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _resolve_env_vars(value):
    """Replace ${VAR} or ${VAR:-default} placeholders with environment variables."""
    if isinstance(value, str):
        def repl(match: re.Match) -> str:
            var_name, default = match.group(1), match.group(2)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            return default if default is not None else match.group(0)
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return _resolve_env_vars(raw)


def resolve_data_window(data_cfg: dict) -> dict:
    resolved = dict(data_cfg)

    end_value = resolved.get("end", "latest")
    if end_value in {None, "", "latest", "now"}:
        end_ts = pd.Timestamp.now(tz="UTC").floor("min")
    else:
        end_ts = pd.Timestamp(end_value)
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")

    if resolved.get("start") not in {None, ""}:
        start_ts = pd.Timestamp(resolved["start"])
        start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
    elif resolved.get("lookback_years") is not None:
        start_ts = end_ts - pd.DateOffset(years=int(resolved["lookback_years"]))
    else:
        raise ValueError("data.start or data.lookback_years must be provided")

    if end_ts <= start_ts:
        raise ValueError("Resolved data window is invalid: end must be later than start")

    resolved["start"] = start_ts.strftime("%Y-%m-%d %H:%M:%S")
    resolved["end"] = end_ts.strftime("%Y-%m-%d %H:%M:%S")
    resolved["resolved_start_utc"] = str(start_ts)
    resolved["resolved_end_utc"] = str(end_ts)
    return resolved


def setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def prepare_output_dirs(project_root: Path, config: dict) -> Dict[str, Path]:
    base_dir = project_root / config.get("output", {}).get("base_dir", "outputs")
    intermediate_dir = base_dir / "intermediate"
    plots_dir = base_dir / "plots"

    for path in [base_dir, intermediate_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "base": base_dir,
        "intermediate": intermediate_dir,
        "plots": plots_dir,
    }


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def save_dataframe(frame: pd.DataFrame | pd.Series, path: str | Path, index: bool = True) -> None:
    path = ensure_parent_dir(path)
    if isinstance(frame, pd.Series):
        frame.to_frame().to_csv(path, index=index)
        return
    frame.to_csv(path, index=index)


def save_text(content: str, path: str | Path) -> None:
    with ensure_parent_dir(path).open("w", encoding="utf-8") as handle:
        handle.write(content)


def save_json(content: dict, path: str | Path) -> None:
    with ensure_parent_dir(path).open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2)


def save_config_snapshot(config: dict, path: str | Path) -> None:
    with ensure_parent_dir(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def save_provenance_manifest(
    config: dict,
    path: str | Path,
    schema_version: int,
    pipeline_version: str,
    statistical_method: str,
    input_manifest_path: str | Path | None = None,
    random_seed: int | None = None,
    train_start: str | None = None,
    train_end: str | None = None,
    oos_start: str | None = None,
    oos_end: str | None = None,
) -> None:
    config_bytes = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    input_hash = None
    if input_manifest_path is not None and Path(input_manifest_path).is_file():
        input_hash = hashlib.sha256(Path(input_manifest_path).read_bytes()).hexdigest()
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_worktree_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
        git_worktree_dirty = None
    data_cfg = config.get("data", {})
    split_cfg = config.get("sample_split", {})
    payload = {
        "schema_version": int(schema_version),
        "pipeline_version": pipeline_version,
        "git_commit": git_commit,
        "git_worktree_dirty": git_worktree_dirty,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "input_manifest_sha256": input_hash,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_start": train_start or split_cfg.get("train_start", data_cfg.get("train_start")),
        "train_end": train_end or split_cfg.get("train_end", split_cfg.get("fit_end", data_cfg.get("train_end"))),
        "oos_start": oos_start or split_cfg.get("oos_start", data_cfg.get("oos_start")),
        "oos_end": oos_end or split_cfg.get("oos_end", data_cfg.get("oos_end")),
        "statistical_method": statistical_method,
        "random_seed": random_seed,
        "status": "corrected",
    }
    save_json(payload, path)


def save_input_manifest(paths: Iterable[str | Path], path: str | Path) -> Path:
    entries = []
    for raw_path in paths:
        source = Path(raw_path)
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        try:
            display_path = source.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            display_path = source.name
        entries.append(
            {
                "path": display_path,
                "size": source.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    destination = ensure_parent_dir(path)
    save_json({"files": entries}, destination)
    return destination


def load_table_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def timeframe_to_pandas_freq(timeframe: str) -> str:
    mapping = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
        "1w": "W-MON",
    }
    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return mapping[timeframe]


def timeframe_to_minutes(timeframe: str) -> int:
    mapping = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
        "1w": 10080,
    }
    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return mapping[timeframe]


def horizon_to_label(minutes: int) -> str:
    minutes = int(minutes)
    if minutes % 1440 == 0:
        return f"{minutes // 1440}d"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}m"


def plot_csad_vs_market(analysis_frame: pd.DataFrame, path: str | Path) -> None:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 6))

    if analysis_frame.empty:
        _render_empty_plot(fig, ax, "CSAD vs |Market Return|", "No observations available.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    plot_frame = analysis_frame.copy()
    if "event_type" in plot_frame.columns:
        plot_frame["event_type"] = plot_frame["event_type"].fillna("none")
    else:
        plot_frame["event_type"] = "none"

    color_map = {
        "none": "#4C78A8",
        "low_dispersion": "#54A24B",
        "shock": "#E45756",
    }

    for event_type, color in color_map.items():
        subset = plot_frame[plot_frame["event_type"] == event_type]
        if subset.empty:
            continue
        ax.scatter(
            subset["abs_market_return"],
            subset["csad"],
            s=12,
            alpha=0.55,
            c=color,
            label=event_type,
        )

    ax.set_title("CSAD vs Absolute Market Return")
    ax.set_xlabel("|Market Return|")
    ax.set_ylabel("CSAD")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_event_returns(
    event_study_results: pd.DataFrame,
    path: str | Path,
    label_column: str = "event_type",
) -> None:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(11, 6))

    if event_study_results.empty:
        _render_empty_plot(fig, ax, "Event Study Mean Returns", "No event-study results available.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    ordered = event_study_results.sort_values(["horizon_minutes", label_column]).copy()
    ordered["mean_return_pct"] = ordered["mean_return"] * 100.0
    horizons = ordered["horizon_label"].drop_duplicates().tolist()
    labels = ordered[label_column].drop_duplicates().tolist()
    x = np.arange(len(horizons))
    width = 0.8 / max(len(labels), 1)

    palette = ["#54A24B", "#E45756", "#4C78A8", "#F58518", "#72B7B2"]
    for idx, label in enumerate(labels):
        subset = ordered[ordered[label_column] == label].set_index("horizon_label").reindex(horizons)
        offset = (idx - (len(labels) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            subset["mean_return_pct"].fillna(0.0),
            width=width,
            label=label,
            color=palette[idx % len(palette)],
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_ylabel("Mean Forward Return (%)")
    ax.set_xlabel("Holding Period")
    ax.set_title("Event Study Mean Forward Returns")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_event_occurrences(
    analysis_frame: pd.DataFrame,
    market_index: pd.Series,
    path: str | Path,
    label_column: str = "event_type",
) -> None:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(12, 6))

    if market_index.empty:
        _render_empty_plot(fig, ax, "Event Occurrences", "No market index observations available.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    ax.plot(market_index.index, market_index.values, color="#4C78A8", linewidth=1.1, label="market_index")

    if not analysis_frame.empty and label_column in analysis_frame.columns:
        marker_map = {
            "low_dispersion": ("o", "#54A24B"),
            "shock": ("x", "#E45756"),
            "bullish_herd": ("^", "#F58518"),
            "panic_herd": ("v", "#B279A2"),
        }
        event_rows = analysis_frame[analysis_frame[label_column].fillna("none") != "none"]
        for label, subset in event_rows.groupby(label_column):
            if label not in marker_map or subset.empty:
                continue
            marker, color = marker_map[label]
            aligned_index = market_index.reindex(subset.index).dropna()
            if aligned_index.empty:
                continue
            ax.scatter(
                aligned_index.index,
                aligned_index.values,
                marker=marker,
                color=color,
                s=36,
                alpha=0.85,
                label=label,
            )

    ax.set_title("Event Occurrences on Market Index")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Market Index")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_event_paths(
    path_frame: pd.DataFrame,
    path: str | Path,
    label_column: str = "event_type",
) -> None:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(11, 6))

    if path_frame.empty:
        _render_empty_plot(fig, ax, "Event-Time Average Return Paths", "No event paths available.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    palette = ["#54A24B", "#E45756", "#4C78A8", "#F58518", "#B279A2"]
    for idx, (label, subset) in enumerate(path_frame.groupby(label_column)):
        ax.plot(
            subset["event_time_minutes"],
            subset["mean_cumulative_return"] * 100.0,
            linewidth=1.6,
            color=palette[idx % len(palette)],
            label=label,
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_title("Average Event-Time Cumulative Return Paths")
    ax.set_xlabel("Minutes After Event")
    ax.set_ylabel("Average Cumulative Return (%)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_backtest_cumulative_pnl(curve_frame: pd.DataFrame, path: str | Path) -> None:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(11, 6))

    if curve_frame.empty:
        _render_empty_plot(fig, ax, "Backtest Cumulative PnL", "No backtest curves available.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    plot_frame = curve_frame[curve_frame["curve_group"] == "all_signals"].copy()
    if plot_frame.empty:
        plot_frame = curve_frame.copy()

    palette = ["#4C78A8", "#54A24B", "#E45756", "#F58518", "#72B7B2", "#B279A2"]
    for idx, (label, subset) in enumerate(plot_frame.groupby("horizon_label")):
        ax.plot(
            subset["trade_number"],
            subset["cumulative_pnl"] * 100.0,
            linewidth=1.5,
            color=palette[idx % len(palette)],
            label=label,
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_title("Simple Event-Driven Backtest Cumulative PnL")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Cumulative PnL (%)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def list_existing_plot_paths(plot_dir: str | Path) -> list[str]:
    plot_dir = Path(plot_dir)
    if not plot_dir.exists():
        return []
    return sorted(str(path) for path in plot_dir.glob("*.png"))


def _render_empty_plot(fig, ax, title: str, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    ax.set_title(title)
    fig.tight_layout()


def _get_pyplot():
    import matplotlib.pyplot as plt

    return plt
