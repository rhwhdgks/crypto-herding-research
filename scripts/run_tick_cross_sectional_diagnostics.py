from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

from tick_cross_sectional_diagnostics import (
    build_cross_sectional_report,
    load_micro_frames,
    load_symbol_focus,
    plot_down_focus,
    plot_group_structure,
    summarize_down_hours,
    summarize_down_prior_state,
    summarize_structure_by_group,
    summarize_structure_by_symbol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tick cross-sectional diagnostics.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tick/multi_asset_365d/cross_sectional_diagnostics.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    base_dir = ROOT / config["output"]["base_dir"]
    plot_dir = base_dir / "plots"
    base_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    symbol_focus = load_symbol_focus(
        path=ROOT / config["input"]["symbol_focus_path"],
        focus_horizon_minutes=int(config["analysis"]["focus_horizon_minutes"]),
    )
    micro_frame = load_micro_frames(config)

    down_hour_summary = summarize_down_hours(
        micro_frame=micro_frame,
        focus_horizon_minutes=int(config["analysis"]["focus_horizon_minutes"]),
        min_event_count=int(config["analysis"]["min_hour_event_count"]),
    )
    down_prior_summary = summarize_down_prior_state(
        micro_frame=micro_frame,
        focus_horizon_minutes=int(config["analysis"]["focus_horizon_minutes"]),
        min_event_count=int(config["analysis"]["min_prior_event_count"]),
    )
    structure_symbol_summary = summarize_structure_by_symbol(
        micro_frame=micro_frame,
        symbol_focus=symbol_focus,
    )
    structure_group_summary = summarize_structure_by_group(
        micro_frame=micro_frame,
        focus_horizon_minutes=int(config["analysis"]["focus_horizon_minutes"]),
    )

    down_focus = symbol_focus.loc[symbol_focus["event_label"] == "down"].copy()
    down_focus_path = base_dir / "down_focus_summary.csv"
    down_hours_path = base_dir / "down_hour_summary.csv"
    down_prior_path = base_dir / "down_prior_state_summary.csv"
    structure_symbol_path = base_dir / "structure_symbol_summary.csv"
    structure_group_path = base_dir / "structure_group_summary.csv"

    down_focus.to_csv(down_focus_path, index=False)
    down_hour_summary.to_csv(down_hours_path, index=False)
    down_prior_summary.to_csv(down_prior_path, index=False)
    structure_symbol_summary.to_csv(structure_symbol_path, index=False)
    structure_group_summary.to_csv(structure_group_path, index=False)

    down_plot = plot_dir / "down_focus.png"
    group_plot = plot_dir / "group_structure.png"
    plot_down_focus(down_focus, down_plot)
    plot_group_structure(structure_group_summary, group_plot)

    report = build_cross_sectional_report(
        config=config,
        symbol_focus=symbol_focus,
        down_hour_summary=down_hour_summary,
        down_prior_summary=down_prior_summary,
        structure_symbol_summary=structure_symbol_summary,
        structure_group_summary=structure_group_summary,
        plot_paths=[str(down_plot), str(group_plot)],
    )
    (base_dir / "tick_cross_sectional_diagnostics_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
