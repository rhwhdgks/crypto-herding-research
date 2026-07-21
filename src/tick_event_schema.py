from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


TICK_EVENT_SCHEMA_VERSION = 2
TICK_PIPELINE_VERSION = "2.0"
LEGACY_EVENT_LABELS = {"micro_herding_up", "micro_herding_down", "micro_herding_zero"}
FILTER_COLUMNS = {
    "run_clustering_side",
    "price_direction",
    "aggressor_direction",
}


class LegacyTickSchemaError(ValueError):
    pass


def validate_tick_analysis_config(analysis_cfg: Mapping) -> None:
    ambiguous = sorted({"direction", "directions"}.intersection(analysis_cfg))
    if ambiguous:
        raise ValueError(
            "Ambiguous tick event config is forbidden in schema v2: "
            + ", ".join(ambiguous)
            + ". Use analysis.event_filter with explicit semantic columns."
        )
    event_filter = analysis_cfg.get("event_filter", {})
    if event_filter is None or event_filter == "any":
        return
    if not isinstance(event_filter, Mapping):
        raise ValueError("analysis.event_filter must be a mapping or 'any'")
    unknown = sorted(set(event_filter).difference(FILTER_COLUMNS))
    if unknown:
        raise ValueError(f"Unsupported event_filter columns: {', '.join(unknown)}")


def require_tick_schema_v2(frame: pd.DataFrame) -> None:
    required = {
        "schema_version",
        "run_clustering_side",
        "price_direction",
        "aggressor_direction",
        "signal_timestamp",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        legacy_values = set(frame.get("event_label", pd.Series(dtype=str)).dropna().astype(str).unique())
        legacy_hint = " Legacy micro_herding_up/down labels were detected." if legacy_values & LEGACY_EVENT_LABELS else ""
        raise LegacyTickSchemaError(
            f"Tick frame is not schema v2; missing columns: {', '.join(missing)}.{legacy_hint} Rebuild it instead of reusing the cache."
        )
    versions = pd.to_numeric(frame["schema_version"], errors="coerce").dropna().unique()
    if len(versions) != 1 or int(versions[0]) != TICK_EVENT_SCHEMA_VERSION:
        raise LegacyTickSchemaError(f"Expected tick schema version {TICK_EVENT_SCHEMA_VERSION}, got {versions.tolist()}")


def build_event_mask(frame: pd.DataFrame, event_filter: Mapping | str | None = None) -> pd.Series:
    require_tick_schema_v2(frame)
    mask = frame["is_micro_run_clustering_event"].fillna(False).astype(bool)
    if event_filter is None or event_filter == "any":
        return mask
    for column, accepted in event_filter.items():
        if accepted is None or accepted == "any":
            continue
        values = [accepted] if isinstance(accepted, str) else list(accepted)
        mask &= frame[column].isin(values)
    return mask


def build_run_side_event_mask(frame: pd.DataFrame, run_side: str) -> pd.Series:
    normalized = str(run_side).strip().lower()
    if normalized == "all":
        return build_event_mask(frame)
    if normalized not in {"up", "down", "zero"}:
        raise ValueError(f"Unsupported run_clustering_side: {run_side}")
    return build_event_mask(frame, {"run_clustering_side": [normalized]})
