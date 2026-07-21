from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
from scipy.stats import binomtest, chi2_contingency
from statsmodels.stats.proportion import proportion_confint

from frequency_sensitivity import benjamini_hochberg
from tick_event_schema import require_tick_schema_v2


REQUIRED_COLUMNS = [
    "bucket_start",
    "symbol",
    "schema_version",
    "signal_timestamp",
    "is_micro_run_clustering_event",
    "is_control_bucket",
    "run_clustering_side",
    "price_direction",
    "aggressor_direction",
    "aggressor_imbalance",
    "bucket_return",
    "transaction_count",
    "total_quote_quantity",
    "forward_return_30m",
]


def load_tick_micro_frame(path: str | Path, expected_symbols: Sequence[str]) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        header = pq.ParquetFile(source).schema.names
    else:
        header = pd.read_csv(source, nrows=0).columns
    missing = sorted(set(REQUIRED_COLUMNS).difference(header))
    if missing:
        raise ValueError(f"Tick micro frame is missing required columns: {', '.join(missing)}")

    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source, columns=REQUIRED_COLUMNS)
        frame["bucket_start"] = pd.to_datetime(frame["bucket_start"], utc=True)
        frame["signal_timestamp"] = pd.to_datetime(frame["signal_timestamp"], utc=True)
    else:
        frame = pd.read_csv(
            source,
            usecols=REQUIRED_COLUMNS,
            parse_dates=["bucket_start", "signal_timestamp"],
        )
    frame["is_micro_run_clustering_event"] = _coerce_bool(frame["is_micro_run_clustering_event"])
    frame["is_control_bucket"] = _coerce_bool(frame["is_control_bucket"])
    frame = frame.sort_values(["symbol", "bucket_start"]).reset_index(drop=True)
    require_tick_schema_v2(frame)

    actual_symbols = set(frame["symbol"].dropna().astype(str).unique())
    missing_symbols = sorted(set(expected_symbols).difference(actual_symbols))
    if missing_symbols:
        raise ValueError(f"Tick micro frame is missing configured symbols: {', '.join(missing_symbols)}")
    return frame


def summarize_schema_coverage(frame: pd.DataFrame, source_label: str) -> pd.DataFrame:
    aggressor_available = frame["aggressor_imbalance"].notna()
    return pd.DataFrame(
        [
            {
                "source": source_label,
                "rows": int(len(frame)),
                "start": frame["bucket_start"].min(),
                "end": frame["bucket_start"].max(),
                "symbols": int(frame["symbol"].nunique()),
                "events": int(frame["is_micro_run_clustering_event"].sum()),
                "controls": int(frame["is_control_bucket"].sum()),
                "aggressor_available_rows": int(aggressor_available.sum()),
                "aggressor_available_share": float(aggressor_available.mean()),
            }
        ]
    )


def analyze_run_price_semantics(
    frame: pd.DataFrame,
    expected_symbols: Sequence[str],
    minimum_events: int,
    family_size: int,
    fdr_alpha: float,
    proxy_minimum_concordance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = frame.loc[frame["is_micro_run_clustering_event"]].copy()
    scopes = [("pooled", events)]
    scopes.extend((symbol, events.loc[events["symbol"] == symbol]) for symbol in expected_symbols)
    if len(scopes) != int(family_size):
        raise ValueError("Price-direction family size does not match pooled plus symbol scopes")

    rows: list[dict] = []
    contingency_rows: list[dict] = []
    for scope, subset in scopes:
        table = pd.crosstab(subset["run_clustering_side"], subset["price_direction"])
        table = table.reindex(index=["up", "down", "zero"], columns=["up", "down", "flat"], fill_value=0)
        chi2, chi_p = _chi_square_nonzero_margins(table)
        cramers_v = _cramers_v(float(chi2), int(table.to_numpy().sum()), table.shape)

        for run_side in table.index:
            for price_direction in table.columns:
                contingency_rows.append(
                    {
                        "scope": scope,
                        "run_clustering_side": run_side,
                        "price_direction": price_direction,
                        "count": int(table.loc[run_side, price_direction]),
                    }
                )

        directional = subset.loc[
            subset["run_clustering_side"].isin(["up", "down"])
            & subset["price_direction"].isin(["up", "down"])
        ]
        directional_count = int(len(directional))
        concordant_count = int(
            directional["run_clustering_side"].eq(directional["price_direction"]).sum()
        )
        eligible = directional_count >= int(minimum_events)
        concordance = concordant_count / directional_count if directional_count else np.nan
        ci_lower, ci_upper = (
            proportion_confint(concordant_count, directional_count, alpha=0.05, method="wilson")
            if directional_count
            else (np.nan, np.nan)
        )
        p_value = float(binomtest(concordant_count, directional_count, 0.5).pvalue) if eligible else np.nan
        rows.append(
            {
                "scope": scope,
                "event_count": int(len(subset)),
                "zero_run_event_count": int((subset["run_clustering_side"] == "zero").sum()),
                "zero_run_event_share": float((subset["run_clustering_side"] == "zero").mean()) if len(subset) else np.nan,
                "directional_event_count": directional_count,
                "concordant_event_count": concordant_count,
                "directional_concordance": concordance,
                "directional_concordance_ci_lower": float(ci_lower),
                "directional_concordance_ci_upper": float(ci_upper),
                "binomial_p_value": p_value,
                "inference_eligible": eligible,
                "cramers_v_all_categories": cramers_v,
                "chi_square_p_value_all_categories": float(chi_p),
            }
        )

    summary = pd.DataFrame(rows)
    family_p = summary["binomial_p_value"].fillna(1.0)
    summary["binomial_q_value_bh_fdr"] = benjamini_hochberg(family_p)
    summary.loc[~summary["inference_eligible"], "binomial_q_value_bh_fdr"] = np.nan
    summary["supports_price_direction_proxy"] = (
        summary["inference_eligible"]
        & (summary["directional_concordance"] >= float(proxy_minimum_concordance))
        & (summary["binomial_q_value_bh_fdr"] <= float(fdr_alpha))
    )
    return summary, pd.DataFrame(contingency_rows)


def run_market_neutral_predictive_regression(
    frame: pd.DataFrame,
    expected_symbols: Sequence[str],
    trailing_volatility_buckets: int,
    family_size: int,
    fdr_alpha: float,
    economic_effect_threshold_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = frame.copy().sort_values(["symbol", "bucket_start"]).reset_index(drop=True)
    working["future_market_loo"] = _leave_one_out_cross_sectional_mean(
        working, "forward_return_30m"
    )
    working["current_market_loo"] = _leave_one_out_cross_sectional_mean(working, "bucket_return")
    working["future_excess_return_30m"] = (
        working["forward_return_30m"] - working["future_market_loo"]
    )
    working["trailing_volatility"] = working.groupby("symbol", sort=False)["bucket_return"].transform(
        lambda values: values.rolling(
            int(trailing_volatility_buckets),
            min_periods=int(trailing_volatility_buckets),
        ).std(ddof=1)
    )
    working["abs_bucket_return"] = working["bucket_return"].abs()
    working["abs_current_market_loo"] = working["current_market_loo"].abs()
    working["log_transaction_count"] = np.log1p(pd.to_numeric(working["transaction_count"], errors="coerce"))
    working["log_quote_quantity"] = np.log1p(pd.to_numeric(working["total_quote_quantity"], errors="coerce"))
    working["hour_utc"] = working["bucket_start"].dt.hour.astype(str)
    working["utc_day"] = working["bucket_start"].dt.floor("D")

    eligible = working["is_micro_run_clustering_event"] | working["is_control_bucket"]
    model_frame = working.loc[eligible].copy()
    for side in ("up", "down", "zero"):
        model_frame[f"event_{side}"] = (
            model_frame["is_micro_run_clustering_event"]
            & model_frame["run_clustering_side"].eq(side)
        ).astype(float)

    continuous = [
        "bucket_return",
        "abs_bucket_return",
        "current_market_loo",
        "abs_current_market_loo",
        "trailing_volatility",
        "log_transaction_count",
        "log_quote_quantity",
    ]
    required = ["future_excess_return_30m", "utc_day", *continuous]
    model_frame = model_frame.dropna(subset=required)
    if model_frame.empty:
        raise ValueError("No complete rows remain for the predictive regression")

    standardized = model_frame[continuous].copy()
    for column in continuous:
        scale = float(standardized[column].std(ddof=1))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Cannot standardize degenerate control: {column}")
        standardized[column] = (standardized[column] - standardized[column].mean()) / scale

    categorical = pd.get_dummies(
        model_frame[["symbol", "hour_utc", "price_direction", "run_clustering_side"]].astype(str),
        prefix=["symbol", "hour", "price", "run_side"],
        drop_first=True,
        dtype=float,
    )
    event_columns = ["event_up", "event_down", "event_zero"]
    design = pd.concat([model_frame[event_columns], standardized, categorical], axis=1)
    design = sm.add_constant(design, has_constant="add").astype(float)
    outcome = model_frame["future_excess_return_30m"].astype(float)
    model = sm.OLS(outcome, design).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_frame["utc_day"], "use_correction": True},
    )

    if len(event_columns) != int(family_size):
        raise ValueError("Predictive family size does not match the three run-side event coefficients")
    ci = model.conf_int()
    control_mask = model_frame["is_control_bucket"]
    rows: list[dict] = []
    for side, term in zip(("up", "down", "zero"), event_columns):
        event_mask = model_frame[term].eq(1.0)
        coefficient = float(model.params[term])
        rows.append(
            {
                "run_clustering_side": side,
                "term": term,
                "event_count": int(event_mask.sum()),
                "control_count": int(control_mask.sum()),
                "event_mean_future_excess_return": float(outcome.loc[event_mask].mean()),
                "control_mean_future_excess_return": float(outcome.loc[control_mask].mean()),
                "unadjusted_difference": float(
                    outcome.loc[event_mask].mean() - outcome.loc[control_mask].mean()
                ),
                "adjusted_coefficient": coefficient,
                "adjusted_coefficient_bps": coefficient * 10_000.0,
                "cluster_std_error": float(model.bse[term]),
                "cluster_t_stat": float(model.tvalues[term]),
                "cluster_p_value": float(model.pvalues[term]),
                "ci_lower": float(ci.loc[term, 0]),
                "ci_upper": float(ci.loc[term, 1]),
                "ci_lower_bps": float(ci.loc[term, 0] * 10_000.0),
                "ci_upper_bps": float(ci.loc[term, 1] * 10_000.0),
            }
        )

    coefficients = pd.DataFrame(rows)
    coefficients["cluster_q_value_bh_fdr"] = benjamini_hochberg(coefficients["cluster_p_value"])
    coefficients["supports_economic_predictive_association"] = (
        (coefficients["cluster_q_value_bh_fdr"] <= float(fdr_alpha))
        & (coefficients["adjusted_coefficient_bps"].abs() >= float(economic_effect_threshold_bps))
    )
    coefficients["ci_inside_predeclared_economic_band"] = (
        (coefficients["ci_lower_bps"] > -float(economic_effect_threshold_bps))
        & (coefficients["ci_upper_bps"] < float(economic_effect_threshold_bps))
    )
    diagnostics = pd.DataFrame(
        [
            {
                "observations": int(model.nobs),
                "utc_day_clusters": int(model_frame["utc_day"].nunique()),
                "symbols": int(model_frame["symbol"].nunique()),
                "rsquared": float(model.rsquared),
                "adj_rsquared": float(model.rsquared_adj),
                "outcome": "target 30m return minus leave-one-out market 30m return",
                "covariance": "UTC-day clustered",
            }
        ]
    )
    return coefficients, diagnostics


def summarize_raw_aggressor_pilot(
    frame: pd.DataFrame,
    expected_symbols: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = frame.loc[
        frame["is_micro_run_clustering_event"] & frame["aggressor_imbalance"].notna()
    ].copy()
    scopes = [("pooled", events)]
    scopes.extend((symbol, events.loc[events["symbol"] == symbol]) for symbol in expected_symbols)
    rows: list[dict] = []
    tables: list[dict] = []
    for scope, subset in scopes:
        for side in ("up", "down", "zero"):
            side_frame = subset.loc[subset["run_clustering_side"] == side]
            expected = "buy" if side == "up" else "sell" if side == "down" else None
            directional = side_frame["aggressor_direction"].isin(["buy", "sell"])
            agreement = (
                side_frame.loc[directional, "aggressor_direction"].eq(expected).mean()
                if expected is not None and directional.any()
                else np.nan
            )
            rows.append(
                {
                    "scope": scope,
                    "run_clustering_side": side,
                    "event_count": int(len(side_frame)),
                    "mean_aggressor_imbalance": float(side_frame["aggressor_imbalance"].mean()),
                    "median_aggressor_imbalance": float(side_frame["aggressor_imbalance"].median()),
                    "buy_share": float((side_frame["aggressor_direction"] == "buy").mean()) if len(side_frame) else np.nan,
                    "sell_share": float((side_frame["aggressor_direction"] == "sell").mean()) if len(side_frame) else np.nan,
                    "directional_agreement": float(agreement) if pd.notna(agreement) else np.nan,
                    "inference_status": "development_pilot_no_p_value",
                }
            )
        table = pd.crosstab(subset["run_clustering_side"], subset["aggressor_direction"])
        table = table.reindex(index=["up", "down", "zero"], columns=["buy", "sell", "balanced"], fill_value=0)
        for run_side in table.index:
            for aggressor_direction in table.columns:
                tables.append(
                    {
                        "scope": scope,
                        "run_clustering_side": run_side,
                        "aggressor_direction": aggressor_direction,
                        "count": int(table.loc[run_side, aggressor_direction]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(tables)


def analyze_run_aggressor_semantics(
    frame: pd.DataFrame,
    expected_symbols: Sequence[str],
    minimum_events: int,
    family_size: int,
    fdr_alpha: float,
    proxy_minimum_concordance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = frame.loc[
        frame["is_micro_run_clustering_event"] & frame["aggressor_imbalance"].notna()
    ].copy()
    scopes = [("pooled", events)]
    scopes.extend((symbol, events.loc[events["symbol"] == symbol]) for symbol in expected_symbols)
    if len(scopes) != int(family_size):
        raise ValueError("Aggressor-direction family size does not match pooled plus symbol scopes")

    rows: list[dict] = []
    contingency_rows: list[dict] = []
    expected_direction = {"up": "buy", "down": "sell"}
    for scope, subset in scopes:
        table = pd.crosstab(subset["run_clustering_side"], subset["aggressor_direction"])
        table = table.reindex(
            index=["up", "down", "zero"],
            columns=["buy", "sell", "balanced"],
            fill_value=0,
        )
        chi2, chi_p = _chi_square_nonzero_margins(table)
        cramers_v = _cramers_v(float(chi2), int(table.to_numpy().sum()), table.shape)
        for run_side in table.index:
            for aggressor_direction in table.columns:
                contingency_rows.append(
                    {
                        "scope": scope,
                        "run_clustering_side": run_side,
                        "aggressor_direction": aggressor_direction,
                        "count": int(table.loc[run_side, aggressor_direction]),
                    }
                )

        directional = subset.loc[
            subset["run_clustering_side"].isin(["up", "down"])
            & subset["aggressor_direction"].isin(["buy", "sell"])
        ].copy()
        directional_count = int(len(directional))
        expected = directional["run_clustering_side"].map(expected_direction)
        concordant_count = int(directional["aggressor_direction"].eq(expected).sum())
        eligible = directional_count >= int(minimum_events)
        concordance = concordant_count / directional_count if directional_count else np.nan
        ci_lower, ci_upper = (
            proportion_confint(concordant_count, directional_count, alpha=0.05, method="wilson")
            if directional_count
            else (np.nan, np.nan)
        )
        p_value = (
            float(binomtest(concordant_count, directional_count, 0.5).pvalue)
            if eligible
            else np.nan
        )
        rows.append(
            {
                "scope": scope,
                "event_count": int(len(subset)),
                "aggressor_available_share": float(subset["aggressor_imbalance"].notna().mean()) if len(subset) else np.nan,
                "zero_run_event_count": int((subset["run_clustering_side"] == "zero").sum()),
                "zero_run_event_share": float((subset["run_clustering_side"] == "zero").mean()) if len(subset) else np.nan,
                "directional_event_count": directional_count,
                "concordant_event_count": concordant_count,
                "directional_concordance": concordance,
                "directional_concordance_ci_lower": float(ci_lower),
                "directional_concordance_ci_upper": float(ci_upper),
                "binomial_p_value": p_value,
                "inference_eligible": eligible,
                "cramers_v_all_categories": cramers_v,
                "chi_square_p_value_all_categories": float(chi_p),
            }
        )

    summary = pd.DataFrame(rows)
    summary["binomial_q_value_bh_fdr"] = benjamini_hochberg(
        summary["binomial_p_value"].fillna(1.0)
    )
    summary.loc[~summary["inference_eligible"], "binomial_q_value_bh_fdr"] = np.nan
    summary["supports_aggressor_direction_proxy"] = (
        summary["inference_eligible"]
        & (summary["directional_concordance"] >= float(proxy_minimum_concordance))
        & (summary["binomial_q_value_bh_fdr"] <= float(fdr_alpha))
    )
    return summary, pd.DataFrame(contingency_rows)


def plot_contingency_heatmap(
    contingency: pd.DataFrame,
    column_name: str,
    path: str | Path,
    title: str,
) -> None:
    pooled = contingency.loc[contingency["scope"] == "pooled"]
    table = pooled.pivot(index="run_clustering_side", columns=column_name, values="count").fillna(0)
    table = table.div(table.sum(axis=1).replace(0, np.nan), axis=0)
    _plot_heatmap_table(table, path, title)


def plot_predictive_coefficients(coefficients: pd.DataFrame, path: str | Path) -> None:
    _configure_font()
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(coefficients))
    values = coefficients["adjusted_coefficient_bps"].to_numpy(dtype=float)
    errors = np.vstack(
        [
            values - coefficients["ci_lower_bps"].to_numpy(dtype=float),
            coefficients["ci_upper_bps"].to_numpy(dtype=float) - values,
        ]
    )
    axis.errorbar(x, values, yerr=errors, fmt="o", capsize=5, linewidth=1.8, color="#1f5a75")
    axis.axhline(0.0, color="#8b1e1e", linestyle="--", linewidth=1.2)
    axis.set_xticks(x, coefficients["run_clustering_side"].tolist())
    axis.set_xlabel("run clustering side")
    axis.set_ylabel("조정 30분 시장중립 수익률 (bp)")
    axis.set_title("Run-clustering event의 조정 30분 반응")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def build_tick_semantic_report(
    schema_coverage: pd.DataFrame,
    price_summary: pd.DataFrame,
    predictive_coefficients: pd.DataFrame,
    predictive_diagnostics: pd.DataFrame,
    raw_aggressor_summary: pd.DataFrame,
    config: Mapping,
    plot_paths: Sequence[str],
) -> str:
    pooled_price = price_summary.loc[price_summary["scope"] == "pooled"].iloc[0]
    zero_share = float(pooled_price["zero_run_event_share"])
    direction_proxy = bool(pooled_price["supports_price_direction_proxy"])
    predictive_survivors = predictive_coefficients.loc[
        predictive_coefficients["supports_economic_predictive_association"]
    ]
    raw_pooled = raw_aggressor_summary.loc[raw_aggressor_summary["scope"] == "pooled"]
    all_inside_economic_band = bool(
        predictive_coefficients["ci_inside_predeclared_economic_band"].all()
    )

    lines = [
        "# Tick run clustering 의미 검증 보고서",
        "",
        "## 자료 범위",
        "",
        "| 자료 | 행 | 기간 | 종목 | 이벤트 | aggressor 가용률 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in schema_coverage.itertuples(index=False):
        lines.append(
            f"| {row.source} | {row.rows:,} | {row.start} ~ {row.end} | {row.symbols} | "
            f"{row.events:,} | {row.aggressor_available_share:.2%} |"
        )

    lines.extend(
        [
            "",
            "## 1. Run side는 가격 방향인가",
            "",
            f"- 5년 cache event 중 run side=zero 비중은 {zero_share:.2%}입니다.",
            f"- pooled up/down directional event는 {int(pooled_price['directional_event_count']):,}건이며, "
            f"같은 방향 일치율은 {pooled_price['directional_concordance']:.2%} "
            f"(95% CI {pooled_price['directional_concordance_ci_lower']:.2%}~{pooled_price['directional_concordance_ci_upper']:.2%}, "
            f"BH q={pooled_price['binomial_q_value_bh_fdr']:.4g})입니다.",
            "- 사전 기준에 따른 가격 방향 proxy 판정: " + ("통과" if direction_proxy else "통과하지 못함"),
            "",
            "| 범위 | 전체 이벤트 | zero 비중 | directional n | 일치율 | BH q | proxy 판정 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in price_summary.itertuples(index=False):
        decision = "통과" if row.supports_price_direction_proxy else "통과 못함"
        q_value = f"{row.binomial_q_value_bh_fdr:.4g}" if pd.notna(row.binomial_q_value_bh_fdr) else "N/A"
        concordance = f"{row.directional_concordance:.2%}" if pd.notna(row.directional_concordance) else "N/A"
        zero_share_display = f"{row.zero_run_event_share:.2%}" if pd.notna(row.zero_run_event_share) else "N/A"
        lines.append(
            f"| {row.scope} | {row.event_count:,} | {zero_share_display} | "
            f"{row.directional_event_count:,} | {concordance} | {q_value} | {decision} |"
        )

    lines.extend(
        [
            "",
            "## 2. 시장요인 제거 후 30분 반응",
            "",
            f"- 회귀 관측치는 {int(predictive_diagnostics.iloc[0]['observations']):,}건, "
            f"UTC-day cluster는 {int(predictive_diagnostics.iloc[0]['utc_day_clusters']):,}개입니다.",
            "- 결과변수는 각 자산 30분 수익률에서 다른 6개 자산의 같은 시각 평균 수익률을 뺀 값입니다.",
            "",
            "| run side | 이벤트 | 조정계수(bp) | 95% CI(bp) | cluster p | BH q | 5bp+ 판정 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in predictive_coefficients.itertuples(index=False):
        decision = "검토 대상" if row.supports_economic_predictive_association else "근거 부족"
        lines.append(
            f"| {row.run_clustering_side} | {row.event_count:,} | {row.adjusted_coefficient_bps:.3f} | "
            f"[{row.ci_lower_bps:.3f}, {row.ci_upper_bps:.3f}] | {row.cluster_p_value:.4g} | "
            f"{row.cluster_q_value_bh_fdr:.4g} | {decision} |"
        )
    lines.append(
        "- 사전 기준을 모두 충족한 coefficient: "
        + (", ".join(predictive_survivors["run_clustering_side"]) if not predictive_survivors.empty else "없음")
    )
    if all_inside_economic_band:
        lines.append(
            f"- 세 coefficient의 95% 신뢰구간이 모두 사전 경제성 기준 ±{float(config['analysis']['economic_effect_threshold_bps']):.1f}bp 안에 있습니다. "
            "통계적 비유의뿐 아니라 이 모형에서 해당 크기 효과가 없다는 practical-null 근거입니다."
        )

    lines.extend(["", "## 3. Raw aggressor pilot", ""])
    if raw_pooled.empty:
        lines.append("- raw pilot 결과가 없어 aggressor를 평가하지 못했습니다.")
    else:
        lines.append("- 2024년 4월 한 달은 parser와 변수 생성 확인용이며 formal p-value를 계산하지 않았습니다.")
        raw_total = int(raw_pooled["event_count"].sum())
        raw_zero = int(
            raw_pooled.loc[raw_pooled["run_clustering_side"] == "zero", "event_count"].sum()
        )
        lines.append(
            f"- raw event {raw_total:,}건 중 zero-run은 {raw_zero:,}건({raw_zero / raw_total:.2%})이고, "
            f"up/down event는 {raw_total - raw_zero:,}건뿐입니다. 이 pilot으로 방향형 aggressor 결론을 내릴 수 없습니다."
        )
        lines.append("")
        lines.append("| run side | 이벤트 | 평균 imbalance | 중앙값 | buy 비중 | 방향 일치율 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in raw_pooled.itertuples(index=False):
            agreement = f"{row.directional_agreement:.2%}" if pd.notna(row.directional_agreement) else "N/A"
            lines.append(
                f"| {row.run_clustering_side} | {row.event_count:,} | {row.mean_aggressor_imbalance:.4f} | "
                f"{row.median_aggressor_imbalance:.4f} | {row.buy_share:.2%} | {agreement} |"
            )

    lines.extend(
        [
            "",
            "## 종합 판정",
            "",
            "- `run_clustering_side`는 조건부 run z가 가장 낮은 tick 범주이며 가격 방향 이름으로 사용하면 안 됩니다.",
            "- 동시적 가격·aggressor 연관성이 존재하더라도 미래 예측이나 인과적 정보전달을 뜻하지 않습니다.",
            "- 5년 cache의 aggressor는 unavailable이므로 raw pilot을 넘어선 aggressor 결론은 보류합니다.",
            "- 다음 confirmatory 단계는 동일 2년 raw 월별 bucket cache를 완성한 뒤 같은 분석을 변경 없이 재실행하는 것입니다.",
            "",
            "## 그림",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in plot_paths)
    lines.append("")
    return "\n".join(lines)


def _leave_one_out_cross_sectional_mean(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("bucket_start", sort=False)[column]
    total = grouped.transform("sum")
    count = grouped.transform("count")
    own = pd.to_numeric(frame[column], errors="coerce")
    denominator = count - own.notna().astype(int)
    return (total - own.fillna(0.0)).where(denominator > 0) / denominator.where(denominator > 0)


def _cramers_v(chi2: float, count: int, shape: tuple[int, int]) -> float:
    denominator = count * min(shape[0] - 1, shape[1] - 1)
    return float(np.sqrt(chi2 / denominator)) if denominator > 0 and np.isfinite(chi2) else np.nan


def _chi_square_nonzero_margins(table: pd.DataFrame) -> tuple[float, float]:
    trimmed = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if trimmed.shape[0] < 2 or trimmed.shape[1] < 2:
        return np.nan, np.nan
    chi2, p_value, _, _ = chi2_contingency(trimmed)
    return float(chi2), float(p_value)


def _plot_heatmap_table(table: pd.DataFrame, path: str | Path, title: str) -> None:
    _configure_font()
    figure, axis = plt.subplots(figsize=(7.5, 5.3))
    values = table.to_numpy(dtype=float)
    image = axis.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=max(0.5, float(np.nanmax(values))))
    axis.set_xticks(np.arange(len(table.columns)), table.columns)
    axis.set_yticks(np.arange(len(table.index)), table.index)
    axis.set_xlabel(table.columns.name or "category")
    axis.set_ylabel("run clustering side")
    axis.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = "N/A" if not np.isfinite(value) else f"{value:.1%}"
            axis.text(column, row, label, ha="center", va="center", color="black")
    figure.colorbar(image, ax=axis, label="run side 내부 비중")
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _configure_font() -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized.dropna().unique()).difference({"true", "false", "1", "0"}))
    if unknown:
        raise ValueError(f"Cannot coerce boolean values: {unknown[:5]}")
    return normalized.isin({"true", "1"})
