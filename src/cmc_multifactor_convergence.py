from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from cmc_factor_convergence import expected_absolute_normal


def load_multifactor_history(
    member_rows: pd.DataFrame,
    snapshot_dir: str | Path,
    start: str,
    end: str,
    momentum_calendar_days: int,
) -> pd.DataFrame:
    relevant_ids = set(member_rows["cmc_id"].astype(int).unique())
    paths = sorted(Path(snapshot_dir).glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No CMC snapshot parquet files under {snapshot_dir}")
    pieces = []
    for path in paths:
        snapshot = pd.read_parquet(
            path,
            columns=[
                "snapshot_date",
                "cmc_id",
                "name",
                "symbol",
                "price_usd",
                "market_cap_usd",
                "volume_24h_usd",
            ],
        )
        snapshot = snapshot.loc[snapshot["cmc_id"].isin(relevant_ids)]
        if not snapshot.empty:
            pieces.append(snapshot)
    history = pd.concat(pieces, ignore_index=True)
    history["snapshot_date"] = pd.to_datetime(history["snapshot_date"], utc=True)
    history["cmc_id"] = history["cmc_id"].astype(int)
    history = history.sort_values(["cmc_id", "snapshot_date"]).reset_index(drop=True)
    if history.duplicated(["snapshot_date", "cmc_id"]).any():
        raise ValueError("CMC snapshot history contains duplicate date/CMC-ID rows")

    grouped = history.groupby("cmc_id", sort=False)
    prior_date = grouped["snapshot_date"].shift(1)
    prior_price = grouped["price_usd"].shift(1)
    prior_cap = grouped["market_cap_usd"].shift(1)
    prior_volume = grouped["volume_24h_usd"].shift(1)
    exact_prior = history["snapshot_date"].sub(prior_date).eq(pd.Timedelta(days=1))
    history["asset_return"] = np.log(history["price_usd"] / prior_price).where(
        exact_prior & history["price_usd"].gt(0.0) & prior_price.gt(0.0)
    )
    history["lagged_size"] = prior_cap.where(exact_prior & prior_cap.gt(0.0))
    history["lagged_turnover"] = (prior_volume / prior_cap).where(
        exact_prior & prior_volume.ge(0.0) & prior_cap.gt(0.0)
    )

    momentum_days = int(momentum_calendar_days)
    old_price = grouped["price_usd"].shift(momentum_days + 1)
    old_date = grouped["snapshot_date"].shift(momentum_days + 1)
    exact_momentum = history["snapshot_date"].sub(old_date).eq(
        pd.Timedelta(days=momentum_days + 1)
    )
    history["lagged_momentum"] = np.log(prior_price / old_price).where(
        exact_prior & exact_momentum & prior_price.gt(0.0) & old_price.gt(0.0)
    )

    start_ts = _as_utc(start)
    end_ts = _as_utc(end)
    history = history.loc[
        history["snapshot_date"].between(start_ts, end_ts, inclusive="both")
    ].copy()
    daily_market = (
        member_rows[["snapshot_date", "market_return"]]
        .drop_duplicates("snapshot_date")
        .set_index("snapshot_date")["market_return"]
    )
    history["market_return"] = history["snapshot_date"].map(daily_market)
    target = member_rows[
        [
            "snapshot_date",
            "cmc_id",
            "loo_market_return",
            "active_assets",
            "lagged_market_cap_usd",
        ]
    ].rename(columns={"loo_market_return": "target_loo_market_return"})
    target["is_target_member"] = True
    history = history.merge(target, on=["snapshot_date", "cmc_id"], how="left")
    history["is_target_member"] = history["is_target_member"].fillna(False).astype(bool)
    return history.sort_values(["cmc_id", "snapshot_date"]).reset_index(drop=True)


def build_point_in_time_factors(
    history: pd.DataFrame,
    tail_fraction: float,
    minimum_leg_assets: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tail = float(tail_fraction)
    if not 0.0 < tail < 0.5:
        raise ValueError("tail_fraction must lie between zero and 0.5")
    minimum_leg = int(minimum_leg_assets)
    target = history.loc[history["is_target_member"]].copy()
    target["size_rank"] = target.groupby("snapshot_date")["lagged_size"].rank(
        method="average", pct=True, ascending=True
    )
    target["liquidity_rank"] = target.groupby("snapshot_date")["lagged_turnover"].rank(
        method="average", pct=True, ascending=True
    )
    target["momentum_rank"] = target.groupby("snapshot_date")["lagged_momentum"].rank(
        method="average", pct=True, ascending=True
    )
    specifications = {
        "factor_size": (target["size_rank"].le(tail), target["size_rank"].gt(1.0 - tail)),
        "factor_liquidity": (
            target["liquidity_rank"].gt(1.0 - tail),
            target["liquidity_rank"].le(tail),
        ),
        "factor_momentum": (
            target["momentum_rank"].gt(1.0 - tail),
            target["momentum_rank"].le(tail),
        ),
    }
    factor_series = {
        "factor_market": target.groupby("snapshot_date")["market_return"].first()
    }
    diagnostic_rows = []
    loo_columns = ["snapshot_date", "cmc_id"]
    for factor_name, (long_mask, short_mask) in specifications.items():
        long_column = f"{factor_name}_long_leg"
        short_column = f"{factor_name}_short_leg"
        target[long_column] = long_mask.fillna(False)
        target[short_column] = short_mask.fillna(False)
        grouped = target.groupby("snapshot_date", sort=False)
        long_count = target[long_column].astype(int).groupby(target["snapshot_date"]).transform("sum")
        short_count = target[short_column].astype(int).groupby(target["snapshot_date"]).transform("sum")
        long_sum = target["asset_return"].where(target[long_column], 0.0).groupby(target["snapshot_date"]).transform("sum")
        short_sum = target["asset_return"].where(target[short_column], 0.0).groupby(target["snapshot_date"]).transform("sum")
        global_factor = (long_sum / long_count) - (short_sum / short_count)
        global_valid = long_count.ge(minimum_leg) & short_count.ge(minimum_leg)
        global_factor = global_factor.where(global_valid)
        factor_series[factor_name] = global_factor.groupby(target["snapshot_date"]).first()

        adjusted_long_count = long_count - target[long_column].astype(int)
        adjusted_short_count = short_count - target[short_column].astype(int)
        adjusted_long_sum = long_sum - target["asset_return"].where(target[long_column], 0.0)
        adjusted_short_sum = short_sum - target["asset_return"].where(target[short_column], 0.0)
        loo_factor = adjusted_long_sum / adjusted_long_count - adjusted_short_sum / adjusted_short_count
        loo_valid = adjusted_long_count.ge(minimum_leg) & adjusted_short_count.ge(minimum_leg)
        target[f"{factor_name}_loo"] = loo_factor.where(loo_valid)
        loo_columns.append(f"{factor_name}_loo")

        counts = pd.DataFrame(
            {
                "snapshot_date": target["snapshot_date"],
                "long_count": long_count,
                "short_count": short_count,
                "valid_factor": global_valid,
            }
        ).drop_duplicates("snapshot_date")
        diagnostic_rows.append(
            {
                "factor": factor_name,
                "calendar_days": len(counts),
                "valid_days": int(counts["valid_factor"].sum()),
                "minimum_long_assets": int(counts.loc[counts["valid_factor"], "long_count"].min()),
                "minimum_short_assets": int(counts.loc[counts["valid_factor"], "short_count"].min()),
                "mean_long_assets": float(counts.loc[counts["valid_factor"], "long_count"].mean()),
                "mean_short_assets": float(counts.loc[counts["valid_factor"], "short_count"].mean()),
            }
        )

    factors = pd.concat(factor_series, axis=1).sort_index()
    factors.index.name = "timestamp"
    enriched = history.merge(
        factors.reset_index().rename(columns={"timestamp": "snapshot_date"}),
        on="snapshot_date",
        how="left",
    )
    enriched = enriched.merge(target[loo_columns], on=["snapshot_date", "cmc_id"], how="left")
    enriched["factor_market"] = enriched["target_loo_market_return"].where(
        enriched["is_target_member"], enriched["factor_market"]
    )
    for factor_name in specifications:
        enriched[factor_name] = enriched[f"{factor_name}_loo"].where(
            enriched["is_target_member"], enriched[factor_name]
        )
        enriched = enriched.drop(columns=[f"{factor_name}_loo"])
    enriched = enriched.drop(columns=["target_loo_market_return"])
    return (
        enriched.sort_values(["cmc_id", "snapshot_date"]).reset_index(drop=True),
        factors,
        pd.DataFrame(diagnostic_rows),
    )


def estimate_multifactor_models(
    history: pd.DataFrame,
    factor_columns: Sequence[str],
    window_observations: int,
    minimum_observations: int,
    maximum_condition_number: float,
    minimum_residual_sigma: float,
    empirical_minimum_residuals: int | None = None,
) -> pd.DataFrame:
    factors = list(factor_columns)
    window = int(window_observations)
    minimum = int(minimum_observations)
    p = len(factors) + 1
    if minimum <= p or window < minimum:
        raise ValueError("Rolling window is too short for the multifactor design")
    pieces = []
    for _, asset in history.groupby("cmc_id", sort=False):
        asset = asset.sort_values("snapshot_date").copy()
        x = asset[factors].to_numpy(dtype=float)
        y = asset["asset_return"].to_numpy(dtype=float)
        complete = np.isfinite(y) & np.isfinite(x).all(axis=1)
        coefficient, sigma, count, condition = _rolling_multivariate_ols(
            x,
            y,
            complete,
            window,
            minimum,
            float(maximum_condition_number),
            float(minimum_residual_sigma),
        )
        prediction = coefficient[:, 0] + np.einsum(
            "nk,nk->n", coefficient[:, 1:], x
        )
        prediction[~np.isfinite(coefficient).all(axis=1) | ~np.isfinite(x).all(axis=1)] = np.nan
        error = y - prediction
        asset["factor_window_observations"] = count
        asset["factor_condition_number"] = condition
        asset["factor_alpha"] = coefficient[:, 0]
        for position, name in enumerate(factors, start=1):
            asset[f"beta_{name}"] = coefficient[:, position]
        asset["factor_beta"] = asset["beta_factor_market"]
        asset["factor_residual_sigma"] = sigma
        asset["factor_predicted_return"] = prediction
        asset["factor_prediction_error"] = error
        mu = prediction - asset["market_return"].to_numpy(dtype=float)
        asset["observed_abs_deviation"] = np.abs(
            y - asset["market_return"].to_numpy(dtype=float)
        )
        asset["factor_point_deviation"] = np.abs(mu)
        asset["expected_abs_deviation"] = expected_absolute_normal(mu, sigma)
        if empirical_minimum_residuals is not None:
            empirical, empirical_count = _empirical_expected_abs_deviation(
                x,
                y,
                coefficient,
                mu,
                window,
                int(empirical_minimum_residuals),
            )
            asset["expected_abs_deviation_empirical"] = empirical
            asset["empirical_residual_count"] = empirical_count
        pieces.append(asset)
    result = pd.concat(pieces, ignore_index=True)
    result = result.loc[result["is_target_member"]].copy()
    return result.sort_values(["snapshot_date", "cmc_id"]).reset_index(drop=True)


def _rolling_multivariate_ols(
    x: np.ndarray,
    y: np.ndarray,
    complete: np.ndarray,
    window: int,
    minimum: int,
    maximum_condition_number: float,
    minimum_residual_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nobs, factor_count = x.shape
    p = factor_count + 1
    masked_x = np.where(complete[:, None], x, np.nan)
    masked_y = np.where(complete, y, np.nan)
    count = _prior_rolling_sum(complete.astype(float), window, minimum)
    sum_y = _prior_rolling_sum(masked_y, window, minimum)
    sum_yy = _prior_rolling_sum(masked_y**2, window, minimum)
    sum_x = np.column_stack(
        [_prior_rolling_sum(masked_x[:, j], window, minimum) for j in range(factor_count)]
    )
    sum_xy = np.column_stack(
        [
            _prior_rolling_sum(masked_x[:, j] * masked_y, window, minimum)
            for j in range(factor_count)
        ]
    )
    sum_xx = np.empty((nobs, factor_count, factor_count), dtype=float)
    for left in range(factor_count):
        for right in range(left, factor_count):
            values = _prior_rolling_sum(
                masked_x[:, left] * masked_x[:, right], window, minimum
            )
            sum_xx[:, left, right] = values
            sum_xx[:, right, left] = values

    normal = np.full((nobs, p, p), np.nan, dtype=float)
    rhs = np.full((nobs, p), np.nan, dtype=float)
    normal[:, 0, 0] = count
    normal[:, 0, 1:] = sum_x
    normal[:, 1:, 0] = sum_x
    normal[:, 1:, 1:] = sum_xx
    rhs[:, 0] = sum_y
    rhs[:, 1:] = sum_xy
    candidates = (
        count >= minimum
    ) & np.isfinite(normal).all(axis=(1, 2)) & np.isfinite(rhs).all(axis=1)
    condition = np.full(nobs, np.nan)
    coefficient = np.full((nobs, p), np.nan)
    if candidates.any():
        positions = np.flatnonzero(candidates)
        candidate_normal = normal[positions]
        candidate_condition = np.linalg.cond(candidate_normal)
        condition[positions] = candidate_condition
        stable_positions = positions[
            np.isfinite(candidate_condition)
            & (candidate_condition <= maximum_condition_number)
        ]
        for position in stable_positions:
            try:
                coefficient[position] = np.linalg.solve(normal[position], rhs[position])
            except np.linalg.LinAlgError:
                continue
    sse = sum_yy - np.einsum("np,np->n", coefficient, rhs)
    degrees = count - p
    sigma = np.sqrt(np.maximum(sse, 0.0) / degrees)
    sigma = np.where(
        np.isfinite(coefficient).all(axis=1) & (degrees > 0),
        np.maximum(sigma, minimum_residual_sigma),
        np.nan,
    )
    return coefficient, sigma, count, condition


def _prior_rolling_sum(values: np.ndarray, window: int, minimum: int) -> np.ndarray:
    return (
        pd.Series(values)
        .shift(1)
        .rolling(int(window), min_periods=int(minimum))
        .sum()
        .to_numpy(dtype=float)
    )


def _empirical_expected_abs_deviation(
    x: np.ndarray,
    y: np.ndarray,
    coefficient: np.ndarray,
    current_mu: np.ndarray,
    window: int,
    minimum_residuals: int,
) -> tuple[np.ndarray, np.ndarray]:
    nobs, factor_count = x.shape
    y_windows = _prior_windows(y, window)
    x_windows = [_prior_windows(x[:, j], window) for j in range(factor_count)]
    predicted_history = coefficient[:, [0]]
    for position, values in enumerate(x_windows, start=1):
        predicted_history = predicted_history + coefficient[:, [position]] * values
    residuals = y_windows - predicted_history
    finite = np.isfinite(residuals)
    count = finite.sum(axis=1)
    absolute = np.abs(current_mu[:, None] + residuals)
    total = np.nansum(absolute, axis=1)
    expected = np.divide(
        total,
        count,
        out=np.full(nobs, np.nan),
        where=count >= int(minimum_residuals),
    )
    expected[~np.isfinite(coefficient).all(axis=1) | ~np.isfinite(current_mu)] = np.nan
    return expected, count


def _prior_windows(values: np.ndarray, window: int) -> np.ndarray:
    padded = np.concatenate([np.full(int(window), np.nan), np.asarray(values, dtype=float)])
    return sliding_window_view(padded, int(window))[: len(values)]


def build_factor_correlation(factors: pd.DataFrame) -> pd.DataFrame:
    return factors.corr().rename_axis("factor").reset_index()


def build_single_multi_comparison(
    single_daily: pd.DataFrame,
    multi_normal_daily: pd.DataFrame,
    multi_empirical_daily: pd.DataFrame,
    single_targets: pd.DataFrame,
    multi_normal_targets: pd.DataFrame,
    multi_empirical_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    series = {
        "single_factor_normal": single_daily.loc[
            single_daily["eligible_factor_day"], "convergence_ratio"
        ],
        "multi_factor_normal": multi_normal_daily.loc[
            multi_normal_daily["eligible_factor_day"], "convergence_ratio"
        ],
        "multi_factor_empirical": multi_empirical_daily.loc[
            multi_empirical_daily["eligible_factor_day"], "convergence_ratio"
        ],
    }
    common = pd.concat(series, axis=1, join="inner").dropna()
    summary_rows = []
    target_sources = {
        "single_factor_normal": single_targets,
        "multi_factor_normal": multi_normal_targets,
        "multi_factor_empirical": multi_empirical_targets,
    }
    for name in series:
        target = target_sources[name]
        full = target.loc[target["period"].eq("full_sample")].iloc[0]
        summary_rows.append(
            {
                "model": name,
                "common_days": len(common),
                "mean_convergence_ratio_common_days": float(common[name].mean()),
                "positive_convergence_share_common_days": float(common[name].gt(0.0).mean()),
                "full_sample_delta2": float(full["delta2"]),
                "full_sample_delta2_t_stat": float(full["delta2_t_stat"]),
                "full_sample_delta2_q_value": float(full["delta2_q_value_bh_fdr"]),
                "full_sample_pass": bool(full["supports_extreme_excess_convergence"]),
            }
        )
    correlation = common.corr().rename_axis("model").reset_index()
    return pd.DataFrame(summary_rows), correlation


def build_multifactor_report(
    quality: pd.DataFrame,
    normal_targets: pd.DataFrame,
    empirical_targets: pd.DataFrame,
    comparison: pd.DataFrame,
    factor_diagnostics: pd.DataFrame,
    plot_paths: Sequence[str],
) -> str:
    primary = normal_targets.loc[normal_targets["window"].eq("window_365")]
    normal_full = primary.loc[primary["period"].eq("full_sample")].iloc[0]
    empirical_full = empirical_targets.loc[
        empirical_targets["period"].eq("full_sample")
    ].iloc[0]
    primary_pass = bool(
        normal_full["supports_extreme_excess_convergence"]
        and empirical_full["supports_extreme_excess_convergence"]
    )
    quality_row = quality.loc[quality["window"].eq("window_365")].iloc[0]
    lines = [
        "# CMC 다요인 초과 수렴 강건성 연구",
        "",
        "## 한눈에 보는 결론",
        "",
        f"- 365관측 다요인 모형의 적격 날짜는 {int(quality_row['eligible_factor_days']):,}일, 평균 coverage는 {quality_row['mean_model_coverage']:.2%}입니다.",
        f"- 정규잔차 반사실적 전체 delta2는 {normal_full['delta2']:.3f} (t={normal_full['delta2_t_stat']:.2f}, q={normal_full['delta2_q_value_bh_fdr']:.4g})입니다.",
        f"- 과거잔차 반사실적 전체 delta2는 {empirical_full['delta2']:.3f} (t={empirical_full['delta2_t_stat']:.2f}, q={empirical_full['delta2_q_value_bh_fdr']:.4g})입니다.",
        f"- 사전 이중 기준 판정은 {'통과' if primary_pass else '미통과'}입니다.",
        "",
        "다요인 통제와 두 잔차분포에서 모두 통과해야 단일 시장요인 결과의 강건성 근거로 봅니다. 통과하더라도 투자자의 의도적 모방이나 미래수익률 alpha를 직접 식별하지는 않습니다.",
        "",
        "## Factor 구성",
        "",
        "당일 동적 universe를 전일 시가총액, 전일 거래대금/시가총액, 전일까지 30일 momentum으로 정렬했습니다. 각 factor는 상·하위 30% equal-weight spread이며 자산별 회귀에서는 자기 자신을 factor leg에서 제외했습니다.",
        "",
        "| factor | 유효 일수 | 최소 long | 최소 short | 평균 long | 평균 short |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in factor_diagnostics.itertuples(index=False):
        lines.append(
            f"| {row.factor} | {row.valid_days:,} | {row.minimum_long_assets} | "
            f"{row.minimum_short_assets} | {row.mean_long_assets:.1f} | {row.mean_short_assets:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 365관측 주 결과",
            "",
            "| 구간 | 정규 delta2 | 정규 q | 정규 판정 | 과거잔차 delta2 | 과거잔차 q | 과거잔차 판정 |",
            "|---|---:|---:|---|---:|---:|---|",
        ]
    )
    empirical_index = empirical_targets.set_index("period")
    for row in primary.itertuples(index=False):
        empirical = empirical_index.loc[row.period]
        lines.append(
            f"| {row.period} | {row.delta2:.3f} | {row.delta2_q_value_bh_fdr:.4g} | "
            f"{'통과' if row.supports_extreme_excess_convergence else '미통과'} | "
            f"{empirical['delta2']:.3f} | {empirical['delta2_q_value_bh_fdr']:.4g} | "
            f"{'통과' if empirical['supports_extreme_excess_convergence'] else '미통과'} |"
        )
    lines.extend(
        [
            "",
            "## 단일요인과 비교",
            "",
            "| 모형 | 공통 날짜 평균 수렴 | 양의 수렴 비중 | 전체 delta2 | 전체 q | 판정 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.mean_convergence_ratio_common_days:.2%} | "
            f"{row.positive_convergence_share_common_days:.2%} | {row.full_sample_delta2:.3f} | "
            f"{row.full_sample_delta2_q_value:.4g} | {'통과' if row.full_sample_pass else '미통과'} |"
        )
    lines.extend(
        [
            "",
            "## 한계",
            "",
            "- 규모·유동성·모멘텀 spread는 사전 고정한 단순 factor이며 완전한 암호화폐 가격모형이 아닙니다.",
            "- CMC 거래대금은 거래소별 실제 체결 유동성과 다릅니다.",
            "- 과거잔차 반사실적도 과거 분포가 현재 조건부 분포를 대표한다는 가정이 있습니다.",
            "- 구조 구간은 같은 원표본에서 선택됐으므로 구간별 결과는 설명적입니다.",
            "- 미래수익률을 사용하지 않았으므로 alpha·백테스트·자동매매 근거가 아닙니다.",
            "",
            "## 그림",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in plot_paths)
    return "\n".join(lines) + "\n"


def plot_multifactor_timeseries(
    normal_daily: pd.DataFrame,
    empirical_daily: pd.DataFrame,
    path: str | Path,
) -> None:
    normal = normal_daily.loc[normal_daily["eligible_factor_day"], "convergence_ratio"]
    empirical = empirical_daily.loc[
        empirical_daily["eligible_factor_day"], "convergence_ratio"
    ]
    frame = pd.concat(
        {
            "normal residual": normal,
            "empirical residual": empirical,
        },
        axis=1,
    ).rolling(30, min_periods=15).mean()
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(frame.index, frame["normal residual"], color="#0d5c63", linewidth=1.7, label="Normal residual")
    axis.plot(frame.index, frame["empirical residual"], color="#c26d38", linewidth=1.5, label="Empirical residual")
    axis.axhline(0.0, color="#222222", linewidth=0.9)
    axis.set_title("Multi-factor excess convergence (30-day mean)")
    axis.set_ylabel("Convergence ratio")
    axis.legend(frameon=False)
    axis.grid(alpha=0.18)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_primary_model_comparison(
    single_targets: pd.DataFrame,
    normal_targets: pd.DataFrame,
    empirical_targets: pd.DataFrame,
    path: str | Path,
) -> None:
    order = ["full_sample", "regime_1", "regime_2", "regime_3", "regime_4", "regime_5"]
    sources = [
        ("Single-factor normal", single_targets, "#7b8d8e"),
        ("Multi-factor normal", normal_targets, "#0d5c63"),
        ("Multi-factor empirical", empirical_targets, "#c26d38"),
    ]
    positions = np.arange(len(order), dtype=float)
    figure, axis = plt.subplots(figsize=(12, 6))
    for offset, (label, source, color) in enumerate(sources):
        subset = source.set_index("period").reindex(order)
        x = positions + (offset - 1) * 0.22
        lower = subset["delta2"] - subset["delta2_ci_lower"]
        upper = subset["delta2_ci_upper"] - subset["delta2"]
        axis.errorbar(
            x,
            subset["delta2"],
            yerr=np.vstack([lower, upper]),
            fmt="o",
            capsize=3,
            color=color,
            label=label,
        )
    axis.axhline(0.0, color="#222222", linewidth=0.9)
    axis.set_xticks(positions, order)
    axis.set_ylabel("Quadratic coefficient delta2 (95% HAC CI)")
    axis.set_title("Single- vs multi-factor extreme convergence")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.18)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _as_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
