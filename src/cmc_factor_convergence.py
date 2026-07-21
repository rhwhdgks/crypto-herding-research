from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import ndtr

from frequency_sensitivity import benjamini_hochberg


REQUIRED_MEMBER_COLUMNS = {
    "snapshot_date",
    "cmc_id",
    "name",
    "symbol",
    "asset_return",
    "lagged_market_cap_usd",
    "valid_return_weight",
    "eligible_day",
}


def load_factor_member_rows(
    path: str | Path,
    start: str,
    end: str,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted(REQUIRED_MEMBER_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Member panel is missing required columns: {missing}")
    frame = frame.copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], utc=True)
    start_ts = _as_utc(start)
    end_ts = _as_utc(end)
    frame = frame.loc[
        frame["snapshot_date"].between(start_ts, end_ts, inclusive="both")
        & frame["eligible_day"].astype(bool)
        & frame["valid_return_weight"].astype(bool)
    ].copy()
    frame = frame.dropna(
        subset=["asset_return", "lagged_market_cap_usd", "cmc_id"]
    )
    frame = frame.loc[frame["lagged_market_cap_usd"].gt(0.0)]
    frame["cmc_id"] = frame["cmc_id"].astype(int)
    frame = frame.sort_values(["cmc_id", "snapshot_date"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No eligible member rows remain for factor decomposition")
    if frame.duplicated(["snapshot_date", "cmc_id"]).any():
        raise ValueError("Member panel contains duplicate date/CMC-ID rows")
    return add_leave_one_out_market_return(frame)


def load_point_in_time_estimation_history(
    member_rows: pd.DataFrame,
    snapshot_dir: str | Path,
    start: str,
    end: str,
) -> pd.DataFrame:
    relevant_ids = set(member_rows["cmc_id"].astype(int).unique())
    paths = sorted(Path(snapshot_dir).glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No CMC snapshot parquet files under {snapshot_dir}")
    pieces = []
    for path in paths:
        snapshot = pd.read_parquet(
            path,
            columns=["snapshot_date", "cmc_id", "name", "symbol", "price_usd"],
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
    previous_date = grouped["snapshot_date"].shift(1)
    previous_price = grouped["price_usd"].shift(1)
    exact_previous_day = history["snapshot_date"].sub(previous_date).eq(
        pd.Timedelta(days=1)
    )
    valid_prices = history["price_usd"].gt(0.0) & previous_price.gt(0.0)
    history["asset_return"] = np.log(history["price_usd"] / previous_price).where(
        exact_previous_day & valid_prices
    )

    daily_market = (
        member_rows[["snapshot_date", "market_return"]]
        .drop_duplicates("snapshot_date")
        .set_index("snapshot_date")["market_return"]
    )
    history = history.loc[
        history["snapshot_date"].between(_as_utc(start), _as_utc(end), inclusive="both")
    ].copy()
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
    history["loo_market_return"] = history["target_loo_market_return"].where(
        history["is_target_member"],
        history["market_return"],
    )
    history = history.drop(columns=["target_loo_market_return"])
    return history.sort_values(["cmc_id", "snapshot_date"]).reset_index(drop=True)


def add_leave_one_out_market_return(member_rows: pd.DataFrame) -> pd.DataFrame:
    frame = member_rows.copy()
    weighted_return = frame["asset_return"] * frame["lagged_market_cap_usd"]
    grouped = frame.groupby("snapshot_date", sort=False)
    frame["market_weight_sum"] = grouped["lagged_market_cap_usd"].transform("sum")
    frame["market_weighted_return_sum"] = weighted_return.groupby(
        frame["snapshot_date"], sort=False
    ).transform("sum")
    frame["active_assets"] = grouped["cmc_id"].transform("size").astype(int)
    frame["market_return"] = (
        frame["market_weighted_return_sum"] / frame["market_weight_sum"]
    )
    denominator = frame["market_weight_sum"] - frame["lagged_market_cap_usd"]
    numerator = (
        frame["market_weighted_return_sum"]
        - frame["asset_return"] * frame["lagged_market_cap_usd"]
    )
    frame["loo_market_return"] = numerator / denominator.where(denominator.gt(0.0))
    frame.loc[frame["active_assets"].lt(2), "loo_market_return"] = np.nan
    return frame


def estimate_point_in_time_factor_model(
    member_rows: pd.DataFrame,
    window_observations: int,
    minimum_observations: int,
    minimum_regressor_variance: float,
    minimum_residual_sigma: float,
) -> pd.DataFrame:
    window = int(window_observations)
    minimum = int(minimum_observations)
    if window < minimum or minimum < 3:
        raise ValueError("Factor window must be >= minimum_observations >= 3")

    pieces = []
    for _, asset in member_rows.groupby("cmc_id", sort=False):
        asset = asset.sort_values("snapshot_date").copy()
        x = asset["loo_market_return"].astype(float)
        y = asset["asset_return"].astype(float)
        valid = x.notna() & y.notna()
        xv = x.where(valid)
        yv = y.where(valid)

        # shift(1) is the core no-look-ahead rule: date t never estimates its own model.
        count = valid.astype(float).shift(1).rolling(window, min_periods=minimum).sum()
        sum_x = xv.shift(1).rolling(window, min_periods=minimum).sum()
        sum_y = yv.shift(1).rolling(window, min_periods=minimum).sum()
        sum_xx = xv.pow(2).shift(1).rolling(window, min_periods=minimum).sum()
        sum_xy = (xv * yv).shift(1).rolling(window, min_periods=minimum).sum()
        sum_yy = yv.pow(2).shift(1).rolling(window, min_periods=minimum).sum()

        centered_xx = sum_xx - sum_x.pow(2) / count
        centered_xy = sum_xy - sum_x * sum_y / count
        centered_yy = sum_yy - sum_y.pow(2) / count
        beta = centered_xy / centered_xx
        alpha = sum_y / count - beta * sum_x / count
        residual_sse = centered_yy - beta * centered_xy
        residual_variance = residual_sse.clip(lower=0.0) / (count - 2.0)
        residual_sigma = np.sqrt(residual_variance).clip(
            lower=float(minimum_residual_sigma)
        )

        model_valid = (
            count.ge(minimum)
            & centered_xx.gt(float(minimum_regressor_variance))
            & alpha.notna()
            & beta.notna()
            & residual_sigma.notna()
            & x.notna()
        )
        asset["factor_window_observations"] = count
        asset["factor_alpha"] = alpha.where(model_valid)
        asset["factor_beta"] = beta.where(model_valid)
        asset["factor_residual_sigma"] = residual_sigma.where(model_valid)
        asset["factor_predicted_return"] = (
            alpha + beta * x
        ).where(model_valid)
        asset["factor_prediction_error"] = (
            y - asset["factor_predicted_return"]
        ).where(model_valid)
        pieces.append(asset)

    result = pd.concat(pieces, ignore_index=True)
    if "is_target_member" in result:
        result = result.loc[result["is_target_member"]].copy()
    result = result.sort_values(["snapshot_date", "cmc_id"]).reset_index(drop=True)
    return add_counterfactual_deviations(result)


def add_counterfactual_deviations(member_models: pd.DataFrame) -> pd.DataFrame:
    frame = member_models.copy()
    frame["observed_abs_deviation"] = (
        frame["asset_return"] - frame["market_return"]
    ).abs()
    frame["factor_point_deviation"] = (
        frame["factor_predicted_return"] - frame["market_return"]
    ).abs()
    mu = frame["factor_predicted_return"] - frame["market_return"]
    sigma = frame["factor_residual_sigma"]
    frame["expected_abs_deviation"] = expected_absolute_normal(mu, sigma)
    return frame


def expected_absolute_normal(
    mean: pd.Series | np.ndarray,
    sigma: pd.Series | np.ndarray,
) -> np.ndarray:
    mean_array = np.asarray(mean, dtype=float)
    sigma_array = np.asarray(sigma, dtype=float)
    result = np.full(np.broadcast_shapes(mean_array.shape, sigma_array.shape), np.nan)
    valid = np.isfinite(mean_array) & np.isfinite(sigma_array) & (sigma_array > 0.0)
    z = np.divide(
        mean_array,
        sigma_array,
        out=np.zeros_like(mean_array),
        where=valid,
    )
    result[valid] = (
        sigma_array[valid]
        * math.sqrt(2.0 / math.pi)
        * np.exp(-0.5 * z[valid] ** 2)
        + mean_array[valid] * (2.0 * ndtr(z[valid]) - 1.0)
    )
    return result


def aggregate_daily_convergence(
    member_models: pd.DataFrame,
    minimum_daily_model_coverage: float,
    minimum_cross_section_assets: int,
) -> pd.DataFrame:
    frame = member_models.copy()
    frame["has_factor_model"] = frame["expected_abs_deviation"].notna()
    rows = []
    for date, daily in frame.groupby("snapshot_date", sort=True):
        modelled = daily.loc[daily["has_factor_model"]]
        active = int(len(daily))
        model_count = int(len(modelled))
        coverage = model_count / active if active else 0.0
        eligible = bool(
            model_count >= int(minimum_cross_section_assets)
            and coverage >= float(minimum_daily_model_coverage)
        )
        row = {
            "timestamp": date,
            "market_return": float(daily["market_return"].iloc[0]),
            "active_assets": active,
            "modelled_assets": model_count,
            "model_coverage": coverage,
            "eligible_factor_day": eligible,
        }
        if eligible:
            observed = float(modelled["observed_abs_deviation"].mean())
            factor_point = float(modelled["factor_point_deviation"].mean())
            expected = float(modelled["expected_abs_deviation"].mean())
            residual_center = float(modelled["factor_prediction_error"].mean())
            residual_csad = float(
                (modelled["factor_prediction_error"] - residual_center).abs().mean()
            )
            abnormal = expected - observed
            row.update(
                {
                    "observed_csad_modelled_assets": observed,
                    "factor_point_csad": factor_point,
                    "expected_csad_null": expected,
                    "abnormal_convergence": abnormal,
                    "convergence_ratio": abnormal / expected if expected > 0.0 else np.nan,
                    "residual_csad": residual_csad,
                    "mean_factor_beta": float(modelled["factor_beta"].mean()),
                    "median_factor_beta": float(modelled["factor_beta"].median()),
                    "mean_residual_sigma": float(modelled["factor_residual_sigma"].mean()),
                }
            )
        rows.append(row)
    result = pd.DataFrame(rows).set_index("timestamp").sort_index()
    numeric = [
        "observed_csad_modelled_assets",
        "factor_point_csad",
        "expected_csad_null",
        "abnormal_convergence",
        "convergence_ratio",
        "residual_csad",
        "mean_factor_beta",
        "median_factor_beta",
        "mean_residual_sigma",
    ]
    for column in numeric:
        if column not in result:
            result[column] = np.nan
    return result


def build_fixed_regimes(
    start: str,
    end: str,
    break_dates: pd.DataFrame,
) -> list[dict]:
    if "next_regime_start" not in break_dates:
        raise ValueError("Structural break table lacks next_regime_start")
    starts = sorted(
        pd.to_datetime(break_dates["next_regime_start"], utc=True).dropna().unique()
    )
    sample_start = _as_utc(start)
    sample_end = _as_utc(end)
    boundaries = [sample_start, *starts, sample_end + pd.Timedelta(days=1)]
    regimes = [{"period": "full_sample", "start": sample_start, "end": sample_end}]
    for number, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        regimes.append(
            {
                "period": f"regime_{number}",
                "start": left,
                "end": right - pd.Timedelta(days=1),
            }
        )
    return regimes


def run_convergence_regressions(
    daily_by_window: Mapping[str, pd.DataFrame],
    regimes: Sequence[Mapping],
    regression_cfg: Mapping,
    primary_window_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_rows = []
    coefficient_rows = []
    mean_rows = []
    for window_name, daily in daily_by_window.items():
        window_targets = []
        window_means = []
        for regime in regimes:
            subset = daily.loc[
                daily.index.to_series().between(regime["start"], regime["end"])
                & daily["eligible_factor_day"]
            ].dropna(subset=["convergence_ratio", "market_return"])
            if len(subset) < 30:
                if primary_window_name is None or window_name == primary_window_name:
                    raise ValueError(
                        f"Too few factor-adjusted observations for {window_name}/{regime['period']}"
                    )
                window_targets.append(
                    {
                        "window": window_name,
                        "period": regime["period"],
                        "start": subset.index.min(),
                        "end": subset.index.max(),
                        "observations": len(subset),
                        "delta2": np.nan,
                        "delta2_std_error": np.nan,
                        "delta2_t_stat": np.nan,
                        "delta2_p_value_hac": np.nan,
                        "delta2_ci_lower": np.nan,
                        "delta2_ci_upper": np.nan,
                        "mean_convergence_ratio": float(subset["convergence_ratio"].mean()) if len(subset) else np.nan,
                        "mean_abnormal_convergence": float(subset["abnormal_convergence"].mean()) if len(subset) else np.nan,
                        "positive_convergence_share": float(subset["abnormal_convergence"].gt(0.0).mean()) if len(subset) else np.nan,
                        "rsquared": np.nan,
                        "hac_maxlags": np.nan,
                        "analysis_status": "insufficient_history",
                    }
                )
                window_means.append(
                    {
                        "window": window_name,
                        "period": regime["period"],
                        "observations": len(subset),
                        "mean_convergence_ratio": np.nan,
                        "mean_std_error_hac": np.nan,
                        "mean_t_stat_hac": np.nan,
                        "mean_p_value_hac": np.nan,
                        "mean_ci_lower": np.nan,
                        "mean_ci_upper": np.nan,
                        "hac_maxlags": np.nan,
                        "analysis_status": "insufficient_history",
                    }
                )
                continue
            x = pd.DataFrame(
                {
                    "const": 1.0,
                    "abs_market_return": subset["market_return"].abs(),
                    "market_return_sq": subset["market_return"].pow(2),
                },
                index=subset.index,
            )
            model = _fit_hac(
                subset["convergence_ratio"],
                x,
                regression_cfg.get("hac_maxlags", "auto"),
            )
            confidence = model.conf_int()
            for term in model.params.index:
                coefficient_rows.append(
                    {
                        "window": window_name,
                        "period": regime["period"],
                        "start": subset.index.min(),
                        "end": subset.index.max(),
                        "term": term,
                        "coefficient": float(model.params[term]),
                        "std_error": float(model.bse[term]),
                        "t_stat": float(model.tvalues[term]),
                        "p_value_hac": float(model.pvalues[term]),
                        "ci_lower": float(confidence.loc[term, 0]),
                        "ci_upper": float(confidence.loc[term, 1]),
                        "observations": int(model.nobs),
                        "rsquared": float(model.rsquared),
                        "hac_maxlags": int(model.cov_kwds["maxlags"]),
                    }
                )
            target = "market_return_sq"
            window_targets.append(
                {
                    "window": window_name,
                    "period": regime["period"],
                    "start": subset.index.min(),
                    "end": subset.index.max(),
                    "observations": int(model.nobs),
                    "delta2": float(model.params[target]),
                    "delta2_std_error": float(model.bse[target]),
                    "delta2_t_stat": float(model.tvalues[target]),
                    "delta2_p_value_hac": float(model.pvalues[target]),
                    "delta2_ci_lower": float(confidence.loc[target, 0]),
                    "delta2_ci_upper": float(confidence.loc[target, 1]),
                    "mean_convergence_ratio": float(subset["convergence_ratio"].mean()),
                    "mean_abnormal_convergence": float(subset["abnormal_convergence"].mean()),
                    "positive_convergence_share": float(subset["abnormal_convergence"].gt(0.0).mean()),
                    "rsquared": float(model.rsquared),
                    "hac_maxlags": int(model.cov_kwds["maxlags"]),
                    "analysis_status": "complete",
                }
            )

            mean_model = _fit_hac(
                subset["convergence_ratio"],
                pd.DataFrame({"const": 1.0}, index=subset.index),
                regression_cfg.get("hac_maxlags", "auto"),
            )
            mean_confidence = mean_model.conf_int().loc["const"]
            window_means.append(
                {
                    "window": window_name,
                    "period": regime["period"],
                    "observations": int(mean_model.nobs),
                    "mean_convergence_ratio": float(mean_model.params["const"]),
                    "mean_std_error_hac": float(mean_model.bse["const"]),
                    "mean_t_stat_hac": float(mean_model.tvalues["const"]),
                    "mean_p_value_hac": float(mean_model.pvalues["const"]),
                    "mean_ci_lower": float(mean_confidence[0]),
                    "mean_ci_upper": float(mean_confidence[1]),
                    "hac_maxlags": int(mean_model.cov_kwds["maxlags"]),
                    "analysis_status": "complete",
                }
            )
        target_frame = pd.DataFrame(window_targets)
        target_frame["delta2_q_value_bh_fdr"] = benjamini_hochberg(
            target_frame["delta2_p_value_hac"]
        )
        target_frame["supports_extreme_excess_convergence"] = (
            target_frame["delta2"].gt(0.0)
            & target_frame["delta2_q_value_bh_fdr"].le(
                float(regression_cfg["fdr_alpha"])
            )
        )
        mean_frame = pd.DataFrame(window_means)
        mean_frame["mean_q_value_bh_fdr"] = benjamini_hochberg(
            mean_frame["mean_p_value_hac"]
        )
        mean_frame["mean_above_zero"] = mean_frame["mean_convergence_ratio"].gt(0.0)
        target_rows.append(target_frame)
        mean_rows.append(mean_frame)
    return (
        pd.concat(target_rows, ignore_index=True),
        pd.DataFrame(coefficient_rows),
        pd.concat(mean_rows, ignore_index=True),
    )


def build_quality_summary(
    daily_by_window: Mapping[str, pd.DataFrame],
    primary_window_name: str,
) -> pd.DataFrame:
    rows = []
    for name, daily in daily_by_window.items():
        eligible = daily.loc[daily["eligible_factor_day"]]
        rows.append(
            {
                "window": name,
                "is_primary": name == primary_window_name,
                "calendar_days": len(daily),
                "eligible_factor_days": len(eligible),
                "first_eligible_day": eligible.index.min(),
                "last_eligible_day": eligible.index.max(),
                "mean_model_coverage": float(eligible["model_coverage"].mean()),
                "minimum_model_coverage": float(eligible["model_coverage"].min()),
                "minimum_modelled_assets": int(eligible["modelled_assets"].min()),
                "nonpositive_expected_csad_days": int(
                    eligible["expected_csad_null"].le(0.0).sum()
                ),
                "nonfinite_ratio_days": int(
                    (~np.isfinite(eligible["convergence_ratio"])).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_factor_convergence_report(
    config: Mapping,
    quality: pd.DataFrame,
    targets: pd.DataFrame,
    mean_tests: pd.DataFrame,
    plot_paths: Sequence[str],
) -> str:
    primary_observations = int(config["factor_model"]["primary_window_observations"])
    primary_name = f"window_{primary_observations}"
    primary = targets.loc[targets["window"].eq(primary_name)].copy()
    full = primary.loc[primary["period"].eq("full_sample")].iloc[0]
    full_mean = mean_tests.loc[
        mean_tests["window"].eq(primary_name)
        & mean_tests["period"].eq("full_sample")
    ].iloc[0]
    supported = primary.loc[primary["supports_extreme_excess_convergence"]]
    quality_row = quality.loc[quality["window"].eq(primary_name)].iloc[0]
    lines = [
        "# CMC 시장요인 조정 초과 수렴 연구",
        "",
        "## 한눈에 보는 결론",
        "",
        f"- 주 분석은 매일 직전 {primary_observations}개 관측만으로 각 코인의 시장 beta를 추정했습니다.",
        f"- rolling burn-in 이후 사용 가능한 날짜는 {int(quality_row['eligible_factor_days']):,}일이며 평균 모형 가용률은 {quality_row['mean_model_coverage']:.2%}입니다.",
        f"- 전체 표본의 평균 초과 수렴비율은 {full_mean['mean_convergence_ratio']:.2%} (HAC t={full_mean['mean_t_stat_hac']:.2f}, q={full_mean['mean_q_value_bh_fdr']:.4g})입니다.",
        f"- 극단적 시장 움직임의 비선형 계수 delta2는 {full['delta2']:.3f} (HAC t={full['delta2_t_stat']:.2f}, q={full['delta2_q_value_bh_fdr']:.4g})입니다.",
        f"- 사전 기준을 통과한 전체·구조 구간은 {len(supported)}/6개입니다.",
        "",
        "이 결과는 공통 시장요인과 과거 고유변동성을 감안한 뒤에도 실제 수익률이 더 모였는지를 측정합니다. 그러나 가격자료만으로 투자자가 의도적으로 타인을 모방했다고 식별할 수는 없습니다.",
        "",
        "## 측정 방법",
        "",
        "각 자산의 수익률을 자기 자신을 제외한 시가총액 가중 시장수익률에 회귀했습니다. 현재 날짜는 회귀창에서 제외했으며, 과거 회귀의 alpha·beta·잔차표준편차로 그날의 정상 기대 절대편차를 계산했습니다.",
        "",
        "`초과 수렴비율 = (정상 기대 CSAD - 실제 CSAD) / 정상 기대 CSAD`입니다. 양수면 실제 횡단면 분산이 단일 시장요인 모형보다 낮았다는 뜻이고, 음수면 오히려 더 넓게 흩어졌다는 뜻입니다.",
        "",
        "## 주 분석 결과",
        "",
        "| 구간 | 관측치 | 평균 초과 수렴 | 양의 수렴 비중 | delta2 | HAC t | BH q | 사전 기준 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.period} | {row.observations:,} | {row.mean_convergence_ratio:.2%} | "
            f"{row.positive_convergence_share:.2%} | {row.delta2:.3f} | "
            f"{row.delta2_t_stat:.2f} | {row.delta2_q_value_bh_fdr:.4g} | "
            f"{'통과' if row.supports_extreme_excess_convergence else '미통과'} |"
        )
    lines.extend(
        [
            "",
            "`delta2 > 0`은 시장 움직임이 커질수록 정상 기대 대비 수렴이 비선형적으로 증가하는 방향입니다. 여섯 회귀의 two-sided HAC p-value를 하나의 BH-FDR family로 보정했습니다.",
            "",
            "## 창 길이 민감도",
            "",
            "| 창 | 적격 일수 | 전체 평균 수렴 | 전체 delta2 | 전체 q | 통과 구간 수 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for window, subset in targets.groupby("window", sort=False):
        qrow = quality.loc[quality["window"].eq(window)].iloc[0]
        full_row = subset.loc[subset["period"].eq("full_sample")].iloc[0]
        lines.append(
            f"| {window} | {int(qrow['eligible_factor_days']):,} | "
            f"{full_row['mean_convergence_ratio']:.2%} | {full_row['delta2']:.3f} | "
            f"{full_row['delta2_q_value_bh_fdr']:.4g} | "
            f"{int(subset['supports_extreme_excess_convergence'].sum())}/"
            f"{int(subset['analysis_status'].eq('complete').sum())} |"
        )
    lines.extend(
        [
            "",
            "180·730 관측창은 민감도 분석입니다. 주 결론은 365 관측창으로만 판정합니다.",
            "",
            "## 한계와 올바른 해석",
            "",
            "- 단일 시장요인에 없는 섹터·규모·유동성 요인이 초과 수렴에 포함될 수 있습니다.",
            "- 정상 잔차의 조건부 정규분포는 계산 가능한 반사실적을 위한 단순화입니다.",
            "- 구조 구간은 같은 표본의 선행 분석에서 선택됐으므로 구간별 p/q는 설명적 강건성 결과입니다.",
            "- 이 분석은 미래수익률, 비용, 실행 가능성을 다루지 않으므로 alpha 또는 거래전략 근거가 아닙니다.",
            "- 의도적 허딩을 식별하려면 투자자별 주문흐름·보유자료 같은 직접 행동자료가 추가로 필요합니다.",
            "",
            "## 그림",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in plot_paths)
    return "\n".join(lines) + "\n"


def plot_convergence_timeseries(
    daily: pd.DataFrame,
    break_dates: pd.DataFrame,
    path: str | Path,
) -> None:
    eligible = daily.loc[daily["eligible_factor_day"]].copy()
    smooth = eligible["convergence_ratio"].rolling(30, min_periods=15).mean()
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(eligible.index, eligible["convergence_ratio"], color="#9eb5b2", alpha=0.35, linewidth=0.7, label="Daily")
    axis.plot(smooth.index, smooth, color="#0d5c63", linewidth=1.8, label="30-day mean")
    axis.axhline(0.0, color="#222222", linewidth=0.9)
    for value in pd.to_datetime(break_dates["next_regime_start"], utc=True):
        axis.axvline(value, color="#c94c32", linestyle="--", alpha=0.7, linewidth=1.0)
    axis.set_title("Factor-adjusted excess convergence ratio")
    axis.set_ylabel("(Expected CSAD - observed CSAD) / expected CSAD")
    axis.legend(frameon=False)
    axis.grid(alpha=0.18)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_regime_delta2(targets: pd.DataFrame, path: str | Path) -> None:
    order = ["full_sample", "regime_1", "regime_2", "regime_3", "regime_4", "regime_5"]
    windows = list(targets["window"].drop_duplicates())
    colors = ["#0d5c63", "#d28f2c", "#8d5b4c"]
    figure, axis = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(order), dtype=float)
    width = 0.24
    for offset, (window, color) in enumerate(zip(windows, colors)):
        subset = targets.set_index(["window", "period"]).loc[window].reindex(order)
        x = positions + (offset - (len(windows) - 1) / 2.0) * width
        lower = subset["delta2"] - subset["delta2_ci_lower"]
        upper = subset["delta2_ci_upper"] - subset["delta2"]
        axis.errorbar(
            x,
            subset["delta2"],
            yerr=np.vstack([lower, upper]),
            fmt="o",
            capsize=3,
            color=color,
            label=window,
        )
    axis.axhline(0.0, color="#222222", linewidth=0.9)
    axis.set_xticks(positions, order)
    axis.set_ylabel("Quadratic coefficient delta2 (95% HAC CI)")
    axis.set_title("Extreme-market excess convergence by fixed regime")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.18)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _fit_hac(
    dependent: pd.Series,
    design: pd.DataFrame,
    configured_maxlags: int | str | None,
):
    nobs = len(dependent)
    if configured_maxlags not in {None, "", "auto"}:
        maxlags = max(int(configured_maxlags), 1)
    else:
        maxlags = max(int(np.floor(4.0 * ((nobs / 100.0) ** (2.0 / 9.0)))), 1)
    return sm.OLS(dependent.astype(float), design.astype(float)).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags},
    )


def _as_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
