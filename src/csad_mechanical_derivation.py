from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.special import gamma


MECHANICAL_MODELS = (
    "standard_csad",
    "no_intercept_csad",
    "intercept_restored",
    "scsad",
)


def half_normal_moment(order: int, scale: float) -> float:
    """Return E[|Z|**order] for Z ~ N(0, scale**2)."""
    if order < 0:
        raise ValueError("order must be non-negative")
    if scale <= 0:
        raise ValueError("scale must be positive")
    return float(
        scale**order
        * 2.0 ** (order / 2.0)
        * gamma((order + 1.0) / 2.0)
        / np.sqrt(np.pi)
    )


def gaussian_market_scale(asset_scale: float, assets: int) -> float:
    _validate_gaussian_inputs(asset_scale, assets)
    return float(asset_scale / np.sqrt(assets))


def gaussian_expected_csad(asset_scale: float, assets: int) -> float:
    _validate_gaussian_inputs(asset_scale, assets)
    residual_scale = asset_scale * np.sqrt(1.0 - 1.0 / assets)
    return float(residual_scale * np.sqrt(2.0 / np.pi))


def gaussian_closed_form_coefficients(
    asset_scale: float,
    assets: int,
) -> dict[str, float]:
    """Population coefficients under the iid Gaussian equal-weight null."""
    market_scale = gaussian_market_scale(asset_scale, assets)
    expected_csad = gaussian_expected_csad(asset_scale, assets)
    u = np.sqrt(2.0 / np.pi)
    denominator = 3.0 - 8.0 / np.pi
    return {
        "market_scale": market_scale,
        "expected_csad": expected_csad,
        "standard_target": 0.0,
        "no_intercept_signed": 0.0,
        "no_intercept_abs": float(expected_csad / market_scale * u / denominator),
        "no_intercept_target": float(
            expected_csad
            / market_scale**2
            * (1.0 - 4.0 / np.pi)
            / denominator
        ),
        "restored_intercept": expected_csad,
        "restored_signed": 0.0,
        "restored_abs": 0.0,
        "restored_target": 0.0,
        "scsad_intercept": 0.0,
        "scsad_linear": float(1.5 * expected_csad * u / market_scale),
        "scsad_quadratic": 0.0,
        "scsad_target": float(-expected_csad * u / (6.0 * market_scale**3)),
    }


def gaussian_moment_projection_coefficients(
    asset_scale: float,
    assets: int,
) -> dict[str, float]:
    """Solve the two population normal equations without simplifying moments."""
    market_scale = gaussian_market_scale(asset_scale, assets)
    expected_csad = gaussian_expected_csad(asset_scale, assets)
    moments = {order: half_normal_moment(order, market_scale) for order in range(1, 7)}

    no_intercept_design = np.array(
        [[moments[2], moments[3]], [moments[3], moments[4]]], dtype=float
    )
    no_intercept_rhs = expected_csad * np.array([moments[1], moments[2]])
    no_intercept_abs, no_intercept_target = np.linalg.solve(
        no_intercept_design, no_intercept_rhs
    )

    scsad_design = np.array(
        [[moments[2], moments[4]], [moments[4], moments[6]]], dtype=float
    )
    scsad_rhs = expected_csad * np.array([moments[1], moments[3]])
    scsad_linear, scsad_target = np.linalg.solve(scsad_design, scsad_rhs)
    return {
        "no_intercept_abs": float(no_intercept_abs),
        "no_intercept_target": float(no_intercept_target),
        "scsad_linear": float(scsad_linear),
        "scsad_target": float(scsad_target),
    }


def build_gaussian_theory_table(
    asset_counts: Iterable[int],
    asset_scale: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    term_map = {
        "standard_csad": ("market_return_sq", "standard_target"),
        "no_intercept_csad": ("market_return_sq", "no_intercept_target"),
        "intercept_restored": ("market_return_sq", "restored_target"),
        "scsad": ("market_return_cu", "scsad_target"),
    }
    for assets in asset_counts:
        values = gaussian_closed_form_coefficients(asset_scale, int(assets))
        for model, (target_term, key) in term_map.items():
            rows.append(
                {
                    "assets": int(assets),
                    "asset_scale": float(asset_scale),
                    "market_scale": values["market_scale"],
                    "expected_csad": values["expected_csad"],
                    "model": model,
                    "target_term": target_term,
                    "theoretical_target_coefficient": values[key],
                    "theoretical_sign": _coefficient_sign(values[key]),
                }
            )
    return pd.DataFrame(rows)


def verify_gaussian_equations(
    asset_counts: Iterable[int],
    asset_scale: float,
    tolerance: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for assets in asset_counts:
        closed = gaussian_closed_form_coefficients(asset_scale, int(assets))
        solved = gaussian_moment_projection_coefficients(asset_scale, int(assets))
        comparisons = {
            "no_intercept_abs": (closed["no_intercept_abs"], solved["no_intercept_abs"]),
            "no_intercept_target": (
                closed["no_intercept_target"],
                solved["no_intercept_target"],
            ),
            "scsad_linear": (closed["scsad_linear"], solved["scsad_linear"]),
            "scsad_target": (closed["scsad_target"], solved["scsad_target"]),
        }
        maximum_difference = max(abs(left - right) for left, right in comparisons.values())
        rows.append(
            {
                "assets": int(assets),
                "asset_scale": float(asset_scale),
                "market_scale": closed["market_scale"],
                "expected_csad": closed["expected_csad"],
                "no_intercept_closed_target": closed["no_intercept_target"],
                "no_intercept_moment_target": solved["no_intercept_target"],
                "scsad_closed_target": closed["scsad_target"],
                "scsad_moment_target": solved["scsad_target"],
                "maximum_absolute_difference": maximum_difference,
                "no_intercept_target_negative": closed["no_intercept_target"] < 0.0,
                "scsad_target_negative": closed["scsad_target"] < 0.0,
                "equation_gate_pass": bool(
                    maximum_difference <= tolerance
                    and closed["no_intercept_target"] < 0.0
                    and closed["scsad_target"] < 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def build_theory_identities() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "identity": "gaussian_mean_residual_independence",
                "assumptions": "iid Gaussian assets; equal weights; N>=3",
                "result": "M is independent of the residual vector and CSAD",
                "implication": "E[CSAD|M] is a positive constant",
            },
            {
                "identity": "no_intercept_projection",
                "assumptions": "positive constant projected on M, |M|, M^2 without intercept",
                "result": "delta2*=(c/s^2)(1-4/pi)/(3-8/pi)<0",
                "implication": "negative quadratic curvature is mechanical",
            },
            {
                "identity": "scsad_projection",
                "assumptions": "sign(M)*constant projected on 1, M, M^2, M^3",
                "result": "gamma3*=-c*sqrt(2/pi)/(6*s^3)<0",
                "implication": "negative cubic curvature approximates a sign step",
            },
            {
                "identity": "intercept_restoration",
                "assumptions": "E[CSAD|M] is constant and the design includes an intercept",
                "result": "standard and restored nonlinear pseudo-true targets equal zero",
                "implication": "the intercept absorbs the positive CSAD level",
            },
        ]
    )


def _validate_gaussian_inputs(asset_scale: float, assets: int) -> None:
    if asset_scale <= 0:
        raise ValueError("asset_scale must be positive")
    if assets < 3:
        raise ValueError("assets must be at least 3")


def _coefficient_sign(value: float, tolerance: float = 1e-14) -> str:
    if value < -tolerance:
        return "negative"
    if value > tolerance:
        return "positive"
    return "zero"
