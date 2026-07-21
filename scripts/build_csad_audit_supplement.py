from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/crypto-herding-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from csad_audit_reporting import (  # noqa: E402
    build_korean_audit_report,
    build_structural_robustness_decisions,
    plot_empirical_null_comparison,
    plot_intercept_mechanics,
    plot_random_effects,
)
from csad_specification_audit import (  # noqa: E402
    _descriptive_univariate_meta_regressions,
)
from run_csad_specification_audit import _sha256, _write_artifact_manifest  # noqa: E402
from utils import load_config  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v2" / "csad_specification_audit_v1"
CONFIG_PATH = PROJECT_ROOT / "configs" / "research" / "csad_specification_audit_v1.yaml"


def main() -> None:
    targets = [
        OUTPUT_DIR / "descriptive_univariate_meta_coefficients.csv",
        OUTPUT_DIR / "descriptive_univariate_meta_diagnostics.csv",
        OUTPUT_DIR / "structural_robustness_decisions_v1_1.csv",
        OUTPUT_DIR / "csad_specification_audit_report_v1_1.md",
        OUTPUT_DIR / "audit_amendment_2026-07-20.md",
        OUTPUT_DIR / "plots" / "intercept_mechanics_font_fixed.png",
        OUTPUT_DIR / "plots" / "descriptive_random_effects_font_fixed.png",
        OUTPUT_DIR / "artifact_manifest.csv",
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite supplement artifacts: {existing}")

    provenance = json.loads((OUTPUT_DIR / "provenance.json").read_text(encoding="utf-8"))
    if provenance["config_sha256"] != _sha256(CONFIG_PATH):
        raise ValueError("Current config no longer matches the frozen output snapshot")
    protocol_path = PROJECT_ROOT / "research_protocols" / "csad_specification_audit_v1.md"
    if provenance["protocol_sha256"] != _sha256(protocol_path):
        raise ValueError("Current protocol no longer matches the frozen output snapshot")
    config = load_config(CONFIG_PATH)

    tables = _load_tables()
    univariate_coefficients, univariate_diagnostics = (
        _descriptive_univariate_meta_regressions(
            tables["empirical_estimates_with_moderators"]
        )
    )
    univariate_coefficients.to_csv(targets[0], index=False)
    univariate_diagnostics.to_csv(targets[1], index=False)
    decisions = build_structural_robustness_decisions(
        tables["model_diagnostics"],
        tables["volatility_regime_results"],
        tables["false_positive_summary"],
        config,
    )
    decisions.to_csv(targets[2], index=False)
    tables["structural_robustness_decisions"] = decisions
    tables["descriptive_univariate_meta_coefficients"] = univariate_coefficients
    tables["descriptive_univariate_meta_diagnostics"] = univariate_diagnostics

    plot_intercept_mechanics(
        tables["intercept_mechanical_comparison"], targets[5]
    )
    plot_random_effects(tables["descriptive_random_effects"], targets[6])
    plot_paths = [
        "plots/false_positive_rates.png",
        "plots/intercept_mechanics_font_fixed.png",
        "plots/empirical_vs_null.png",
        "plots/descriptive_random_effects_font_fixed.png",
    ]
    report = build_korean_audit_report(
        tables,
        plot_paths,
        protocol_hash=provenance["protocol_sha256"],
        config_hash=provenance["config_sha256"],
    )
    targets[3].write_text(report, encoding="utf-8")
    targets[4].write_text(
        "\n".join(
            [
                "# CSAD audit v1 비수치 보완 기록",
                "",
                "- 작성일: 2026-07-20",
                "- 동결된 protocol, config, empirical 계수, Monte Carlo 반복과 판정 기준은 변경하지 않았습니다.",
                "- 기존 전체 moderator 동시 meta-regression은 provider와 universe가 교락되어 3/3 모형에서 rank deficient였습니다.",
                "- 해당 계수를 인과적으로 해석하지 않도록 moderator별 HC3 단변량 회귀와 model별 BH-FDR 표를 추가했습니다.",
                "- 기존 corrected 기준(no-intercept와 SCSAD daily·weekly 4셀)을 별도 열로 명시했지만, 사전등록된 구조적 강건성 최종 판정은 변경하지 않았습니다.",
                "- 한글 글리프 의존이 없도록 두 그림의 제목을 영문으로 바꿔 새 파일에 렌더링했습니다. 기존 그림은 보존했습니다.",
                "- 보완 보고서: `csad_specification_audit_report_v1_1.md`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_artifact_manifest(OUTPUT_DIR)


def _load_tables() -> dict[str, pd.DataFrame]:
    names = [
        "model_diagnostics",
        "intercept_mechanical_comparison",
        "intercept_verdict",
        "false_positive_summary",
        "empirical_vs_null",
        "conditional_concentration_results",
        "volatility_regime_results",
        "descriptive_random_effects",
        "descriptive_meta_regression_diagnostics",
        "input_integrity",
        "empirical_estimates_with_moderators",
    ]
    return {name: pd.read_csv(OUTPUT_DIR / f"{name}.csv") for name in names}


if __name__ == "__main__":
    main()
