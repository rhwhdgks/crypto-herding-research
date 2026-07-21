# CSAD 기계적 계수 v1.1 유한표본 수렴 보충 보고서

## 1. 왜 보충 분석을 했나

원 v1의 사전등록 composite gate는 5/6이었습니다. N=62 SCSAD의 Monte Carlo 평균은 이론값과 1.02% 차이였지만 200회 평균의 99% CI가 이론값을 아주 근소하게 포함하지 못했습니다.

원 판정 `mechanical_null_not_confirmed`은 그대로 보존합니다. 이번 결과로 원 연구를 사후 통과 처리하지 않습니다.

## 2. 새로 고정한 검증

- 독립 seed 2026072103
- iid Gaussian, equal weight, sigma=0.025
- N=14·50·62 전부 유지
- T=3,000·12,000·48,000
- 셀당 300회
- T=48,000만 primary supplement gate

## 3. Primary supplement 결과

| 모형 | N | 이론값 | MC 평균 | 상대오차 | 99% CI 포함 | 음수율 | raw FPR | BH3 FPR | gate |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| Intercept-restored | 14 | 0 | -0.067383 | 해당 없음 | 예 | 52.7% | 2.3% | 진단용 | 통과 |
| Intercept-restored | 50 | 0 | 0.055409 | 해당 없음 | 예 | 49.0% | 2.3% | 진단용 | 통과 |
| Intercept-restored | 62 | 0 | 0.0501415 | 해당 없음 | 예 | 49.0% | 2.7% | 진단용 | 통과 |
| No-intercept | 14 | -259.407 | -259.522 | 0.04% | 예 | 100.0% | 100.0% | 100.0% | 통과 |
| No-intercept | 50 | -951.764 | -952.605 | 0.09% | 예 | 100.0% | 100.0% | 100.0% | 통과 |
| No-intercept | 62 | -1182.52 | -1183.13 | 0.05% | 예 | 100.0% | 100.0% | 100.0% | 통과 |
| SCSAD | 14 | -8569.36 | -8583.08 | 0.16% | 예 | 100.0% | 100.0% | 100.0% | 통과 |
| SCSAD | 50 | -59417.8 | -59575.1 | 0.26% | 예 | 100.0% | 100.0% | 100.0% | 통과 |
| SCSAD | 62 | -82206.4 | -82362.6 | 0.19% | 예 | 100.0% | 100.0% | 100.0% | 통과 |
| Standard | 14 | 0 | -0.0670135 | 해당 없음 | 예 | 53.3% | 2.3% | 2.3% | 통과 |
| Standard | 50 | 0 | 0.0556099 | 해당 없음 | 예 | 48.7% | 2.3% | 2.3% | 통과 |
| Standard | 62 | 0 | 0.0487854 | 해당 없음 | 예 | 48.7% | 2.7% | 2.7% | 통과 |

## 4. 표본 길이에 따른 경로

| 모형 | N | T=3,000 오차 | T=12,000 오차 | T=48,000 오차 |
|---|---:|---:|---:|---:|
| No-intercept | 14 | 0.08% | 0.07% | 0.04% |
| No-intercept | 50 | 0.41% | 0.30% | 0.09% |
| No-intercept | 62 | 0.18% | 0.14% | 0.05% |
| SCSAD | 14 | 1.60% | 0.45% | 0.16% |
| SCSAD | 50 | 1.96% | 0.92% | 0.26% |
| SCSAD | 62 | 1.69% | 0.58% | 0.19% |

오차가 표본 길이마다 단조롭게 감소해야 한다는 조건은 사전 gate로 두지 않았습니다. Monte Carlo 평균에는 별도의 simulation 변동이 있기 때문입니다. 핵심은 가장 긴 T=48,000에서 이론값과 크기·CI·부호가 함께 맞는지입니다.

## 5. 판정

- 원 v1 수렴 gate: `5/6`이며 실패 판정을 보존
- 폐형식 검증: `3/3`
- 보충 기계적 셀: `6/6`
- 보충 control 셀: `6/6`
- 보충 판정: `finite_sample_convergence_supported`
- 통합 해석: `analytic_mechanism_established_and_finite_sample_convergence_supported`

원 v1의 한 셀 실패는 population 폐형식을 반박하지 않습니다. 폐형식은 Gaussian 모멘트 정규방정식에서 정확히 도출되며, 보충 simulation은 유한표본 추정치가 그 값으로 접근하는지를 별도로 확인합니다.

## 6. 연구상 의미

1. No-intercept의 음의 제곱항은 양의 CSAD 수준을 절편 없이 근사한 결과입니다.
2. SCSAD의 음의 세제곱항은 sign-step을 저차 홀수 다항식으로 근사한 결과입니다.
3. 두 음의 계수와 작은 HAC p-value는 intentional herding의 충분한 증거가 아닙니다.
4. Standard CSAD도 현실적인 공통요인·이분산·fat-tail null로 검정 크기를 교정해야 합니다.

## 7. 재현성

- Protocol SHA-256: `b51450e787b8d7884b9ab72d710913a1a7904d6c2d8fd8f22e2c068c711a6d81`
- Config SHA-256: `2b6cfa360829da47c81f03f0329b8bd2c8357196f6947c346e47e1b6bbdcf2a9`
- 그림: `plots/finite_sample_convergence_ladder.png`

```bash
PYTHONPATH=src .venv/bin/python scripts/run_csad_mechanical_supplement.py
PYTHONPATH=src .venv/bin/python scripts/verify_csad_mechanical_derivation_v1_1.py
```
