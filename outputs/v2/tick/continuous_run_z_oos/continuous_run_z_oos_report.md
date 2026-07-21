# 연속형 Run-Z OOS 연구 보고서

## 연구 질문

Winner label을 완전히 버리고 up, down, zero run clustering intensity를 각각 연속형으로 사용했을 때, 실제 aggressor 방향이나 30분 후 시장중립 움직임과 연관되는지 검정합니다.

## 사전 고정 설계

- 개발: 2024-04-08T00:00:00Z ~ 2025-04-08T00:00:00Z 미만
- OOS: 2025-04-08T00:00:00Z ~ 2026-04-08T00:00:00Z 미만
- feature: `-run_z_up`, `-run_z_down`, `-run_z_zero` 3개를 공동 회귀
- clipping·표준화: 개발 구간에서만 적합 후 OOS에 고정
- 가설 family: aggressor, 30분 수익률, 30분 절대수익률 각 3개 BH-FDR
- OOS는 이 feature specification에 대한 held-out이며 완전히 새로운 외부 데이터는 아님

## 표본 품질

| split | 행 | 기간 | run-z 완전률 | aggressor 가용률 |
|---|---:|---|---:|---:|
| development | 245,280 | 2024-04-08 00:00:00+00:00 ~ 2025-04-07 23:45:00+00:00 | 100.00% | 100.00% |
| oos | 245,280 | 2025-04-08 00:00:00+00:00 ~ 2026-04-07 23:45:00+00:00 | 100.00% | 100.00% |

## OOS 핵심 결과

### Aggressor 구성타당도

| feature | 개발 계수 | OOS 계수 | OOS 95% CI | OOS BH q | 판정 |
|---|---:|---:|---:|---:|---|
| up | -0.0394 | -0.0185 | [-0.0234, -0.0136] | 2.025e-13 | 미통과 |
| down | 0.0432 | 0.0204 | [0.0156, 0.0252] | 2.424e-16 | 미통과 |
| zero | -0.0030 | 0.0088 | [0.0047, 0.0129] | 2.549e-05 | 미통과 |

### 30분 시장중립 수익률

| feature | 개발 계수 | OOS 계수 | OOS 95% CI | OOS BH q | 판정 |
|---|---:|---:|---:|---:|---|
| up | 0.0275 | -0.2942 | [-0.8952, 0.3067] | 0.7655 | 미통과 |
| down | -0.4399 | 0.1208 | [-0.4362, 0.6778] | 0.7655 | 미통과 |
| zero | -0.6627 | -0.1601 | [-1.2120, 0.8919] | 0.7655 | 미통과 |

### 30분 시장중립 절대수익률

| feature | 개발 계수 | OOS 계수 | OOS 95% CI | OOS BH q | 판정 |
|---|---:|---:|---:|---:|---|
| up | -0.7913 | 1.0443 | [0.2760, 1.8126] | 0.01158 | 미통과 |
| down | 0.6968 | 0.1083 | [-0.9284, 1.1451] | 0.8378 | 미통과 |
| zero | -5.8681 | -3.4271 | [-4.0386, -2.8156] | 1.365e-27 | 미통과 |

## 결과 해석

- 개발 구간에서 사전 기준을 통과한 유일한 미래 계수는 `run_intensity_zero`의 30분 시장중립 절대수익률 -5.87bp였지만, OOS에서는 -3.43bp로 축소되어 5bp 경제성 기준을 재현하지 못했습니다.
- OOS 미래 계수 6개 중 6개의 95% 신뢰구간이 사전 경제성 경계인 ±5bp 안에 완전히 들어갑니다.
- BH-FDR상 유의하지만 5bp보다 작은 OOS 결과는 up 30분 시장중립 절대수익률 1.04bp, zero 30분 시장중립 절대수익률 -3.43bp입니다. 통계적 유의성과 거래 가능한 효과를 구분해야 합니다.
- Up/down intensity의 aggressor 공동 회귀 부호는 개발과 OOS 모두 사전 방향과 반대였습니다. 방향 proxy 구성타당도를 지지하지 않습니다.
- OOS up/down intensity 상관은 0.946으로 높아, 개별 계수는 다른 intensity를 고정한 잔차적 차이로 해석해야 합니다.
- 종목별 계수 부호가 미래 family에서 일관되지 않으므로 pooled 소규모 효과를 보편적 자산 신호로 해석하지 않습니다.

## 종합 판정

- 사전 통계·경제성 기준을 모두 통과한 OOS 미래 반응은 없습니다.
- 전체 OOS 통과 계수: 0개
- Aggressor 동시 연관성은 구성타당도이지 미래 예측력이 아닙니다.
- 통과 결과가 있어도 외부 표본 검증 전에는 tracker·paper-sim·자동매매를 활성화하지 않습니다.

## 모형 진단

| split | family | 관측치 | UTC-day | R2 | condition number |
|---|---|---:|---:|---:|---:|
| development | construct_aggressor | 240,408 | 365 | 0.2432 | 50.9 |
| development | future_excess_return | 240,387 | 365 | 0.0007 | 50.9 |
| development | future_excess_abs_return | 240,387 | 365 | 0.1327 | 50.9 |
| oos | construct_aggressor | 233,201 | 365 | 0.1891 | 46.2 |
| oos | future_excess_return | 233,180 | 365 | 0.0006 | 46.2 |
| oos | future_excess_abs_return | 233,180 | 365 | 0.1192 | 46.2 |

## 재현 정보

- scaling fit: 2024-04-08T23:45:00+00:00 ~ 2025-04-07T23:45:00+00:00
- winsor: 0.500% ~ 99.500%
- 사전 프로토콜: `research_protocols/tick_continuous_run_z_oos_v1.md`
- 종목별 계수는 `symbol_descriptive_coefficients.csv`에 기술용으로만 저장

## 그림

- `outputs/v2/tick/continuous_run_z_oos/plots/oos_coefficients.png`
- `outputs/v2/tick/continuous_run_z_oos/plots/development_oos_comparison.png`
- `outputs/v2/tick/continuous_run_z_oos/plots/run_intensity_correlations.png`
