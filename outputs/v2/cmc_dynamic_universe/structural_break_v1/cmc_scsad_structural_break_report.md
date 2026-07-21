# CMC SCSAD 구조적 변화 분석

## 한 문장 결론

> BIC가 4개 break를 선택했고 no-break 대비 ΔBIC=194.64로 사전 10 기준을 넘어 SCSAD 관계의 구조 변화를 지지합니다.

## 사전 고정 설계

- 표본: 2018-01-01~2024-04-09, 2,290관측
- 모형: scsad_full_coefficient_change
- trimming: 15%, 최소 regime 344관측
- 최대 break: 5, primary selection: BIC
- break 위치와 regime 계수를 같은 표본에서 추정하므로 regime p/q는 post-selection descriptive inference입니다.

## Break 수 선택

| breaks | segments | RSS | AIC | BIC | HQIC | BIC selected |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 | 1.046695 | -17603.63 | -17580.69 | -17595.26 | False |
| 1 | 2 | 1.003969 | -17689.07 | -17637.45 | -17670.25 | False |
| 2 | 3 | 0.949543 | -17806.71 | -17726.40 | -17777.42 | False |
| 3 | 4 | 0.915163 | -17881.16 | -17772.17 | -17841.41 | False |
| 4 | 5 | 0.898596 | -17912.99 | -17775.32 | -17862.79 | True |
| 5 | 6 | 0.894838 | -17912.59 | -17746.24 | -17851.93 | False |

## BIC 선택 Break

| # | 이전 regime 종료 | 새 regime 시작 | 매칭 논문 break | 차이(일) |
|---:|---|---|---|---:|
| 1 | 2019-06-27 | 2019-06-28 | 2019-06-26 | +2 |
| 2 | 2020-06-05 | 2020-06-06 | 2020-06-03 | +3 |
| 3 | 2021-05-16 | 2021-05-17 | 2021-05-12 | +5 |
| 4 | 2022-11-10 | 2022-11-11 | 2022-12-07 | -26 |

## Regime별 Corrected Herding

| regime | period | n | gamma3 | standardized | HAC t | BH q | herding |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | 2018-01-01~2019-06-27 | 543 | -12.212 | -0.275 | -3.144 | 0.002782 | True |
| 2 | 2019-06-28~2020-06-05 | 344 | -3.547 | -0.687 | -2.726 | 0.006408 | True |
| 3 | 2020-06-06~2021-05-16 | 344 | -53.517 | -0.435 | -6.158 | 2.352e-09 | True |
| 4 | 2021-05-17~2022-11-10 | 543 | -6.861 | -0.220 | -2.829 | 0.00584 | True |
| 5 | 2022-11-11~2024-04-09 | 516 | -137.972 | -0.465 | -6.119 | 2.352e-09 | True |

## 논문과 동일한 4-Break 수 고정 비교

- break 1: 2019-06-28 vs paper 2019-06-26, +2일
- break 2: 2020-06-06 vs paper 2020-06-03, +3일
- break 3: 2021-05-17 vs paper 2021-05-12, +5일
- break 4: 2022-11-11 vs paper 2022-12-07, -26일

## 논문 Break 날짜 그대로 적용한 계수 재현

| regime | n(paper/our) | paper gamma3 | our gamma3 | paper t | our t | sign match |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 541/541 | -13.377 | -12.208 | -3.752 | -3.355 | True |
| 2 | 343/343 | -2.636 | -2.909 | -2.130 | -2.085 | True |
| 3 | 343/342 | -54.097 | -55.507 | -5.247 | -5.789 | True |
| 4 | 574/574 | -7.882 | -6.887 | -1.741 | -2.717 | True |
| 5 | 490/490 | -125.620 | -139.496 | -7.244 | -6.064 | True |

## 쉽게 읽는 결과

- AIC·BIC·HQIC의 break 수 일치: True. 세 기준 모두 4개를 선택했습니다.
- 선택된 5개 regime의 gamma3는 모두 음수이고 BH-FDR를 통과했습니다.
- 표준화 cubic effect가 가장 음수인 구간은 regime 2 (2019-06-28~2020-06-05, -0.687)입니다.
- 가장 0에 가까운 구간은 regime 4 (2021-05-17~2022-11-10, -0.220)입니다.
- 첫 세 break는 논문과 5일 이내이고, 네 번째는 우리 dynamic universe가 26일 빨리 감지했습니다.

## 보조 안정성 진단

- hansen_parameter_instability: statistic=6.177, 5% stability rejection=True
- cusum_ols_residuals: statistic=1.858, p=0.002002, 5% stability rejection=True

## 해석 제약

- boundary solution: False. True면 더 많은 break 가능성을 배제하지 못합니다.
- break 날짜와 시장 이벤트의 근접은 인과관계를 증명하지 않습니다.
- 이 분석은 herding 구조를 설명하며 forward return·거래비용·alpha를 검증하지 않습니다.

## 그림

- `outputs/v2/cmc_dynamic_universe/structural_break_v1/plots/scsad_structural_regimes.png`
- `outputs/v2/cmc_dynamic_universe/structural_break_v1/plots/regime_gamma3_coefficients.png`
