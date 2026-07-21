# 동일 2년 Raw Tick Confirmatory 보고서

## 한 문장 결론

현재 `min(up, down, zero)` run-clustering winner는 방향성 herding 대리변수가 아니며, 시장요인을 제거한 30분 미래 수익률도 예측하지 못합니다.

## 쉽게 읽는 요약

- 이벤트 25,460건 중 91.82%가 가격이 변하지 않은 tick의 연속인 `zero-run`이었습니다.
- 방향이 정의된 run event의 가격 방향 일치율은 49.23%로 50%와 구분되지 않았습니다.
- 실제 순매수·순매도 aggressor 방향 일치율도 48.13%로 60% 사전 기준을 충족하지 못했습니다.
- 시장중립 30분 회귀의 up, down, zero 계수는 모두 BH-FDR 5%와 5bp 경제성 기준을 통과하지 못했습니다.
- 따라서 현재 이벤트를 매매 신호로 승격시키지 않습니다.

## 고정 표본

- 기간: 2024-04-08 00:00:00+00:00 ~ 2026-04-07 23:45:00+00:00
- 15분 bucket: 490,560개, 7종목이 각각 70,080개
- point-in-time event: 25,460건, control: 131,544건
- buyer-maker 기반 aggressor 가용률: 100.00%
- 월별 checkpoint 175개, 심볼·시간 중복 0건, 15분 전체 grid 완전
- signal timestamp는 모든 행에서 bucket end와 일치
- 가격, aggressor, 미래 반응은 서로 분리된 사전 검정 family로 판정

## 1. Run side와 가격 방향

- pooled directional n=2,070, 일치율=49.23%
- 95% CI 47.08%~51.38%, BH q=1
- 가격 방향 proxy 판정: 통과하지 못함

| 범위 | 이벤트 | zero-run | 방향 n | 일치율 | BH q | proxy |
|---|---:|---:|---:|---:|---:|---|
| pooled | 25,460 | 91.82% | 2,070 | 49.23% | 1 | 미통과 |
| BTCUSDT | 3,833 | 100.00% | 0 | N/A | N/A | 미통과 |
| ETHUSDT | 3,728 | 100.00% | 0 | N/A | N/A | 미통과 |
| XRPUSDT | 3,488 | 90.02% | 346 | 50.00% | 1 | 미통과 |
| SOLUSDT | 3,543 | 97.32% | 95 | 51.58% | 1 | 미통과 |
| DOGEUSDT | 3,756 | 69.09% | 1,157 | 49.35% | 1 | 미통과 |
| ADAUSDT | 3,421 | 86.32% | 463 | 47.95% | 1 | 미통과 |
| AVAXUSDT | 3,691 | 99.73% | 9 | 44.44% | N/A | 미통과 |

## 2. Run side와 Aggressor 방향

- pooled directional n=2,082, 일치율=48.13%
- 95% CI 45.99%~50.27%, BH q=0.2439
- aggressor 방향 proxy 판정: 통과하지 못함

| 범위 | 방향 n | 일치율 | 95% CI | BH q | proxy |
|---|---:|---:|---:|---:|---|
| pooled | 2,082 | 48.13% | 45.99%~50.27% | 0.2439 | 미통과 |
| BTCUSDT | 0 | N/A | N/A | N/A | 미통과 |
| ETHUSDT | 0 | N/A | N/A | N/A | 미통과 |
| XRPUSDT | 348 | 41.38% | 36.33%~46.62% | 0.01221 | 미통과 |
| SOLUSDT | 95 | 50.53% | 40.65%~60.36% | 1 | 미통과 |
| DOGEUSDT | 1,161 | 50.82% | 47.94%~53.69% | 1 | 미통과 |
| ADAUSDT | 468 | 45.94% | 41.48%~50.47% | 0.2439 | 미통과 |
| AVAXUSDT | 10 | 50.00% | 23.66%~76.34% | N/A | 미통과 |

- XRPUSDT에서는 50% 반대방향으로 유의했지만, 이는 사전의 60% 순방향 proxy 가설을 지지하지 않습니다.
- 역방향 구조는 사후적 단일 종목 결과이므로 별도 탐색 가설로만 보존합니다.

## 3. 시장중립 30분 반응

- 회귀 관측치 156,990, UTC-day cluster 725
- 각 자산의 30분 수익률에서 나머지 6자산의 동시간 평균을 빼 시장 공통 움직임을 제거
- 현재 수익률·시장수익률·변동성·거래량·종목·UTC hour·가격 방향·일반 run side를 통제

| run side | 이벤트 | 조정계수(bp) | 95% CI(bp) | BH q | 사전 판정 |
|---|---:|---:|---:|---:|---|
| up | 1,050 | 0.550 | [-2.454, 3.554] | 0.9336 | 근거 부족 |
| down | 1,032 | 1.144 | [-1.527, 3.814] | 0.9336 | 근거 부족 |
| zero | 23,366 | -0.033 | [-0.804, 0.738] | 0.9336 | 근거 부족 |

- 세 계수의 95% 신뢰구간이 모두 사전 경제성 밴드 ±5bp 안에 있습니다.
- 즉 p-value가 크다는 것을 넘어, 현재 모형에서 5bp 규모 효과도 제외할 수 있는 practical-null 결과입니다.

## 해석

- `run_clustering_side`는 가격·주문 방향이 아니라 조건부 run z가 가장 낮은 tick category입니다.
- winner-take-all 방식이 zero-tick 구조에 지배되므로, 현재 이벤트는 방향성 herding보다 호가단위·유동성·거래분할 미시구조를 측정할 가능성이 높습니다.
- 방향 프록시와 미래 반응이 둘 다 사전 기준을 통과하지 못했으므로, 기존 방향성 alpha·tracker를 복구할 근거가 없습니다.

## 종합 판정

- 가격 방향 proxy: 지지하지 않음
- aggressor 방향 proxy: 지지하지 않음
- 통계·경제 기준을 모두 통과한 미래 반응: 없음
- tracker·paper-sim·자동매매 상태: 비활성 유지
- 이 결과는 고정 Binance 표본의 조건부 연관성 판정이며 인과관계나 거래 전략 성과가 아닙니다.

## 다음 연구

1. winner label을 버리고 `run_z_up`, `run_z_down`, `run_z_zero`를 각각 연속형 설명변수로 다룬니다.
2. zero-run은 방향성 herding과 분리해 spread proxy, tick size, 거래량, 후속 변동성과의 관계를 미시구조 주제로 연구합니다.
3. up/down category 가설은 새 프로토콜·검정 family·untouched OOS를 먼저 고정한 뒤 다시 검정합니다.
4. Binance 내부에서 사전 기준을 통과한 결과가 생긴 후에만 다른 거래소·거래비용·paper-sim으로 넘어갑니다.

## 재현 정보

- 사전 프로토콜: `research_protocols/tick_raw_confirmatory_2y_v1.md`
- 입력 manifest: `outputs/v2/tick/semantic_validation/confirmatory_2y/input_manifest.json`
- 월별 백필 상태: `outputs/v2/tick/semantic_validation/raw_2y/backfill_state.json`
- 월별 해시 manifest: `outputs/v2/tick/semantic_validation/raw_2y/input_manifest.json`

## 그림

- `outputs/v2/tick/semantic_validation/confirmatory_2y/plots/run_side_vs_price_direction.png`
- `outputs/v2/tick/semantic_validation/confirmatory_2y/plots/run_side_vs_aggressor_direction.png`
- `outputs/v2/tick/semantic_validation/confirmatory_2y/plots/market_neutral_predictive_coefficients.png`
