# CSAD 빈도·시장 구성 민감도 분석계획 v1

## 상태

- 동결일: 2026-07-16
- 성격: 기존 1분봉 baseline 결과를 확인한 뒤 설계한 prospective extension
- 주의: 1분봉 `beta2 > 0`과 과거 paper-like 결과를 이미 알고 있으므로 완전한 blind preregistration으로 표현하지 않는다.

## 연구 질문

1. 동일한 기간과 종목을 사용해도 수익률 집계 주기에 따라 CSAD의 이차항 부호와 크기가 달라지는가?
2. BTC와 ETH를 시장 구성에서 제외하면 그 결론이 달라지는가?

## 고정 표본

- 원천: baseline의 완전 정렬된 1분 로그수익률 패널
- 기간: 2024-04-08 00:00:00 UTC 이상, 2026-04-08 00:00:00 UTC 미만
- 주기: 1분, 5분, 15분, 1시간, 4시간, 1일
- 집계: UTC 기준 비중첩 구간에서 1분 로그수익률을 합산
- 완전성: 구간을 구성하는 모든 1분 수익률과 모든 해당 universe 종목이 존재하는 구간만 사용

## 고정 시장 구성

- `ew14`: baseline 14종목 동일가중
- `ew12_ex_btc_eth`: BTC와 ETH를 제외한 나머지 12종목 동일가중

현재 시점의 시가총액 순위를 이용한 종목 선택이나 결과 확인 후의 종목 제외는 허용하지 않는다.

## 모형과 비교 척도

각 12개 셀에서 다음 회귀를 Newey-West HAC covariance로 추정한다.

```text
CSAD_t = alpha + beta1 * |R_m,t| + beta2 * R_m,t^2 + epsilon_t
```

- 원 `beta2`는 기존 문헌식 해석을 위해 보존한다.
- 주기별 수익률 단위 차이는 `beta2 * SD(R_m^2) / SD(CSAD)`로 표준화해 비교한다.
- 표준화는 계수 단위만 바꾸므로 해당 계수의 t-stat과 p-value는 원 회귀와 같다.

## 다중검정과 판정

- 검정군은 `6개 주기 × 2개 universe = 12개 셀`로 고정한다.
- 12개 양측 HAC p-value 전체에 Benjamini-Hochberg FDR을 한 번 적용한다.
- 특정 주기의 광범위한 herding 근거는 두 universe 모두에서 `beta2 < 0`이고 `q <= 0.05`일 때만 인정한다.
- 한 셀의 raw p-value, 가장 좋은 주기 또는 가장 큰 효과만 따로 골라 결론을 내리지 않는다.

## 산출물

- 12개 셀 회귀 및 FDR 요약
- 주기별 universe 비교표
- 표준화 beta2와 95% 신뢰구간 그림
- 판정 기준을 포함한 한국어 보고서

## 다음 단계

본 분석이 끝난 뒤에만 raw tick 7종목의 run side, 가격 방향, aggressor imbalance를 분리한 의미 검증으로 이동한다. 빈도 분석 결과를 보고 tick threshold, 시간대 또는 종목을 변경하지 않는다.
