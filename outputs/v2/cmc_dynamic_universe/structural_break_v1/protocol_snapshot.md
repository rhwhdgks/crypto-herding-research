# CMC SCSAD Structural Break Protocol v1

- 동결일: 2026-07-19
- 연구 성격: CMC point-in-time 동적 universe corrected herding의 시간별 구조 변화
- 입력: `cmc_dynamic_universe_replication_v1`의 daily market return과 CSAD
- 알고 있는 결과: full sample과 COVID 고정 구간의 no-intercept·SCSAD는 herding을 지지한다. 선행논문 fixed-62 SCSAD break는 2019-06-26, 2020-06-03, 2021-05-12, 2022-12-07에 시작한다.
- 알지 못하는 결과: dynamic Top-200 SCSAD의 BIC 선택 break 수·날짜·regime별 cubic coefficient.

## 연구 질문

CMC 동적 universe에서 발견한 corrected herding은 전체 표본에 동일하게 유지되는가, 아니면 SCSAD와 시장수익률의 관계가 특정 시점에서 구조적으로 변하는가?

## 주 모형

```text
SCSAD_t = CSAD_t   if R_m,t >= 0
SCSAD_t = -CSAD_t  if R_m,t < 0

SCSAD_t = alpha_j + gamma1_j R_m,t + gamma2_j R_m,t^2
          + gamma3_j R_m,t^3 + epsilon_t,  t in regime j
```

- break에서 `alpha`, `gamma1`, `gamma2`, `gamma3` 전체가 함께 변할 수 있다.
- break 탐색은 OLS residual sum of squares(RSS)를 전역 최소화하는 Bai–Perron 계열 정확 동적계획법을 사용한다.
- regime 계수 추론은 break 선택 후 Newey–West HAC로 다시 적합한다.

## Break 탐색 규칙

- 분석 빈도: daily only
- 표본: 2018-01-01~2024-04-09의 품질 gate 통과 관측
- trimming fraction: 0.15
- 최소 regime 크기: `ceil(T * 0.15)` observations; 현재 T=2,290에서 344
- 최대 break 수: 5, 단 최소 regime 크기로 가능한 수를 초과하지 않는다.
- candidate date: 모든 관측일; 주말·월말로 강제 정렬하지 않는다.
- missing calendar day는 0으로 채우지 않고 적격 관측 순서로 처리한다.

## Break 수 선택

0부터 최대 break까지 각각 global minimum RSS를 구한다.

```text
BIC(m) = T * log(RSS_m / T) + p_m * log(T)
p_m = (m + 1) * k + m
k = 4 regression coefficients
```

- primary break 수는 BIC가 최소인 `m`.
- `BIC(0) - BIC(selected) >= 10`이고 selected `m >= 1`일 때만 구조 변화 근거로 판정한다.
- AIC·HQIC는 민감도로 보고하지만 primary 선택을 대체하지 않는다.
- BIC가 최대 break 경계를 선택하면 `boundary_solution=True`로 보고한다.

## Regime 추론과 다중검정

- 선택된 모든 regime의 `gamma3` two-sided HAC p-value를 하나의 BH-FDR family로 보정한다.
- `gamma3 < 0` and `q <= 0.05`를 만족하면 해당 regime의 corrected herding으로 표시한다.
- break와 regime을 같은 표본에서 선택했으므로 p/q-value는 post-selection descriptive inference이며 독립 confirmatory p-value로 표현하지 않는다.

## 외부 Benchmark

Natashekara and Sampath (2026) Table A.2의 regime start를 외부 비교로 고정한다.

- 2019-06-26
- 2020-06-03
- 2021-05-12
- 2022-12-07

우리 각 break와 가장 가까운 논문 break의 calendar-day 차이를 보고하되, 이 거리로 break 수를 재선택하지 않는다. 논문과 동일하게 4개 break를 강제한 해도 secondary benchmark로 별도 저장한다.

## 보조 안정성 진단

- no-break OLS residual의 Hansen parameter-instability test와 CUSUM test를 보고한다.
- 이 테스트는 break 숫자·날짜를 선택하지 않으며 BIC 결정의 보조 진단이다.

## 산출물

- break-count information criteria
- BIC-selected break dates and paper-distance comparison
- paper-fixed four-break global optimum
- regime-level HAC coefficients, diagnostics, BH decisions
- no-break stability diagnostics
- observed/fitted SCSAD regime plot
- cubic coefficient confidence-interval plot
- config·protocol snapshot, input manifest, provenance
- 한국어 구조 변화 보고서

## 해석 제약

- 이 분석은 구조 변화와 regime별 herding 크기를 설명하며 수익률 alpha를 검증하지 않는다.
- break 날짜에 이벤트 인과를 부여하지 않는다.
- current metadata와 CMC partial snapshot 한계는 원 CMC 재현 프로토콜을 승계한다.
