# Tick run clustering 의미 검증계획 v1

## 상태와 기존 지식

- 동결일: 2026-07-17
- 5년 migrated schema-v2 cache의 lead-lag 결과와 FDR 0건을 이미 확인한 뒤 설계한 의미 검증이다.
- 따라서 완전한 blind preregistration이 아니라, 기존 방향 오해를 바로잡기 위한 prospective diagnostic protocol이다.
- 구형 `micro_herding_up/down` 결과는 가격 방향 증거로 재사용하지 않는다.

## 연구 질문

1. `run_clustering_side=up/down`은 같은 15분 구간의 가격 방향과 충분히 일치하는가?
2. run-clustering threshold event는 현재 시장 상태를 통제한 뒤에도 30분 미래 시장중립 수익률과 연관되는가?
3. raw buyer-maker 자료에서 run side와 aggressor direction은 어떤 관계를 보이는가?

## 고정 자료

### 주 분석

- 2021-05-09부터 2026-05-07까지 7종목 15분 migrated schema-v2 cache
- 종목: BTC, ETH, XRP, SOL, DOGE, ADA, AVAX USDT
- 가격 방향과 conditional run z는 v2로 재구성됐지만 aggressor 정보는 없다.

### Raw 개발용 pilot

- 2024년 4월 월별 Binance `aggTrades` 7종목
- 첫 완전 월을 parser·aggressor 구성타당도 점검용으로 고정한다.
- 이 한 달 결과에는 confirmatory p-value나 alpha 주장을 부여하지 않는다.

## 검정 1: 가격 방향 의미

- threshold event만 사용한다.
- `run side=up/down`과 `price direction=up/down`만 남겨 동일 방향 비율을 계산한다.
- 귀무가설은 방향 일치 확률 50%이며 양측 exact binomial test를 사용한다.
- pooled와 7개 종목, 총 8개 p-value 전체에 BH-FDR을 적용한다.
- 최소 30개 directional event가 없는 행은 추론하지 않고 family에서는 보수적으로 p=1로 처리한다.
- 가격 방향 proxy로 부르려면 pooled 일치율이 60% 이상이고 q<=0.05여야 한다.
- 전체 3×3 run side·price direction Cramer's V는 구성 설명용으로 함께 기록한다.

## 검정 2: 30분 시장중립 반응

- 결과변수는 각 자산 30분 수익률에서 다른 6개 자산의 동일 시각 평균 30분 수익률을 뺀 값이다.
- 표본은 point-in-time threshold가 존재하는 event와 control bucket으로 제한한다.
- 설명변수는 event-up, event-down, event-zero와 현재 자산 수익률, leave-one-out 시장수익률, 절댓값, 24시간 trailing volatility, 거래 건수, 거래대금, 가격 방향, 일반 run side, 종목 및 UTC hour fixed effects다.
- 표준오차는 UTC day cluster covariance를 사용해 중첩 30분 결과를 허용한다.
- 세 event coefficient의 p-value 전체에 BH-FDR을 적용한다.
- q<=0.05이고 절대 계수가 5bp 이상일 때만 경제적으로 검토할 조건부 연관성으로 표시한다.
- 이 기준을 통과해도 인과적 lead-lag 또는 거래 alpha로 표현하지 않는다.

## 검정 3: Aggressor 의미

- raw pilot에서 aggressor imbalance의 가용률, run side별 평균·중앙값, 방향 일치율을 기술한다.
- 한 달 pilot에는 formal p-value를 사용하지 않는다.
- 결론은 동일 2년 raw bucket cache가 완성된 후 이 protocol의 새 버전에서만 내린다.

## 금지 사항

- 결과 확인 후 threshold, 세션, horizon 또는 종목 변경
- raw p-value만 골라 alpha로 표현
- migrated cache에서 unavailable인 aggressor를 추정하거나 대체
- 동시 연관성을 미래 예측 또는 인과관계로 표현
