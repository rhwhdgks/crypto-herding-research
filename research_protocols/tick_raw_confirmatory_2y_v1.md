# 동일 2년 Raw Tick Confirmatory Protocol v1

- 동결일: 2026-07-18
- 기존 지식: 5년 migrated cache의 가격 일치율 null과 2024-04 raw pilot의 zero-run 우세를 알고 있다.
- 미확인 정보: 동일 2년 전체 raw의 가격·aggressor 일치율과 조정 30분 계수는 보지 않은 상태에서 이 규격을 고정한다.

## 표본

- Binance spot `aggTrades`, 2024-04-08 00:00 UTC부터 2026-04-07 23:59 UTC
- BTC, ETH, XRP, SOL, DOGE, ADA, AVAX USDT 7종목
- 15분 bucket 490,560개와 30분 exact-clock forward return
- 5일 trailing 15th percentile run-clustering threshold, UTC 16~23시, 최소 거래 200건
- buyer-maker 기반 aggressor imbalance 가용률 100%를 실행 전제조건으로 둔다.

## Family A: 가격 방향

- event의 run up/down과 같은 bucket price up/down 일치율
- pooled+7종목 8개 양측 exact-binomial p-value에 BH-FDR
- 최소 30개 directional event, 일치율 60% 이상, q<=0.05일 때만 가격 proxy로 인정

## Family B: Aggressor 방향

- run up은 buyer aggressor, run down은 seller aggressor와 일치하는지 검정
- pooled+7종목 8개 양측 exact-binomial p-value에 별도 BH-FDR
- 최소 30개 directional event, 일치율 60% 이상, q<=0.05일 때만 aggressor proxy로 인정

## Family C: 미래 반응

- 각 자산 30분 수익률에서 다른 6자산 동시간 평균을 뺀 시장중립 결과변수
- 현재 자산·시장 수익률, 절댓값, 24시간 변동성, 거래 건수·대금, 가격 방향, 일반 run side, 종목, UTC hour 통제
- UTC-day cluster covariance
- event-up/down/zero 3개 p-value에 별도 BH-FDR
- q<=0.05이고 절대 조정계수가 5bp 이상일 때만 경제적 조건부 연관성으로 표시

세 family의 결과를 서로 대체하거나 가장 좋은 결과만 선택하지 않는다. 통과 결과도 인과적 lead-lag 또는 거래 alpha로 표현하지 않는다.
