# 논문 유사 환경 허딩 요약

## 설정
- 이 결과는 논문을 완전히 복제한 것이 아니라, 논문에 가깝게 맞춘 근사 환경입니다. CoinMarketCap 일봉이나 Bitstamp tick data 대신 Binance OHLCV와 현재 설정 종목을 사용합니다.
- 주기: 1w
- 패널 모드: expanding_universe
- 최소 활성 자산 수: 4
- 시장수익률 계산 방식: equal_weighted
- 공분산 추정량: Newey-West HAC
- 비고: 주간 자료는 마지막 `2024-04-08` 라벨 바를 포함하기 위해 내부 요청 종료 시각을 다음 주 경계로 둡니다.

## 패널
- 요청 구간: 2018-01-01 00:00:00+00:00 ~ 2024-04-08 00:00:00+00:00
- 실제 분석 구간: 2018-01-08 00:00:00+00:00 ~ 2024-04-08 00:00:00+00:00
- 최대 활성 자산 수: 14
- 활성 자산 수 중앙값: 14.0

## 전체 표본
- standard_csad: market_return_sq=-0.340271, t-통계량=-3.369, p-값=0.000753805, R-squared=0.2285
- no_intercept_csad: market_return_sq=-0.847877, t-통계량=-4.122, p-값=3.76177e-05, R-squared=0.7184
- scsad: market_return_cu=-0.583287, t-통계량=-1.826, p-값=0.0678276, R-squared=0.6821

## 하위 구간
- pre_covid / standard_csad: market_return_sq=-0.211066, t-통계량=-1.201, p-값=0.229592
- covid / standard_csad: market_return_sq=-0.590885, t-통계량=-3.235, p-값=0.00121749
- post_covid / standard_csad: market_return_sq=0.101098, t-통계량=0.466, p-값=0.641165
- pre_covid / no_intercept_csad: market_return_sq=-1.349186, t-통계량=-5.848, p-값=4.96734e-09
- covid / no_intercept_csad: market_return_sq=-0.995070, t-통계량=-6.988, p-값=2.77918e-12
- post_covid / no_intercept_csad: market_return_sq=-1.504105, t-통계량=-3.351, p-값=0.000806018
- pre_covid / scsad: market_return_cu=-2.199358, t-통계량=-4.552, p-값=5.32303e-06
- covid / scsad: market_return_cu=-1.313393, t-통계량=-3.015, p-값=0.00257364
- post_covid / scsad: market_return_cu=-3.049409, t-통계량=-2.258, p-값=0.0239586
