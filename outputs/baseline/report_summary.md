# 연구 요약

## 패널
- 모드: intersection
- 최소 활성 자산 수: 14
- 설정된 전체 자산 수: 14
- 요청 구간: 2024-04-08 00:00:00+00:00 ~ 2026-04-07 23:59:00+00:00
- 실제 분석 구간: 2024-04-08 00:01:00+00:00 ~ 2026-04-07 23:59:00+00:00
- 비고: 정렬 이후에도 요청한 전체 구간이 그대로 유지됐습니다.

## 회귀
- beta2: 4.534759
- beta2 t-통계량: 11.250
- beta2 p-값: 2.32213e-29
- 공분산 추정: HAC (maxlags=31)
- 결정계수 R-squared: 0.6036
- 해석: beta2가 음수가 아니므로 baseline 회귀는 herding을 지지하지 않습니다.

## 이벤트 수
- low_dispersion: 24945
- shock: 22047

## 보유기간 하이라이트
- shock / 6h: 평균 수익률 0.0075%, UTC-day block p=0.8641, 95% CI [-0.0583%, 0.0750%], non-overlap n=2665, 승률 51.87%
- shock / 5m: 평균 수익률 0.0017%, UTC-day block p=0.3656, 95% CI [-0.0022%, 0.0057%], non-overlap n=22047, 승률 50.52%
- shock / 30m: 평균 수익률 0.0009%, UTC-day block p=0.7992, 95% CI [-0.0063%, 0.0080%], non-overlap n=22046, 승률 50.86%
- low_dispersion / 15m: 평균 수익률 0.0005%, UTC-day block p=0.8192, 95% CI [-0.0032%, 0.0044%], non-overlap n=24945, 승률 50.91%

## 플롯
- outputs/baseline/plots/csad_vs_market_return.png
- outputs/baseline/plots/event_occurrences.png
- outputs/baseline/plots/event_forward_returns.png
- outputs/baseline/plots/event_paths.png

## 확장 포인트
- baseline CSAD / event-study 파이프라인은 유지한 채로, 나중에 sentiment 특성을 별도 모듈로 추가할 수 있습니다.
- liquidation 또는 order-flow 스트레스 특성도 이후에 선택 모듈로 붙일 수 있습니다.
- full backtest는 이벤트 정의와 수익률 패턴이 더 안정화된 뒤에 추가하는 것이 좋습니다.
