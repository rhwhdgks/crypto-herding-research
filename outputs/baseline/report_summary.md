# 연구 요약

## 패널
- 모드: intersection
- 최소 활성 자산 수: 14
- 설정된 전체 자산 수: 14
- 요청 구간: 2024-04-08 09:14:00+00:00 ~ 2026-04-08 09:13:00+00:00
- 실제 분석 구간: 2024-04-08 09:15:00+00:00 ~ 2026-04-08 09:13:00+00:00
- 비고: 정렬 이후에도 요청한 전체 구간이 그대로 유지됐습니다.

## 회귀
- beta2: 4.534877
- beta2 t-통계량: 559.160
- beta2 p-값: 0
- 결정계수 R-squared: 0.6036
- 해석: beta2가 음수가 아니므로 baseline 회귀는 herding을 지지하지 않습니다.

## 이벤트 수
- herding: 24948
- shock: 22045

## 보유기간 하이라이트
- shock / 6h: 평균 수익률 0.0093%, t-통계량 0.74, 승률 51.91%
- shock / 5m: 평균 수익률 0.0017%, t-통계량 0.90, 승률 50.51%
- shock / 30m: 평균 수익률 0.0009%, t-통계량 0.23, 승률 50.87%
- herding / 15m: 평균 수익률 0.0005%, t-통계량 0.25, 승률 50.91%

## 플롯
- /home/jonghan/바탕화면/파알/herding/outputs/plots/csad_vs_market_return.png
- /home/jonghan/바탕화면/파알/herding/outputs/plots/event_occurrences.png
- /home/jonghan/바탕화면/파알/herding/outputs/plots/event_forward_returns.png
- /home/jonghan/바탕화면/파알/herding/outputs/plots/event_paths.png

## 확장 포인트
- baseline CSAD / event-study 파이프라인은 유지한 채로, 나중에 sentiment 특성을 별도 모듈로 추가할 수 있습니다.
- liquidation 또는 order-flow 스트레스 특성도 이후에 선택 모듈로 붙일 수 있습니다.
- full backtest는 이벤트 정의와 수익률 패턴이 더 안정화된 뒤에 추가하는 것이 좋습니다.
