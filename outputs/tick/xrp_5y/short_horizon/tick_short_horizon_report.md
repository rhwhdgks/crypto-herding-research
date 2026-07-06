# Tick 단기 미시구조 연구

## 설정
- 종목: XRPUSDT
- 거래 데이터 종류: aggTrades
- 구간: 2021-04-09 ~ latest
- 버킷: 15m
- 목표 세션 UTC: 16~23시
- trailing herding score percentile: 0.15

## 15분 버킷 요약
- all / 30m: 이벤트 0.0151%, 대조군 0.0093%, 차이 0.0058%, 차이 t=0.53, 이벤트 8465건 / 대조군 52435건
- up / 30m: 이벤트 0.0438%, 대조군 0.0093%, 차이 0.0345%, 차이 t=2.15, 이벤트 4031건 / 대조군 52435건
- down / 30m: 이벤트 -0.0111%, 대조군 0.0093%, 차이 -0.0203%, 차이 t=-1.43, 이벤트 4434건 / 대조군 52435건

## 해석
- 15분 버킷에서는 up 이벤트의 30m 반응이 가장 좋았고, 이벤트-대조군 차이는 0.0345%입니다.
- 이 실험은 자동매매 구현이 아니라, short-horizon alpha가 tick run 구조에서 더 또렷해지는지 확인하는 연구 단계입니다.

## 플롯
- /home/jonghan/findalpha/herding/outputs/tick/xrp_5y/short_horizon/plots/tick_micro_delta_15m.png
