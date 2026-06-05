# Tick 단기 미시구조 연구

## 설정
- 종목: XRPUSDT
- 거래 데이터 종류: aggTrades
- 구간: 2021-04-09 ~ latest
- 버킷: 15m
- 목표 세션 UTC: 16~23시
- trailing herding score percentile: 0.15

## 15분 버킷 요약
- all / 30m: 이벤트 0.0124%, 대조군 0.0103%, 차이 0.0020%, 차이 t=0.18, 이벤트 8082건 / 대조군 50226건
- up / 30m: 이벤트 0.0417%, 대조군 0.0103%, 차이 0.0314%, 차이 t=1.87, 이벤트 3838건 / 대조군 50226건
- down / 30m: 이벤트 -0.0142%, 대조군 0.0103%, 차이 -0.0245%, 차이 t=-1.67, 이벤트 4244건 / 대조군 50226건

## 해석
- 15분 버킷에서는 up 이벤트의 30m 반응이 가장 좋았고, 이벤트-대조군 차이는 0.0314%입니다.
- 이 실험은 자동매매 구현이 아니라, short-horizon alpha가 tick run 구조에서 더 또렷해지는지 확인하는 연구 단계입니다.

## 플롯
- /home/jonghan/바탕화면/파알/herding/outputs/tick/xrp_5y/short_horizon/plots/tick_micro_delta_15m.png
