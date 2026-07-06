# Tick 겹침 핵심 후보 검증

## 규칙
- 규칙 이름: `overlap_core`
- 정의: `prev_neg AND ratio_1_40_16_18`
- 의미: 직전 15분이 음수였고, 이벤트 강도가 threshold 대비 충분히 강하며, 시각대가 UTC 16-18인 XRP 이벤트만 남긴 규칙입니다.
- 이벤트가 존재한 날짜 수: 251일
- 기준 비용: round-trip 4.0 bps

## OOS 요약
- full_sample: 거래 319건, 평균 순수익 0.0632%, 승률 54.55%, 누적 21.19%, t=1.48
- development_30d: 거래 277건, 평균 순수익 0.0441%, 승률 54.51%, 누적 12.09%, t=0.96
- holdout_30d: 거래 42건, 평균 순수익 0.1889%, 승률 54.76%, 누적 8.13%, t=1.65
- development_60d: 거래 239건, 평균 순수익 0.0405%, 승률 55.23%, 누적 9.43%, t=0.84
- holdout_60d: 거래 80건, 평균 순수익 0.1310%, 승률 52.50%, 누적 10.75%, t=1.44

## 비용 내구성
- full_sample: 평균 총수익 0.1032%, 손익분기 비용 10.32 bps
- holdout_30d: 평균 총수익 0.2289%, 손익분기 비용 22.89 bps
- development_30d: 평균 총수익 0.0841%, 손익분기 비용 8.41 bps
- holdout_60d: 평균 총수익 0.1710%, 손익분기 비용 17.10 bps
- development_60d: 평균 총수익 0.0805%, 손익분기 비용 8.05 bps
- full_sample 기준 2bps에서 평균 순수익 0.0832%, 10bps에서는 0.0032%입니다.
- holdout_30d 기준 2bps에서 평균 순수익 0.2089%, 10bps에서는 0.1289%입니다.

## Paper Simulation
- 양(+) 월 비중 62.90%, 월 평균 누적 순수익 0.3186%, 월 수 62
- 30일 블록 기준 양(+) 블록 비중 56.99%, 블록 평균 순수익 0.0542%

## 해석
- 전체 표본에서는 평균 순수익 0.0632%로, 기존 `prev_neg` 전체보다 훨씬 압축된 고확신 규칙인지 확인할 수 있습니다.
- 최근 30일과 60일 holdout 모두 양(+)이면, 최근 구간에서도 규칙이 무너지지 않았다고 볼 수 있습니다.
- 이 리포트는 겹침 핵심 65건이 단순한 우연인지, 아니면 시간순 누적 기준에서도 추적할 가치가 있는지 보는 용도입니다.

## 플롯
- /home/jonghan/findalpha/herding/outputs/tick/xrp_5y/overlap_core/plots/overlap_oos.png
- /home/jonghan/findalpha/herding/outputs/tick/xrp_5y/overlap_core/plots/overlap_blocks.png
- /home/jonghan/findalpha/herding/outputs/tick/xrp_5y/overlap_core/plots/cost_curve.png
- /home/jonghan/findalpha/herding/outputs/tick/xrp_5y/overlap_core/plots/equity_curve.png
- /home/jonghan/findalpha/herding/outputs/tick/xrp_5y/overlap_core/plots/equity_curve_recent_90d.png
- /home/jonghan/findalpha/herding/outputs/tick/xrp_5y/overlap_core/plots/monthly_bars.png
