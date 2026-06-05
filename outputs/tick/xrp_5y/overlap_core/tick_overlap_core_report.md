# Tick 겹침 핵심 후보 검증

## 규칙
- 규칙 이름: `overlap_core`
- 정의: `prev_neg AND ratio_1_40_16_18`
- 의미: 직전 15분이 음수였고, 이벤트 강도가 threshold 대비 충분히 강하며, 시각대가 UTC 16-18인 XRP 이벤트만 남긴 규칙입니다.
- 이벤트가 존재한 날짜 수: 240일
- 기준 비용: round-trip 4.0 bps

## OOS 요약
- full_sample: 거래 301건, 평균 순수익 0.0631%, 승률 55.48%, 누적 19.81%, t=1.41
- development_30d: 거래 263건, 평균 순수익 0.0503%, 승률 54.75%, 누적 13.31%, t=1.09
- holdout_30d: 거래 38건, 평균 순수익 0.1512%, 승률 60.53%, 누적 5.74%, t=0.99
- development_60d: 거래 224건, 평균 순수익 0.0384%, 승률 55.36%, 누적 8.34%, t=0.79
- holdout_60d: 거래 77건, 평균 순수익 0.1347%, 승률 55.84%, 누적 10.59%, t=1.32

## 비용 내구성
- full_sample: 평균 총수익 0.1031%, 손익분기 비용 10.31 bps
- holdout_30d: 평균 총수익 0.1912%, 손익분기 비용 19.12 bps
- development_30d: 평균 총수익 0.0903%, 손익분기 비용 9.03 bps
- holdout_60d: 평균 총수익 0.1747%, 손익분기 비용 17.47 bps
- development_60d: 평균 총수익 0.0784%, 손익분기 비용 7.84 bps
- full_sample 기준 2bps에서 평균 순수익 0.0831%, 10bps에서는 0.0031%입니다.
- holdout_30d 기준 2bps에서 평균 순수익 0.1712%, 10bps에서는 0.0912%입니다.

## Paper Simulation
- 양(+) 월 비중 61.67%, 월 평균 누적 순수익 0.3099%, 월 수 60
- 30일 블록 기준 양(+) 블록 비중 56.42%, 블록 평균 순수익 0.0520%

## 해석
- 전체 표본에서는 평균 순수익 0.0631%로, 기존 `prev_neg` 전체보다 훨씬 압축된 고확신 규칙인지 확인할 수 있습니다.
- 최근 30일과 60일 holdout 모두 양(+)이면, 최근 구간에서도 규칙이 무너지지 않았다고 볼 수 있습니다.
- 이 리포트는 겹침 핵심 65건이 단순한 우연인지, 아니면 시간순 누적 기준에서도 추적할 가치가 있는지 보는 용도입니다.

## 플롯
- /home/jonghan/바탕화면/파알/herding/outputs/tick/xrp_5y/overlap_core/plots/overlap_oos.png
- /home/jonghan/바탕화면/파알/herding/outputs/tick/xrp_5y/overlap_core/plots/overlap_blocks.png
- /home/jonghan/바탕화면/파알/herding/outputs/tick/xrp_5y/overlap_core/plots/cost_curve.png
- /home/jonghan/바탕화면/파알/herding/outputs/tick/xrp_5y/overlap_core/plots/equity_curve.png
- /home/jonghan/바탕화면/파알/herding/outputs/tick/xrp_5y/overlap_core/plots/equity_curve_recent_90d.png
- /home/jonghan/바탕화면/파알/herding/outputs/tick/xrp_5y/overlap_core/plots/monthly_bars.png
