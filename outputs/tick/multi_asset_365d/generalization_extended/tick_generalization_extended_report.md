# 확장 Tick 일반화 연구

## 대상 심볼
- BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT
- 구간: 2025-04-09 ~ 2026-04-09
- 이벤트 정의: 15분 up/down micro-herding
- 반응 horizon: 다음 30분

## up 이벤트 비교
- XRPUSDT: 차이 0.0383%, t=0.95, 이벤트 900건, 판정 positive_weak
- SOLUSDT: 차이 0.0167%, t=0.57, 이벤트 824건, 판정 positive_weak
- AVAXUSDT: 차이 0.0148%, t=0.35, 이벤트 979건, 판정 positive_weak
- ETHUSDT: 차이 -0.0076%, t=-0.35, 이벤트 860건, 판정 negative
- BTCUSDT: 차이 -0.0134%, t=-0.86, 이벤트 784건, 판정 negative
- ADAUSDT: 차이 -0.0227%, t=-0.45, 이벤트 896건, 판정 negative
- DOGEUSDT: 차이 -0.0336%, t=-0.91, 이벤트 1087건, 판정 negative

## all 이벤트 비교
- AVAXUSDT: 차이 0.0314%, t=1.21, 이벤트 1915건
- XRPUSDT: 차이 0.0140%, t=0.59, 이벤트 1902건
- DOGEUSDT: 차이 0.0090%, t=0.36, 이벤트 1987건
- ADAUSDT: 차이 -0.0012%, t=-0.04, 이벤트 1823건
- BTCUSDT: 차이 -0.0036%, t=-0.33, 이벤트 1616건
- ETHUSDT: 차이 -0.0163%, t=-1.00, 이벤트 1862건
- SOLUSDT: 차이 -0.0203%, t=-1.09, 이벤트 1868건

## down 이벤트 비교
- DOGEUSDT: 차이 0.0606%, t=1.92, 이벤트 900건
- AVAXUSDT: 차이 0.0488%, t=1.75, 이벤트 936건
- ADAUSDT: 차이 0.0195%, t=0.73, 이벤트 927건
- BTCUSDT: 차이 0.0057%, t=0.39, 이벤트 832건
- XRPUSDT: 차이 -0.0077%, t=-0.29, 이벤트 1002건
- ETHUSDT: 차이 -0.0238%, t=-1.04, 이벤트 1002건
- SOLUSDT: 차이 -0.0495%, t=-2.14, 이벤트 1044건

## 해석
- up micro-herding 기준 가장 강한 심볼은 XRPUSDT이며, 차이는 0.0383%입니다.
- up 기준 양(+) 반응 심볼은 3개, 음(-) 반응 심볼은 4개입니다.
- 이 결과는 XRP 신호가 시장 전체 공통 구조인지, 특정 알트군에 더 가까운 구조인지 확인하는 확장 일반화 실험입니다.
