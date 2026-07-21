# 논문 정렬 점검 보고서

## 전체 표본 정렬 여부
- daily_standard_full: 불일치 | model=standard_csad | value=-0.209709 | t=-3.534 | p=0.00040948
  논문 기준: 본문 기준: standard CSAD는 전체 표본에서 herding을 보이지 않아야 합니다.
- daily_scsad_full_main: 일치 | model=scsad | value=-2.439450 | t=-3.940 | p=8.15317e-05
  논문 기준: 본문 기준: daily SCSAD는 전체 표본에서 herding을 지지해야 합니다.
- daily_scsad_full_appendix: 불일치 | model=scsad | value=-2.439450 | t=-3.940 | p=8.15317e-05
  논문 기준: 부록 A.6 기준: daily SCSAD는 확장 robustness 표본 전체에서는 herding이 약할 수 있습니다.
- weekly_standard_full: 일치 | model=standard_csad | value=-0.340271 | t=-3.369 | p=0.000753805
  논문 기준: weekly standard CSAD는 음의 herding 계수를 보여야 합니다.
- weekly_scsad_full: 불일치 | model=scsad | value=-0.583287 | t=-1.826 | p=0.0678276
  논문 기준: weekly SCSAD는 음수이면서 유의한 cubic term을 보여야 합니다.
- daily_no_intercept_full: 일치 | model=no_intercept_csad | value=-1.258817 | t=-3.861 | p=0.000112967
  논문 기준: 절편 없는 모형은 전체 표본에서 herding을 지지해야 합니다.
- weekly_no_intercept_full: 일치 | model=no_intercept_csad | value=-0.847877 | t=-4.122 | p=3.76177e-05
  논문 기준: weekly 절편 제거 모형은 herding을 강하게 지지해야 합니다.

## Daily vs Weekly 비교
- daily 전체표본 SCSAD: value=-2.439450, t=-3.940, p=8.15317e-05
- weekly 전체표본 SCSAD: value=-0.583287, t=-1.826, p=0.0678276
- daily 전체표본 no-intercept: value=-1.258817, t=-3.861
- weekly 전체표본 no-intercept: value=-0.847877, t=-4.122
- 해석: weekly는 standard와 no-intercept가 더 안정적으로 음수를 유지하고, SCSAD도 방향은 음수라서 daily보다 논문 방향에 조금 더 가깝습니다.

## 하위 구간 강도 순위
- daily / standard_csad: 관측 순서 post_covid > covid | 논문 최상위 구간 일치 여부=True
- daily / no_intercept_csad: 관측 순서 post_covid > covid | 논문 최상위 구간 일치 여부=True
- daily / scsad: 관측 순서 post_covid > covid | 논문 최상위 구간 일치 여부=True
- weekly / standard_csad: 관측 순서 covid > pre_covid > post_covid | 논문 최상위 구간 일치 여부=False
- weekly / no_intercept_csad: 관측 순서 post_covid > pre_covid > covid | 논문 최상위 구간 일치 여부=True
- weekly / scsad: 관측 순서 post_covid > pre_covid > covid | 논문 최상위 구간 일치 여부=True

## 요약
- 현재 paper-like 근사 결과는 추적한 주장 7개 중 4개와 일치합니다.
- 주요 차이는 Binance OHLCV, 14개 종목, equal-weighted 시장수익률을 사용한다는 점에서 생깁니다. 논문은 더 넓은 유니버스와 market-cap weighted 설정에 가깝습니다.
