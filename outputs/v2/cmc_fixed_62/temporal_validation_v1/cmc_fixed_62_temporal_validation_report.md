# CMC Fixed-62 시간 외부표본 검증

## 결론

- direct-replication primary corrected 4개 셀: 4/4 통과, 시간 외부표본 재현
- no-look-ahead sensitivity corrected 4개 셀: 4/4 통과, timing 강건성 통과
- corrected 표준화 절대크기는 과거 대비 1.64~1.86배입니다.
- 이 검정은 corrected CSAD 관계의 시간 지속성을 다루며 미래수익률 alpha 검정이 아닙니다.

## 표본과 품질

- 분석 기간: 2024-04-10~2026-07-18
- 고정 universe: 62개 legacy CMC ID, 구성종목 교체 없음
- 전체 asset-day coverage: 100.00%
- 자산별 최저 coverage: 100.00%
- quality gate: 6/6 통과
- replication_primary: daily 830개, weekly 118개
- no_lookahead_sensitivity: daily 830개, weekly 118개

## Full Holdout 회귀

- replication_primary / daily / standard_csad: 계수 -0.2922, t=-0.33, q=0.743, 통과=False
- replication_primary / daily / no_intercept_csad: 계수 -11.2649, t=-7.15, q=1.71e-12, 통과=True
- replication_primary / daily / scsad: 계수 -59.1645, t=-7.23, q=1.5e-12, 통과=True
- replication_primary / weekly / standard_csad: 계수 -0.5398, t=-0.77, q=0.532, 통과=False
- replication_primary / weekly / no_intercept_csad: 계수 -9.5161, t=-7.63, q=1.38e-13, 통과=True
- replication_primary / weekly / scsad: 계수 -35.0194, t=-5.91, q=5.12e-09, 통과=True
- no_lookahead_sensitivity / daily / standard_csad: 계수 -0.2284, t=-0.25, q=0.804, 통과=False
- no_lookahead_sensitivity / daily / no_intercept_csad: 계수 -11.3162, t=-7.00, q=5e-12, 통과=True
- no_lookahead_sensitivity / daily / scsad: 계수 -59.7998, t=-7.01, q=5e-12, 통과=True
- no_lookahead_sensitivity / weekly / standard_csad: 계수 -0.3881, t=-0.55, q=0.699, 통과=False
- no_lookahead_sensitivity / weekly / no_intercept_csad: 계수 -9.5952, t=-7.82, q=3.06e-14, 통과=True
- no_lookahead_sensitivity / weekly / scsad: 계수 -36.1310, t=-5.98, q=3.44e-09, 통과=True

## 2018-2024 대비 Holdout

- daily / no_intercept_csad: 과거 -1.8373, holdout -11.2649, 차이 -9.4276, 표준화 계수 -0.700→-1.303, 표준화 절대크기 비율 1.86
- daily / scsad: 과거 -2.9023, holdout -59.1645, 차이 -56.2622, 표준화 계수 -0.204→-0.345, 표준화 절대크기 비율 1.70
- daily / standard_csad: 과거 -0.0579, holdout -0.2922, 차이 -0.2344, 표준화 계수 -0.022→-0.034, 표준화 절대크기 비율 1.53
- weekly / no_intercept_csad: 과거 -2.0787, holdout -9.5161, 차이 -7.4374, 표준화 계수 -1.181→-2.163, 표준화 절대크기 비율 1.83
- weekly / scsad: 과거 -3.3227, holdout -35.0194, 차이 -31.6968, 표준화 계수 -0.307→-0.503, 표준화 절대크기 비율 1.64
- weekly / standard_csad: 과거 -0.1257, holdout -0.5398, 차이 -0.4141, 표준화 계수 -0.071→-0.123, 표준화 절대크기 비율 1.72

## 기간 진단

- holdout_year1 / daily / no_intercept_csad: 계수 -11.3424, t=-6.69, q=1.34e-10, 통과=True
- holdout_year1 / daily / scsad: 계수 -71.0352, t=-5.69, q=2.55e-08, 통과=True
- holdout_year1 / weekly / no_intercept_csad: 계수 -8.7997, t=-6.06, q=3.98e-09, 통과=True
- holdout_year1 / weekly / scsad: 계수 -31.1860, t=-5.28, q=1.97e-07, 통과=True
- holdout_later / daily / no_intercept_csad: 계수 -11.2826, t=-6.31, q=8.39e-10, 통과=True
- holdout_later / daily / scsad: 계수 -61.4335, t=-8.04, q=5.35e-15, 통과=True
- holdout_later / weekly / no_intercept_csad: 계수 -12.9935, t=-5.29, q=1.81e-07, 통과=True
- holdout_later / weekly / scsad: 계수 -61.5037, t=-5.38, q=1.45e-07, 통과=True

## 해석 제한

- 동일 CMC 공급자를 사용한 시간 외부검증이며 공급자 외부검증은 아닙니다.
- 당일 시총가중 primary는 논문 재현용이며 예측 가능한 투자 가중치가 아닙니다.
- corrected CSAD는 intentional imitation이나 거래 가능한 수익률을 직접 식별하지 않습니다.
- 비선형 원계수는 수익률 변동성 단위에 민감하므로 시기별 강도 비교는 표준화 계수를 우선합니다.
- 하위기간 결과는 진단이며 full holdout 판정을 대체하지 않습니다.

## 그림

- `outputs/v2/cmc_fixed_62/temporal_validation_v1/plots/historical_vs_holdout_coefficients.png`
