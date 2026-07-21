# CMC 고정 62종목 선행논문 재현

## 결론

- primary 6개 사양 중 4개가 통과해 결과는 모형 민감적입니다.
- no-look-ahead sensitivity는 6개 중 4개가 통과했습니다.
- 이 결과는 동시적 횡단면 수렴 관계이며 미래수익률 alpha가 아닙니다.

## 데이터 품질

- 고정 universe: 62개 CMC legacy ID
- 자산별 최저 coverage: 83.90%
- quality gate: 6/6 통과
- replication_primary: daily 2,291개, weekly 328개
- no_lookahead_sensitivity: daily 2,291개, weekly 328개

## 전체표본 Target

- replication_primary / daily / standard_csad: 계수 -0.0579, t=-0.40, q=0.686, n=2291, 통과=False
- replication_primary / daily / no_intercept_csad: 계수 -1.8373, t=-3.56, q=0.000564, n=2291, 통과=True
- replication_primary / daily / scsad: 계수 -2.9023, t=-4.62, q=1.15e-05, n=2291, 통과=True
- replication_primary / weekly / standard_csad: 계수 -0.1257, t=-0.48, q=0.686, n=328, 통과=False
- replication_primary / weekly / no_intercept_csad: 계수 -2.0787, t=-5.90, q=2.24e-08, n=328, 통과=True
- replication_primary / weekly / scsad: 계수 -3.3227, t=-3.63, q=0.000564, n=328, 통과=True
- no_lookahead_sensitivity / daily / standard_csad: 계수 -0.0527, t=-0.39, q=0.7, n=2291, 통과=False
- no_lookahead_sensitivity / daily / no_intercept_csad: 계수 -1.8259, t=-3.56, q=0.000747, n=2291, 통과=True
- no_lookahead_sensitivity / daily / scsad: 계수 -2.6768, t=-4.21, q=7.67e-05, n=2291, 통과=True
- no_lookahead_sensitivity / weekly / standard_csad: 계수 -0.2304, t=-1.09, q=0.329, n=328, 통과=False
- no_lookahead_sensitivity / weekly / no_intercept_csad: 계수 -2.0510, t=-5.75, q=5.3e-08, n=328, 통과=True
- no_lookahead_sensitivity / weekly / scsad: 계수 -3.0660, t=-3.29, q=0.00152, n=328, 통과=True

## 논문 Benchmark 비교

- full_sample / daily / no_intercept_csad: 논문 -1.850, 우리 -1.837, 차이 +0.013, 관측 수 차이 +0
- pre_covid / daily / no_intercept_csad: 논문 -4.634, 우리 -4.593, 차이 +0.041, 관측 수 차이 +0
- covid / daily / no_intercept_csad: 논문 -1.757, 우리 -1.753, 차이 +0.004, 관측 수 차이 +0
- post_covid / daily / no_intercept_csad: 논문 -6.543, 우리 -6.542, 차이 +0.001, 관측 수 차이 +0
- full_sample / daily / scsad: 논문 -2.924, 우리 -2.902, 차이 +0.022, 관측 수 차이 +0
- pre_covid / daily / scsad: 논문 -14.892, 우리 -14.698, 차이 +0.194, 관측 수 차이 +0
- covid / daily / scsad: 논문 -4.807, 우리 -4.801, 차이 +0.006, 관측 수 차이 +0
- post_covid / daily / scsad: 논문 -30.813, 우리 -30.816, 차이 -0.003, 관측 수 차이 +0
- full_sample / weekly / standard_csad: 논문 -0.115, 우리 -0.126, 차이 -0.011, 관측 수 차이 +21
- full_sample / weekly / no_intercept_csad: 논문 -2.096, 우리 -2.079, 차이 +0.017, 관측 수 차이 +21
- full_sample / weekly / scsad: 논문 -3.359, 우리 -3.323, 차이 +0.036, 관측 수 차이 +21

## 2×2 방법 감사

- log_contemporaneous: benchmark 5개 평균 절대 계수차 0.0197, 평균 절대 t-stat 차이 0.054
- log_lagged: benchmark 5개 평균 절대 계수차 0.1449, 평균 절대 t-stat 차이 0.297
- simple_lagged: benchmark 5개 평균 절대 계수차 1.0787, 평균 절대 t-stat 차이 1.150
- simple_contemporaneous: benchmark 5개 평균 절대 계수차 1.2363, 평균 절대 t-stat 차이 1.444
- Table 1 기술통계와 위 방법 감사를 함께 근거로 log_contemporaneous를 direct-replication primary로 교정했습니다.

## 해석 제한

- 논문의 정확한 시점별 $100M 선정 규칙과 weekly 307개 생성 규칙은 공개되지 않았습니다.
- primary 당일 시총가중은 논문 직접 재현용이며 예측 가능한 가중치가 아닙니다.
- 고정 62종목은 사후 생존·선정 편향 가능성이 있어 투자 universe로 사용하지 않습니다.
- CMC 무료 웹 endpoint는 공식 Pro API 계약 endpoint가 아니므로 raw checkpoint와 hash를 보존했습니다.

## 그림

- `outputs/v2/cmc_fixed_62/replication_v1/plots/daily_active_assets.png`
- `outputs/v2/cmc_fixed_62/replication_v1/plots/target_coefficients.png`
