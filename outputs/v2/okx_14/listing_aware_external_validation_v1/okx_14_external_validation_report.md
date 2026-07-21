# OKX 14종목 상장 인지형 외부검증

## 결론

- 동일가중 primary corrected 4개 셀: 3/4 통과
- 전기 quote-volume sensitivity corrected 4개 셀: 1/4 통과
- 사전 고정 primary strict criterion: 미통과
- OKX 동일가중 표준화 계수는 Binance와 매우 근접해 weekly SCSAD 미통과가 두 거래소에서 반복됐습니다.
- Quote-volume sensitivity는 full sample 1/4이지만 late half 4/4여서 가중법·기간 불안정성이 큽니다.
- 미관찰 거래소 회귀 결과이지만 corrected CSAD는 미래수익률 alpha나 intentional imitation의 직접 검정이 아닙니다.

## 표본과 품질

- 분석 기간: 2021-04-10~2026-04-09
- candidate universe: 14개 OKX USDT 현물, listing-aware active panel
- 자산별 최저 coverage: 100.00%
- quality gate: 10/10 통과
- equal_weight_primary: daily 1,826개, weekly 261개, 활성 종목 13~14개
- lagged_quote_volume_sensitivity: daily 1,826개, weekly 261개, 활성 종목 13~14개

## Strict criterion 미통과 셀

- equal_weight_primary / weekly / scsad: 계수 -0.4318, t=-1.89, q=0.0592
- lagged_quote_volume_sensitivity / daily / no_intercept_csad: 계수 -0.4835, t=-0.56, q=0.733
- lagged_quote_volume_sensitivity / daily / scsad: 계수 0.7992, t=0.51, q=0.733
- lagged_quote_volume_sensitivity / weekly / scsad: 계수 -0.9383, t=-2.31, q=0.062

## 5년 전체표본 회귀

- equal_weight_primary / daily / standard_csad: 계수 -0.1498, 표준화 -0.071, t=-2.18, q=0.0397, 통과=True
- equal_weight_primary / daily / no_intercept_csad: 계수 -1.2505, 표준화 -0.593, t=-3.84, q=0.000376, 통과=True
- equal_weight_primary / daily / scsad: 계수 -1.9845, 표준화 -0.158, t=-2.98, q=0.00585, 통과=True
- equal_weight_primary / weekly / standard_csad: 계수 -0.1915, 표준화 -0.190, t=-2.13, q=0.0397, 통과=True
- equal_weight_primary / weekly / no_intercept_csad: 계수 -0.6683, 표준화 -0.664, t=-6.85, q=4.35e-11, 통과=True
- equal_weight_primary / weekly / scsad: 계수 -0.4318, 표준화 -0.131, t=-1.89, q=0.0592, 통과=False
- lagged_quote_volume_sensitivity / daily / standard_csad: 계수 1.2455, 표준화 0.368, t=1.20, q=0.462, 통과=False
- lagged_quote_volume_sensitivity / daily / no_intercept_csad: 계수 -0.4835, 표준화 -0.143, t=-0.56, q=0.733, 통과=False
- lagged_quote_volume_sensitivity / daily / scsad: 계수 0.7992, 표준화 0.038, t=0.51, q=0.733, 통과=False
- lagged_quote_volume_sensitivity / weekly / standard_csad: 계수 -0.0376, 표준화 -0.026, t=-0.24, q=0.809, 통과=False
- lagged_quote_volume_sensitivity / weekly / no_intercept_csad: 계수 -0.8858, 표준화 -0.614, t=-6.19, q=3.71e-09, 통과=True
- lagged_quote_volume_sensitivity / weekly / scsad: 계수 -0.9383, 표준화 -0.167, t=-2.31, q=0.062, 통과=False

## 기간 진단

- early_half / equal_weight_primary / daily / no_intercept_csad: 표준화 -0.657, t=-3.84, q=0.000366, 통과=True
- early_half / equal_weight_primary / daily / scsad: 표준화 -0.231, t=-3.66, q=0.000507, 통과=True
- early_half / equal_weight_primary / weekly / no_intercept_csad: 표준화 -0.790, t=-8.89, q=3.72e-18, 통과=True
- early_half / equal_weight_primary / weekly / scsad: 표준화 -0.174, t=-1.36, q=0.174, 통과=False
- late_half / equal_weight_primary / daily / no_intercept_csad: 표준화 -0.622, t=-3.16, q=0.00313, 통과=True
- late_half / equal_weight_primary / daily / scsad: 표준화 -0.117, t=-1.63, q=0.103, 통과=False
- late_half / equal_weight_primary / weekly / no_intercept_csad: 표준화 -0.971, t=-5.67, q=8.38e-08, 통과=True
- late_half / equal_weight_primary / weekly / scsad: 표준화 -0.261, t=-4.30, q=5.17e-05, 통과=True
- early_half / lagged_quote_volume_sensitivity / daily / no_intercept_csad: 표준화 -0.063, t=-0.24, q=0.813, 통과=False
- early_half / lagged_quote_volume_sensitivity / daily / scsad: 표준화 0.071, t=0.94, q=0.485, 통과=False
- early_half / lagged_quote_volume_sensitivity / weekly / no_intercept_csad: 표준화 -0.656, t=-8.28, q=7.16e-16, 통과=True
- early_half / lagged_quote_volume_sensitivity / weekly / scsad: 표준화 -0.211, t=-1.76, q=0.234, 통과=False
- late_half / lagged_quote_volume_sensitivity / daily / no_intercept_csad: 표준화 -0.883, t=-5.66, q=9.04e-08, 통과=True
- late_half / lagged_quote_volume_sensitivity / daily / scsad: 표준화 -0.263, t=-4.79, q=5.06e-06, 통과=True
- late_half / lagged_quote_volume_sensitivity / weekly / no_intercept_csad: 표준화 -1.195, t=-4.57, q=9.53e-06, 통과=True
- late_half / lagged_quote_volume_sensitivity / weekly / scsad: 표준화 -0.304, t=-4.29, q=2.72e-05, 통과=True

## 공급자별 표준화 계수

- cmc_2018_2024 / daily / no_intercept_csad: 표준화 -0.700, t=-3.56, q=0.000564
- cmc_2018_2024 / daily / scsad: 표준화 -0.204, t=-4.62, q=1.15e-05
- cmc_2018_2024 / weekly / no_intercept_csad: 표준화 -1.181, t=-5.90, q=2.24e-08
- cmc_2018_2024 / weekly / scsad: 표준화 -0.307, t=-3.63, q=0.000564
- cmc_2024_2026 / daily / no_intercept_csad: 표준화 -1.303, t=-7.15, q=1.71e-12
- cmc_2024_2026 / daily / scsad: 표준화 -0.345, t=-7.23, q=1.5e-12
- cmc_2024_2026 / weekly / no_intercept_csad: 표준화 -2.163, t=-7.63, q=1.38e-13
- cmc_2024_2026 / weekly / scsad: 표준화 -0.503, t=-5.91, q=5.12e-09
- binance_2021_2026 / daily / no_intercept_csad: 표준화 -0.612, t=-3.95, q=0.00023
- binance_2021_2026 / daily / scsad: 표준화 -0.166, t=-3.13, q=0.00346
- binance_2021_2026 / weekly / no_intercept_csad: 표준화 -0.694, t=-6.84, q=4.69e-11
- binance_2021_2026 / weekly / scsad: 표준화 -0.129, t=-1.61, q=0.108
- okx_2021_2026 / daily / no_intercept_csad: 표준화 -0.593, t=-3.84, q=0.000376
- okx_2021_2026 / daily / scsad: 표준화 -0.158, t=-2.98, q=0.00585
- okx_2021_2026 / weekly / no_intercept_csad: 표준화 -0.664, t=-6.85, q=4.35e-11
- okx_2021_2026 / weekly / scsad: 표준화 -0.131, t=-1.89, q=0.0592

## 해석 제한

- 현재 live instrument에서 고른 후보군이므로 상장폐지 자산 survivor bias가 남습니다.
- Listing-aware 편입은 시가총액 기반 point-in-time 동적 universe와 동일하지 않습니다.
- OKX 단일 거래소·USDT 현물과 선택한 14개 후보에 조건부인 결과입니다.
- 전기 quote volume은 시점 안전하지만 시장수익률의 유일한 경제적 가중법은 아닙니다.
- 동시적 CSAD 관계를 미래수익률 alpha, 인과효과 또는 의도적 모방으로 확대하지 않습니다.

## 그림

- `outputs/v2/okx_14/listing_aware_external_validation_v1/plots/cross_provider_standardized_coefficients.png`
