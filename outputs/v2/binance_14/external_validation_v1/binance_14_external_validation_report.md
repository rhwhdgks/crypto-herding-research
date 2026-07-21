# Binance 14종목 공급자·유니버스 외부 강건성 검증

## 결론

- 동일가중 primary corrected 4개 셀: 3/4 통과
- 전기 거래대금가중 sensitivity corrected 4개 셀: 3/4 통과
- 사전 고정 strict criterion: 미통과, 공급자·universe 외부 재현은 부분적
- 이 검정은 corrected CSAD 관계의 공급자·universe 강건성을 다루며 미래수익률 alpha 검정이 아닙니다.
- 기존에 연구한 Binance 패널이므로 결과는 secondary external robustness로 분류합니다.

## 표본과 품질

- 분석 기간: 2021-04-10~2026-04-09
- 고정 universe: 14개 Binance USDT 현물 페어
- 전체 asset-day coverage: 100.00%
- quality gate: 8/8 통과
- equal_weight_primary: daily 1,826개, weekly 261개
- lagged_turnover_sensitivity: daily 1,826개, weekly 261개

## Strict criterion 미통과 셀

- equal_weight_primary / weekly / scsad: 계수 -0.4042, t=-1.61, q=0.108
- lagged_turnover_sensitivity / daily / scsad: 계수 -3.0244, t=-1.95, q=0.076
- no-intercept는 두 가중법·두 빈도에서 모두 통과했지만 SCSAD는 가중법과 빈도에 따라 한 셀씩 탈락했습니다.

## 5년 전체표본 회귀

- equal_weight_primary / daily / standard_csad: 계수 -0.1765, 표준화 -0.086, t=-2.65, q=0.0121, 통과=True
- equal_weight_primary / daily / no_intercept_csad: 계수 -1.2587, 표준화 -0.612, t=-3.95, q=0.00023, 통과=True
- equal_weight_primary / daily / scsad: 계수 -2.0315, 표준화 -0.166, t=-3.13, q=0.00346, 통과=True
- equal_weight_primary / weekly / standard_csad: 계수 -0.1994, 표준화 -0.206, t=-2.51, q=0.0145, 통과=True
- equal_weight_primary / weekly / no_intercept_csad: 계수 -0.6725, 표준화 -0.694, t=-6.84, q=4.69e-11, 통과=True
- equal_weight_primary / weekly / scsad: 계수 -0.4042, 표준화 -0.129, t=-1.61, q=0.108, 통과=False
- lagged_turnover_sensitivity / daily / standard_csad: 계수 0.3973, 표준화 0.115, t=1.41, q=0.191, 통과=False
- lagged_turnover_sensitivity / daily / no_intercept_csad: 계수 -1.6506, 표준화 -0.477, t=-3.13, q=0.00351, 통과=True
- lagged_turnover_sensitivity / daily / scsad: 계수 -3.0244, 표준화 -0.118, t=-1.95, q=0.076, 통과=False
- lagged_turnover_sensitivity / weekly / standard_csad: 계수 0.0040, 표준화 0.003, t=0.04, q=0.966, 통과=False
- lagged_turnover_sensitivity / weekly / no_intercept_csad: 계수 -1.0064, 표준화 -0.689, t=-4.47, q=4.63e-05, 통과=True
- lagged_turnover_sensitivity / weekly / scsad: 계수 -1.4578, 표준화 -0.236, t=-3.16, q=0.00351, 통과=True

## 기간 진단

- cmc_historical_overlap / equal_weight_primary / daily / no_intercept_csad: 표준화 -0.694, t=-3.86, q=0.000224, 통과=True
- cmc_historical_overlap / equal_weight_primary / daily / scsad: 표준화 -0.237, t=-3.94, q=0.000224, 통과=True
- cmc_historical_overlap / equal_weight_primary / weekly / no_intercept_csad: 표준화 -0.813, t=-8.55, q=7.27e-17, 통과=True
- cmc_historical_overlap / equal_weight_primary / weekly / scsad: 표준화 -0.164, t=-1.16, q=0.247, 통과=False
- cmc_holdout_overlap / equal_weight_primary / daily / no_intercept_csad: 표준화 -0.544, t=-2.99, q=0.00563, 통과=True
- cmc_holdout_overlap / equal_weight_primary / daily / scsad: 표준화 -0.085, t=-1.24, q=0.217, 통과=False
- cmc_holdout_overlap / equal_weight_primary / weekly / no_intercept_csad: 표준화 -1.001, t=-5.81, q=3.64e-08, 통과=True
- cmc_holdout_overlap / equal_weight_primary / weekly / scsad: 표준화 -0.241, t=-4.38, q=3.52e-05, 통과=True
- cmc_historical_overlap / lagged_turnover_sensitivity / daily / no_intercept_csad: 표준화 -0.473, t=-3.11, q=0.00371, 통과=True
- cmc_historical_overlap / lagged_turnover_sensitivity / daily / scsad: 표준화 -0.124, t=-1.65, q=0.15, 통과=False
- cmc_historical_overlap / lagged_turnover_sensitivity / weekly / no_intercept_csad: 표준화 -0.830, t=-5.54, q=1.78e-07, 통과=True
- cmc_historical_overlap / lagged_turnover_sensitivity / weekly / scsad: 표준화 -0.375, t=-3.37, q=0.00223, 통과=True
- cmc_holdout_overlap / lagged_turnover_sensitivity / daily / no_intercept_csad: 표준화 -0.776, t=-5.01, q=3.26e-06, 통과=True
- cmc_holdout_overlap / lagged_turnover_sensitivity / daily / scsad: 표준화 -0.241, t=-3.77, q=0.000293, 통과=True
- cmc_holdout_overlap / lagged_turnover_sensitivity / weekly / no_intercept_csad: 표준화 -1.095, t=-4.48, q=2.21e-05, 통과=True
- cmc_holdout_overlap / lagged_turnover_sensitivity / weekly / scsad: 표준화 -0.264, t=-3.73, q=0.000293, 통과=True

## CMC와 표준화 계수 비교

- cmc_2018_2024 / daily / no_intercept_csad: 표준화 계수 -0.700, t=-3.56, q=0.000564
- cmc_2018_2024 / daily / scsad: 표준화 계수 -0.204, t=-4.62, q=1.15e-05
- cmc_2018_2024 / weekly / no_intercept_csad: 표준화 계수 -1.181, t=-5.90, q=2.24e-08
- cmc_2018_2024 / weekly / scsad: 표준화 계수 -0.307, t=-3.63, q=0.000564
- cmc_2024_2026 / daily / no_intercept_csad: 표준화 계수 -1.303, t=-7.15, q=1.71e-12
- cmc_2024_2026 / daily / scsad: 표준화 계수 -0.345, t=-7.23, q=1.5e-12
- cmc_2024_2026 / weekly / no_intercept_csad: 표준화 계수 -2.163, t=-7.63, q=1.38e-13
- cmc_2024_2026 / weekly / scsad: 표준화 계수 -0.503, t=-5.91, q=5.12e-09
- binance_2021_2026 / daily / no_intercept_csad: 표준화 계수 -0.612, t=-3.95, q=0.00023
- binance_2021_2026 / daily / scsad: 표준화 계수 -0.166, t=-3.13, q=0.00346
- binance_2021_2026 / weekly / no_intercept_csad: 표준화 계수 -0.694, t=-6.84, q=4.69e-11
- binance_2021_2026 / weekly / scsad: 표준화 계수 -0.129, t=-1.61, q=0.108

## 해석 제한

- Binance 거래소 가격과 14개 USDT 페어에 조건부인 결과입니다.
- 고정 14종목은 survivor·listing selection bias를 제거하지 못합니다.
- 동일가중 시장수익률은 CMC 시총가중 사양과 직접 동일하지 않습니다.
- 거래대금 sensitivity는 close×base volume 근사치를 사용하며 전기 값만 사용합니다.
- corrected CSAD를 intentional imitation, 인과효과 또는 거래 가능한 alpha로 해석하지 않습니다.

## 그림

- `outputs/v2/binance_14/external_validation_v1/plots/cross_provider_standardized_coefficients.png`
