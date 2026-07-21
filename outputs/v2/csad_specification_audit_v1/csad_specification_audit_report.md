# Corrected CSAD Specification and Mechanism Audit v1

## 한눈에 보는 결론

- 사전등록한 구조적 강건성 기준을 모두 통과한 10개 empirical 사양은 **0개**입니다.
- 절편 제약의 실질적 기계성 판정은 **True**입니다. Empirical 변화 비율은 100.0%, null 변화 비율은 100.0%입니다.
- 4개 비허딩 DGP × 고정 시나리오에서 사전 허용치 7.5%를 넘은 기존 모형 false-positive 셀은 **136개/192개**입니다.
- Empirical 계수가 대응 simulation null보다 더 음수라고 BH-FDR로 판정된 비교는 **0개/240개**입니다.
- 이 결과는 동시적 CSAD형 비선형 관계의 사양 민감도 감사이며, intentional imitation이나 미래수익률 alpha의 증거가 아닙니다.

## 무엇을 감사했나

- CMC fixed-62 historical·holdout, Binance fixed-14, OKX listing-aware 14, Binance archive point-in-time Top-50의 기존 중간 패널을 통합했습니다.
- 각 표본의 equal/current/lagged weighting과 daily·weekly를 동일한 코드로 재계산했습니다.
- Standard, no-intercept, SCSAD와 함께 no-intercept의 설명변수에 절편만 복원한 기계성 control을 적합했습니다.
- 자기 자신을 시장수익률에서 제외한 leave-one-out, 과거정보 전용 volatility regime, N·HHI·BTC weight 교호항을 검사했습니다.
- 독립 Gaussian, 공통요인, 이분산 공통요인, fat-tail 상관 DGP에는 자산 간 모방 규칙을 넣지 않았습니다.

## 입력 무결성

- 읽은 패널: 20개, 저장 market/CSAD와 재계산값이 허용오차 내 일치한 패널: 20/20개
- protocol SHA-256: `0fdf5de62c408d1cc0d0bf820e6d0b8c2e8c4fe42fbab45bdaab0c587b1617a5`
- config SHA-256: `9f7f39a2054b626d0c0852c62e621fd8eb6e2d01ef1b1398e6de7f1768a16b4c`

## 절편 제거의 영향

- Baseline에서 음의 BH 지지를 보인 no-intercept 셀은 16개였고, 이 중 지지가 사라지거나 표준화 절대크기가 50% 이상 줄어든 셀은 16개였습니다.
- 비허딩 null의 DGP·시나리오 64개 중 사전 기계성 조건을 만족한 셀은 64개였습니다.
- No-intercept의 음의 계수 전체를 절편 하나로 설명할 수 있는지와, 일부 사양에서 절편 제약이 계수를 증폭하는지는 구분해서 봐야 합니다.

## 구조적 강건성 판정

- `binance_fixed14_equal`: baseline 5/6, LOO 5/6, 최악 null FPR 100.0%, 반대 regime 1개, 최종 통과=False
- `binance_fixed14_lagged`: baseline 3/6, LOO 3/6, 최악 null FPR 100.0%, 반대 regime 0개, 최종 통과=False
- `binance_pit50_equal`: baseline 2/6, LOO 2/6, 최악 null FPR 100.0%, 반대 regime 1개, 최종 통과=False
- `binance_pit50_lagged`: baseline 0/6, LOO 0/6, 최악 null FPR 100.0%, 반대 regime 3개, 최종 통과=False
- `cmc_fixed62_historical_current`: baseline 4/6, LOO 4/6, 최악 null FPR 100.0%, 반대 regime 1개, 최종 통과=False
- `cmc_fixed62_historical_lagged`: baseline 4/6, LOO 4/6, 최악 null FPR 100.0%, 반대 regime 1개, 최종 통과=False
- `cmc_fixed62_holdout_current`: baseline 4/6, LOO 4/6, 최악 null FPR 100.0%, 반대 regime 0개, 최종 통과=False
- `cmc_fixed62_holdout_lagged`: baseline 4/6, LOO 4/6, 최악 null FPR 100.0%, 반대 regime 0개, 최종 통과=False
- `okx_listing14_equal`: baseline 5/6, LOO 5/6, 최악 null FPR 100.0%, 반대 regime 2개, 최종 통과=False
- `okx_listing14_lagged`: baseline 1/6, LOO 1/6, 최악 null FPR 100.0%, 반대 regime 0개, 최종 통과=False

## Synthetic null false positive

- Standard CSAD: 최댓값 11.3% (heteroskedastic_factor, n14_daily_equal), 허용치 초과 8/64개
- No-intercept CSAD: 최댓값 100.0% (independent_gaussian, n14_daily_equal), 허용치 초과 64/64개
- SCSAD: 최댓값 100.0% (independent_gaussian, n14_daily_equal), 허용치 초과 64/64개

## 조건부 분석

- N·HHI·BTC weight 교호항 중 식별 가능: 156개, BH-FDR 통과: 52개
- 사전 고정 volatility regime 회귀 중 추정 가능: 168개, 유의한 반대 양의 비선형항: 9개
- 고정 14·62 universe에서 N 변화가 거의 없는 경우는 억지로 추정하지 않고 `not_identified`로 남겼습니다.

## 이질성

- No-intercept CSAD / daily: 표준화 random-effects -0.613, I-squared 96.9%
- No-intercept CSAD / weekly: 표준화 random-effects -0.969, I-squared 87.0%
- SCSAD / daily: 표준화 random-effects -0.176, I-squared 81.2%
- SCSAD / weekly: 표준화 random-effects -0.245, I-squared 73.1%
- Standard CSAD / daily: 표준화 random-effects 0.093, I-squared 95.9%
- Standard CSAD / weekly: 표준화 random-effects -0.027, I-squared 69.3%
- 표본이 중첩되고 provider·universe·가중법이 서로 교락되어 있으므로 위 수치는 독립 연구 메타분석이나 인과효과가 아닙니다.

## 해석 한계

- Null DGP는 알려진 네 가지 기계성을 분리하는 반사실적이며 실제 암호화폐 시장 전체를 완전하게 생성하지 않습니다.
- CMC contemporaneous cap weighting은 논문 재현용이고 미래 시점에 실행 가능한 weighting이 아닙니다.
- Survivor bias는 fixed와 point-in-time 표본 차이에 섞여 있으나 provider 차이도 동시에 존재해 단독 인과효과를 식별하지 못합니다.
- 음의 계수가 null보다 극단적이어도 누락 공통요인, 가격결정 구조와 microstructure를 배제하지 못합니다.
- 이 감사에는 미래수익률, 비용, portfolio backtest가 없으므로 자동매매 근거로 사용할 수 없습니다.

## 재현 산출물

- `plots/false_positive_rates.png`
- `plots/intercept_mechanics.png`
- `plots/empirical_vs_null.png`
- `plots/descriptive_random_effects.png`
- 주요 CSV와 simulation parquet, input manifest, config/protocol snapshot은 같은 출력 폴더에 보존했습니다.

## 최종 연구 판단

이 감사의 판정은 사전등록 기준을 기계적으로 적용한 것입니다. 통과 셀이 있더라도 표현은 `CSAD형 비선형 수렴 관계`로 제한하며, 통과하지 못한 경우에는 같은 표본에서 threshold나 종목을 다시 최적화하지 않습니다.
