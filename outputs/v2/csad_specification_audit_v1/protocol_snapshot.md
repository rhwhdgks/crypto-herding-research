# Corrected CSAD Specification and Mechanism Audit v1 사전등록

## 1. 등록 시점과 목적

- 등록일: 2026-07-20
- 결과 경로: `outputs/v2/csad_specification_audit_v1`
- 이 문서와 `configs/research/csad_specification_audit_v1.yaml`을 새 감사 결과 계산 전에 동결한다.
- 목적은 corrected CSAD의 음의 비선형 계수가 모형, universe, 가중법과 비허딩 자료생성과정에서도 나타나는 기계적 현상인지 구분하는 것이다.
- 이 감사는 미래수익률 alpha, intentional imitation 또는 거래전략을 검정하지 않는다.

## 2. 동결된 empirical 표본

기존 중간 패널을 그대로 읽으며 원자료를 다시 선택하거나 종목을 사후 제외하지 않는다.

1. CMC fixed-62 historical: replication primary와 lagged-cap sensitivity
2. CMC fixed-62 non-overlapping temporal holdout: replication primary와 lagged-cap sensitivity
3. Binance fixed-14: equal weight와 전기 turnover weight
4. OKX listing-aware fixed-14: equal weight와 전기 quote-volume weight
5. Binance archive point-in-time Top-50: equal weight와 전기 quote-volume weight

Primary audit는 각 표본의 기존 전체기간만 사용한다. Stablecoin 이름 추가 제외, 최대 비중 cap, 임의 날짜 삭제 및 결과를 본 뒤의 universe 변경은 금지한다. CMC contemporaneous cap은 논문 직접재현 사양으로만 보존하며 예측 가능한 가중법으로 해석하지 않는다.

## 3. 동결된 모형

모든 회귀는 기존 구현과 동일하게 HAC(Newey-West), 자동 lag를 사용한다.

### Standard CSAD

\[
CSAD_t=\alpha+\beta_1|R_{m,t}|+\beta_2R_{m,t}^2+\epsilon_t
\]

### No-intercept CSAD

\[
CSAD_t=\beta_0R_{m,t}+\beta_1|R_{m,t}|+\beta_2R_{m,t}^2+\epsilon_t
\]

### Intercept-restored mechanical control

\[
CSAD_t=\alpha+\beta_0R_{m,t}+\beta_1|R_{m,t}|+\beta_2R_{m,t}^2+\epsilon_t
\]

이는 no-intercept와 설명변수를 같게 유지하고 절편만 복원한다. Standard CSAD와 no-intercept를 직접 비교해 절편 효과라고 결론 내리지 않는다.

### SCSAD

\[
SCSAD_t=\alpha+\gamma_1R_{m,t}+\gamma_2R_{m,t}^2+\gamma_3R_{m,t}^3+\epsilon_t
\]

여기서 `SCSAD=CSAD` if `Rm>=0`, otherwise `-CSAD`이다. Target은 standard/no-intercept/intercept-restored의 제곱항과 SCSAD의 세제곱항이다. 원계수와 함께 `coefficient * SD(target) / SD(dependent)` 표준화 계수를 보고한다.

## 4. 모형 진단

각 표본·가중법·빈도·모형에 대해 다음을 저장한다.

- 절편과 HAC p-value, no-intercept residual mean
- target 계수, HAC t/p, 95% CI, 표준화 계수
- SSE, RMSE, R-squared, AIC, BIC
- residual mean, skewness, excess kurtosis, Jarque-Bera p-value
- Durbin-Watson, Ljung-Box p-value, Breusch-Pagan p-value
- target VIF, 최대 VIF, standardized design condition number
- no-intercept와 intercept-restored 간 target 변화, SSE 변화 및 절편 복원 여부

`intercept_restored`는 진단용이므로 기존 3모형의 confirmatory family에는 넣지 않는다.

## 5. Leave-one-out market sensitivity

각 시점과 자산 `i`에서 자기 수익률과 자기 가중치를 제외한 시장수익률을 계산한다.

\[
R_{m,-i,t}=\frac{\sum_j w_{j,t}R_{j,t}-w_{i,t}R_{i,t}}{\sum_jw_{j,t}-w_{i,t}}
\]

- `LOO CSAD`는 `mean_i |R_i - R_m,-i|`이다.
- 회귀용 `LOO market`은 같은 시점의 자산별 `R_m,-i` 단순평균이다. CSAD 자체가 자산별 단순평균이므로 동일한 측도를 사용한다.
- 최소 3개 유효 자산이 있어야 계산한다.
- Equal weight에서 LOO market이 full market과 같고 LOO CSAD가 `N/(N-1)`배가 되는 것은 예상되는 수학적 성질이며 독립적 강건성으로 과장하지 않는다.
- Baseline과 LOO는 각각 기존 3모형 6개 셀을 별도 BH family로 처리한다.

## 6. Universe 크기와 가중 집중도

시점별로 유효 자산 수, HHI, 최대 단일 가중치, BTC 가중치를 계산한다. 각 metric은 표본 내부 평균과 표준편차로 표준화한다.

- 각 원모형에 target 비선형항과 metric의 교호항 하나만 추가한다.
- metric별 단일 교호모형으로 적합하여 HHI, BTC 집중도와 universe 크기 사이의 공선성으로부터 해석을 분리한다.
- metric의 표준편차가 `1e-10` 이하이거나 유효 관측이 100개 미만이면 `not_identified`로 보존한다.
- 각 표본·가중법·빈도 내 모든 식별 가능한 교호항을 하나의 BH family로 처리한다.
- 이 분석은 조건부 연관이며 survivor bias의 인과효과를 식별하지 않는다.

## 7. 사전 고정 변동성 regime

현재 관측을 사용하지 않는 시장 변동성 상태를 만든다.

- Daily: 직전 30개 시장수익률의 표준편차, 최소 20개
- Weekly: 직전 8개 시장수익률의 표준편차, 최소 6개
- realized volatility는 한 시점 shift한다.
- Low/high 경계는 이전 volatility 관측만으로 계산한 expanding 1/3 및 2/3 quantile이다.
- Expanding history 최소치는 daily 180개, weekly 52개이다.
- Regime 회귀 최소 관측은 30개이다.
- 각 표본·가중법·빈도의 low/mid/high × 3모형을 하나의 최대 9개 BH family로 처리한다.

## 8. Synthetic null Monte Carlo

모든 DGP에는 intentional herding 또는 자산 간 모방 규칙이 없다. Seed는 `20260720`, 반복은 DGP·시나리오당 300회로 고정한다.

### DGP

1. `independent_gaussian`: 서로 독립인 Gaussian 자산수익률
2. `common_factor`: 하나의 Gaussian 시장요인과 독립 고유충격
3. `heteroskedastic_factor`: 공통 확률변동성 AR(1)이 시장요인과 고유충격 크기만 변화
4. `fat_tail_correlated`: 자유도 5 Student-t 공통요인과 고유충격

### 시나리오

- N=14, T=1,826 daily equal
- N=14, T=261 weekly equal
- N=14, T=1,826 daily lagged lognormal liquidity
- N=14, T=261 weekly lagged lognormal liquidity
- N=50, T=1,826 daily equal
- N=50, T=261 weekly equal
- N=50, T=1,826 daily lagged lognormal liquidity
- N=50, T=261 weekly lagged lognormal liquidity
- N=62, T=2,291 daily contemporaneous evolving size
- N=62, T=328 weekly contemporaneous evolving size
- N=62, T=2,291 daily lagged evolving size
- N=62, T=328 weekly lagged evolving size
- N=62, T=830 holdout daily contemporaneous/lagged evolving size
- N=62, T=118 holdout weekly contemporaneous/lagged evolving size

Lagged weight는 반드시 `t-1`의 상태를 사용한다. Contemporaneous evolving size는 CMC direct-replication weighting의 기계적 민감도를 보기 위한 null이며 예측 사양이 아니다.

### False positive 정의

- Raw false positive: target 계수 `<0`이고 HAC two-sided `p<=0.05`
- BH false positive: 같은 replicate의 기존 3모형 p-value에 BH를 적용한 뒤 계수 `<0`, `q<=0.05`
- DGP·시나리오·모형별 BH false-positive rate가 7.5%를 초과하면 명목 5% 대비 `inflated`, 10% 이상이면 `materially_inflated`로 분류한다.
- Monte Carlo binomial 95% Wilson interval을 함께 보고한다.

## 9. Empirical-vs-null 비교

- 각 empirical 표본은 빈도, 대략적 N, 가중법이 일치하는 사전 고정 시나리오에 매핑한다.
- 각 DGP·모형별 simulation 표준화 target 계수 분포에서 empirical percentile과 음의 단측 Monte Carlo p-value `(1 + count(null<=empirical))/(B+1)`를 계산한다.
- 동일 empirical 표본·가중법 안에서 빈도 × 모형 × DGP를 하나의 BH family로 처리한다.
- Null보다 극단적이라는 사실만으로 intentional herding을 식별하지 않는다.

## 10. 공급자·universe·가중법 이질성

- 표준화 계수의 provider, sample, fixed/listing-aware/point-in-time universe, equal/current/lagged weighting별 기술통계를 보고한다.
- model·frequency별 DerSimonian-Laird descriptive random-effects 요약과 I-squared를 계산한다.
- 각 model별로 표준화 계수를 provider, frequency, universe type, weighting class, 평균 N, 평균 HHI, 평균 BTC weight에 회귀하고 HC3 표준오차를 사용한다.
- 표본이 중첩되고 공급자와 universe가 교락되므로 meta-regression은 설명표이며 독립 연구의 인과 메타분석이 아니다.

## 11. 사전 성공·판정 기준

### 절편 기계성

다음 두 증거를 분리해 보고한다.

1. Empirical: no-intercept에서 음의 BH 지지를 보인 셀 중 절편 복원 후 지지가 사라지거나 표준화 target 절대값이 50% 이상 감소한 비율
2. Null: no-intercept BH false-positive rate가 7.5%를 넘는 DGP·시나리오 수와 intercept-restored 대비 차이

두 축 모두 절반 이상의 비교에서 조건을 만족할 때만 `절편 제약의 실질적 기계성 증거`로 판정한다.

### 구조적 강건성

한 empirical 표본·가중법이 구조적으로 강건하려면 다음을 모두 만족해야 한다.

1. Standard, no-intercept, SCSAD가 daily와 weekly 모두 음의 BH-FDR 지지
2. LOO에서도 같은 6개 셀이 모두 유지
3. 대응 null scenario에서 각 모형의 최악 DGP BH false-positive rate가 7.5% 이하
4. Low/mid/high 중 관측 가능한 regime에서 계수 부호가 서로 반대로 유의하지 않음

이 기준을 통과해도 표현은 `CSAD형 비선형 수렴 관계가 구조적으로 강건함`으로 제한한다.

## 12. 보고 원칙

- 모든 실패, 반대 부호, 식별 불가와 결측 진단을 숨기지 않는다.
- 기존 결과를 덮어쓰지 않는다.
- 새 threshold, 종목 제외, 가중 cap 또는 DGP 조정은 결과 확인 후 허용하지 않는다.
- 오류 수정이 필요하면 amendment 파일에 시각, 이유, 영향 범위를 남기고 구 결과도 보존한다.
- 최종 보고서는 기술 독자가 아닌 사람도 이해할 수 있는 한국어로 작성한다.
