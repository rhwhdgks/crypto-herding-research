# CMC Dynamic Universe Replication Protocol v1

- 동결일: 2026-07-19
- 연구 성격: CoinMarketCap point-in-time 동적 universe를 이용한 선행논문 재현·확장
- 이 프로토콜 동결 전에 알고 있는 결과: Binance 14종목 fixed-universe standard CSAD는 음의 곡률을 지지하지 않았고, 과거 Binance paper-like 근사는 일부 corrected model에서 음의 계수를 보였다.
- 이 프로토콜 동결 전에 알지 못하는 결과: CMC point-in-time monthly universe로 구성한 daily·weekly Standard, no-intercept, SCSAD 계수.

## 연구 질문

Binance 14종목 fixed universe에서 보이지 않았던 classical herding이 CoinMarketCap 시가총액 기반 point-in-time 동적 universe와 Wang-Hudson corrected CSAD specification에서는 나타나는지 검증한다.

## 선행연구 정렬

- Natashekara and Sampath (2026): 2018-01-01부터 2024-04-09까지 CMC daily·weekly, market-cap weighted market return, Standard/no-intercept/SCSAD 비교.
- Wang and Hudson (2024): standard CCK CSAD test의 asset-pricing misspecification 편향을 지적하고 no-intercept와 SCSAD 등 corrected test를 제안.
- Jeon et al. (2026 working paper): monthly-rebalanced Top-200 market-cap universe와 structural break를 이용.

이 연구는 Natashekara and Sampath의 기간·CMC·corrected specification과 Jeon et al.의 monthly dynamic universe를 결합한 확장 재현이다. 원논문의 fixed 62-coin sample을 완전히 복제한 것으로 표현하지 않는다.

## 데이터

- 소스: CoinMarketCap historical snapshot 페이지가 사용하는 public JSON endpoint
- URL: `https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listings/historical`
- 호출 단위: UTC calendar day, USD quote, rank 1-500
- cache 기간: 2017-12-31부터 2024-04-09까지 포함
- 분석 수익률 기간: 2018-01-01부터 2024-04-09까지 포함
- 저장: 날짜별 Parquet checkpoint, 파일 SHA-256 manifest, 중단 후 재개
- CMC ID를 자산 기본키로 사용하며 symbol 중복을 합치지 않는다.

### Source-discovered deviation (2026-07-19, regression 실행 전 동결)

CMC public historical endpoint는 일부 날짜에 `limit=500`을 요청해도 이를 제거하고 더 낮은 행 수를 반환한다. 원응답 검증에서 2021-09-28은 318행, 일부 2023-2024 날짜는 497-499행이었고 누락 rank는 다른 rank로 보충되지 않았다. 연구자가 임의로 행을 채우지 않고 API가 반환한 관측만 보존한다.

- 요청 범위는 여전히 rank 1-500, USD로 고정한다.
- checkpoint 최소 행 수 gate를 300으로 변경하고 날짜별 행 수·누락 rank 수를 manifest에 저장한다.
- USD price·market cap 양수 행 95% gate와 분석일 membership coverage 80% gate는 그대로 유지한다.
- 2021-09-28은 전월 말 formation date가 아니므로 월별 universe 선정에 직접 사용되지 않는다.
- 이 deviation은 회귀 계수를 실행·관찰하기 전에 데이터 소스 검증으로 결정했다.

Public web endpoint는 정식 Pro API 계약 endpoint가 아니므로 향후 변경될 수 있다. 다만 연구 재현성을 위해 검증된 checkpoint와 hash를 보존한다.

## Point-in-time Universe

각 calendar month M의 구성종목은 M 직전 calendar month-end snapshot으로 결정한다.

1. 직전 월말 rank 상위 200개를 후보로 선택한다.
2. 형성 시점 시가총액 USD 100 million 미만을 제외한다.
3. CMC current metadata에 `stablecoin`, `wrapped-tokens`, `liquid-staking-derivatives` tag가 있는 자산을 제외한다.
4. Metadata가 없는 자산은 slug·name 고정 규칙과 형성일 이전 30일 USD price를 이용한 peg 규칙을 적용한다.
5. Peg 규칙은 30일 중 20일 이상 관측, 중앙값 0.90-1.10 USD, 최대·최소 비율 1.10 이하로 고정한다.
6. 구성종목은 한 달 동안 고정하고, 중간 순위 변화로 교체하지 않는다.

Wrapped·staked derivative 제외는 동일 기초자산의 기계적 중복을 herding으로 오해하는 것을 막기 위한다.

## Return Panel

- CMC daily USD snapshot price의 log return을 사용한다.
- 일별 자산 return은 같은 CMC ID의 t-1, t price가 모두 있을 때만 계산한다.
- 해당 월 point-in-time membership 밖의 자산은 NaN으로 둔다.
- 해당 날 활성 자산이 20개 미만이거나 월 구성종목의 80%보다 적으면 그 날을 분석에서 제외한다.
- 상위 500 일별 snapshot에서 누락된 구성종목을 0으로 전체하지 않는다.

## Market Return and CSAD

일별 시장수익률은 t-1 시가총액 가중을 사용한다.

```text
R_m,t = sum_i(w_i,t-1 * R_i,t) / sum_i(w_i,t-1)
CSAD_t = mean_i(|R_i,t - R_m,t|)
```

CSAD는 논문과 같이 활성 자산의 단순 평균 절대편차를 사용한다.

## Frequency

- Daily: 원본 일별 log return
- Weekly: UTC Monday label, 일별 log return의 비중첩 7일 합계
- Weekly membership는 해당 daily panel에 이미 point-in-time으로 적용된 membership을 유지한다.
- 월 경계를 가로지르는 week에서도 각 daily return은 그 날의 사전 고정 membership만 사용한다.

## Regression Specifications

### Standard CSAD

```text
CSAD_t = alpha + beta1 * |R_m,t| + beta2 * R_m,t^2 + epsilon_t
```

Target: `beta2 < 0`.

### No-intercept corrected CSAD

```text
CSAD_t = gamma1 * R_m,t + gamma2 * |R_m,t| + gamma3 * R_m,t^2 + epsilon_t
```

Target: `gamma3 < 0`.

### Symmetrical CSAD

```text
SCSAD_t = CSAD_t  if R_m,t >= 0
SCSAD_t = -CSAD_t if R_m,t < 0

SCSAD_t = alpha + delta1 * R_m,t + delta2 * R_m,t^2
          + delta3 * R_m,t^3 + epsilon_t
```

Target: `delta3 < 0`.

모든 회귀는 Newey-West HAC covariance를 사용하고 lag는 기존 프로젝트 automatic rule로 결정한다.

## Confirmatory Family

- Full-sample daily 3 target terms + full-sample weekly 3 target terms = 6 hypotheses
- 6개 two-sided p-value를 하나의 Benjamini-Hochberg FDR family로 보정한다.
- `q <= 0.05`이고 target coefficient가 음수일 때만 corrected herding support로 표시한다.
- Standard, no-intercept, SCSAD가 모두 통과해야만 강한 모형 간 일치로 표현한다.
- 일부 specification만 통과하면 method-sensitive evidence로 표현한다.

## Subperiod Diagnostics

- pre-COVID: 2018-01-01 - 2020-03-10
- COVID: 2020-03-11 - 2022-02-23
- post-COVID: 2022-02-24 - 2024-04-09
- 각 subperiod에서도 daily·weekly 6개 target p-value를 독립 BH family로 보정한다.
- Subperiod 결과는 full-sample primary를 대체하지 않는다.

## Data Quality Gates

- 요청 calendar day 중 snapshot cache 완료율 100%
- 각 snapshot 최소 300 rows, CMC ID·rank 중복 0건; 날짜별 실제 행 수·누락 rank 수 manifest 보고
- USD price·market cap 양수 행 95% 이상
- 월별 universe 최소 20자산
- 분석 기간 daily return 적격일 95% 이상
- 일별 동일 ID 중복 0건
- 요청 기간, active count, coverage failure day를 보고서에 명시

품질 gate를 통과하지 못하면 회귀 결과를 생성하지 않는다. Gate를 낮추는 것은 프로토콜 deviation으로 기록한다.

## Outputs

- 수집 상태·checkpoint manifest
- 날짜별 normalized CMC snapshot
- 자산 metadata·제외 사유
- 월별 point-in-time membership·형성 시가총액
- daily·weekly return panel, market return, CSAD/SCSAD
- full/subperiod regression coefficients·diagnostics·BH decisions
- universe size·turnover·coverage plots
- CMC dynamic universe 재현 보고서
- config·protocol snapshot, input manifest, provenance

## 해석 제약

- CMC historical snapshot 수정 가능성과 current metadata tag의 후행적 분류 한계를 보고한다.
- Dynamic universe 결과를 원논문 fixed 62-coin 완전 재현이라고 부르지 않는다.
- 이 연구는 시장 현상 재현이며 수익률 alpha 백테스트가 아니다.
- Structural break·intentional/spurious decomposition은 이 재현의 데이터·모형 검증 완료 후 별도 프로토콜로 진행한다.
