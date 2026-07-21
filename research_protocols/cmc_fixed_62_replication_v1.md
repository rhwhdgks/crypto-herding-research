# CMC Fixed-62 Exact-Universe Replication Protocol v1

- 동결일: 2026-07-19
- 연구 성격: Natashekara and Sampath (2026)의 Table 1 고정 62종목 저빈도 CSAD 결과 직접 재현
- 사전 인지 결과: 논문이 보고한 계수와 기존 CMC 동적 Top-200 결과
- 사전 미인지 결과: 공개 CMC 개별 역사 자료로 재구성한 fixed-62 daily·weekly 회귀계수

### Source-alignment amendment (2026-07-19)

최초 실행 전 논문이 수익률 산식을 명시하지 않아 primary를 단순수익률로 고정했다. 최초 결과 확인 뒤 계수 자체가 아니라 논문 Table 1의 공개된 자산별 평균·표준편차를 재현하는 2×2 방법 감사를 실시했다. BTC 평균 0.07%, BNB 0.18% 등 Table 1 통계는 단순수익률이 아니라 일별 로그수익률과 일치했다. 시총 가중시점까지 교차한 네 조합 중 `로그수익률 + 당일 시가총액`이 Table 1 통계와 회귀표를 동시에 재현했다.

- 최종 direct-replication primary를 `로그수익률 + 당일 시가총액`으로 교정한다.
- 최초 `단순수익률 + 당일 시가총액`과 나머지 두 조합은 `method_audit`에 모두 보존한다.
- 이 교정은 유의한 alpha를 선택한 것이 아니라 논문에 공개된 동시적 기술통계의 측정 단위를 복원한 것이다.
- primary의 BH 판정은 교정 결과를 사용하되 amendment 이전 사양과 결과도 보고서에서 숨기지 않는다.

## 연구 질문

기존 동적 Top-200에서 확인된 모형 민감적 herding 결과가 논문 Table 1과 동일한 62개 CMC ID를 고정했을 때 논문의 계수에 더 가까워지는가?

## 데이터와 고정 Universe

- 소스: CMC 웹사이트가 사용하는 무료 JSON endpoint
- URL: `https://api.coinmarketcap.com/data-api/v3.1/cryptocurrency/historical`
- 원자료 기간: 2017-12-31~2024-04-09, UTC 일별 OHLCV와 market cap
- 분석 기간: 2018-01-01~2024-04-09
- API는 종료일 기준 최대 400개 일별 관측을 반환하므로 서로 겹치지 않는 최대 400일 구간으로 수집한다.
- 원응답은 gzip JSON checkpoint로 보존하고 SHA-256 manifest를 생성한다.
- CMC ID를 기본키로 사용하며 symbol 변경이나 동일 symbol의 신규 코인을 합치지 않는다.
- Table 1의 62종목과 legacy CMC ID는 config에 명시적으로 동결한다.
- 가격·시가총액 누락을 보간하거나 수익률 0으로 대체하지 않는다.

이 endpoint는 정식 Pro API 계약 endpoint가 아니므로 변경될 수 있다. 재현성은 저장한 원응답·정규화 Parquet·hash로 확보한다.

## 수익률과 시장수익률

### Primary replication

- 자산수익률: 일별 close-to-close 로그수익률 `log(P_t / P_t-1)`
- 시장수익률: 해당 날짜 관측 시가총액 가중
- CSAD: 해당 날짜 활성 자산의 `mean(|R_i,t - R_m,t|)`

논문은 market-cap weighted return이라고 명시하지만 가중치 시점을 밝히지 않는다. 2×2 방법 감사에서 당일 시총가중이 공개 회귀표와 일치해 직접 재현 primary로 사용한다.

### No-look-ahead sensitivity

- 자산수익률: primary와 동일한 일별 close-to-close 로그수익률
- 시장수익률: 정확히 전일에 관측된 시가총액 가중
- CSAD 계산은 동일하다.

이 민감도 결과는 실전 예측에서 사용 가능한 시점 정렬을 확인하기 위한 것이며 primary를 대체하지 않는다.

## 결측과 품질 Gate

- 자산별 요청 구간 checkpoint 완료율 100%
- 원응답 status 성공, 일자·CMC ID 중복 0건
- 양의 종가 비율 99% 이상
- 전체 62종목 asset-day coverage 95% 이상
- 분석일 최소 55종목, 적격 분석일 99% 이상
- 정확한 전일 가격이 없으면 해당 자산 수익률은 결측으로 둔다.
- 품질 gate 실패 시 회귀를 생성하지 않는다.

## Weekly 구성

- UTC Monday-start calendar week로 묶는다.
- Primary weekly 수익률은 주 첫 관측 직전 종가와 주 마지막 종가의 단순수익률이다.
- No-look-ahead sensitivity는 주간 일별 로그수익률 합이다.
- 첫 주와 마지막 주의 부분 주간도 논문의 포괄 기간에 맞춰 포함하고 별도 coverage에 표시한다.
- 논문의 307주와 달리 달력상 가능한 주 수가 다르면 관측 수 차이를 결과에 명시하며 임의로 21주를 제거하지 않는다.

## 회귀 모형

1. Standard CSAD: `CSAD = alpha + beta1*|Rm| + beta2*Rm^2 + error`
2. No-intercept: `CSAD = gamma1*Rm + gamma2*|Rm| + gamma3*Rm^2 + error`
3. SCSAD: `SCSAD = alpha + delta1*Rm + delta2*Rm^2 + delta3*Rm^3 + error`

SCSAD는 `Rm >= 0`이면 CSAD, `Rm < 0`이면 `-CSAD`다. 모든 t-stat은 Newey-West HAC를 사용한다.

## Confirmatory 판정

- primary full-sample daily 3개와 weekly 3개 target을 하나의 6-test BH-FDR family로 둔다.
- target 계수가 음수이고 `q <= 0.05`일 때만 해당 specification이 herding과 일치한다고 판정한다.
- 세 모형 모두 통과하면 모형 간 강한 일치, corrected 모형만 통과하면 method-sensitive evidence로 표현한다.
- no-look-ahead sensitivity와 세 하위기간은 각각 별도 6-test BH family이며 강건성 진단이다.
- 결과는 동시적 분산 관계이지 미래수익률 alpha가 아니다.

## 하위기간

- pre-COVID: 2018-01-01~2020-03-10
- COVID-19: 2020-03-11~2022-02-23
- post-COVID: 2022-02-24~2024-04-09

## 논문 Benchmark

- Daily no-intercept gamma3: full -1.850, pre -4.634, COVID -1.757, post -6.543
- Daily SCSAD gamma3: full -2.924, pre -14.892, COVID -4.807, post -30.813
- Weekly standard beta2: -0.115
- Weekly no-intercept gamma3: -2.096
- Weekly SCSAD gamma3: -3.359

## 해석 제한

- 논문이 고정 62종목 선정에 사용한 정확한 시점별 $100M 조건은 공개되지 않아 동일한 62개 목록만 복제한다.
- 고정 62종목은 2024년까지 생존한 자산을 사후 선택했을 가능성이 있어 point-in-time 투자 universe가 아니다.
- CMC 역사값 수정, weekly 307개 생성 규칙 미공개, 당일 시총가중의 동시성 한계를 보고한다.
- 이 연구를 intentional imitation, 인과효과 또는 거래 alpha로 표현하지 않는다.
