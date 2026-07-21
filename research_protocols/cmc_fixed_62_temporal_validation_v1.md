# CMC Fixed-62 Temporal Validation Protocol v1

- 동결일: 2026-07-19
- 연구 성격: 선행논문 종료 다음 날부터의 완전한 시간 외부표본 검증
- 알고 있는 결과: 2018-01-01~2024-04-09 fixed-62에서 standard 0/2, corrected 4/4 통과
- 알지 못하는 결과: 2024-04-10 이후 fixed-62 corrected 계수·유의성·기간 안정성

## 연구 질문

논문 표본에서 거의 정확히 복제된 no-intercept와 SCSAD의 음의 비선형 계수가 논문 종료 이후의 새로운 시장 기간에도 유지되는가?

## 표본

- 데이터 소스: CMC 무료 개별 역사 JSON endpoint
- 원자료: 2024-04-09~2026-07-18 UTC 일별 close·market cap
- 분석: 2024-04-10~2026-07-18
- universe: fixed-62 복제 config의 동일한 62개 legacy CMC ID
- 구성종목 추가·삭제·교체 금지
- 2024-04-09는 첫 holdout 수익률 계산에만 사용하고 회귀에는 포함하지 않는다.
- 400일 이하 raw gzip checkpoint와 SHA-256 manifest를 별도 temporal cache에 저장한다.
- 누락 가격·시총은 보간하거나 0으로 대체하지 않는다.

이 검증은 새 시간 표본이지만 동일한 CMC 데이터 공급자를 사용하므로 공급자 외부검증은 아니다.

## 사양

### Direct-replication primary

- 일별 close-to-close 로그수익률
- 당일 시가총액 가중 시장수익률
- 활성 자산의 equal-weight CSAD

### No-look-ahead sensitivity

- 일별 close-to-close 로그수익률
- 정확한 전일 시가총액 가중 시장수익률
- 활성 자산의 equal-weight CSAD

### Frequency

- Daily: UTC 일별
- Weekly: UTC Monday-start, 주 마지막 관측 종가의 비중첩 로그수익률
- 첫·마지막 partial week 포함 여부는 fixed-62 복제 규칙을 그대로 유지한다.

## 회귀와 주 판정

각 variant에서 standard, no-intercept, SCSAD를 daily·weekly로 실행하고 6개 target의 two-sided p-value를 하나의 BH-FDR family로 처리한다.

Primary persistence criterion:

1. full holdout daily no-intercept target < 0 and `q <= 0.05`
2. full holdout daily SCSAD target < 0 and `q <= 0.05`
3. full holdout weekly no-intercept target < 0 and `q <= 0.05`
4. full holdout weekly SCSAD target < 0 and `q <= 0.05`

네 조건을 모두 만족해야 corrected herding relation이 시간 외부표본에서 재현됐다고 판정한다. Standard 두 셀은 같은 BH family에 포함하지만 persistence 필수조건은 아니다. No-look-ahead sensitivity도 같은 네 조건을 별도로 모두 만족해야 timing-robust로 표시한다.

## 기간 진단

- holdout_full: 2024-04-10~2026-07-18, confirmatory
- holdout_year1: 2024-04-10~2025-04-09, diagnostic
- holdout_later: 2025-04-10~2026-07-18, diagnostic

두 하위기간은 각 variant·period별 별도 6-test BH family이며 full holdout 판정을 대체하지 않는다.

## 품질 Gate

- checkpoint 완료율 100%
- 62개 CMC ID와 date-ID 중복 0건
- 양의 종가 99% 이상
- asset-day coverage 95% 이상
- 일별 최소 활성 자산 55개
- 적격 분석일 99% 이상

## 비교

- 2018-2024 fixed-62 full-sample target과 holdout target의 부호, 계수 차이, 절대크기 비율을 보고한다.
- 같은 방향·유의성은 관계의 지속성을 의미하지만 계수 동일성이나 구조 안정성을 자동으로 의미하지 않는다.
- 별도의 coefficient-break 검정 없이 단순 차이를 구조 변화로 부르지 않는다.

## 해석 제한

- corrected CSAD는 동시적 횡단면 수렴 관계이며 미래수익률 alpha가 아니다.
- 당일 시총가중 primary는 논문 재현용이고 예측 가능한 투자 가중치가 아니다.
- 고정 62종목은 사후 생존·선정 편향 가능성이 있다.
- 동일 CMC 공급자의 역사자료 수정·산식 일관성이 결과 유사성을 높일 수 있다.
- 결과를 intentional imitation, 인과효과, 거래전략으로 확대하지 않는다.
