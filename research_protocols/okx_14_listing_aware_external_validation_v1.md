# OKX 14종목 상장 인지형 외부검증 프로토콜 v1

- 동결일: 2026-07-20
- 연구 성격: CMC corrected CSAD 관계의 미관찰 거래소·상이한 universe 외부검증
- 결과 확인 전 점검 범위: OKX 현재 현물 instrument metadata와 BTC·BNB 각 5개 일봉만 확인
- 알고 있는 결과: CMC fixed-62 corrected 4/4, Binance 14종목 두 가중법 corrected 3/4
- 알지 못하는 결과: OKX 전체 5년 CSAD·회귀계수·유의성·하위기간 안정성

## 연구 질문

CMC에서 재현됐지만 Binance에서 부분 재현에 그친 corrected CSAD 관계가, 아직 회귀 결과를 보지 않은 OKX 현물 가격과 상장 인지형 14종목 후보 universe에서도 유지되는가?

## 표본

- 공급자: OKX 공개 `GET /api/v5/market/history-candles`
- bar: UTC 0시 기준 `1Dutc`
- 원자료: 2021-04-09~2026-04-09
- 분석: `[2021-04-10, 2026-04-10)`, 정확한 5년
- candidate universe: BTC, ETH, XRP, ADA, SOL, DOGE, LINK, AVAX, DOT, LTC, TRX, ATOM, NEAR, BNB의 USDT 현물 페어
- 2021-04-09는 첫 수익률 계산에만 사용한다.
- 현재 instrument metadata의 `listTime`과 실제 complete candle을 함께 보존한다.
- BNB-USDT는 2022-12-20 상장 뒤 complete UTC 일봉과 정확한 전기 가격이 존재할 때부터 자연 편입한다.
- 다른 자산도 상장 전·자료 부재 기간을 보간하지 않으며 활성 자산만 사용한다.

현재 공개 instrument 목록에는 이미 상장폐지된 페어가 없으므로 완전한 point-in-time exchange universe가 아니라 current-survivor candidate set의 listing-aware panel이다.

## 원자료 보존

- 종목별 최대 100일 고정 window JSON gzip checkpoint를 저장한다.
- checkpoint별 관측 수·첫/마지막 timestamp·파일 크기·SHA-256을 manifest에 기록한다.
- API 오류, 중복 timestamp, 미완성 candle, window 밖 candle은 검증한다.
- 빈 checkpoint도 상장 전 구간이면 정상 응답 원문과 hash를 보존한다.
- 결과 확인 후 종목·기간·window를 교체하지 않는다.

## 사양

### Primary

- close-to-close 로그수익률
- 시점별 활성 종목 동일가중 시장수익률
- 활성 종목 동일가중 CSAD

### No-look-ahead liquidity sensitivity

- close-to-close 로그수익률
- OKX가 제공한 실제 quote-currency volume의 정확한 전기 값으로 시장수익률 가중
- 활성 종목 동일가중 CSAD
- 당기 volume은 당기 시장수익률 가중치에 사용하지 않는다.

### 빈도

- Daily: UTC 일별 비중첩 수익률
- Weekly: UTC Monday-start, 주 마지막 종가의 비중첩 로그수익률
- Weekly sensitivity는 전주 quote-volume 합계만 사용한다.

## 회귀와 주 판정

각 variant에서 standard, no-intercept, SCSAD를 daily·weekly로 실행하고 6개 target의 two-sided p-value를 variant·period별 하나의 BH-FDR family로 처리한다.

Primary external-validation criterion:

1. full_5y daily no-intercept target < 0 and `q <= 0.05`
2. full_5y daily SCSAD target < 0 and `q <= 0.05`
3. full_5y weekly no-intercept target < 0 and `q <= 0.05`
4. full_5y weekly SCSAD target < 0 and `q <= 0.05`

네 조건을 모두 만족할 때 corrected relation이 미관찰 OKX 표본에서 재현됐다고 판정한다. Standard 두 셀은 같은 BH family에 포함하지만 필수조건은 아니다. 전기 quote-volume sensitivity도 동일한 네 조건을 별도로 평가한다.

## 기간 진단

- full_5y: 2021-04-10~2026-04-09, 주 판정
- early_half: 2021-04-10~2023-10-09, 진단
- late_half: 2023-10-10~2026-04-09, 진단

두 하위기간은 결과를 보기 전에 달력상 절반으로 고정하며 각 variant·period별 별도 6-test BH family로 처리한다. 하위기간은 full_5y 판정을 대체하지 않는다.

## 품질 Gate

- metadata와 14×고정 window checkpoint 완료율 100%
- timestamp-instrument 중복 0건
- complete candle만 사용
- 양의 종가 100%
- 각 종목의 상장 후 기대 asset-day coverage 99% 이상
- 일별 최소 활성 종목 12개
- 적격 분석일 99% 이상

## 해석 제한

- corrected CSAD는 동시적 횡단면 수렴 관계이며 미래수익률 alpha가 아니다.
- current-survivor candidate set이므로 상장폐지 자산 선택 편향이 남는다.
- OKX 단일 거래소·USDT 현물에 조건부인 결과다.
- 상장 인지형 편입은 시가총액 기반 point-in-time 동적 universe와 동일하지 않다.
- 결과를 intentional imitation, 인과효과 또는 거래전략으로 확대하지 않는다.
