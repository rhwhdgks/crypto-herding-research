# Binance 14종목 공급자·유니버스 외부 강건성 프로토콜 v1

- 동결일: 2026-07-19
- 연구 성격: CMC corrected CSAD 관계의 독립 거래소·상이한 universe 강건성 검증
- 알고 있는 결과: CMC fixed-62에서 standard 0/2, no-intercept·SCSAD 4/4 통과
- 사전 주의: 같은 Binance 패널이 과거 baseline 연구에 사용됐으므로 완전한 미관찰 confirmatory 검정이 아니라 secondary external robustness다.

## 연구 질문

CMC 시총자료와 62개 생존목록에서 확인된 corrected CSAD의 음의 비선형 관계가 Binance USDT 현물 가격과 고정 14종목 universe에서도 유지되는가?

## 표본

- 공급자: Binance 현물 OHLCV 로컬 parquet
- 원자료: 2021-04-09~2026-04-09 UTC 일봉
- 분석: 2021-04-10~2026-04-09, 정확히 5년
- universe: baseline에 이미 고정된 14개 USDT 페어
- 2021-04-09는 첫 수익률 계산에만 사용하고 회귀에는 포함하지 않는다.
- 구성종목 추가·삭제·교체 및 누락값 보간을 금지한다.
- 각 입력 parquet의 SHA-256을 manifest에 저장한다.

이 표본은 CMC와 데이터 공급자, 통화 표시, 자산 수, universe 구성이 모두 다르다. 다만 Binance 상장·USDT 거래 가능성 및 고정 survivor list에 조건부인 결과다.

## 사양

### Primary

- close-to-close 로그수익률
- 14종목 동일가중 시장수익률
- 활성 종목의 동일가중 CSAD

### No-look-ahead liquidity sensitivity

- close-to-close 로그수익률
- 직전 기간 거래대금 proxy로 가중한 시장수익률
- 거래대금 proxy는 해당 기간 `close × base volume`이며 당기 가중치에는 사용하지 않는다.
- 활성 종목의 동일가중 CSAD

### 빈도

- Daily: UTC 일별 비중첩 수익률
- Weekly: UTC Monday-start, 주 마지막 종가의 비중첩 로그수익률
- Weekly 거래대금 proxy는 주간 `close × base volume` 합계이며 정확한 전주 값만 사용한다.

## 회귀와 주 판정

각 variant에서 standard, no-intercept, SCSAD를 daily·weekly로 실행하고 6개 target의 two-sided p-value를 variant·period별 하나의 BH-FDR family로 처리한다.

Primary external-robustness criterion:

1. full_5y daily no-intercept target < 0 and `q <= 0.05`
2. full_5y daily SCSAD target < 0 and `q <= 0.05`
3. full_5y weekly no-intercept target < 0 and `q <= 0.05`
4. full_5y weekly SCSAD target < 0 and `q <= 0.05`

네 조건을 모두 만족할 때 corrected relation이 Binance 14종목에서도 강건하다고 판정한다. Standard 두 셀은 같은 BH family에 포함하지만 필수조건은 아니다. 거래대금가중 sensitivity도 동일한 네 조건을 별도로 평가한다.

## 기간 진단

- full_5y: 2021-04-10~2026-04-09, 주 판정
- cmc_historical_overlap: 2021-04-10~2024-04-09, 진단
- cmc_holdout_overlap: 2024-04-10~2026-04-09, 진단

하위기간은 각각 별도의 6-test BH family이며 full_5y 판정을 대체하지 않는다.

## 품질 Gate

- 입력 파일 14/14 존재
- timestamp-symbol 중복 0건
- 양의 종가 100%
- asset-day coverage 100%
- 일별 최소 활성 종목 14개
- 적격 분석일 100%

## 해석 제한

- corrected CSAD는 동시적 횡단면 수렴 관계이며 미래수익률 alpha가 아니다.
- 동일가중 시장수익률은 CMC 시총가중 사양과 직접 동일하지 않다.
- 거래대금 sensitivity의 `close × base volume`은 Binance 공식 quote volume의 근사치다.
- 14종목은 고정 생존목록이며 상장폐지·신규상장 코인을 대표하지 않는다.
- 결과를 intentional imitation, 인과효과 또는 거래전략으로 확대하지 않는다.

## Source-alignment amendment

- 2026-07-19 최초 실행 뒤 달력 경계를 감사해, 최초 종료일 `2026-04-08`이 분석일 1,825개로 정확한 5년보다 하루 짧음을 확인했다.
- 최종 종료일을 `2026-04-09`로 하루 연장해 `[2021-04-10, 2026-04-10)`의 정확한 5년으로 교정했다.
- 최초 strict 결과는 primary 3/4, sensitivity 3/4였다. 이 결과를 확인한 상태에서 수정했음을 공개하며 종료일 외 사양·판정 기준은 바꾸지 않았다.
