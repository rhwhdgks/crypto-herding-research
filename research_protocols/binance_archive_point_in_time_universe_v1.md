# Binance Archive Point-in-Time Universe 외부검증 프로토콜 v1

- 동결일: 2026-07-20
- 연구 성격: 상장폐지와 신규상장을 포함한 archive-observed point-in-time 거래소 universe 외부검증
- 결과 확인 전 점검 범위: Binance Vision S3 symbol prefix, LUNAUSDT 2022-04 일봉 1개월, archive 파일 checksum 형식만 확인
- 알고 있는 결과: CMC fixed-62 corrected 4/4, Binance fixed-14 3/4·3/4, OKX listing-aware 3/4·1/4
- 알지 못하는 결과: archive point-in-time membership, CSAD 계수, BH-FDR, pooled/meta 결과

## Source-quality amendment 1 (2026-07-20, 회귀 실행 전)

- 21,479개 ZIP 전수 원시행 감사에서 `AXSUSDT-1d-2026-02.zip`의 2026-02-10 행이 byte-for-byte 완전히 동일한 형태로 1회 반복됐다.
- 완전히 동일한 원시행은 하나로 축약하되 source-quality audit에 제거 건수를 보존한다.
- 같은 open timestamp이면서 하나라도 값이 다른 충돌 행은 임의로 합치지 않고 파이프라인을 즉시 실패시킨다.
- 이 amendment 작성 시점에 archive CSAD 회귀는 한 번도 실행되지 않았다. 중복 정책 외의 universe·기간·회귀·판정 규칙은 변경하지 않는다.

## 연구 질문

현재 생존 종목만 고정한 표본을 버리고 당시 Binance에 실제 일봉 archive가 존재한 종목을 전월 유동성으로 선택할 때, standard·no-intercept·SCSAD 관계가 daily·weekly에서 유지되는가?

## 무료 원자료와 완전성 범위

- 공급자: Binance 공식 `data.binance.vision` spot monthly kline archive
- S3 bucket inventory에서 `data/spot/monthly/klines/{SYMBOL}/1d/` prefix를 전수 열거한다.
- 현재 REST `exchangeInfo`가 아니라 archive prefix를 사용하므로 현재 API에서 사라진 BCCUSDT, LUNAUSDT, FTTUSDT 같은 symbol도 후보에 남는다.
- 각 monthly ZIP의 1d OHLCV와 quote asset volume을 사용한다.
- ZIP별 local SHA-256과 S3 ETag를 manifest에 기록하고, 최종 membership에 사용된 ZIP은 공식 `.CHECKSUM`도 검증한다.
- bucket inventory XML, symbol key inventory XML, ZIP, checksum을 checkpoint로 보존한다.
- Binance archive가 과거 모든 상장·상장폐지 symbol을 영구적으로 완전 보존한다는 보장은 없다. 따라서 결과는 `archive-observed point-in-time universe`로 부르고 완전한 법적 listing master로 표현하지 않는다.

## 기간

- 유동성 형성용 source 시작: 2021-03-01
- 분석 수익률 시작: 2021-04-10
- 분석 종료: 2026-04-09
- 분석 구간: `[2021-04-10, 2026-04-10)`, 정확한 5년
- 2021-03은 2021-04 membership 산출에만 사용한다.
- 2021-04-01~09는 첫 분석 수익률과 월 membership의 시점 정렬에 사용한다.

## Candidate 필터

- symbol 문자열이 `USDT`로 끝나는 spot monthly kline prefix만 후보로 사용한다.
- base asset은 마지막 `USDT`를 제거해 정의한다.
- leveraged-token suffix `UP`, `DOWN`, `BULL`, `BEAR`, `HEDGE`, `HALF`는 제외한다.
- stablecoin·fiat base는 config의 사전 고정 목록으로 제외한다.
- wrapped·staked duplicate base는 config의 사전 고정 목록으로 제외한다.
- 결과를 본 뒤 제외 목록을 추가·삭제하지 않는다. 오분류 가능성은 limitation으로 남긴다.

## Ticker 재사용과 Listing Episode

- 같은 symbol에서 관측일 간 공백이 7일을 초과하면 새로운 `listing_episode`로 분리한다.
- asset key는 `{symbol}#{episode_number}`로 저장한다.
- 정확한 전일 또는 전주 가격이 없으면 수익률을 만들지 않는다.
- 서로 다른 episode의 가격과 거래량을 연결하거나 보간하지 않는다.
- 이 규칙은 LUNA처럼 ticker가 재사용된 경우 old/new asset의 수익률 연결을 막기 위한 사전 고정 규칙이다.

## 월별 Point-in-Time Membership

각 달 `m`의 membership은 달력상 직전 달 `m-1` 정보만 사용한다.

1. 직전 달에 양의 종가와 quote volume이 있는 episode를 집계한다.
2. 직전 달 관측일이 최소 20일인 episode만 적격이다.
3. 직전 달 quote volume 합계 내림차순으로 정렬한다.
4. 상위 50개 episode를 다음 달 membership으로 고정한다.
5. 동률은 asset key 사전순으로 결정한다.
6. 당월 신규상장은 다음 달부터만 편입 가능하다.
7. 당월 중도 상장폐지 episode는 마지막 exact return 이후 자동 비활성화하며 미래 가격을 채우지 않는다.

Membership 50개를 채우지 못한 달은 실제 적격 수만 사용한다. 일별 분석은 최소 30개 활성 수익률을 요구한다.

## 수익률·CSAD 사양

### Equal-weight primary

- close-to-close 로그수익률
- 당월 membership 중 당일 exact return이 있는 episode의 동일가중 시장수익률
- 같은 활성 episode의 동일가중 CSAD

### Lagged-liquidity sensitivity

- close-to-close 로그수익률
- 당월 membership 중 정확한 전일 quote volume으로 가중한 시장수익률
- weekly는 정확한 전주 quote volume 합계로 가중한다.
- 당일·당주 volume은 같은 기간 시장수익률 가중치에 사용하지 않는다.
- CSAD는 활성 episode 동일가중을 유지한다.

### 빈도

- Daily: UTC 일별 비중첩 수익률
- Weekly: UTC Monday-start, 주 마지막 관측 종가의 비중첩 로그수익률
- Weekly membership은 해당 주 마지막 날짜가 속한 월의 사전 고정 membership을 사용한다.

## 회귀와 BH-FDR

각 variant에서 아래 3개 모형을 daily·weekly로 실행한다.

1. standard CSAD
2. no-intercept CSAD
3. SCSAD

각 variant·period의 6개 target two-sided p-value를 하나의 BH-FDR family로 처리한다.

Primary strict criterion:

1. full_5y daily no-intercept target < 0 and `q <= 0.05`
2. full_5y daily SCSAD target < 0 and `q <= 0.05`
3. full_5y weekly no-intercept target < 0 and `q <= 0.05`
4. full_5y weekly SCSAD target < 0 and `q <= 0.05`

네 조건을 모두 만족해야 point-in-time corrected relation이 재현됐다고 판정한다. Lagged-liquidity sensitivity도 동일한 네 조건을 별도로 평가한다. Standard는 같은 BH family에 포함하지만 필수조건은 아니다.

## 기간 진단

- full_5y: 2021-04-10~2026-04-09, 주 판정
- early_half: 2021-04-10~2023-10-09, 진단
- late_half: 2023-10-10~2026-04-09, 진단

하위기간은 full_5y 판정을 대체하지 않으며 variant·period별 별도 6-test BH family를 사용한다.

## 품질 Gate

- bucket inventory pagination 완료
- 후보 symbol key inventory 100%
- 발견된 분석 범위 ZIP download 100%
- membership 사용 ZIP 공식 checksum 100%
- date-asset key 중복 0건
- 양의 종가 100%
- monthly membership look-ahead 위반 0건
- 일별 최소 활성 episode 30개
- 적격 분석일 99% 이상
- episode 경계를 넘는 수익률 0건

## Pooled/Meta 통합분석

아래 사전 고정 estimate를 daily·weekly no-intercept·SCSAD 4개 cell별로 통합한다.

- CMC fixed-62 historical primary
- CMC fixed-62 temporal holdout primary
- Binance fixed-14 equal weight
- Binance fixed-14 lagged liquidity
- OKX listing-aware equal weight
- OKX listing-aware lagged quote volume
- Binance archive point-in-time equal weight
- Binance archive point-in-time lagged liquidity

원계수 단위 차이를 피하기 위해 standardized target coefficient와 같은 비율로 변환한 standard error를 사용한다.

- inverse-variance fixed-effect estimate
- DerSimonian-Laird random-effects estimate
- Cochran Q와 I-squared
- two-sided p-value와 4-cell BH-FDR
- source·universe·weighting·frequency·model별 descriptive comparison

표본·기간·공급자가 일부 중첩하므로 meta p-value를 독립 연구의 확정적 합성으로 부르지 않고 descriptive pooled evidence로만 해석한다.

## 해석 제한

- Archive presence는 거래 가능 상태의 감사 가능한 proxy지만 공식 historical exchangeInfo master가 아니다.
- Delisting 종료일은 마지막 complete daily candle로 근사하며 공지 시각과 다를 수 있다.
- Static symbol-name exclusion은 point-in-time token taxonomy를 완전히 복원하지 못한다.
- Corrected CSAD는 동시적 횡단면 수렴 관계이며 미래수익률 alpha가 아니다.
- 결과를 intentional imitation, 인과효과 또는 거래전략으로 확대하지 않는다.
