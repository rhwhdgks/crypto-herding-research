# CMC Multi-Factor Convergence Protocol v1

- 동결일: 2026-07-19
- 연구 성격: 단일 시장요인 초과 수렴 결과에 대한 point-in-time 다요인 반사실적 강건성 검증
- 알고 있는 결과: 365관측 단일요인 전체표본 `delta2=6.772`, BH `q=2.91e-05`; 180관측 전체도 통과, 730관측 전체는 미통과.
- 알지 못하는 결과: 규모·유동성·모멘텀을 추가한 정상 기대 CSAD, 과거잔차 반사실적 CSAD, 전체·고정구간별 `delta2`.

## 연구 질문

단일 시장 beta를 넘어 point-in-time 규모·유동성·모멘텀 노출까지 통제해도 극단적 시장 움직임에서 정상 기대보다 강한 횡단면 수렴이 남는가?

본 분석은 intentional herding을 직접 식별하지 않는다. 다요인 통제 뒤의 초과 수렴도 누락요인, factor specification, 잔차분포 변화로 설명될 수 있다.

## Point-in-time factor

날짜 `t`의 factor portfolio 구성원은 기존 CMC 전월 말 Top-200 동적 universe의 당일 적격 구성원으로 고정한다. 정렬 특성은 모두 `t-1`까지 알려진 값만 사용한다.

```text
MKT_t = 기존 전일 시가총액 가중 시장수익률
SIZE_t = 하위 30% 시가총액 코인 EW return - 상위 30% EW return
LIQ_t = 상위 30% lagged turnover 코인 EW return - 하위 30% EW return
MOM_t = 상위 30% lagged 30일 momentum 코인 EW return - 하위 30% EW return

lagged turnover_t = volume_24h_(t-1) / market_cap_(t-1)
lagged momentum_t = log(price_(t-1) / price_(t-31))
```

- 분위수는 날짜별 유효 자산의 rank percentile로 정한다.
- 각 long/short leg에 최소 10개 자산을 요구한다.
- target 자산이 factor leg에 속하면 그 자산을 leg 수익률에서 제외한 leave-one-out factor를 해당 자산 회귀에 사용한다.
- target 자산이 당시 universe 밖이었던 과거 날짜에는 전체 point-in-time factor를 사용한다.
- 가격·시가총액·거래대금·momentum을 보간하지 않는다.

## Rolling 다요인 모형

```text
R_i,s = alpha_i,t-1
        + beta_MKT,i,t-1 MKT_(-i),s
        + beta_SIZE,i,t-1 SIZE_(-i),s
        + beta_LIQ,i,t-1 LIQ_(-i),s
        + beta_MOM,i,t-1 MOM_(-i),s
        + epsilon_i,s
```

- 주 분석창: 직전 365개 행, 최소 180개 완전 관측
- 민감도: 180개 행·최소 120개, 730개 행·최소 365개
- 모든 sufficient statistic은 1행 shift 후 rolling 계산한다.
- 현재 날짜는 alpha, beta, residual scale 추정에 포함하지 않는다.
- 회귀행렬이 비가역이거나 조건수가 `1e10`을 넘으면 해당 자산-일 모형을 사용하지 않는다.
- 날짜별 구성종목의 80% 이상, 최소 20개 모형이 있어야 일간 결과를 사용한다.

## 두 반사실적

### 1. 정규잔차 반사실적

과거 다요인 회귀의 잔차표준편차를 사용해 `epsilon ~ N(0, sigma^2)`에서 기대 절대편차를 계산한다. 단일요인 연구와 직접 비교 가능한 주 반사실적이다.

### 2. 과거잔차 반사실적

365관측 주 창에 한해 현재 계수로 과거 rolling window의 잔차를 다시 계산하고 아래 empirical mean을 사용한다.

```text
expected_abs_deviation_empirical_i,t
  = mean_s |predicted_return_i,t - R_m,t + residual_i,s(current coefficients)|
```

- 최소 60개 유효 과거잔차를 요구한다.
- resampling seed나 분포 가정이 없는 residual empirical integration이다.
- 현재 날짜 잔차는 포함하지 않는다.

## 사전 고정 주 판정

전체표본과 기존 다섯 구조 구간에서 각각 아래를 적합한다.

```text
convergence_ratio_t = delta0 + delta1 |R_m,t| + delta2 R_m,t^2 + u_t
```

- Newey-West HAC 표준오차
- 반사실적별 6개 회귀를 각각 하나의 BH-FDR family로 보정
- 다요인 강건성의 주 통과 조건:
  - 365관측 정규잔차 전체표본 `delta2 > 0`, `q <= 0.05`
  - 365관측 과거잔차 전체표본 `delta2 > 0`, `q <= 0.05`
- 180·730관측 결과와 개별 regime은 강건성·이질성 진단이며 주 기준을 대체하지 않는다.
- 단일요인과 다요인 계수 차이는 기술 비교이며 별도 유의성 선택에 사용하지 않는다.

## 산출물

- 날짜별 point-in-time factor return과 factor 상관
- 주 분석창 자산별 다요인 beta·예측·잔차 scale
- 정규잔차 창별 일간 초과 수렴
- 365관측 과거잔차 일간 초과 수렴
- 전체·구간별 HAC 계수와 BH-FDR 판정
- 단일요인 대비 비교표
- 시계열·구간별 계수 그림
- config·protocol snapshot, input manifest, provenance
- 한국어 보고서

## 해석 제약

- factor portfolio는 연구자가 고정한 단순 characteristic spread이며 암호화폐의 완전한 가격결정모형이 아니다.
- 거래대금은 CMC 집계값이고 거래소별 체결 유동성과 다르다.
- 과거잔차 반사실적도 조건부 분포가 시간에 따라 안정적이라는 가정이 남는다.
- 미래수익률·비용·실행을 검증하지 않으므로 alpha 또는 전략으로 해석하지 않는다.
- 결과를 본 뒤 factor 방향, 30% cut, momentum 기간, 창 길이, 판정 기준을 변경하지 않는다.
