# CMC Factor-Adjusted Convergence Protocol v1

- 동결일: 2026-07-19
- 연구 성격: CMC point-in-time 동적 universe에서 공통 시장요인으로 설명되는 동조화와 모형 대비 초과 수렴의 분해
- 입력: `cmc_dynamic_universe_replication_v1`의 일간 구성종목 패널
- 알고 있는 결과: corrected SCSAD는 전체 표본과 기존 BIC 선택 5개 구조 구간에서 음의 cubic coefficient를 보였다.
- 알지 못하는 결과: 과거정보 전용 시장모형의 정상 기대 CSAD, 초과 수렴도, 구간별 초과 수렴 회귀계수.

### 실행 전 품질 교정 기록

최초 구현 점검에서 구성종목 패널 안의 과거 행만 beta 학습에 사용하면 신규 편입 자산의 편입 전 공개 가격이 누락되어 긴 coverage 공백이 생김을 확인했다. 최종 실행은 당일 횡단면 구성은 point-in-time universe로 고정하되, beta 학습에는 해당 날짜 이전 CMC snapshot에 실제 존재했던 자산 가격 이력을 사용한다. 창 길이, 최소 관측, coverage gate, 회귀식과 판정 기준은 변경하지 않았다. 교정 전 preliminary 계수는 연구 결과로 사용하지 않는다.

## 연구 질문

관측된 낮은 횡단면 분산은 각 암호화폐가 시장 공통요인에 노출된 정도와 과거 고유변동성만으로 설명되는가, 아니면 그 반사실적 기대보다 실제 수익률이 더 강하게 수렴하는가?

가격자료만으로 투자자의 의도적 모방을 식별할 수 없으므로, 본 연구는 `intentional herding`을 직접 측정한다고 표현하지 않는다. 양의 초과 수렴은 공통요인 모형을 넘어선 동조화와 일치하지만, 누락요인·비선형 beta·분포 가정 실패도 가능한 설명이다.

## 시점별 시장모형

자산 `i`의 날짜 `t` 예측은 날짜 `t`보다 앞선 관측만 사용한다.

```text
R_i,s = alpha_i,t-1 + beta_i,t-1 R_(-i),s + epsilon_i,s

R_(-i),s = 시가총액 가중 시장수익률에서 자산 i를 제외한 수익률
```

- 주 분석창: 직전 365개 유효 관측
- 최소 과거 관측: 180개
- 민감도 창: 180개와 730개, 각각 최소 120개와 365개
- 모든 rolling sufficient statistic은 1행 shift한 뒤 계산한다.
- 현재 날짜 수익률은 회귀계수와 잔차표준편차 추정에 포함하지 않는다.
- intercept와 beta를 함께 추정한다.
- 자산 자신이 시장수익률에 미치는 기계적 상관을 줄이기 위해 leave-one-out 시장수익률을 사용한다.
- 자산이 당시 universe 구성원이 아니었던 과거 날짜에는 그 자산이 시장수익률 산식에 들어 있지 않으므로 전체 point-in-time 시장수익률을 사용한다.
- 편입 전 가격 이력은 그날 CMC snapshot에 실제 관측된 경우에만 사용하며 과거 가격을 보간하지 않는다.

## 정상 기대 CSAD와 초과 수렴

과거 회귀 잔차가 조건부 정규분포라는 단순 반사실적을 둔다.

```text
mu_i,t = alpha_hat_i,t-1 + beta_hat_i,t-1 R_(-i),t - R_m,t
epsilon_i,t ~ N(0, sigma_hat_i,t-1^2)

expected_abs_deviation_i,t = E|mu_i,t + epsilon_i,t|
expected_CSAD_t = mean_i(expected_abs_deviation_i,t)
observed_CSAD_t = mean_i|R_i,t - R_m,t|

abnormal_convergence_t = expected_CSAD_t - observed_CSAD_t
convergence_ratio_t = abnormal_convergence_t / expected_CSAD_t
```

- `abnormal_convergence > 0`: 실제 횡단면 분산이 시장모형과 과거 잔차위험이 기대한 수준보다 낮음.
- `abnormal_convergence < 0`: 실제 분산이 정상 기대보다 큼.
- 날짜별 적격 구성종목의 80% 이상에 과거모형이 존재할 때만 일간 분해값을 사용한다.

## 사전 고정 주 검정

전체 표본과 기존 구조분석에서 이미 고정된 다섯 구간에 대해 아래 회귀를 적합한다.

```text
convergence_ratio_t = delta0 + delta1 |R_m,t| + delta2 R_m,t^2 + u_t
```

- Newey-West HAC 표준오차를 사용한다.
- 주 분석은 365 관측창이다.
- `delta2 > 0`이고 전체 6개 회귀 family의 BH-FDR `q <= 0.05`이면, 극단적 시장 움직임에서 모형 대비 초과 수렴이 증가한 것으로 판정한다.
- 전체 평균 초과 수렴도와 평균이 0인지에 대한 HAC 검정은 보조 진단이다.
- 180·730 관측창은 방향과 크기의 민감도 분석이며 주 판정을 대체하지 않는다.

## 보조 진단

- 관측 CSAD, 시장모형 점예측 CSAD, 정상 기대 CSAD를 함께 저장한다.
- 자산별 one-step prediction error와 beta, 과거 잔차표준편차를 주 분석창에 한해 저장한다.
- 기존 SCSAD 구조 구간과의 비교는 동일 날짜 구간을 그대로 사용하며 break를 다시 탐색하지 않는다.
- 모형 가용률, beta 분포, 정상 기대 CSAD 양수 여부를 품질 gate로 확인한다.

## 구조 구간

`cmc_scsad_structural_break_v1`의 BIC 선택 결과를 고정 입력으로 사용한다.

- 2018-01-01~2019-06-27
- 2019-06-28~2020-06-05
- 2020-06-06~2021-05-16
- 2021-05-17~2022-11-10
- 2022-11-11~2024-04-09

첫 구간은 rolling burn-in 때문에 실제 회귀 시작일이 늦어진다.

## 산출물

- 창별 일간 factor-adjusted convergence series
- 주 분석창 자산별 rolling beta와 one-step prediction
- 전체·구간별 HAC 회귀계수와 BH-FDR 판정
- 평균 초과 수렴 HAC 진단
- rolling-window 민감도 비교
- 시계열·구간별 계수 그림
- config·protocol snapshot, input manifest, provenance
- 한국어 연구 보고서

## 해석 제약

- 결과는 수익률 alpha나 거래전략을 검증하지 않는다.
- 초과 수렴을 투자자의 의도적 모방으로 단정하지 않는다.
- 조건부 정규 잔차, 단일 시장요인, rolling-window 선택은 모형 가정이다.
- 동일 표본에서 발견된 구조 구간을 사용하므로 구간별 결과는 설명적 강건성 분석이다.
- 결과를 본 뒤 창 길이·최소 관측·구간·판정 방향을 변경하지 않는다.
