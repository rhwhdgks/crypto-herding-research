# Continuous Run-Z OOS Protocol v1

- 동결일: 2026-07-19
- 입력: 동일 2년 Binance spot raw `aggTrades` 7종목 15분 schema v2 frame
- 기존에 알고 있는 결과: winner label의 91.82%가 zero-run이고, winner의 가격·aggressor 방향 일치율이 약 49%이며, winner 기반 30분 효과가 practical null이다.
- 아직 보지 않은 결과: `run_z_up`, `run_z_down`, `run_z_zero`를 동시에 연속형 feature로 넣은 개발·OOS 계수.

## 연구 성격

이 분석은 winner-take-all label을 폐기하고 세 category의 run clustering을 각각 연속형으로 분리하는 새 feature specification 연구입니다. OOS 기간의 winner 요약은 이미 확인했으므로 완전히 untouched된 외부 검증은 아닙니다. 다만 연속형 공동 회귀 specification은 계수를 보기 전에 동결하며, 둘째 해를 `specification-held-out OOS`로 표현합니다.

## 표본과 분할

- 전체: 2024-04-08 00:00 UTC ~ 2026-04-07 23:45 UTC
- 개발: 2024-04-08 00:00 UTC ~ 2025-04-08 00:00 UTC 미만
- OOS: 2025-04-08 00:00 UTC ~ 2026-04-08 00:00 UTC 미만
- 종목: BTC, ETH, XRP, SOL, DOGE, ADA, AVAX USDT
- 버킷: 15분 UTC complete grid
- 전 UTC hour를 사용하고 hour fixed effect를 포함합니다.
- 최소 체결 수 200개인 bucket만 모형에 사용합니다.
- 개발 행의 30분 outcome 가용 시각이 OOS 경계에 닿거나 넘으면 개발 표본에서 제외합니다.

## Feature 정의

조건부 category-run z가 더 작을수록 clustering이 강하므로 해석을 쉽게 하기 위해 다음과 같이 부호를 바꿉니다.

- `run_intensity_up = -run_z_up`
- `run_intensity_down = -run_z_down`
- `run_intensity_zero = -run_z_zero`

세 feature를 하나의 회귀에 동시에 넣습니다. `min(up, down, zero)`, `run_clustering_side`, 이벤트 percentile winner는 설명변수로 사용하지 않습니다.

## Point-in-time 변환

- 연속형 설명변수의 0.5%/99.5% winsorization 경계는 개발 구간에서만 적합합니다.
- 평균과 표준편차도 개발 구간에서만 적합하고 OOS에 고정 적용합니다.
- 회귀 계수는 개발 구간 1표준편차 변화당 outcome 변화를 의미합니다.
- 24시간 trailing volatility는 현재 bucket까지의 96개 15분 수익률로 계산합니다.

## 공통 통제변수

- 현재 bucket return과 절댓값
- 나머지 6자산의 현재 동시간 평균 return과 절댓값
- 24시간 trailing volatility
- `log1p(transaction_count)`
- `log1p(total_quote_quantity)`
- 종목 fixed effect
- UTC hour fixed effect
- 가격 방향 fixed effect

미래 outcome family에는 현재 `aggressor_imbalance`를 추가해 run intensity가 현재 order-flow를 넘어서도 증분 정보를 갖는지 봅니다.

## Family A: 동시 Aggressor 구성타당도

- outcome: 현재 bucket `aggressor_imbalance`
- feature: up/down/zero intensity 3개 공동 계수
- covariance: UTC-day cluster
- 사전 방향: up intensity 계수 > 0, down intensity 계수 < 0
- zero intensity는 사전 방향을 두지 않으며 양측 검정합니다.
- 개발과 OOS에서 각각 3개 p-value를 별도 BH-FDR family로 보정합니다.
- OOS `q<=0.05`, 사전 방향 일치, 절댓계수 0.02 이상을 모두 충족할 때만 구성타당도 후보로 표시합니다.

## Family B: 시장중립 30분 수익률

- outcome: 각 자산 30분 return - 나머지 6자산 동시간 30분 return 평균
- feature: up/down/zero intensity 3개 공동 계수
- covariance: UTC-day cluster
- 개발과 OOS에서 각각 3개 양측 p-value를 별도 BH-FDR family로 보정합니다.
- OOS `q<=0.05`이고 절대 조정계수가 5bp 이상일 때만 경제적 조건부 연관성으로 표시합니다.
- 경제성 기준을 통과해도 거래비용·외부 검증 전에는 alpha로 부르지 않습니다.

## Family C: 시장중립 30분 절대수익률

- outcome: `abs(각 자산 30분 return) - 나머지 6자산 abs(30분 return) 평균`
- 의미: 시장 공통 변동을 제거한 자산 고유 후속 절대움직임
- feature: up/down/zero intensity 3개 공동 계수
- covariance: UTC-day cluster
- 개발과 OOS에서 각각 3개 양측 p-value를 별도 BH-FDR family로 보정합니다.
- OOS `q<=0.05`이고 절대 조정계수가 5bp 이상일 때만 경제적 조건부 연관성으로 표시합니다.

## 추론 적격성

- 각 split의 최소 완전 관측치 100,000개
- 각 split의 최소 UTC-day cluster 300개
- 사전 검정 크기: split별 Family A 3개, Family B 3개, Family C 3개
- family 간 q-value를 섞지 않습니다.
- 개발 결과로 OOS feature를 선택하지 않고 세 feature를 모두 OOS에서 검정합니다.

## 추가 진단

- split별 feature 상관행렬
- split별 공동 회귀 condition number
- 종목별 계수는 이질성 기술용으로만 저장하고 사전 family 결론을 대체하지 않습니다.
- 월별 계수나 최적 threshold를 탐색하지 않습니다.

## 승격 금지

사전 기준을 통과한 OOS 계수가 있어도 이 분석 하나만으로 tracker, paper-sim, 실시간 신호, 자동매매를 활성화하지 않습니다. 다른 기간 또는 거래소의 외부 검증을 추가로 요구합니다.
