# Zero-Run 조건부 배열 Null 기계성 감사 프로토콜 v1

- 동결일: 2026-07-20
- 연구 지위: 기존 zero-run 동시점 미시구조 5/5 결과를 확인한 뒤 수행하는 후속 mechanism audit
- 목적: 관찰된 관계가 거래 수와 zero-tick 수를 조건으로 보존한 무작위 배열에서도 생기는지 분리한다.
- 금지: 본 감사를 완전히 미관찰된 confirmatory 연구, 방향성 alpha 연구 또는 자동매매 검증으로 표현하지 않는다.

## 1. 이미 알고 있는 결과

- 2024-04-08 이상 2026-04-08 미만 Binance spot 7자산 raw aggTrades를 490,560개 15분 bucket으로 재구축했다.
- `zero_run_intensity=-run_z_zero`의 OOS 동시점 mechanism 5개는 모두 BH-FDR와 0.05 SD 기준을 통과했다.
- 5·15·30분 절대수익률과 실현변동성의 미래 family는 0/2였고 개발→OOS RMSE 개선률은 사실상 0%였다.
- 따라서 이번 감사는 동시점 5/5를 다시 선택하거나 미래 예측 후보를 되살리는 작업이 아니다.

## 2. 입력과 고정 표본

- 자산: BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT
- 분석 구간: 기존 zero-run 연구의 OOS인 2025-04-08 00:00 UTC 이상 2026-04-08 00:00 UTC 미만
- 최소 aggregate trade 수: 기존과 동일한 bucket당 200개
- 원본 입력: schema v2 tick frame의 `transaction_count`, `zero_ticks`, `zero_runs`, `run_z_zero`
- 파생 입력: 기존 zero-run `analysis_frame.parquet`의 동일 시점 mechanism outcome과 개발구간 scaler
- 기존 2년 표본의 threshold, 자산, horizon, winsor 범위 또는 outcome을 변경하지 않는다.

## 3. 정확한 조건부 배열 Null

각 bucket에서 전체 aggregate trade 수를 `n`, zero-tick 수를 `k`, non-zero 수를 `m=n-k`로 둔다. `n`과 `k`를 고정한 상태에서 zero/non-zero의 모든 이진 배열을 같은 확률로 본다.

Zero category run 개수 `R`의 정확한 조건부분포는 다음과 같다.

```text
P(R=r | n,k) = C(k-1,r-1) C(m+1,r) / C(n,k)
r = 1, ..., min(k,m+1)
```

각 draw에서 이 분포로 `R`을 직접 표본추출하고 기존 코드와 동일하게 계산한다.

```text
E[R|n,k] = k(m+1)/n
Var[R|n,k] = km(k-1)(m+1) / (n^2(n-1))
Z = (R-E[R|n,k]) / sqrt(Var[R|n,k])
null_zero_run_intensity = -Z
```

원본 `zero_runs`로 재계산한 Z와 저장된 `run_z_zero`의 최대 절대차는 `1e-12` 이하여야 한다.

## 4. Monte Carlo 표본과 재현성

- 반복 수: 999회
- random seed: 20260720
- OOS에서 자산 × transaction-count quintile × zero-tick-share quintile별 최대 80개를 고정 seed로 추출한다.
- 각 stratum의 모집단 수/표본 수를 WLS weight로 사용해 OOS 모집단을 복원한다.
- quintile 경계와 추출 row key를 산출물에 저장한다.
- exact combinatorial PMF는 `gammaln`과 `logsumexp`로 계산하며 정규근사로 대체하지 않는다.

## 5. Null FPR과 empirical percentile

기존 clustering cutoff `run_z_zero <= -1.96`의 조건부 null false-positive rate를 다음 23개 그룹에서 계산한다.

- pooled 1개
- 자산 7개
- transaction-count quintile 5개
- zero-tick-share quintile 5개
- liquidity quintile 5개 (`log_quote_volume` 기준)

그룹별로 아래를 저장한다.

- null FPR 평균, 2.5%·97.5% Monte Carlo 분위수
- 실제 empirical clustering share
- 실제 평균 zero-run intensity와 null 분포
- 실제 통계량의 Monte Carlo percentile
- one-sided empirical p-value

평균 intensity와 clustering share의 23개 p-value는 각각 별도 BH-FDR family로 보정한다. 그룹이 200개 미만이면 추론하지 않지만 선언 family에서 삭제하지 않고 부적격으로 표시한다.

### Calibration 기준

- pooled null FPR이 1.5% 이상 3.5% 이하
- 23개 그룹 중 90% 이상이 null FPR 5% 이하
- 위 두 조건을 모두 만족할 때 `-1.96` cutoff가 조건부 배열 null에서 실용적으로 calibration됐다고 판단한다.

### Empirical excess clustering 기준

- pooled에서 평균 intensity와 clustering share가 각각 BH q<=0.05 및 null 95th percentile 이상
- 7개 자산 중 최소 5개가 두 통계량 모두 같은 기준을 충족
- 두 조건을 모두 만족할 때 배열 순서에 count만으로 설명되지 않는 excess clustering이 있다고 판단한다.

## 6. 동시점 Mechanism 5개와 Null 비교

기존과 같은 다섯 outcome을 변경 없이 사용한다.

1. `log_amihud_illiquidity`
2. `zero_tick_share`
3. `log_mean_intertrade_ms`
4. `log_quote_volume`
5. `abs_aggressor_imbalance`

감사 표본에서 기존 개발 scaler를 적용한 outcome, 현재 절대수익률, 다른 6자산의 동시간 평균 절대수익률을 사용한다. 자산·UTC hour·UTC weekday fixed effect와 stratum WLS weight를 포함한다.

- 실제 zero-run intensity의 WLS coefficient를 계산한다.
- 각 exact-null draw의 coefficient 999개를 같은 식으로 계산한다.
- 실제 계수의 양측 empirical p-value를 산출하고 5개를 하나의 BH-FDR family로 보정한다.
- 기존 full-OOS 계수와 감사 표본 계수의 부호 일치 및 절대크기 비율도 저장한다.

개별 mechanism은 아래를 모두 충족할 때 count-conditioned arrangement null을 넘어선 관계로 표시한다.

- 기존 full-OOS mechanism 기준 통과
- 감사 표본 계수가 기존과 같은 부호
- 감사 표본 절대계수가 full-OOS 절대계수의 50% 이상
- null empirical BH q<=0.05
- 실제 계수가 null 95% interval 밖

Mechanism family는 5개 중 3개 이상이 통과하고, 그중 `log_amihud_illiquidity` 또는 `abs_aggressor_imbalance`가 포함될 때만 broad structural association으로 유지한다. `zero_tick_share`, 거래간격, 거래대금은 run-z 입력과 직접 또는 간접적으로 연결되므로 단독 근거로 사용하지 않는다.

## 7. 최종 판정 규칙

- cutoff calibration 실패: zero-run threshold 자체가 finite-sample 조건부 null에서 불안정
- calibration 통과 + excess clustering 실패: 실제 run 배열이 count-conditioned null과 구분되지 않음
- calibration 통과 + excess clustering 통과 + mechanism 미통과: 배열 clustering은 있으나 기존 동시점 경제 해석은 산식/구성으로 설명 가능
- calibration 통과 + excess clustering 통과 + mechanism 통과: 배열 순서와 동시점 시장구조의 관계가 count-conditioned null을 넘어섬

어느 판정에서도 미래 예측력, 의도적 herding, 인과관계 또는 거래 alpha를 주장하지 않는다. 기존 미래 family 0/2는 변경하지 않는다.

## 8. 사후 변경 금지와 한계

- 결과 확인 후 반복 수, cutoff, quintile 수, 표본 크기, 최소 거래 수, 자산 또는 mechanism outcome을 바꾸지 않는다.
- aggTrades는 동일 taker order의 fill을 집계하므로 개별 주문·호가 배열과 같지 않다.
- bid·ask quote가 없어 spread와 depth는 직접 검증할 수 없다.
- 조건부 null은 `n`과 `k`만 보존하며 intrabucket duration, trade size, aggressor sequence는 보존하지 않는다.
- 이 감사가 통과해도 새로운 quote/order-book 표본 없이는 시장 미시구조 인과 연구로 확대하지 않는다.
