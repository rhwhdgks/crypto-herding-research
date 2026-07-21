# Zero-Run 시장 미시구조 사전등록 프로토콜 v1

- 동결일: 2026-07-20
- 분석 목적: 방향성 수익률 alpha가 아니라 zero-run intensity의 시장 미시구조적 의미와 미래 가격 움직임의 크기 예측력을 검증한다.
- 출력 위치: `outputs/v2/zero_run_microstructure_v1`

## 1. 기존에 알고 있는 정보와 연구 지위

- 동일 2년 7자산 raw `aggTrades` schema v2 frame에서 winner event의 91.82%가 zero-run이라는 사실을 알고 있다.
- 기존 continuous run-z OOS 연구에서 30분 시장중립 절대수익률에 대한 `run_intensity_zero` 계수가 작고 OOS에서 축소됐다는 사실을 알고 있다.
- 따라서 이번 연구는 완전히 미관찰된 독립 confirmatory 연구가 아니다. 다만 5분·15분 결과, 1분 수익률 기반 실현변동성, 미시구조 대리변수 family, LOAO, permutation, placebo, 개발→OOS 예측평가는 아래 규격으로 결과 확인 전에 고정한다.
- 어떤 결과도 방향성 수익률 alpha, 매수·매도 신호 또는 herding의 의도적 모방 증거로 표현하지 않는다.

## 2. 표본과 입력

- Binance spot 7자산: BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT
- tick 표본: 2024-04-08 00:00 UTC 이상, 2026-04-08 00:00 UTC 미만
- 15분 complete grid, 자산당 70,080행, 전체 490,560행
- 개발 구간: 2024-04-08 00:00 UTC 이상, 2025-04-08 00:00 UTC 미만
- OOS 구간: 2025-04-08 00:00 UTC 이상, 2026-04-08 00:00 UTC 미만
- zero-run 분석에는 15분 bucket당 aggregate trade 200건 이상인 행만 사용한다.
- 미래 가격 결과는 동일 기간 Binance 1분 OHLCV의 종가를 사용한다. 신호시각 `t`의 기준가격은 `t-1분` bar의 종가이며, 5·15·30분 종료가격은 각각 `t+4`, `t+14`, `t+29분` bar의 종가다.
- 결과 가용시각이 개발/OOS 경계에 닿거나 넘는 행은 해당 split에서 제외한다.

## 3. 핵심 설명변수

- `zero_run_intensity = -run_z_zero`
- 값이 클수록 조건부 zero-price-change run clustering이 강하다.
- 연속형 변수만 사용하며 percentile event, winner label, 임계값 최적화 또는 종목 선택을 하지 않는다.
- 0.5%·99.5% winsorization과 평균·표준편차는 개발 구간에서만 적합하고 OOS·LOAO·placebo에 고정 적용한다.

## 4. 미시구조 대리변수 Family M

동시점 연관성을 다음 5개 outcome으로 검정한다.

1. `log_amihud_illiquidity`: `log1p(|15분 수익률| × 1,000,000 / USDT 거래대금)`
2. `zero_tick_share`: 연속 aggregate trade 중 가격 변화가 0인 비중
3. `log_mean_intertrade_ms`: `log(15분 / aggregate trade 수)`인 평균 aggregate-trade 간격 대리변수
4. `log_quote_volume`: `log1p(15분 USDT 거래대금)`
5. `abs_aggressor_imbalance`: buyer-maker 정보로 만든 signed imbalance의 절댓값

각 outcome과 `zero_run_intensity`를 개발 구간 scaler로 표준화한다. 회귀는 현재 절대수익률과 다른 6자산의 동시간 평균 절대수익률을 통제하고 자산, UTC hour, UTC weekday fixed effect를 포함한다. 표준오차는 UTC date로 군집화한다.

Family M의 5개 양측 p-value는 split별 BH-FDR로 보정한다. OOS `q<=0.05`이고 절대 표준화 계수가 0.05 이상이면 의미 있는 미시구조 연관성으로 표시한다. 이 family는 동시점 구성타당도이며 미래 예측 또는 인과 증거가 아니다.

## 5. 미래 움직임 Family A와 V

### Family A: 미래 절대수익률

- `abs_forward_return_5m_bps`
- `abs_forward_return_15m_bps`
- `abs_forward_return_30m_bps`

### Family V: 미래 실현변동성

- `realized_volatility_5m_bps`
- `realized_volatility_15m_bps`
- `realized_volatility_30m_bps`
- 각 값은 신호 직후 1분 로그수익률 제곱합의 제곱근이다.

두 family 모두 다음 현재시점 정보만 통제한다.

- 현재 15분 절대수익률
- 다른 6자산의 동시간 평균 절대수익률
- 현재까지 96개 15분 bucket의 trailing volatility
- `log1p(transaction_count)`
- `log1p(total_quote_quantity)`
- `zero_tick_share`
- `abs_aggressor_imbalance`
- 자산, UTC hour, UTC weekday fixed effect

표준오차는 UTC date로 군집화한다. 각 split에서 3개 horizon을 family별 BH-FDR로 보정한다.

## 6. 시간 OOS 예측평가

- 개발 구간에서 baseline 모형과 `zero_run_intensity`를 추가한 augmented 모형을 각각 적합한다.
- 개발 scaler와 계수를 고정해 OOS를 예측한다.
- RMSE, MAE, OOS R-squared와 augmented 대비 개선률을 저장한다.
- OOS RMSE 개선률 0.25% 이상을 최소 증분 예측력 기준으로 둔다.
- OOS 자체 회귀의 유의성만으로 예측력이라고 부르지 않는다.

## 7. Leave-One-Asset-Out 강건성

- OOS에서 한 자산씩 제외하고 Family A·V의 6개 모형을 다시 적합한다.
- 전체 OOS 계수와 같은 부호가 7회 중 최소 6회여야 한다.
- LOAO 계수 절댓값 중앙값이 전체 OOS 계수 절댓값의 50% 이상이어야 한다.
- LOAO p-value는 family별 21개 검정에 BH-FDR을 적용해 저장하지만, 단일 자산 제외 결과를 골라 결론을 바꾸지 않는다.

## 8. Permutation과 placebo

- OOS residualized `zero_run_intensity`를 모든 자산에 동일한 무작위 시차로 circular shift한다.
- 최소 시차는 7일이며 199회, seed 20260720으로 고정한다.
- 이 방식은 횡단면 구조와 각 시계열의 자기상관 구조를 가능한 한 보존하면서 시점 정렬만 끊는다.
- 실제 계수의 양측 empirical p-value를 계산하고 Family A·V별 3개를 BH-FDR 보정한다.
- 별도 placebo는 7일 뒤 `zero_run_intensity`를 현재 결과에 붙이는 미래 lead falsification이다. 이는 거래 feature가 아니라 누락된 느린 regime의 진단이다.
- placebo가 `q<=0.05`, 실제 계수와 같은 부호, 실제 절댓값의 50% 이상이면 해당 horizon을 veto한다.

## 9. 사전 성공 기준

개별 미래 horizon은 아래를 모두 만족해야 통과한다.

1. OOS UTC-day clustered `q<=0.05`
2. permutation BH `q<=0.05`
3. 절대 계수가 `max(1bp, OOS outcome 평균의 5%)` 이상
4. LOAO 부호 일치 6/7 이상 및 LOAO 중앙 절대크기 비율 50% 이상
5. 개발→OOS augmented RMSE 개선률 0.25% 이상
6. 미래 lead placebo veto가 아님

Family A 또는 V의 성공은 15분 horizon이 통과하고 5분 또는 30분 중 하나 이상이 함께 통과할 때만 인정한다. 한 horizon만 통과하면 탐색적 단서로만 남긴다.

미시구조 mechanism family는 5개 중 3개 이상이 OOS 기준을 통과하고, 그중 `log_amihud_illiquidity` 또는 `zero_tick_share`가 포함될 때만 구조적 연관성이 넓게 지지됐다고 판단한다.

## 10. 해석과 금지사항

- Family A·V가 통과해도 예측 대상은 움직임의 크기이지 방향이 아니다.
- 결과가 양호해도 tracker, paper-sim, 자동매매 또는 방향성 포지션을 활성화하지 않는다.
- 동일 표본에서 horizon, 최소 거래 수, winsor 범위, 통제변수, 경제성 기준, 자산 목록을 사후 변경하지 않는다.
- aggTrades에는 bid·ask quote가 없으므로 quoted/effective spread를 직접 측정했다고 표현하지 않는다.
- `transaction_count`와 intertrade proxy는 개별 fill이 아니라 Binance aggregate trade 단위임을 명시한다.
- 실패 결과도 그대로 보고하고 zero-run 연구의 종료 또는 외부 quote-data 연구 필요성을 판단한다.
