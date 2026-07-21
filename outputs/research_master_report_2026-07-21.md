# 암호화폐 Herding 연구 통합 보고서

최종 갱신: 2026-07-21

## 1. 이 프로젝트는 무엇을 연구했나

이 프로젝트는 암호화폐들이 같은 시기에 서로 비슷하게 움직이는 현상이 실제 집단추종인지, 단순한 시장요인이나 통계식의 구조 때문에 그렇게 보이는지를 검증했습니다.

연구는 크게 세 축으로 진행했습니다.

1. 여러 코인의 수익률 분산을 이용하는 CSAD 연구
2. 개별 체결의 연속성을 이용하는 tick microstructure 연구
3. 이 결과가 미래수익률 예측과 자동매매 후보로 이어지는지 확인하는 OOS 검증

현재 자동매매와 tracker는 활성 연구 결론이 아닙니다. 사전 기준을 통과한 미래수익률 alpha가 없기 때문입니다.

## 2. 가장 중요한 최신 결론

선행논문의 no-intercept CSAD와 SCSAD 수치는 매우 정확하게 재현했습니다. 그러나 이 두 모형은 실제 herding이 전혀 없는 synthetic data에서도 거의 항상 음의 유의한 계수를 만들었습니다.

따라서 최신 결론은 다음과 같습니다.

- 논문 수치의 **재현에는 성공**했습니다.
- 그 음의 계수가 intentional herding을 식별한다는 주장에는 **실패**했습니다.
- 미래수익률 alpha도 **식별되지 않았습니다**.
- Zero-run은 유동성 높은 동시점 거래 상태를 설명하지만 5·15·30분 절대움직임을 OOS에서 예측하지 못했습니다.
- 현재 남은 연구 가치는 “왜 이 통계량이 음수가 되는가”를 설명한 방법론 감사와 시장 미시구조의 비방향성 연구입니다.

## 3. CSAD란 무엇인가

CSAD는 같은 시점에 여러 코인의 수익률이 시장 평균에서 얼마나 멀리 떨어져 있는지 계산합니다.

```text
CSAD_t = mean_i |R_i,t - R_m,t|
```

전통적인 해석은 시장이 크게 움직일 때 CSAD 증가세가 둔화되면 코인들이 서로 따라 움직일 가능성이 있다는 것입니다.

```text
CSAD_t = alpha + beta1 * |R_m,t| + beta2 * R_m,t^2 + error_t
```

Standard CSAD에서는 `beta2 < 0`이 필요조건으로 사용됩니다. 선행논문은 절편을 제거한 no-intercept 모형과 부호를 변환한 SCSAD도 사용했습니다.

## 4. 처음의 Binance baseline

- 기간: 2024-04-08 포함, 2026-04-08 미포함의 정확한 2년
- 데이터: Binance 1분봉
- universe: BTC, ETH를 포함한 고정 14개 USDT 페어
- 결과: baseline `beta2`는 양수
- 1분, 5분, 15분, 1시간, 4시간, 1일과 EW14/EW12를 조합한 12개 셀도 모두 양수

이 표본에서는 classical CSAD herding이 확인되지 않았습니다.

## 5. 선행논문 재현

CoinMarketCap 자료를 이용해 두 방식으로 재현했습니다.

### 동적 Top-200

- 전월 말 시가총액 순위로 다음 달 universe를 고정
- 전일 시가총액으로 시장수익률 가중
- Standard CSAD는 daily·weekly 모두 미지지
- No-intercept와 SCSAD는 corrected 4/4 통과

### 논문 고정 62종목

- 논문 Table 1과 같은 legacy CMC ID 62개
- 2018-01-01~2024-04-09, daily 2,291개
- Daily SCSAD: 우리 `-2.902 (t=-4.621)`, 논문 `-2.924 (t=-4.471)`
- No-intercept와 SCSAD daily·weekly corrected 4/4 통과
- Standard는 0/2

수치 재현도는 높습니다. 다만 fixed-62는 사후 생존목록이고 당일 시가총액 가중은 예측 가능한 투자 가중법이 아닙니다.

## 6. 시간·거래소·universe 외부검증

| 표본 | Equal/current corrected | Lagged-liquidity corrected | 판정 |
|---|---:|---:|---|
| CMC fixed-62 historical | 4/4 | 4/4 | 수치 재현 |
| CMC fixed-62 temporal holdout | 4/4 | 4/4 | 같은 provider·survivor list에서 시간 재현 |
| Binance fixed-14 | 3/4 | 3/4 | strict 미통과 |
| OKX listing-aware 14 | 3/4 | 1/4 | strict 미통과 |
| Binance archive point-in-time Top-50 | 2/4 | 0/4 | strict 미통과 |

상장폐지를 포함하고 universe와 가중법을 현실적으로 만들수록 corrected 관계가 약해졌습니다. 기존 통합 I-squared도 69.6~96.8%로 매우 높았습니다.

## 7. 최신 specification and mechanism audit

### 사전 고정한 감사

- 기존 10개 empirical 사양
- Daily·weekly 20개 패널
- Standard, no-intercept, SCSAD
- No-intercept와 같은 설명변수에 절편만 복원한 control
- Leave-one-out 시장수익률
- Universe N, 가중 HHI, BTC 비중
- 과거정보만 사용한 low/mid/high volatility regime
- 4개 비허딩 DGP × 16개 N·기간·가중 시나리오 × 300회
- HAC, BH-FDR, 7.5% false-positive 허용치

Protocol과 config의 SHA-256을 결과에 함께 저장했고 기존 결과는 덮어쓰지 않았습니다.

### 입력 무결성

20개 패널 모두 저장된 market return·CSAD와 감사 코드의 독립 재계산값이 `1e-12` 이내에서 일치했습니다.

### 절편 제거

Baseline에서 no-intercept가 음의 BH 지지를 보인 셀은 16개였습니다. 동일한 signed return, absolute return, squared return을 유지하고 절편만 복원하자 16개 모두 지지가 사라지거나 표준화 절대크기가 50% 이상 줄었습니다.

평균 표준화 계수는 다음처럼 바뀌었습니다.

| 빈도 | No-intercept 평균 | 절편 복원 평균 | 중앙 절대크기 감소율 |
|---|---:|---:|---:|
| Daily | -0.617 | +0.126 | 96.5% |
| Weekly | -0.988 | +0.135 | 92.8% |

이는 CSAD가 시장수익률 0 근처에서도 양수인데 회귀선을 원점에 강제로 통과시키면서 제곱항이 음수로 보상하는 구조와 일치합니다.

### 비허딩 synthetic null

의도적 모방 규칙이 없는 네 자료생성과정을 사용했습니다.

1. 독립 Gaussian 자산
2. 공통 시장요인과 독립 고유충격
3. 공통 확률변동성이 있는 이분산 요인
4. 자유도 5 fat-tail 공통요인과 상관관계

| 모형 | BH false-positive 범위 | 중앙값 | 7.5% 초과 |
|---|---:|---:|---:|
| Standard CSAD | 0~11.3% | 1% | 8/64 |
| No-intercept CSAD | 99~100% | 100% | 64/64 |
| SCSAD | 97~100% | 100% | 64/64 |
| Intercept-restored control | raw 0~10.3% | raw 1% | 진단용 |

No-intercept와 SCSAD는 herding이 없어도 거의 결정적으로 음의 유의성을 만들었습니다. 따라서 이 두 모형의 일반적인 p-value를 herding false-positive 확률로 해석할 수 없습니다.

### Empirical과 null의 직접 비교

10개 empirical 사양 × 2빈도 × 3모형 × 4DGP의 240개 비교를 수행했습니다.

- BH-FDR로 simulation null보다 더 음수인 empirical 비교: `0/240`
- No-intercept와 SCSAD empirical percentile 중앙값: `100%`
- 구조적 강건성 사전 기준 통과: `0/10`

여기서 percentile 100%는 empirical 계수가 null보다 더 음수가 아니라, 오히려 null 분포의 오른쪽에 있었다는 뜻입니다.

### Leave-one-out과 regime

- CMC historical·holdout의 corrected 4/4는 leave-one-out에서도 그대로 유지됐습니다.
- 하지만 대응 비허딩 null FPR이 최대 100%여서 식별 근거가 되지 못했습니다.
- 사전 volatility regime 회귀 168개가 추정 가능했고 유의한 반대 양의 비선형항이 9개 있었습니다.
- 고변동 구간에서 Standard CSAD가 양수로 돌아서는 경우가 여러 provider에서 나타났습니다.

### 이질성

Random-effects I-squared는 모형·빈도별 69.3~96.9%였습니다. Provider와 universe를 동시에 넣은 meta-regression은 둘이 교락되어 3/3 모형에서 rank deficient였습니다.

Moderator별 단변량 회귀는 full-rank이지만 CoinMarketCap과 fixed-62 효과가 같은 계수로 나타나므로 둘을 분리해 인과적으로 해석할 수 없습니다.

## 7-1. 음의 계수가 생기는 수학적 원인

2026-07-21에 별도 사전등록 연구로 기존 null 감사의 97~100% 오탐 원인을 폐형식과 Monte Carlo로 검증했습니다.

- 독립 Gaussian 동일가중에서 `M`과 CSAD는 독립이며 `E[CSAD|M]`는 양의 상수입니다.
- No-intercept는 이 상수를 `|M|`와 `M^2`로 원점부터 근사하므로 population 제곱항이 반드시 음수입니다.
- SCSAD는 `sign(M)` 계단을 선형·세제곱 다항식으로 근사하므로 population 세제곱항이 반드시 음수입니다.
- 12,000시점 × 200회 × N=14·50·62 수렴 검증에서 기계적 모형 gate는 `5/6`, 절편 control은 `6/6` 통과했습니다.
- 기계적 두 모형의 Gaussian raw FPR 범위는 `100.0%~100.0%`였습니다.
- 대칭 비허딩 DGP 강건성 셀은 `45/45` 통과했습니다.
- 최종 판정은 `mechanical_null_not_confirmed`입니다.

따라서 no-intercept와 현재 SCSAD는 선행논문 수치 재현·모형 진단용으로만 보존하며 intentional herding 검정으로 사용하지 않습니다. Standard CSAD도 공통요인과 이분산성이 있는 현실적 null로 검정 크기를 먼저 교정해야 합니다.

상세 보고서: `outputs/v2/csad_mechanical_derivation_v1/csad_mechanical_derivation_report.md`

### v1.1 유한표본 수렴 보충

원 v1은 N=62 SCSAD의 99% Monte Carlo 평균 CI 한 셀 때문에 5/6으로 실패했으며 이 판정을 그대로 보존합니다. 이후 별도 등록한 독립 seed의 T=3,000·12,000·48,000, N=14·50·62, 셀당 300회 보충 분석으로 유한표본 수렴을 점검했습니다.

보충 기계적 셀은 `6/6`, control은 `6/6` 통과했습니다. 보충 판정은 `finite_sample_convergence_supported`입니다.

따라서 정확한 Gaussian 폐형식과 유한표본 수렴은 지지되지만, 원 사전등록 결과를 사후 성공으로 바꾸지는 않습니다.

보충 보고서: `outputs/v2/csad_mechanical_derivation_v1/supplement_v1_1/csad_mechanical_convergence_supplement_report.md`
## 8. Tick 연구와 미래수익률

구형 tick 결과에서는 `micro_herding_up/down`을 가격 방향으로 잘못 해석한 문제가 발견됐습니다. Schema v2에서 다음을 분리했습니다.

- `run_clustering_side`: 연속 체결 clustering의 어느 쪽이 극단적인가
- `price_direction`: 같은 구간 가격의 상승·하락
- `aggressor_direction`: buyer-maker 기반 순공격 방향

2년 raw 7종목 490,560개 15분 bucket에서:

- Run side와 가격 방향 일치율 49.23%
- Run side와 aggressor 방향 일치율 48.13%
- 시장중립 30분 미래반응의 세 계수 모두 BH `q=0.9336`
- 연속형 run-z OOS 사전 기준 통과 0/9
- 미래 계수 6개의 95% CI 모두 ±5bp 이내

따라서 기존 XRP·DOGE·AVAX 방향성 전략과 tracker 결론은 corrected 연구 근거로 사용하지 않습니다.

### Zero-run 비방향성 미시구조 연구

같은 2년 7자산을 이용하되 방향성 수익률을 전혀 검정하지 않고 `zero_run_intensity=-run_z_zero`가 거래 구조와 미래 움직임의 크기에 연결되는지 사전등록했습니다.

- 동시점 5개 대리변수는 OOS BH-FDR와 0.05 SD 기준을 모두 통과했습니다.
- 1 SD 강한 zero-run은 Amihud형 비유동성 `-0.207 SD`, 평균 aggregate-trade 간격 `-0.537 SD`, USDT 거래대금 `+0.503 SD`, 절대 aggressor imbalance `-0.120 SD`와 연결됐습니다.
- 이는 강한 zero-run이 거래가 적고 불안정한 충격 상태보다 거래가 활발하고 order-flow가 비교적 균형적인 상태에 가깝다는 해석과 일치합니다.
- 다만 `run_z_zero` 자체가 거래 수와 zero-tick 수를 조건으로 계산되므로 동시점 관계에는 산식의 구조적 결합이 포함됩니다. 인과나 독립 예측력으로 해석하지 않습니다.
- 미래 절대수익률과 실현변동성의 5·15·30분 clustered BH q는 각각 모두 `0.736`과 `0.745` 이상이었습니다.
- 개발→OOS RMSE 개선률은 `-0.0073%~+0.0004%`로 사실상 0이었습니다.
- 42개 LOAO, 199회×6개 circular-shift permutation, 6개 미래-lead placebo를 포함한 최종 미래 family 판정은 `0/2`입니다.

결론적으로 zero-run은 동시점 시장 상태를 기술하는 변수일 수 있지만, 현재 표본에서는 단기 위험 크기를 예측하는 feature도 아닙니다.

## 9. 지금 무엇을 주장할 수 있나

주장 가능한 내용:

- 선행논문의 CMC no-intercept·SCSAD 계수는 높은 정확도로 재현된다.
- 그 결과는 universe, provider, weighting에 매우 민감하다.
- No-intercept의 음수는 절편 제약의 기계성과 강하게 연결된다.
- SCSAD도 현재 형태에서는 비허딩 null을 구분하지 못한다.
- Classical Standard CSAD는 더 정상적인 null FPR을 보이지만 현재 empirical 표본에서 일관된 음의 관계가 없다.
- Zero-run intensity는 거래가 활발하고 aggressor 불균형이 낮은 동시점 상태와 안정적으로 연결된다.

주장할 수 없는 내용:

- 투자자들이 의도적으로 서로를 모방했다.
- Corrected CSAD 음의 계수가 보편적인 crypto herding을 입증한다.
- Herding event 뒤에 거래 가능한 방향성 alpha가 있다.
- 현재 결과로 자동매매를 시작할 수 있다.
- Zero-run이 5·15·30분 미래 변동성이나 절대수익률을 유의미하게 개선한다.

## 10. 재현 파일

- 사전등록: `research_protocols/csad_specification_audit_v1.md`
- 설정: `configs/research/csad_specification_audit_v1.yaml`
- 실행: `scripts/run_csad_specification_audit.py`
- 최신 보고서: `outputs/v2/csad_specification_audit_v1/csad_specification_audit_report_v1_1.md`
- 모형 진단: `outputs/v2/csad_specification_audit_v1/model_diagnostics.csv`
- FPR: `outputs/v2/csad_specification_audit_v1/false_positive_summary.csv`
- Empirical-vs-null: `outputs/v2/csad_specification_audit_v1/empirical_vs_null.csv`
- 구조적 판정: `outputs/v2/csad_specification_audit_v1/structural_robustness_decisions_v1_1.csv`
- Simulation 원행: `outputs/v2/csad_specification_audit_v1/simulation_replicates.parquet`
- 입력 해시: `outputs/v2/csad_specification_audit_v1/input_manifest.csv`
- 교정 그림: `outputs/v2/csad_specification_audit_v1/plots/false_positive_rates_contrast_fixed.png`, `outputs/v2/csad_specification_audit_v1/plots/empirical_vs_null_labels_fixed.png`
- Zero-run 프로토콜: `research_protocols/zero_run_microstructure_v1.md`
- Zero-run 설정: `configs/research/zero_run_microstructure_v1.yaml`
- Zero-run 실행·검증: `scripts/run_zero_run_microstructure.py`, `scripts/verify_zero_run_microstructure.py`
- Zero-run 보고서: `outputs/v2/zero_run_microstructure_v1/zero_run_microstructure_report.md`
- Zero-run 판정: `outputs/v2/zero_run_microstructure_v1/future_decisions.csv`, `outputs/v2/zero_run_microstructure_v1/family_decisions.csv`

## 11. 다음 연구 원칙

1. No-intercept와 SCSAD 음의 계수만 이용한 herding 주장은 종료합니다.
2. 같은 5년 표본에서 threshold, 제외 종목, weight cap을 다시 최적화하지 않습니다.
3. CSAD를 계속 연구하려면 비허딩 null에 맞춰 보정된 새 식별식을 먼저 사전등록해야 합니다.
4. 새 식별식은 완전히 새로운 기간 또는 거래소에서 검증해야 합니다.
5. AggTrades만으로 가능한 zero-run 동시점 구조와 후속 절대변동 연구는 완료됐습니다. 같은 표본에서 horizon·threshold·종목을 다시 최적화하지 않습니다.
6. Zero-run 후속 연구는 bid·ask quote가 있는 완전히 새로운 기간 또는 거래소에서 spread·depth·회복속도를 사전등록하는 경우에만 진행합니다.
7. 사전 통계·경제성 기준을 통과하기 전에는 tracker, paper-sim, 자동매매를 활성화하지 않습니다.

## 12. 최종 평가

이 프로젝트의 가장 큰 성과는 “논문과 같은 음의 계수를 찾은 것”이 아니라, 그 계수가 herding이 없는 자료에서도 거의 자동으로 나타난다는 것을 재현 가능한 실험으로 확인한 것입니다.

즉 연구 결과는 유망한 매매 alpha가 아니라 더 엄격한 방법론적 반증입니다. CSAD 축은 기존 corrected 해석을 제한하는 단계이고, tick 축은 zero-run이 동시점 유동성 상태를 설명해도 미래 움직임의 크기는 예측하지 못한다는 경계까지 확인했습니다.

## 13. Zero-run 조건부 배열 Null 후속 감사

기존 동시점 5/5 결과가 거래 수와 zero-tick 수의 산식 결합만으로도 생기는지 확인하기 위해, null 결과를 보기 전에 exact conditional 배열 감사를 봉인했습니다.

- OOS 적격 모집단: 233,201개 15분 bucket
- 층화 감사 표본: 9,044개, 141개 자산×거래수×zero-share 층
- 조건부 null: 각 row의 `transaction_count`와 `zero_ticks` 고정, 999회 exact 조합분포
- Pooled null FPR: 2.537%
- 실제 `run_z_zero<=-1.96` share: 87.347%
- Excess clustering: 7/7 자산 통과
- Count-conditioned mechanism family: 0/5
- 최종 분류: `clustering_beyond_counts_but_mechanism_not_distinct`

따라서 zero/non-zero 배열 순서가 단순 무작위가 아니라는 점은 지지됩니다. 그러나 기존 비유동성·거래간격·거래대금·aggressor 관계를 독립적인 경제 메커니즘으로 확정할 수는 없습니다. 미래 family 0/2와 tracker 비활성 판정도 변경되지 않습니다.

최종 논문형 원고, 19개 가설 종료표, 입력·산출물 SHA-256 manifest와 읽기 전용 verifier는 `outputs/v2/final_research_completion_v1/`에 보존했습니다.
