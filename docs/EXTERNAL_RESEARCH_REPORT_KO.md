# 암호화폐 Herding 연구 외부 공개 보고서

최종 정리일: 2026-07-21

## 1. 한눈에 보는 결론

이 연구는 암호화폐들이 함께 움직이는 현상이 투자자의 의도적 집단추종인지, 공통 시장충격이나 통계모형의 구조 때문에 그렇게 보이는지를 검증했습니다.

가장 중요한 결론은 세 가지입니다.

1. 선행논문의 CoinMarketCap no-intercept CSAD와 SCSAD 계수는 높은 정확도로 재현됐습니다.
2. 그러나 두 모형은 투자자 모방이 전혀 없는 합성자료에서도 거의 항상 음의 유의한 계수를 만들었습니다.
3. 1분봉과 tick 자료에서 여러 이벤트를 검증했지만, 사전 기준을 통과한 미래수익률 alpha는 확인되지 않았습니다.

따라서 이 프로젝트의 성과는 매매전략 발견이 아니라, 기존 herding 지표가 무엇을 식별하고 무엇을 식별하지 못하는지 밝힌 재현·반증 연구입니다.

## 2. 연구 배경과 목적

시장이 급등하거나 급락할 때 여러 코인이 비슷한 방향으로 움직이면 흔히 “군중이 서로를 따라갔다”고 설명합니다. 하지만 같은 움직임은 다음 상황에서도 나타날 수 있습니다.

- 비트코인이나 거시시장이라는 공통요인에 동시에 반응한 경우
- 변동성이 갑자기 커진 경우
- 거래소 상장종목과 가중방식이 결과를 좌우한 경우
- 회귀식의 절편 또는 변수변환이 음의 계수를 기계적으로 만든 경우

이 연구는 단순히 음의 계수를 찾는 대신 다음 질문을 순서대로 검증했습니다.

- 선행논문의 수치를 같은 자료와 방법으로 재현할 수 있는가?
- 다른 기간, 거래소, universe에서도 결과가 유지되는가?
- herding이 없는 null data에서도 같은 신호가 나타나는가?
- tick 수준의 체결 연속성이 미래수익률이나 미래 변동을 예측하는가?

## 3. CSAD란 무엇인가

CSAD(cross-sectional absolute deviation)는 같은 시점에 여러 자산의 수익률이 시장수익률에서 평균적으로 얼마나 떨어져 있는지 측정합니다.

```text
CSAD_t = (1/N_t) * sum_i |R_i,t - R_m,t|
```

- `R_i,t`: 자산 `i`의 시점 `t` 수익률
- `R_m,t`: 같은 시점의 시장수익률
- `N_t`: 그 시점에 포함된 자산 수

Classical standard CSAD 회귀식은 다음과 같습니다.

```text
CSAD_t = alpha + beta1 * |R_m,t| + beta2 * R_m,t^2 + epsilon_t
```

전통적으로 `beta2 < 0`이면 시장 움직임이 커질 때 횡단면 분산의 증가세가 둔화됐다고 해석합니다. 이는 herding의 필요조건으로 사용할 수 있지만, 그 자체로 투자자가 서로를 보고 의도적으로 모방했다는 충분조건은 아닙니다.

이 연구는 standard 모형 외에 선행논문이 사용한 두 사양을 감사했습니다.

- No-intercept: CSAD 회귀에서 절편을 제거
- SCSAD: 시장수익률 부호를 CSAD에 곱한 뒤 선형항과 세제곱항으로 회귀

## 4. 데이터와 연구 범위

### 4.1 Binance 1분봉 baseline

- 기간: 2024-04-08 포함, 2026-04-08 미포함
- 빈도: 1분
- universe: Binance USDT 현물 14종목
- 종목: BTC, ETH, BNB, XRP, ADA, SOL, DOGE, LINK, AVAX, DOT, LTC, TRX, ATOM, NEAR
- 시장수익률: 동일가중
- 추론: Newey-West HAC

### 4.2 CoinMarketCap 선행논문 재현

두 universe를 구분했습니다.

- Dynamic Top-200: 전월 말 시가총액을 이용해 다음 달 universe를 고정
- Fixed-62: 선행논문 Table 1과 같은 legacy CMC ID 62개를 고정

주요 재현기간은 2018-01-01부터 2024-04-09까지입니다. Fixed-62 직접 재현은 daily 2,291개 관측을 사용했습니다.

### 4.3 외부검증

- CMC temporal holdout: 2024-04-10~2026-07-18
- Binance fixed-14: 2021-04-10~2026-04-09
- OKX listing-aware 14: 2021-04-10~2026-04-09
- Binance archive point-in-time Top-50: 2021-04-10~2026-04-09

Point-in-time 연구는 직전 달 자료로 다음 달 Top-50을 정하고, 중간 상장폐지와 관측 공백을 반영했습니다.

### 4.4 Tick 연구

최종 교정 연구는 Binance raw aggTrades 7종목의 정확한 2년 자료를 사용했습니다.

- 15분 bucket: 490,560개
- 체결 연속성, 가격 방향, aggressor 방향을 별도 변수로 분리
- 미래 horizon: 5분, 15분, 30분 중심
- 개발표본과 OOS를 분리하고 block·cluster 추론 적용

## 5. Binance baseline 결과

고정 14자산의 2년 1분봉에서 standard CSAD의 `beta2`는 양수였습니다. 1분, 5분, 15분, 1시간, 4시간, 1일과 EW14·EW12를 조합한 12개 강건성 셀도 모두 양수였습니다.

따라서 이 표본은 classical CSAD herding을 지지하지 않습니다. 낮은 CSAD 이벤트는 행동적 herding으로 단정하지 않고 `low_dispersion`으로 이름을 교정했습니다. 이벤트 이후 보유기간 수익률도 UTC-day block 추론에서 유의하지 않았습니다.

## 6. 선행논문 재현 결과

### 6.1 Dynamic Top-200

Standard CSAD는 daily와 weekly 모두 herding을 지지하지 않았습니다. 반면 no-intercept와 SCSAD는 daily·weekly의 corrected 4개 검정을 모두 통과했습니다.

### 6.2 Fixed-62 직접 재현

선행논문과 같은 62개 legacy ID를 사용했을 때 수치가 매우 가깝게 재현됐습니다.

| 모형 | 이 연구 daily 계수 | 선행논문 daily 계수 |
|---|---:|---:|
| No-intercept | -1.837 | -1.850 |
| SCSAD | -2.902 | -2.924 |

SCSAD t-통계량도 이 연구 `-4.621`, 논문 `-4.471`로 근접했습니다. 즉 “논문 표의 숫자를 재현할 수 있는가”라는 질문에는 그렇다고 답할 수 있습니다.

그러나 fixed-62는 표본 종료 뒤 살아남은 종목을 과거 전체에 고정한 목록일 수 있고, 당일 시가총액 가중은 실제 시점에서 완전히 예측 가능한 가중법이 아닙니다. 수치 재현은 행동적 herding 식별이나 투자 가능성을 뜻하지 않습니다.

## 7. 기간·거래소·universe 외부검증

| 표본 | Equal/current corrected | Lagged-liquidity corrected | 판정 |
|---|---:|---:|---|
| CMC fixed-62 historical | 4/4 | 4/4 | 수치 재현 |
| CMC fixed-62 temporal holdout | 4/4 | 4/4 | 같은 provider·목록 내 시간 재현 |
| Binance fixed-14 | 3/4 | 3/4 | strict 기준 미통과 |
| OKX listing-aware 14 | 3/4 | 1/4 | strict 기준 미통과 |
| Binance point-in-time Top-50 | 2/4 | 0/4 | strict 기준 미통과 |

CMC holdout에서는 corrected 관계가 유지됐지만 같은 공급자와 생존목록을 사용했습니다. 거래소와 universe를 바꾸고, 과거 정보만으로 종목과 가중치를 정할수록 결과가 약해졌습니다.

연구 간 이질성을 나타내는 I-squared는 약 69~97%였습니다. Provider와 fixed-universe 효과가 서로 교락되어 어느 하나를 독립 원인으로 해석할 수도 없었습니다.

## 8. Specification audit

결과를 보기 전에 protocol과 config를 고정하고 다음을 감사했습니다.

- Empirical 10개 사양 × daily·weekly 20개 패널
- Standard, no-intercept, SCSAD, intercept-restored control
- Leave-one-out 시장수익률
- 자산 수, 가중 HHI, BTC 비중, 과거정보 기반 변동성 regime
- 4개 비허딩 DGP × 16개 시나리오 × 300회
- HAC, BH-FDR, 7.5% false-positive 허용치

20개 패널의 market return과 CSAD를 독립 재계산한 결과 저장값과 `1e-12` 이내로 일치했습니다.

### 8.1 절편 제거의 영향

No-intercept가 음의 BH 지지를 보인 16개 baseline 셀에 같은 설명변수를 유지하고 절편만 복원했습니다. 16개 모두 지지가 사라지거나 표준화 절대크기가 50% 이상 감소했습니다.

| 빈도 | No-intercept 평균 | 절편 복원 평균 | 중앙 절대크기 감소율 |
|---|---:|---:|---:|
| Daily | -0.617 | +0.126 | 96.5% |
| Weekly | -0.988 | +0.135 | 92.8% |

### 8.2 비허딩 synthetic null

다른 자산을 보고 따라가는 규칙이 없는 Gaussian, 공통요인, 확률변동성, fat-tail DGP를 생성했습니다.

| 모형 | BH false-positive 범위 | 중앙값 |
|---|---:|---:|
| Standard CSAD | 0~11.3% | 1% |
| No-intercept CSAD | 99~100% | 100% |
| SCSAD | 97~100% | 100% |

Empirical 계수가 simulation null보다 더 음수인지 비교한 240개 BH-FDR 검정은 `0/240`이었습니다. 즉 실제 자료의 corrected 음수는 비허딩 모형이 만드는 값보다 더 극단적이지 않았습니다.

## 9. 기계적 음의 계수의 수학적 원인

독립·동분산 Gaussian 자산을 동일가중하면 시장평균 `M`과 각 자산의 평균 이탈분이 독립입니다. 따라서 `E[CSAD|M]`는 시장 방향이나 크기와 무관한 양의 상수 `c`입니다.

### 9.1 No-intercept

절편이 없는 회귀는 양의 상수 `c`를 `|M|`과 `M^2`만으로 원점부터 근사해야 합니다. 선형항이 원점에서 빠르게 올라가고 제곱항이 바깥쪽을 다시 눌러 상수에 가까워지므로 population 제곱항이 음수가 됩니다.

### 9.2 SCSAD

`sign(M) * CSAD`는 원점에서 부호가 바뀌는 계단 모양입니다. 이를 선형항과 세제곱항으로 근사하면 선형항이 계단을 따라가고 세제곱항이 양 끝의 과도한 증가를 누르므로 population 세제곱항이 음수가 됩니다.

`s = sigma / sqrt(N)`, `u = sqrt(2/pi)`일 때 폐형식은 다음과 같습니다.

```text
No-intercept delta2* = (c/s^2) * (1 - 4/pi) / (3 - 8/pi) < 0
SCSAD gamma3*        = -c*u / (6*s^3) < 0
Standard beta2*      = 0
Intercept-restored   = 0
```

Gaussian 절대모멘트 정규방정식과 폐형식의 독립 검증은 N=14·50·62에서 `3/3` 통과했습니다.

## 10. 원 v1 5/6과 독립 v1.1 12/12

실패 결과를 숨기지 않는 것이 이 연구의 중요한 원칙입니다.

### 10.1 원 사전등록 v1

- T=12,000
- 반복 200회
- N=14·50·62
- 기계적 수렴 gate: 5/6
- 절편 control gate: 6/6
- 최종 판정: `mechanical_null_not_confirmed`

N=62 SCSAD 셀의 Monte Carlo 평균은 이론값과 상대오차가 1.02%였지만, 매우 좁은 99% Monte Carlo 평균 신뢰구간 gate를 벗어나 실패했습니다. 따라서 원 v1은 사후에 성공으로 바꾸지 않았습니다.

### 10.2 독립 v1.1 보충

원 결과를 덮어쓰지 않고 별도 protocol, 독립 seed, 별도 output으로 T=3,000·12,000·48,000과 N=14·50·62를 재검증했습니다.

- 보충 기계적 셀: 6/6
- 보충 control 셀: 6/6
- Primary T=48,000 gate: 12/12
- 보충 판정: `finite_sample_convergence_supported`

정확한 Gaussian 폐형식과 유한표본 수렴은 지지됩니다. 하지만 이 보충은 원 v1의 5/6 실패와 `mechanical_null_not_confirmed` 판정을 변경하지 않습니다.

## 11. Tick run-clustering 연구

초기 탐색에서는 `micro_herding_up/down`이 체결 run의 방향과 가격 방향을 혼동했습니다. Schema v2에서 다음을 분리했습니다.

- `run_clustering_side`: 어떤 체결 run이 극단적인가
- `price_direction`: 같은 bucket의 가격 상승·하락
- `aggressor_direction`: buyer-maker 기반 순공격 방향

2년 raw 7종목에서 run side와 가격 방향의 일치율은 49.23%, aggressor 방향과의 일치율은 48.13%였습니다. 시장중립 30분 미래반응의 세 계수는 모두 BH `q=0.9336`이었고, 연속형 run-z OOS 기준은 `0/9`였습니다.

따라서 과거의 XRP·DOGE·AVAX 방향성 후보와 tracker는 corrected 연구 근거로 사용하지 않습니다.

## 12. Zero-run 연구

Zero-run intensity는 동일 가격 체결이 연속되는 정도를 비방향성으로 측정합니다.

### 12.1 동시점 관계

OOS에서 강한 zero-run 1표준편차는 다음과 연결됐습니다.

- Amihud형 비유동성: -0.207 SD
- 평균 aggregate-trade 간격: -0.537 SD
- USDT 거래대금: +0.503 SD
- 절대 aggressor imbalance: -0.120 SD

동시점 5개 대리변수는 사전 BH-FDR와 0.05 SD 기준을 통과했습니다. 강한 zero-run은 거래가 적은 충격 상태보다 거래가 활발하고 order flow가 비교적 균형적인 상태에 가까웠습니다.

### 12.2 미래 예측

미래 절대수익률과 실현변동성의 5·15·30분 clustered BH q는 모두 0.736과 0.745 이상이었습니다. 개발표본 대비 OOS RMSE 개선률은 -0.0073%~+0.0004%로 사실상 0이었고, 최종 미래 family 판정은 `0/2`였습니다.

### 12.3 조건부 배열 null

거래 수와 zero-tick 수만으로 동시점 관계가 생기는지 확인하기 위해 결과 전 protocol을 봉인하고 exact conditional 배열 감사를 수행했습니다.

- OOS 모집단 233,201 bucket
- 층화 표본 9,044개, 141개 층
- 각 row의 transaction count와 zero ticks를 고정한 999회 null
- Pooled null FPR 2.537%
- Excess clustering 7/7 자산
- Count-conditioned mechanism family 0/5
- 판정: `clustering_beyond_counts_but_mechanism_not_distinct`

Zero/non-zero 배열이 단순 무작위가 아니라는 점은 지지되지만, 동시점 경제 메커니즘을 독립적으로 확정할 수 없고 미래 alpha도 없습니다.

## 13. 확인된 내용과 확인되지 않은 내용

### 확인된 내용

- CMC fixed-62 no-intercept·SCSAD 수치는 높은 정확도로 재현됩니다.
- 결과는 provider, universe, weighting, 기간에 매우 민감합니다.
- No-intercept 음의 제곱항과 SCSAD 음의 세제곱항은 Gaussian 비허딩 null에서 기계적으로 발생합니다.
- 원 v1 5/6 실패와 독립 v1.1 12/12 수렴 결과가 모두 재현 가능하게 보존돼 있습니다.
- Zero-run은 동시점의 활발하고 비교적 균형적인 거래상태와 연결됩니다.

### 확인되지 않은 내용

- 투자자가 다른 투자자를 의도적으로 모방했다는 증거
- Corrected CSAD 음의 계수가 보편적 crypto herding을 식별한다는 주장
- Herding-like event 이후의 거래 가능한 방향성 수익률
- Run-clustering 또는 zero-run의 5·15·30분 미래 예측력
- 현재 결과를 이용한 자동매매의 경제성

현재 연구 결론은 **alpha 없음**입니다.

## 14. 연구 한계

- CSAD는 횡단면 수익률 동조를 측정하지만 투자자의 정보집합과 의도를 관찰하지 않습니다.
- Fixed-62와 fixed-14에는 survivorship 및 listing bias 가능성이 있습니다.
- Standard CSAD도 현실적인 공통요인, 시변변동성, fat tail 아래에서 검정 크기가 왜곡될 수 있습니다.
- 거래소별 데이터 정의와 CMC 시가총액 산식의 차이를 완전히 제거할 수 없습니다.
- Tick 자료는 aggTrades이며 bid-ask spread, order-book depth, 취소와 회복속도를 직접 측정하지 못합니다.
- 뉴스·Reddit은 충분한 point-in-time archive가 없어 confirmatory feature로 사용하지 않았습니다.
- 다중검정, 겹치는 horizon, 시간의존성은 HAC·BH-FDR·cluster/block 방법으로 완화했지만 완전히 제거할 수 없습니다.

## 15. 재현 방법

### 15.1 설치와 테스트

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
PYTHONPATH=src python -m pytest -q
```

### 15.2 핵심 verifier

```bash
PYTHONPATH=src python scripts/verify_csad_specification_audit.py
PYTHONPATH=src python scripts/verify_csad_mechanical_derivation_v1_1_amended.py
PYTHONPATH=src python scripts/verify_final_research_completion.py
```

첫 번째와 세 번째 strict verifier는 manifest가 가리키는 전체 로컬 empirical·tick 입력도 검사합니다. 공개 Git 저장소에서 제외한 대형 입력을 복원한 연구 환경에서 실행해야 합니다. 기계적 수렴 verifier는 공개한 simulation package만으로 실행할 수 있습니다.

### 15.3 주요 실행기

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/baseline/config.yaml

PYTHONPATH=src python scripts/run_csad_specification_audit.py \
  --config configs/research/csad_specification_audit_v1.yaml

PYTHONPATH=src python scripts/run_csad_mechanical_derivation.py \
  --config configs/research/csad_mechanical_derivation_v1.yaml

PYTHONPATH=src python scripts/run_csad_mechanical_supplement.py \
  --config configs/research/csad_mechanical_convergence_supplement_v1_1.yaml
```

전체 empirical 재실행에는 Git에 포함하지 않은 raw data가 필요합니다. 공개 verifier는 저장된 manifest와 판정 파일을 읽기 전용으로 확인하며 원 결과를 덮어쓰지 않습니다.

## 16. 주요 결과 파일

- [최종 논문형 원고](../outputs/v2/final_research_completion_v1/final_research_manuscript.md)
- [최종 재현 안내](../outputs/v2/final_research_completion_v1/REPRODUCIBILITY.md)
- [19개 가설 종료표](../outputs/v2/final_research_completion_v1/hypothesis_closure_status.csv)
- [Specification audit 보고서](../outputs/v2/csad_specification_audit_v1/csad_specification_audit_report_v1_1.md)
- [Empirical-vs-null 비교](../outputs/v2/csad_specification_audit_v1/empirical_vs_null.csv)
- [기계적 음의 계수 v1 보고서](../outputs/v2/csad_mechanical_derivation_v1/csad_mechanical_derivation_report.md)
- [v1 최종 판정](../outputs/v2/csad_mechanical_derivation_v1/final_mechanical_decision.csv)
- [독립 v1.1 수렴 보충](../outputs/v2/csad_mechanical_derivation_v1/supplement_v1_1/csad_mechanical_convergence_supplement_report.md)
- [Zero-run 미시구조 보고서](../outputs/v2/zero_run_microstructure_v1/zero_run_microstructure_report.md)
- [Zero-run 미래 판정](../outputs/v2/zero_run_microstructure_v1/future_decisions.csv)

## 17. 후속 연구 방향

1. No-intercept와 현재 SCSAD의 일반 p-value를 herding 식별에 사용하지 않습니다.
2. Standard CSAD를 계속 사용할 경우 universe·가중·분포에 맞춘 비허딩 null로 검정 크기를 먼저 교정합니다.
3. 새 식별식은 결과를 보기 전에 protocol, family, 경제성 기준을 고정합니다.
4. 완전히 새로운 거래소 또는 미래기간에서 외부검증합니다.
5. Tick 후속 연구는 bid-ask quote와 order-book depth가 있는 자료로 spread, depth, 충격 회복속도를 검증합니다.
6. 뉴스·Reddit·X sentiment는 원 게시시각과 최초 관측시각이 검증된 archive가 확보된 뒤에만 보조 feature로 평가합니다.
7. 통계·경제성 기준을 통과하기 전에는 tracker, paper-sim, 자동매매를 활성화하지 않습니다.

## 18. 최종 평가

이 프로젝트는 기대했던 매매 alpha를 찾지 못했습니다. 대신 선행논문의 숫자를 재현한 뒤, 그 숫자가 행동적 herding의 증거인지 더 엄격한 null과 외부표본으로 반증했습니다.

연구적으로 가장 중요한 결과는 “음의 계수가 있다”가 아니라 “그 음의 계수가 herding이 없어도 왜 생기는지 설명할 수 있다”는 점입니다. 수치 재현과 현상 식별을 분리하고, 실패한 gate도 성공한 보충결과와 함께 남긴 것이 이 저장소의 핵심 기여입니다.
