# Informed-Trading Axis 통합 — 1차 결과 (VPIN-proxy conditional lead-lag)

작성: 2026-07-06 · 트랙: `experiments/informed_trading/` (baseline / lead_lag_matrix 미변경, 읽기 전용)

## 배경

선행연구("Herding, information cascades, and cryptocurrencies", 2024)는 **VPIN + SUR**로
informed trading 축을 깊게 팠고, 본 연구는 자산 간 **directional lead-lag matrix** 축을 더했다.
발표자료(2026-05-09)에서 "가장 우선" 확장방향으로 두 축의 통합을 지목했다.

핵심 질문:
> 살아남은 `DOGE down event → AVAX/ADA +30min` 공동반응이
> **informed trading에 driving되는가(orchestrated cascade)**, 아니면 **noise herding인가?**

## 방법

- **step1** (`step1_doge_vpin.py`) — DOGE aggTrades 60개월(2021-05 ~ 2026-04, **581M trades**)에서
  15분 버킷별 order-flow imbalance 계산. `is_buyer_maker`로 실제 taker aggressor side를 쓰므로
  BVC 근사가 아닌 정확 분류. 지표:
  - `toxicity = |buyVol − sellVol| / totalVol` — 단일 버킷 order-flow 쏠림 (0~1)
  - `vpin_50` — 직전 50버킷 toxicity 이동평균 (Easley-López de Prado VPIN smoothing 근사)
  - → `outputs/doge_vpin_15m.csv` (175,260 buckets, repo에는 미포함 — step1로 재생성)
- **step2** (`step2_conditional_leadlag.py`) — `src/tick_lead_lag.build_lead_lag_frame`으로
  headline cell을 **그대로 재현**한 뒤(동일 session/control 정의), DOGE down 이벤트를 DOGE
  informed-flow **3분위(low/mid/high)** 로 나눠 각 분위의 AVAX/ADA +30min forward return을
  공통 control(DOGE 이벤트 없음)과 Welch t로 비교.

주의: canonical equal-volume-bucket VPIN이 아니라 event frame(15분 시간버킷)에 정합시킨
15분 order-flow imbalance / **VPIN-proxy**다. 정식 VPIN·SUR는 후속 단계.

## 결과

### Headline 재현 (검증 통과)

| cell | delta (30m) | t | n_event |
|---|---|---|---|
| DOGE down → AVAX | +0.0576% | **+3.42** | 3,920 |
| DOGE down → ADA | +0.0513% | **+3.24** | 4,243 |

→ 5y BH-FDR 통과 수치(`multiple_comparison_5y.csv`)와 정확히 일치.

### 조건부 분해 — DOGE informed-flow 3분위

| target | cond | low | mid | high |
|---|---|---|---|---|
| AVAX | toxicity | **+0.072% (t=2.44)** | +0.066% (t=2.17) | +0.035% (t=1.39) |
| AVAX | vpin_50 | **+0.082% (t=2.58)** | +0.053% (t=2.02) | +0.039% (t=1.42) |
| ADA | toxicity | +0.054% (t=2.15) | +0.076% (t=2.25) | +0.025% (t=1.22) |
| ADA | vpin_50 | **+0.088% (t=3.37)** | +0.035% (t=1.55) | +0.033% (t=1.03) |

**4개 조합(2 target × 2 conditioning var) 모두에서 공동반응이 low-toxicity에 집중되고
high-toxicity에서 유의성을 잃는다.**

## 해석

**Informed-driven cascade 가설은 기각 쪽.** 공동반응은 DOGE의 order-flow가 한쪽으로 쏠린
(toxic/informed 추정) 하락일 때가 아니라, **balanced/noise 하락일 때 가장 강하다.**

경제적 직관과 정합:

- DOGE가 **쏠린 매도(high toxicity)** 로 빠지면 → 하락이 "정보성"에 가까워 알트가 덜 되돌린다
  (mean reversion 약화, t=1.0~1.4로 소멸).
- DOGE가 **balanced/noise 흐름(low toxicity)** 으로 빠지면 → 일시적 liquidity dip일 확률이 커
  알트가 공동으로 더 크게 되돌린다 (t=2.4~3.4).

→ 살아남은 신호는 **informed orchestrator herding이 아니라 noise-driven 공동 mean reversion**이다.
이는 앞선 **CCF lag=0(동시반응=beta)** 결론과 정합하며, 선행연구의 "herding은 정보흐름이 약할 때의
consensus 현상 / informed traders는 정체 상태를 stabilize한다"는 서사와도 방향이 맞는다.
두 축이 같은 결론을 가리킨다: **이 공동반응은 alpha가 아니라 noise/liquidity 성격의 공통 beta.**

## 한계 & 다음 단계

- **Confound 미제어**: toxicity는 volume/volatility regime과 상관 가능. vol regime split에서는
  고변동성에서 효과가 강했는데(t=2.76) 여기서는 low-toxicity가 강함 — 두 변수의 결합 효과를
  분리하려면 **SUR 모형(herding × informed × vol 동시 회귀)** 이 정식 다음 단계.
- **VPIN-proxy ≠ canonical VPIN**: equal-volume bucket 정식화 필요.
- **leader 1종(DOGE)만 계산**: 다른 leader(ADA/SOL 등)의 toxicity로 일반화 검증 필요.

## Step 3 추가 — Vol regime × toxicity 결합 분해 (2026-07-06)

step6(고변동성에서 강함)과 step2(저toxicity에서 강함)의 긴장을 해소하기 위한 결합 분해.
vol regime은 step6와 동일(BTC 96-bar rolling std, median split), control은 regime-matched.

**결과** (`outputs/joint_vol_toxicity_summary.csv`):

1. **교락 없음** — toxicity vs BTC vol_24h spearman −0.13(전체)/−0.09(이벤트 내),
   regime × tercile 분포 균등(0.32~0.36). 두 조건축은 독립.
2. **"high-vol × low-tox" 집중 가설 기각** — 해당 셀 t=1.53(AVAX)/1.01(ADA)로 오히려 약함.
   toxicity 기울기(low > high)는 **low-vol regime에서만** 유지(t=2.20/2.49).
   high-vol regime 내에서는 tox 기울기가 비단조(mid가 최강)로 흐려짐.
3. 셀 최대 delta +0.090~0.099%/30min — 12셀 × 2 target 다중비교를 감안하면
   수수료 장벽(왕복 ~0.15%)을 넘는 조합은 없음.

**해석**: 결합 conditioning으로도 이벤트 정의가 tradable하게 날카로워지지 않는다.
신호는 어느 조건에서도 diffuse하게 남음 — "정제 가능한 alpha가 아니라 확산된 beta 구조"
결론을 세 번째로 재확인. → 같은 데이터 내 추가 조합 탐색은 중단하고,
**새 정보축(funding rate / OI / 레버리지 캐스케이드)** 검증으로 이동한다.

## Step 4~7 — 방향 A: 레버리지 캐스케이드 축 (2026-07-06)

step2/3에서 "OHLCV+flow 내부 조합으로는 alpha로 안 날카로워진다"가 확정되어,
새 정보축인 **선물 레버리지 상태(funding rate / open interest)** 를 추가했다.

### 데이터 (step4)

Binance 공개 아카이브 (`data.binance.vision/data/futures/um/`):
- fundingRate monthly (8h 간격, 2021-05~2026-04)
- metrics daily (5분 간격 OI 등, 아카이브 시작 2021-12 → 커버리지 2021-12~2026-04, 결손일 0)

15분 그리드 변수 — look-ahead 구분이 설계 핵심:
- `funding_pre` (ex-ante): 버킷 시작까지 공표된 마지막 funding rate (merge_asof backward)
- `d_oi_24h_pre` (ex-ante): 버킷 시작 기준 직전 24h OI 증감률
- `d_oi_event` (이벤트 버킷 동시, 단 버킷 종료=진입 시점엔 관측 완료): 버킷 내 OI 증감률

### 단변량 결과 (step5)

| 조건 | AVAX low→high | ADA low→high | 판정 |
|---|---|---|---|
| `d_oi_event` (OI flush) | t=+2.80 → +1.16 | t=+2.03 → +1.72 | flush에서 강함 (H-A1 방향) |
| `funding_pre` | t=+0.28 → +2.05 | t=−0.90 → +2.64 | **단조 증가 — 유일한 ex-ante 기울기** |
| `d_oi_24h_pre` | t=+3.11 → +1.34 | t=+2.21 → +0.91 | 예상 반대 (디레버리징 중 강함) |

진단: `d_oi_event` ↔ toxicity spearman **−0.01** — OI flush는 15분 단위 toxicity와 직교.
step2의 "low-toxicity 집중"과 모순이 아니라 서로 다른 축이었음이 해명됨.

### 상호작용 반전 (step6)

funding을 경제적 카테고리(negative / default≈0.01% / **crowded**>0.01%)로 나누고
OI flush와 교차하자 가설이 비틀렸다:

- 가설했던 **crowded × flush (청산 캐스케이드 프로파일): 죽음** (t=0.33/1.18, n≈118)
- **crowded × no_flush 가 최강**: AVAX **+0.156% (t=2.25)**, ADA **+0.165% (t=2.38)**, n≈243

해석: long 과열(funding>기본요율) 상태에서 DOGE가 급락했는데 **OI가 버티면**(포지션 홀딩)
= noise dip → 대기 매수세가 강하게 되돌림. OI가 실제로 flush되면 = 디레버리징 개시
신호라 반등이 불확실. 청산 "캐스케이드"가 아니라 **청산 "부재"가 반등의 조건**.

### Robustness 스크리닝 (step7)

crowded × no_flush 셀, 타깃 6종 확장 + 시간 반분할:

| target | full delta | full t | H1 t | H2 t |
|---|---|---|---|---|
| SOL | **+0.172%** | **+2.80** | +2.49 | +1.70 |
| ADA | +0.165% | +2.38 | +1.77 | +1.86 |
| AVAX | +0.156% | +2.25 | +1.79 | +1.48 |
| XRP | +0.132% | +1.93 | +1.08 | +1.80 |
| ETH (falsif.) | +0.062% | +1.81 | — | — |
| BTC (falsif.) | +0.047% | +1.68 | — | — |

- 알트 4종 모두 taker 왕복비(~0.15%) 근방~초과, 반분할 부호 유지
- BTC/ETH는 효과 절반 이하 — 기존 "메이저 비일반화" 패턴과 정합

### 검증 1 — Permutation test (step8)

circular shift null (state 시계열을 15분 그리드에서 통째로 회전, 자기상관 보존, 1000회).
통계량은 다중비교를 제거한 **알트 4종 동일가중 basket delta**.

- 관측: +0.157% (n=241) vs null 평균 +0.040% / sd 0.066% / p95 +0.151%
- **p = 0.041 (one-sided), 0.043 (two-sided)** — 통과하나 아슬아슬 (z≈1.8)
- null 평균이 0이 아닌 이유: DOGE down 이벤트는 아무 부분집합이나 기본 반등(t=3.4)을
  가지므로, 이 검정은 **conditioning의 증분 정보**를 잰다. 증분은 실재하나 강하지 않음.

### 검증 2 — 실행 시뮬 (step9, 30min hold, 이벤트당)

| basket | gross | maker (0.04% RT) | taker (0.10% RT) |
|---|---|---|---|
| 알트 4종 | +0.156% (win 56%) | **+0.116%** | +0.056% |
| +DOGE 5종 | +0.171% (win 58%) | **+0.131%** | +0.071% |

- 총 241 이벤트 누적 (5종, maker): **+31.5%**, maxDD 7.9%
- 연도별 일관: 2023 +0.168% / 2024 +0.151%

### 결정적 함정 — 레짐 휴면 (step9)

**cell 이벤트가 전부 2023-01-15 ~ 2024-12-07 구간에만 존재.**
2024-12 이후 17개월간 DOGE funding이 기본요율을 초과한 적이 없어 이벤트 0건.
step7의 "반분할 안정"도 실질적으로는 2023 vs 2024 비교였던 셈.

→ 이 후보는 상시 전략이 아니라 **crowded-long 레짐에서만 발화하는 조건부 후보**다.
살아있는지 여부는 다음 crowded funding 레짐이 와야 확인 가능 — forward tracker 의
역할이며, 그 전까지는 어떤 성과 주장도 하지 않는다.

### 지위: 추적 후보 (승격 아님)

- **처음으로 ex-ante 구현 가능하면서 수수료 장벽을 넘는 조건부 셀** (maker 기준 명확,
  taker 기준 얇게 통과). permutation p=0.041로 증분 정보도 실재.
- 단: (1) 다중비교 스캔 출신 + permutation 아슬아슬, (2) **2024-12 이후 레짐 휴면**
  (활동 기간 2023-01~2024-12 뿐), (3) 활동기에도 연 ~120건 수준.
- 완료된 검증: basket pooled (step8에 포함), permutation (step8), 비용 시뮬 (step9)
- 남은 것: **forward tracker 등록** — 다음 crowded funding 레짐에서의 생존 여부가
  이 후보의 최종 판정. rolling flush 컷 구현도 tracker 단계에서 필요.

## 산출물

| 파일 | 내용 | repo 포함 |
|---|---|---|
| `step1_doge_vpin.py` | DOGE aggTrades → 15분 VPIN-proxy | ✓ |
| `step2_conditional_leadlag.py` | headline 재현 + toxicity 3분위 분해 | ✓ |
| `step3_joint_vol_toxicity.py` | vol regime × toxicity 결합 분해 | ✓ |
| `step4_doge_futures_state.py` | funding/OI 수집 + 15분 상태변수 | ✓ |
| `step5_leverage_conditional.py` | 레버리지 조건부 분해 | ✓ |
| `step6_combined_filter.py` | funding regime × OI flush 교차 | ✓ |
| `step7_crowded_noflush_robustness.py` | 최강 셀 타깃 확장 + 반분할 | ✓ |
| `step8_permutation_test.py` | circular-shift permutation (basket) | ✓ |
| `step9_execution_sim.py` | 비용 시나리오 실행 시뮬 | ✓ |
| `outputs/conditional_leadlag_summary.csv` | toxicity 분위별 delta/t | ✓ |
| `outputs/joint_vol_toxicity_summary.csv` | vol × tox 셀별 결과 | ✓ |
| `outputs/leverage_conditional_summary.csv` | 레버리지 조건별 결과 | ✓ |
| `outputs/combined_filter_summary.csv` | funding × flush 셀별 결과 | ✓ |
| `outputs/crowded_noflush_robustness.csv` | 최강 셀 robustness | ✓ |
| `outputs/permutation_crowded_noflush.csv` | permutation null 분포 (1001 draws) | ✓ |
| `outputs/execution_sim_summary.csv` | basket × fee 시나리오 성과 | ✓ |
| `outputs/execution_sim_events.csv` | cell 이벤트별 상세 (241건) | ✓ |
| `outputs/doge_vpin_15m.csv` | 175,260 buckets VPIN-proxy (23MB) | ✗ (step1로 재생성) |
| `outputs/doge_futures_state_15m.csv` | 154,753 buckets funding/OI 상태 (~15MB) | ✗ (step4로 재생성) |
