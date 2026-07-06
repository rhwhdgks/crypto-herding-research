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

## 산출물

| 파일 | 내용 | repo 포함 |
|---|---|---|
| `step1_doge_vpin.py` | DOGE aggTrades → 15분 VPIN-proxy | ✓ |
| `step2_conditional_leadlag.py` | headline 재현 + toxicity 3분위 분해 | ✓ |
| `outputs/conditional_leadlag_summary.csv` | 분위별 delta/t/win_rate | ✓ |
| `outputs/doge_vpin_15m.csv` | 175,260 buckets VPIN-proxy (23MB) | ✗ (step1로 재생성) |
