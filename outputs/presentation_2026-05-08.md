# 암호화폐 허딩(Herding) 연구 — 발표 자료 (상세본)

발표일: 2026-05-09
프로젝트 위치: `/home/jonghan/findalpha/herding`

---

## 0. 한 줄 요약 (TL;DR)

> 암호화폐 시장 전체에서 강한 허딩은 보이지 않는다. Tick microstructure에서 **DOGE-centric "공동 mean reversion" 패턴**은 5년 데이터 + 다중비교 보정 + stability split 검증을 통과해 통계적으로 robust. 다만 **시간적 lead-lag이 아닌 동시 반응(CCF lag=0)**이라 alpha 트레이딩 신호로는 변환 불가 — beta 구조. 이 연구의 가치는 alpha 발굴이 아니라 **가설 공간의 명확한 축소**.

다섯 가지 결론:
1. **광범위 허딩은 weak** — β2 = +4.53 (105만 obs) → 광범위 herding 가설 ❌
2. 학술 재현은 부합 — 주간 paper-like β2 음수 (실거래 적용 ❌)
3. **Tick candidate**: XRP up / DOGE down / AVAX down (15min → 30min) — 거래비용 후 marginal
4. **Lead-lag matrix (5년)**: DOGE down → AVAX/ADA 등이 BH-FDR q=0.05 통과. 단 시간 의존적 + lag=0 동시 반응
5. **Cointegration 통합**: 두 독립 프로젝트가 같은 자산쌍(AVAX-DOGE) 1위로 일치, 단 단순 결합 alpha ❌

---

## 1. 연구 동기와 질문

### 두 가지 핵심 질문

1. 암호화폐 시장에서 사람들이 서로를 따라 움직이는 **허딩(herding) 비슷한 현상**이 실제로 있는가?
2. 만약 있다면, **아주 짧은 시간 안에 반복되는 가격 반응**이 있는가?

### 프로젝트 성격

- **연구 파이프라인** (production 트레이딩 봇 ❌)
- 재현 가능성, 실험 트랙 분리, 룩어헤드 방지가 최우선
- baseline / paper-like / tick / news / reddit / X 트랙을 코드 레벨에서 모두 독립

---

## 2. 데이터 & 방법 개요

| 항목 | 내용 |
|---|---|
| 거래소 | Binance |
| 자산 universe | 14종 USDT 페어 (BTC, ETH, BNB, XRP, ADA, SOL, DOGE, LINK, AVAX, DOT, LTC, TRX, ATOM, NEAR) |
| Baseline 데이터 | 1분봉 OHLCV, 최근 2년 |
| Tick 데이터 | Binance `aggTrades` (체결 단위), 5년 |
| 텍스트 데이터 | Google News RSS, GDELT, Reddit public JSON |

### CSAD 회귀 (전통적 허딩 측정)

```
CSAD_t = α + β1·|R_m,t| + β2·R_m,t² + ε
```

- β2 < 0 이면 허딩 존재 (변동성이 클수록 cross-sectional dispersion이 줄어듦)
- β2 > 0 이면 허딩 없음 또는 분산 확대

### Tick microstructure event

- 15분 버킷별 herding 점수 계산 (up-tick run vs down-tick run 분포 기반)
- Lower-percentile bucket이 micro-herding event
- Forward 30분 수익률을 event vs control로 비교

---

## 3. 연구 진행 (5단계)

### Stage 1 — 2년 1분봉 Baseline CSAD

- 관측치 약 **105만**
- `β2 = +4.53, t = 559` → **허딩 없음 (반대 부호)**

**해석:** 전체 평균에서는 허딩이 강하게 나타나지 않는다. 1분봉은 micro-noise가 많아 패턴이 묻힌다.

### Stage 2 — Paper-like 저빈도 비교 (논문 재현 트랙)

주간 회귀 결과:
- `standard_csad β2 = -0.34`
- `no_intercept_csad β2 = -0.85`
- `scsad β2 = -0.58`

**해석:** 주간 단위에서는 음(-)의 β2가 잡힌다. 학술 논문 결과와 부합. 단, 실거래 short-horizon에는 직접 사용하기 어려움.

### Stage 3 — XRP Tick Microstructure (mainline)

`15분 micro-herding event → 다음 30분 반응` 구조가 핵심.

XRP dual tracker (5년, as of 2026-04-17):

| 규칙 | 전체 건수 | 누적 수익 | 승률 | 평균 순수익 |
|---|---|---|---|---|
| `time_17_18` | 171건 | 25.47% | 58.48% | 0.1356% |
| `time_17_18_prior_drop_q4` | 44건 | 16.44% | 72.73% | 0.3512% |

**해석:**
- `time_17_18` — 균형 잡힌 mainline candidate
- `time_17_18_prior_drop_q4` — 더 강하지만 sparse, "track, not promote" 원칙 적용

### Stage 4 — 다심볼 일반화 (basket candidate)

7-심볼 universe (BTC/ETH/XRP/SOL/DOGE/ADA/AVAX) 대상.

다심볼 후보 basket (as of 2026-04-16):

| 후보 | 전체 건수 | 평균 순수익 | 누적 | 최근 30일 | 최근 60일 | 최근 90일 |
|---|---|---|---|---|---|---|
| AVAX down 21 | 95 | +0.1215% | 11.89% | +0.5934% (3건) | +0.2463% (7건) | +0.0476% (16건) |
| DOGE down 21-22 | 162 | +0.2183% | 40.67% | +0.1725% (13건) | −0.0751% (30건) | −0.0728% (37건) |
| XRP up 21-22 | 171 | +0.1611% | 29.19% | +0.0957% (8건) | −0.1121% (20건) | +0.0131% (41건) |

**해석:**
- 누적 1위는 `DOGE down 21-22`이지만 최근 60·90일 음수 drift — 주시 필요
- `AVAX down 21`이 최근 30일에서 가장 안정적 (+0.5934%)
- BTC/ETH는 같은 패턴으로 일반화되지 않음 — 알트 특유의 미시구조 효과로 보임

### Stage 5 — 텍스트 보조 레이어 (뉴스 + Reddit)

#### 뉴스 sentiment (Google News + GDELT)

- Google News RSS: 안정적
- GDELT: `429 Too Many Requests` 빈발 → slow collector로 보완
- 누적 headline ~890건
- 강한 feature group은 sample이 작아 단독 alpha 취급 ❌

#### Reddit sentiment (CryptoCurrency / CryptoMarkets)

- 1,946 posts scored, 2025-04-16 ~ 2026-04-20
- Reddit 슬랭 어휘(moon/hodl/rekt/wagmi/ngmi/dump/rug/bagholder/bloodbath/cope) 추가 후 negative event group **1,428 → 2,457 (+72%)**
- 가장 강한 후보: `negative_shock_strong` 6h, 평균 +0.1247%, **t=1.05** (marginal)
- positive 6h, 평균 +0.0423%, **t=1.97** (marginal)

**핵심 결론:** Reddit-only 단독 예측력은 **약하다**. 이 자체가 의미 있는 결과 — 프로젝트의 가정대로 텍스트는 보조 feature이지 standalone alpha가 아니다.

---

## 4. 신규 발견: Lead-Lag 매트릭스

### 동기

단일 심볼 tracker만으로는 drift 원인이 공통 요인인지 심볼 특유 변화인지 구분할 수 없다. **심볼 간 선행–후행 구조**를 보면 trigger 타이밍과 dependency를 같이 판단할 수 있다.

### 방법

- 데이터: **5년 (2021-05 ~ 2026-05)**, 7-심볼, 15-min tick bucket
- 7-심볼 × 6-leader × 2-direction = **84개 directed (leader → target) 셀**
- 각 셀: leader의 micro-herding event 발생 시 target의 다음 30분 mean return을 control과 비교 (Welch t-stat)
- 출력: 7×7 매트릭스 4종 (delta, t-stat, count) × 2 방향(up/down)

### 핵심 결과 (|t| ≥ 2.0인 directed edge)

| 순위 | Leader event | → Target +30m | delta | t-stat | n |
|---|---|---|---|---|---|
| 1 | **DOGE down** | AVAX | **+0.0576%** | **+3.42** | 3920 |
| 2 | DOGE down | ADA | +0.0513% | +3.24 | 4243 |
| 3 | DOGE down | ETH | +0.0315% | +2.86 | 4419 |
| 4 | DOGE down | XRP | +0.0337% | +2.58 | 4416 |
| 5 | DOGE down | BTC | +0.0207% | +2.57 | 4419 |
| 6 | DOGE up | ETH | -0.0252% | -2.36 | 4517 |
| 7 | SOL down | AVAX | +0.0317% | +2.10 | 4294 |

→ **DOGE는 universal leader** — down direction에서 6개 알트 모두 양(+) 반응.

### Robustness 검증

**검증 1: 다중비교 보정 (BH-FDR, Bonferroni)**

84개 cell이라는 큰 가설 공간에서 단일 t-stat은 우연 발견 가능성 있음. 보정 결과:

| 보정 방법 | 통과 cell 수 |
|---|---|
| Uncorrected p<0.05 | 8 |
| Uncorrected p<0.01 | 4 |
| Bonferroni α=0.05 | 0 |
| **BH-FDR q=0.05** | **2** (DOGE down → AVAX, ADA) |
| BH-FDR q=0.10 | 2 |
| BH-FDR q=0.20 | 5 |

→ **DOGE down → AVAX (p=0.0006), DOGE down → ADA (p=0.001)는 BH-FDR q=0.05 통과** — 통계적으로 robust.

**검증 2: Stability split (전반 2.5년 / 후반 2.5년)**

5년 데이터를 절반으로 나눠 (H1: 2021-2023, H2: 2023-2026) 각각 매트릭스 재계산:

| Top edge (5y full) | H1 t (2021-23) | H2 t (2023-26) | 판정 |
|---|---|---|---|
| DOGE down → AVAX (+3.42) | +0.86 | **+3.86** | 같은 부호, 후반 강함 |
| DOGE down → ADA (+3.24) | +1.22 | +3.06 | 같은 부호, 후반 강함 |
| DOGE down → ETH (+2.86) | +1.55 | +2.45 | 양쪽 모두 \|t\|≥1.5 ✓ |
| DOGE down → XRP (+2.58) | +1.10 | +2.46 | 같은 부호, 후반 강함 |
| DOGE down → BTC (+2.57) | +1.20 | +2.49 | 같은 부호, 후반 강함 |
| DOGE up → ETH (-2.36) | -2.24 | -1.13 | 같은 부호, 전반 강함 |

전체: same sign in both halves **47/84 (56%)**.

→ **신호는 시간 의존적 (regime-dependent)**. 모두 같은 부호 유지하지만 강도는 후반에 더 강함.

**검증 3: Vol regime split (논문 비교용)**

선행연구 (Patterson-Sharma 등)는 **저변동성에서 herding 강함**을 주장. 우리도 같은 결과인지 검증:

- BTC 24-hour rolling vol → median split → low / high vol regime
- 각 regime에서 lead-lag matrix 재계산

| Top edge (5y) | low vol t | high vol t | 판정 |
|---|---|---|---|
| DOGE down → AVAX (+3.42) | +2.06 | **+2.76** | 고변동성에서 강함 |
| DOGE down → ADA (+3.24) | +2.23 | +2.48 | 비슷 |
| DOGE down → ETH (+2.86) | +1.80 | +2.21 | 고변동성에서 강함 |
| DOGE down → XRP (+2.58) | +1.00 | +2.25 | 저변동성에서 거의 사라짐 |
| DOGE down → BTC (+2.57) | +1.32 | +2.16 | 고변동성에서 강함 |

전체: |t|≥2.0 in low vol = 8/84, in high vol = 8/84 (비슷). 그러나 **top edges는 일관되게 고변동성에서 강함**.

→ **논문과 반대 방향**. 다른 종류의 herding 측정 (논문 CSAD 시장 dispersion ≠ 우리 lead-lag 자산 간 동시 반응).

**검증 4: 시기 구분 (3-period split)**

5년 데이터를 시기별로 나눠 어느 regime에서 신호가 강한지 확인:

| Period | 기간 | rows | \|t\|≥1.5 | \|t\|≥2.0 |
|---|---|---|---|---|
| P1 (winter) | 2021-05~2022-12 | 404k | 18/84 | 7/84 |
| **P2 (recovery)** | 2023-01~2024-06 | 367k | 18/84 | **9/84** |
| P3 (current) | 2024-07~2026-05 | 454k | 9/84 | 3/84 |

| Top edge | P1 winter | P2 recovery | P3 current |
|---|---|---|---|
| DOGE down → AVAX (+3.42) | +0.72 | **+2.92** | +2.62 |
| DOGE down → ADA (+3.24) | +0.94 | **+2.96** | +2.10 |
| DOGE down → XRP (+2.58) | +1.08 | +2.00 | +1.67 |
| SOL down → AVAX (+2.10) | +2.34 | +1.00 | +0.01 (사라짐) |

→ **2023-2024 recovery 시기에 가장 강함**. 2022 winter에 weak — 논문의 "post-COVID 침체장에서 herding 강함" 주장과 다른 결과.

**검증 5: Tick-level lead time 측정 (CCF + event-triggered curve)**

진짜 시간적 lead-lag인지 동시 반응인지 사이언스 깊이 결판:

- Method: 1분봉 cross-correlation Cov(DOGE_t, AVAX_{t+lag}) for lag ∈ [-60, +60] min
- 결과: **Peak |CCF| = 0.64 at lag = 0 min** (lag=±1에서는 |CCF| 0.04~0.08)

→ DOGE와 AVAX는 **거의 100% 동시 반응**. lead-lag time gap = 0.

Event-triggered average curve (1867 DOGE down events):

| 시점 | AVAX 누적 | DOGE 누적 |
|---|---|---|
| t=-5 → t=0 (pre-event) | -0.019% | DOGE 떨어지는 중 |
| t=0 → t=+30m (post-event) | **+0.215%** | **+0.245%** |

→ AVAX +0.21% lift할 때 **DOGE 자기도 +0.25% 반등**. **공동 mean reversion**, AVAX-only follower 효과 ❌.

### 종합 진단

3가지 robustness 검증 결과 → **"DOGE → 알트 공동 mean reversion" 패턴은 통계적으로 robust하지만, 시간 의존적 + 동시 반응**.

진짜 메커니즘:
1. Macro 충격으로 DOGE 포함 모든 알트 **동시에** 떨어짐 (이게 DOGE micro_herding_down 트리거)
2. 다음 30분 동안 **공동 반등** (lag=0)
3. 5년 평균에서 통계적으로 robust, 단 최근 regime에 강함
4. **시간적 lead-lag은 아님** — alpha 트레이딩 신호로 변환 불가

### 산출물

- 리포트: `outputs/tick/multi_asset_5y/lead_lag_matrix/tick_lead_lag_matrix_report.md`
- Flat CSV: `lead_lag_matrix_summary.csv` (84행)
- Pivot CSV: `lead_lag_matrix_{up,down}_{delta,tstat,count}.csv`
- Heatmap PNG: `plots/lead_lag_matrix_{up,down}.png`
- 다중비교 보정: `experiments/lead_lag_robustness/outputs/multiple_comparison_5y.csv`
- Stability split: `experiments/lead_lag_robustness/outputs/stability_split_5y.csv`
- Tick-level CCF + event curve: `experiments/lead_lag_robustness/outputs/tick_level_lead_time_5y.png`
- **Vol regime split**: `experiments/lead_lag_robustness/outputs/vol_regime_comparison.csv`
- **시기 구분 (3-period)**: `experiments/lead_lag_robustness/outputs/period_top_edges_breakdown.csv`

---

## 5. 선행연구와의 비교

### 참고 논문

> "Herding, information cascades, and cryptocurrencies — New evidence using low frequency and high frequency data"

논문 핵심 메시지:
1. **CSAD만으로는 부족** — SCSAD, GJR-GARCH, idiosyncratic vol grouping, Patterson-Sharma intensity 같이 사용
2. **저변동성·침체장에서 herding 강함** (post-COVID 2022-2023 winter)
3. **VPIN을 informed trading proxy**로 사용
4. **SUR 모형**으로 herding × informed trading 관계 분석
5. **Informed traders의 이중 역할** — 추세 강화 / 안정화

### 논문 도구 vs 우리 사용 여부

| 도구 | 우리 사용 | 갭 |
|---|---|---|
| CSAD (단순) | ✅ baseline | OK |
| SCSAD (Wang-Hudson) | ✅ paper-like 주간 | OK |
| Tick intraday herding | ✅ 15분 bucket event study | OK |
| GJR-GARCH | ❌ | 갭 |
| Vol regime split | ✅ (이번 추가) | OK |
| Idiosyncratic vol grouping | ❌ | 갭 |
| Patterson-Sharma intensity (정식화) | △ (비슷한 score 사용) | 부분 갭 |
| **VPIN** (informed trading proxy) | ❌ | **핵심 갭** |
| **SUR 모형** | ❌ | 갭 |
| 시기 구분 (pre/COVID/post) | ✅ (3-period 추가) | OK |
| Zero-tick state 분석 | ❌ | 갭 |

### 결과 비교 — 다른 결론

| 변수 | 논문 결과 | 우리 결과 | 차이의 원인 |
|---|---|---|---|
| Vol regime | **저변동성에서 herding 강함** | **고변동성에서 강함** | 다른 측정 도구 (CSAD vs lead-lag matrix) |
| 시기 | post-COVID 침체장 (2022-2023) 강함 | 2023-2024 recovery 강함 | 알트 narrative 부활 시기에 자산 간 동시 반응 |

### 차이의 해석

논문과 우리 결과가 직접 모순 ❌ — **다른 종류의 herding 측정**:

- **논문 framework**: 시장 전체 dispersion 압축 (CSAD beta가 음수 = 자산들이 시장 평균에서 덜 벗어남)
  - 저변동성 침체장 = 정보 흐름 약함 → 투자자들이 자기 판단 약화 → 시장 컨센서스 따라감
  - "조용한 컨센서스 herding"
- **우리 framework**: 자산 간 동시 반응 (DOGE down event 후 알트 30분 mean lift)
  - macro shock 필요 → 고변동성에서 강함
  - recovery 시기에 알트 narrative 활성화 → DOGE 같은 meme이 driving force
  - "macro shock 후 알트 공동 반응"

→ **두 종류의 herding은 다른 시간 scale + 다른 자산 차원**을 측정. **두 결과는 보완적**.

### 우리 연구의 미실행 도구 (다음 1년 후보)

논문 framework를 따라잡기 위한 가장 가성비 좋은 추가:
- **VPIN**: aggTrades에서 직접 계산 가능. informed trading proxy
- **SUR 모형**: VPIN + 우리 lead-lag matrix 결합
- **Idiosyncratic vol grouping**: 자산별 고유 변동성 기준 분류 후 그룹별 매트릭스
- **GJR-GARCH**: 변동성 비대칭 모델링
- **Zero-tick state 분석**: 정체 상태에서 informed traders의 안정화 효과

이런 도구들이 추가되면 **"우리 lead-lag matrix가 informed trading에 의해 driving되는지 vs noise herding인지"**를 결판할 수 있음.

---

## 6. 통합 검증: Cointegration × Lead-Lag

### 동기

Delta sprint1 (cointegration pair trading) 결과에서 **AVAX|DOGE**가 가장 강한 페어 (Upbit KRW, Sharpe 2.1, MDD -0.09 @ 10bps).
Herding lead-lag matrix에서도 **DOGE down → AVAX**가 매트릭스 최강 edge (t=3.42).

→ 가설: 두 독립 프로젝트가 같은 자산쌍을 1위로 지목한 건 **같은 mean-reversion 메커니즘의 다른 측정**일 것.

### 실험 설계

- 데이터: Binance USDT, 1분봉 → 15min resample (Sprint1 PPT framework와 정합)
- 기간: 2025-05-08 ~ 2026-04-09 (11개월)
- Cointegration screening: 3-day rolling window (= 288 bars), 1-day step, ADF p<0.05 → **101개 tradable window** (30.2%)
- Backtester: Sprint1 step10 robust (Z_ENTRY=3.0, Z_EXIT=0.5, sigma floor q20)
- Lead-lag filter: 직전 LAG 안에 DOGE micro_herding_down event 발생 시 진입 허용
- LAG sweep: 15분 / 30분 / 1시간 / 2시간 / 4시간

### 결과 (15분봉, slip 5bps, 101 windows)

| Mode | n_entries | filtered_out | cum PnL | Sharpe | MDD |
|---|---|---|---|---|---|
| Baseline (cointegration only) | 32 | 0 | -0.090 | -3.34 | -0.090 |
| Filtered LAG=15min | 2 | 40 | +0.003 | +0.78 | -0.003 |
| Filtered LAG=30min | 2 | 40 | +0.003 | +0.78 | -0.003 |
| Filtered LAG=1h | 4 | 37 | -0.015 | -1.93 | -0.018 |
| Filtered LAG=2h | 5 | 36 | -0.018 | -2.24 | -0.021 |

### Permutation test — 결정적 검증

"Sharpe -3.34 → +0.78 부호 반전"이 진짜 신호인지 우연인지 결판:

- DOGE event timestamp 1000번 random circular shift → 매번 backtest → null 분포
- Real PnL +0.0028 vs Null mean -0.0038
- **p-value = 0.208** — null의 79th percentile, 우연 자주 발생

→ **부호 반전은 filter logic이 아니라 mechanical regularization** (진입 32→2건으로 강하게 줄이면 PnL이 0 근방으로 수렴, random shuffle도 비슷한 효과).

### 결론

- ✅ 두 신호 공동발생 입증 (filter가 진입의 94% 차단, random보다 자주 같이 발생)
- ❌ 그러나 단순 결합 시너지는 통계적으로 입증 ❌
- ❌ Sprint1 Sharpe 2.1은 Binance USDT에서 재현 안 됨 (KIMCHI premium / 거래소 효과)

**함의:** 두 프로젝트의 "AVAX-DOGE 1위 일치"는 같은 raw 시장 비효율(DOGE 큰 움직임 → 알트 공동 반등)을 다른 lens로 본 결과. 단 단순 AND 결합으로 새 alpha는 만들어내지 못함.

---

## 7. 정직한 한계

- **β2 양수 (+4.53)** — 1분봉 baseline은 허딩 가설을 강하게 지지 ❌
- **Tick candidate 최근 60·90일 drift** — DOGE down/XRP up이 약해짐. 일시적 noise인지 구조 변화인지 추가 관찰 필요
- **Lead-lag matrix (5년)** — BH-FDR q=0.05 통과 2 cell. 통계적으로 robust지만 **시간 의존적 (regime-dependent)** + **lag=0 동시 반응**으로 trade alpha 변환 불가
- **Cointegration 통합** — 두 신호 공동발생 입증, 그러나 결합 alpha는 permutation test에서 우연 (p=0.208)
- **텍스트 데이터** — GDELT 429, Google News 비완결, Reddit 표본 한계로 2년 baseline window 전체 커버리지 ❌
- **현재 거래비용·다중비교·시간 안정성을 모두 통과하는 actionable alpha는 없다**
- 모든 candidate는 **research hypothesis**, 라이브 전략 ❌

---

## 8. 정리 — 지금까지의 큰 그림

```
1분봉 baseline      → 허딩 weak (β2 = +4.53, 105만 obs) [확정]
주간 paper-like     → β2 음수, 학술 논문 부합 [확정]
Tick 15→30분        → XRP up / DOGE·AVAX down [거래비용 후 marginal]
다심볼 generalization → BTC/ETH 일반화 ❌, 알트 한정 [확정]
텍스트 sentiment    → 보조 feature, standalone alpha ❌ [확정]
Lead-lag matrix (5y) → DOGE down → AVAX/ADA (BH-FDR q=0.05 통과)
                       ↳ Stability split: regime-dependent (후반에 강함)
                       ↳ Tick CCF: lag=0 peak (동시 반응)
                       ↳ Vol regime: 고변동성에서 강함 (논문과 반대)
                       ↳ 시기 구분: 2023-2024 recovery에 가장 강함
                       → 진짜 메커니즘: 공동 macro shock + 공동 mean reversion
선행연구 비교        → 논문 (저변동성, CSAD) ≠ 우리 (고변동성, lead-lag) — 보완적
통합 검증 (sprint1) → 같은 자산쌍 1위 일치 [재현 가치 있음]
                       ↳ 결합 alpha ❌ (permutation p=0.208)

→ 죽은 가설: 광범위 herding, BTC/ETH 일반화, Reddit standalone, 
  시간적 lead-lag, simple cointegration combination
→ 살아있는 패턴: DOGE-centric 공동 mean reversion (recovery 시기 + 고변동성 + beta 구조)
→ 미실행 (다음 1년 핵심): VPIN, SUR, idiosyncratic vol grouping, GJR-GARCH
```

---

## 9. 다음 단계 (우선순위)

연구 가치를 더 뽑아내려면 (1) **선행연구 framework 따라잡기**, (2) **새 정보축 추가**, (3) **살아있는 신호 정밀화** 세 방향.

### 1순위 — 선행연구 도구 통합 (논문 framework 따라잡기)

- **VPIN** (Volume-synchronized Probability of Informed Trading): aggTrades에서 계산 가능. informed trading의 핵심 proxy
- **SUR 모형**: VPIN + 우리 lead-lag matrix 결합해서 "누가 driving force인지" 결판
- **Idiosyncratic volatility grouping**: 자산별 고유 변동성으로 분류 후 그룹별 herding 측정 (논문이 의미있는 결과 본 분석)
- **GJR-GARCH**: 변동성 비대칭 모델링
- **Zero-tick state 분석**: 정체 상태에서 informed traders의 안정화 효과

### 2순위 — 살아있는 신호 정밀화

- **Recovery vs current regime 분해**: 왜 2023-2024 recovery에 강하고 2024-2026 current에 약화되는가
- **Lag=0 동시 반응의 trade 변환 가능성**: latency 게임 (DOGE shock 직후 즉각 진입)
- 더 narrow filter로 effect size 키우기 (regime conditional)

### 3순위 — 새 데이터 축

- **Funding rate / Perp-spot basis**: Binance API 무료. Leveraged position 쏠림 → cascade 선행 지표
- **Cross-exchange spillover**: Coinbase/OKX/Bybit 동일 페어 spread 변화. 미국 vs 아시아 시간대 dominance 교대
- **On-chain flow**: exchange inflow/outflow

### 하지 말 것

- 같은 데이터에서 hyperparameter 더 튜닝
- Reddit/News sentiment 추가 튜닝 (이미 약함 확정)
- 1분봉 baseline 재실행

---

## 10. 발표 시 안전한 표현

### 권장 (정직 모드)

> 이 프로젝트는 광범위 허딩 가설을 105만 1분봉 관측치로 명확히 기각했고, 학술 논문 방향과 부합하는 주간 결과를 재현했습니다. Tick microstructure에서 알트 한정 candidate를 발견했고, 5년 데이터의 lead-lag matrix에서 BH-FDR q=0.05 통과하는 통계적 신호도 확보했습니다. 다만 이 신호는 (1) 시간 의존적이고 (2) tick-level CCF가 lag=0이라 시간적 lead-lag이 아닌 동시 반응 — 즉 통계적 패턴은 robust하지만 trade alpha로 직결되지는 않는 beta 구조입니다. 선행연구가 측정한 "저변동성 침체장 herding (CSAD 시장 dispersion)"과 우리가 본 "고변동성 recovery 시기 자산 간 동시 반응 (lead-lag matrix)"은 같은 시장의 다른 종류 herding을 측정한 보완적 결과입니다. 이 연구의 진짜 가치는 alpha 발굴이 아니라, 어떤 가설이 죽었고 어떤 방향이 살아있을 수 있는지 명확히 정리한 가설 공간 축소이며, 다음 단계는 선행연구 framework (VPIN, SUR 모형, idiosyncratic vol grouping)와 우리 lead-lag matrix를 통합하는 것입니다.

### 피해야 할 표현

- "우리는 production 트레이딩 전략을 가지고 있습니다" — 아직 사실 ❌
- "DOGE → AVAX lead-lag가 시간적으로 leading한다" — CCF lag=0으로 반증
- "AVAX|DOGE 페어 트레이딩에서 alpha 확인" — Binance 재현 ❌, permutation 우연

### Q&A 예상 + 답변

**Q: 결국 alpha가 있는 건가 없는 건가?**
> 통계적 신호는 5년 데이터에서 robust (BH-FDR q=0.05 통과). 하지만 (1) effect size가 작아 거래비용 후 marginal, (2) tick-level CCF가 lag=0이라 시간적 lead가 아닌 동시 반응 — 이 자체로는 trade로 변환 불가. "공동 mean reversion 구조가 있다"는 fact는 입증되었지만 alpha 도구는 아직 아님.

**Q: 시간 의존적이라는 게 무슨 의미?**
> 5년을 둘로 쪼개니 2021-2023 (전반)은 신호 약함, 2023-2026 (후반)에 강함. 즉 최근 2년에 강하고 더 옛날엔 약함. 단순 5년 평균만 봐선 안 되고 regime을 따로 봐야 한다는 뜻.

**Q: lead-lag인데 시간차가 0이면 무슨 의미가 있나?**
> 정확히 그 점이 핵심. "DOGE → AVAX" 라는 directional 표현은 트리거 자산일 뿐, 실제 가격 반응에는 시간차가 없음. 그래서 alpha 트레이딩 신호는 안 되고, 대신 "알트 간 공통 beta가 매우 높다"는 시장 구조의 명확한 증거.

**Q: AVAX|DOGE 페어가 두 프로젝트에서 일치한 건 우연 아닌가?**
> 일치 자체는 진짜 — 두 방법 모두 "DOGE 큰 움직임에 알트 반응"이라는 같은 raw 비효율을 잡음. 하지만 결합 시 alpha는 안 됨 (permutation p=0.208).

**Q: 다음에 뭘 다르게 해야 하나?**
> 같은 OHLCV에서 hyperparameter 튜닝 ❌. **선행연구 도구 통합 (VPIN, SUR 모형, idiosyncratic vol grouping)** + Funding rate / cross-exchange / on-chain 같은 새 정보축 추가가 답.

**Q: 선행연구는 저변동성에서 herding 강하다고 하는데 너는 반대잖아?**
> 같은 시장의 다른 종류 herding을 측정한 것. 논문은 일별 시장 dispersion 압축 (CSAD) — 저변동성 컨센서스 따라가기. 우리는 micro-event 후 30분 자산 간 동시 반응 (lead-lag matrix) — macro shock 발생이 trigger라 고변동성에서 강함. 모순 아니라 보완적.

**Q: VPIN이나 informed trading 분석 안 했나?**
> 안 함. 논문이 강조한 informed trading proxy (VPIN)는 우리 lead-lag matrix 트랙과 직교. 다음 1년의 명확한 통합 방향 — VPIN으로 informed trading 측정하고 SUR 모형으로 lead-lag와 결합해서 "누가 driving force인지" 결판.

---

## 부록 — 주요 파일 위치

| 분류 | 경로 |
|---|---|
| 마스터 리포트 (최신) | `outputs/research_master_report_2026-04-22.md` |
| Baseline 결과 | `outputs/baseline/report_summary.md` |
| Paper-like 주간 | `outputs/paper_like/weekly/paper_like_summary.md` |
| XRP 5년 tick mainline | `outputs/tick/xrp_5y/short_horizon/tick_short_horizon_report.md` |
| XRP dual tracker | `outputs/tick/xrp_5y/trackers/dual_tracker/tick_dual_tracker_report.md` |
| 다심볼 basket tracker | `outputs/tick/multi_asset_365d/trackers/candidate_basket/tick_candidate_basket_tracker_report.md` |
| 뉴스 sentiment | `outputs/baseline/sentiment_extension/sentiment_extension_report.md` |
| Reddit sentiment | `outputs/baseline/reddit_sentiment_extension/reddit_sentiment_extension_report.md` |
| **Lead-lag matrix (5년)** | `outputs/tick/multi_asset_5y/lead_lag_matrix/tick_lead_lag_matrix_report.md` |
| **Robustness 검증** | `experiments/lead_lag_robustness/outputs/` |
| **Cointegration × Lead-lag 통합** | `experiments/cointegration_lead_lag/outputs/backtest_summary.csv` |
| 압축 발표본 (5-10분) | `outputs/presentation_short_2026-05-09.md` |
