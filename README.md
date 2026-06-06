# Crypto Market Herding Research

> 암호화폐 시장의 **군집행동(herding)** 을 CSAD, tick microstructure, lead-lag matrix, 뉴스/Reddit sentiment feature로 검증하는 재현 가능한 quant research repo.
> 목표는 자동매매 시스템이 아니라, herding-like event가 존재하는지와 그 뒤의 단기 가격 반응이 통계적으로 유지되는지를 검증하는 것이다.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()

---

## TL;DR

1. **광범위 herding 가설은 기각**: 2년 1분봉 Binance 14-자산 universe (105만 obs) CSAD 회귀에서 `β₂ = +4.53` — 시장 전체 herding은 통계적으로 약함.
2. **알트 한정 미시구조 패턴은 통계적으로 살아있음**: 5년 tick aggTrades 데이터로 빌드한 7×6×2 = 84-cell **lead-lag matrix**에서 `DOGE down event → AVAX/ADA +30min` 패턴이 **BH-FDR q=0.05 보정** 통과.
3. **다만 시간적 lead-lag이 아닌 동시 반응 (CCF lag=0 peak)** → alpha 트레이딩 신호로 변환 불가, **알트 간 공통 beta 구조**의 정량적 증거.
4. **연구의 핵심 가치**: alpha를 성급히 주장하지 않고, **죽은 가설의 정리 + robustness 검증 (다중비교 보정 / stability split / permutation test / tick-level CCF) 자동화 파이프라인**을 남김.

선행연구 ("Herding, information cascades, and cryptocurrencies", 2024)는 CSAD/SCSAD + VPIN + SUR 모형으로 informed trading 측면을 다뤘고, 본 연구는 자산 간 **directional lead-lag matrix** 측면을 추가해 보완적 관점을 제공.

---

## Key Findings

| 결과 | 검증 도구 | 결론 |
|---|---|---|
| 1분봉 baseline CSAD β₂ = +4.53 | 105만 obs OLS | 광범위 herding ❌ |
| 주간 paper-like CSAD β₂ 음수 | SCSAD 회귀 | 학술 herding 재현 ✓ |
| Tick 15-min → 30-min event study | XRP 5년 dual tracker | 후보 발견, 거래비용 후 marginal |
| **Lead-lag matrix** `DOGE down → AVAX`, t=3.42 | 5년, 7-자산, 84 cell | **BH-FDR q=0.05 통과 (2 cell)** |
| Stability split (전반 vs 후반 2.5년) | OOS 검증 | 같은 부호 유지, 후반에 강도 ↑ |
| Permutation test (1000 shuffles) | random null 분포 | 시간적 lead-lag 가설 p=0.208 (reject) |
| Tick-level CCF | 1분 단위 cross-correlation | **lag=0 peak → 동시 반응** |

---

## Repository Structure

```
herding/
├── src/                              # Core library modules
│   ├── csad.py                       # CSAD / SCSAD regression
│   ├── event_detection.py            # Herding / shock event detection
│   ├── event_study.py                # Forward-return event study
│   ├── tick_short_horizon.py         # Tick-level micro-herding events
│   ├── tick_lead_lag.py              # 7×7 directed lead-lag matrix
│   ├── tick_archive_backfill.py      # Binance aggTrades downloader
│   ├── news_sentiment.py             # News headline scoring
│   ├── reddit_sentiment.py           # Reddit post sentiment scoring
│   └── ...
│
├── scripts/                          # Pipeline entry points
│   ├── run_pipeline.py               # 1m baseline CSAD + event study
│   ├── run_paper_like_pipeline.py    # SCSAD (Wang-Hudson) weekly
│   ├── run_tick_short_horizon_study.py
│   ├── run_tick_lead_lag_matrix.py   # 7-symbol directed matrix
│   ├── run_tick_dual_tracker.py
│   ├── run_tick_candidate_basket_tracker.py
│   ├── collect_news_headlines.py
│   ├── collect_reddit_posts.py
│   └── ...
│
├── experiments/                      # Hypothesis-specific deep dives
│   ├── lead_lag_robustness/          # 4-step robustness validation
│   │   ├── step1_multiple_comparison.py
│   │   ├── step2_stability_split.py
│   │   ├── step3_tick_level_lead_time.py     # CCF + event-triggered curve
│   │   ├── step4_5y_robustness.py
│   │   ├── step5_5y_ccf.py
│   │   ├── step6_vol_regime.py               # Low vs high vol split
│   │   └── step7_period_split.py             # Bull / winter / recovery
│   │
│   └── cointegration_lead_lag/       # Cross-validation with pair trading
│       ├── step1_15m_screening.py            # Engle-Granger 3-day rolling
│       ├── step2_15m_backtest.py             # Z-score backtest + LAG sweep
│       └── step3_permutation_test.py         # 1000-shuffle null distribution
│
├── configs/                          # YAML configuration files
│   ├── baseline/config.yaml          # 2-year 1m baseline
│   ├── paper_like/{daily,weekly}.yaml
│   └── tick/
│       ├── xrp_5y/                   # XRP 5-year mainline
│       ├── multi_asset_365d/         # 7-symbol 1-year
│       └── multi_asset_5y/           # 7-symbol 5-year (lead-lag matrix)
│
├── outputs/                          # Curated reports (heavy CSVs gitignored)
│   ├── presentation_2026-05-08.md    # Detailed presentation notes
│   ├── presentation_short_2026-05-09.{md,pdf}
│   ├── research_master_report_2026-04-22.md
│   └── tick/multi_asset_5y/lead_lag_matrix/
│
└── data/                             # Raw market data (gitignored)
```

---

## Methodology

### 1. Baseline CSAD (Cross-Sectional Absolute Deviation)

```
CSAD_t = α + β₁ |R_m,t| + β₂ R_m,t² + ε
```

- Binance 14-symbol USDT universe, 1-minute OHLCV, 2-year window
- `β₂ < 0` ⟹ herding (dispersion compresses with market volatility)
- Result: **β₂ = +4.53** (rejects broad herding)

### 2. SCSAD (Standardized CSAD, Wang-Hudson)

Per-asset dispersion을 자산별 변동성으로 정규화해 mechanical scale effect 제거. 주간 회귀에 적용해 학술 herding signal 재현.

### 3. Tick Microstructure Event Study

Binance `aggTrades` → 15-minute buckets → micro-herding score (run-based, Patterson-Sharma–inspired) → event = lowest 15%-tile bucket → 30-minute forward return vs control.

### 4. Lead-Lag Matrix (Novel Contribution)

For each `(leader, target, direction)` ∈ 7 × 6 × 2 = **84 directed cells**:
```
Δ_{L,T,d} = E[R_T(t+30m) | L has micro_herding_d event at t]
           − E[R_T(t+30m) | no event]
```
Welch t-stat 적용, 84개 hypothesis에 대해 BH-FDR 보정.

### 5. Robustness Stack (the core engineering value)

| Step | Test | Implementation |
|---|---|---|
| 1 | Multiple comparison (BH-FDR + Bonferroni) | [`step1_multiple_comparison.py`](experiments/lead_lag_robustness/step1_multiple_comparison.py) |
| 2 | Stability split (in-sample halves) | [`step2_stability_split.py`](experiments/lead_lag_robustness/step2_stability_split.py) |
| 3 | Tick-level CCF + event-triggered curve | [`step3_tick_level_lead_time.py`](experiments/lead_lag_robustness/step3_tick_level_lead_time.py) |
| 4 | 5y window OOS extension | [`step4_5y_robustness.py`](experiments/lead_lag_robustness/step4_5y_robustness.py) |
| 5 | Vol regime split (low vs high) | [`step6_vol_regime.py`](experiments/lead_lag_robustness/step6_vol_regime.py) |
| 6 | Period split (bull/winter/recovery) | [`step7_period_split.py`](experiments/lead_lag_robustness/step7_period_split.py) |
| 7 | Permutation test (1000 shuffles) | [`step3_permutation_test.py`](experiments/cointegration_lead_lag/step3_permutation_test.py) |

---

## Getting Started

### Prerequisites

- Python 3.12+
- ~100 GB free disk space if backfilling full 5-year tick archive (much less for OHLCV only)
- (Optional) MariaDB/MySQL for raw OHLCV persistence

### Install

```bash
git clone https://github.com/rhwhdgks/cryptomarket_herding.git
cd cryptomarket_herding
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Repository rename candidate:

```text
crypto-herding-research
```

Recommended GitHub About:

```text
Reproducible crypto herding research: CSAD/SCSAD, Binance tick microstructure event studies, lead-lag robustness, and news/Reddit sentiment features.
```

### Configure secrets (optional — only if using DB export)

```bash
cp .env.example .env
# edit .env with your DB credentials
export $(cat .env | xargs)
```

### Run baseline pipeline

```bash
# 1-minute Binance baseline CSAD + event study
python scripts/run_pipeline.py --config configs/baseline/config.yaml

# Weekly paper-like SCSAD
python scripts/run_paper_like_pipeline.py --config configs/paper_like/weekly.yaml

# Tick 5-year lead-lag matrix (downloads aggTrades from Binance public archive)
python scripts/run_tick_lead_lag_matrix.py --config configs/tick/multi_asset_5y/lead_lag_matrix.yaml

# Robustness validation suite
python experiments/lead_lag_robustness/step1_multiple_comparison.py
python experiments/lead_lag_robustness/step2_stability_split.py
python experiments/lead_lag_robustness/step3_tick_level_lead_time.py
```

### Reproduce final report

```bash
# Key results in outputs/tick/multi_asset_5y/lead_lag_matrix/
ls outputs/tick/multi_asset_5y/lead_lag_matrix/
# tick_lead_lag_matrix_report.md   ← 7×7 matrix + top edges
# lead_lag_matrix_summary.csv      ← 84 cells flat table
# plots/lead_lag_matrix_{up,down}.png
```

---

## Reference Paper

> **Herding, information cascades, and cryptocurrencies — New evidence using low frequency and high frequency data** (2024)

본 연구는 이 논문의 informed-trading axis (VPIN + SUR)에 더해 **자산 간 directional lead-lag matrix** 차원을 추가. 두 framework는 직교적이고 보완적이며, 다음 단계는 두 axis 통합. 상세 비교: [`outputs/presentation_short_2026-05-09.md`](outputs/presentation_short_2026-05-09.md).

---

## Limitations & Honest Disclosure

- **No live alpha**: BH-FDR 보정 통과한 cell도 effect size가 taker fee (왕복 ~0.15%)에 비해 작음 (5y mean +0.058% / 30min).
- **No VPIN / SUR model yet**: 선행연구의 informed-trading axis는 본 연구와 직교 — 명시적 다음 단계.
- **CCF lag=0**: 통계적으로 유의한 lead-lag 쌍도 동시 반응이고 시간차 leading이 아님. 이를 숨기지 않고 명시.
- **Single exchange (Binance USDT)**: Cross-exchange spillover 미검증.
- 모든 결과는 **research hypothesis**, production strategy 아님.

---

## Roadmap

- [ ] VPIN computation from aggTrades + SUR(herding × VPIN) model
- [ ] Idiosyncratic vol grouping per Patterson-Sharma decomposition
- [ ] Funding-rate / perp-spot basis layer
- [ ] Cross-exchange (Coinbase / OKX / Bybit) spillover
- [ ] GJR-GARCH volatility-asymmetry layer

---

## Author

**고종한 (Jonghan Ko)**
Finance × Quantitative Research

5단 pipeline (baseline → paper-like → tick → multi-asset → sentiment) + 4단 자동화 robustness validation을 결합한 research repo.

---

## License

[MIT](LICENSE)
