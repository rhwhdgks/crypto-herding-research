# 연구 노트 2026-07-06 — Informed-Trading × 레버리지 축 통합

> 하루 세션에서 informed-trading 축(VPIN-proxy)과 레버리지 축(funding/OI)을 추가하고,
> 첫 ex-ante 수수료권 후보를 발견 → 검증 → forward tracker 등록까지 완결한 기록.
> 커밋: `251266e` → `edd09d1` → `7ee841f` → `81fd424` → `fcd22b2`

---

## TL;DR

1. **Informed cascade 기각**: DOGE down → 알트 공동반등은 DOGE의 order-flow가 **balanced(noise)일 때** 강하고, informed(쏠린) flow에서 소멸. CCF lag=0 "beta" 결론을 두 번째 독립 축으로 재확인.
2. **OHLCV+flow 내부 조합 한계 확정**: vol regime × toxicity 결합으로도 수수료 장벽을 못 넘음 (diffuse beta 3차 확인). → 새 정보축 필요.
3. **새 정보축(funding/OI)에서 첫 ex-ante 후보 발견**: `crowded funding × OI no-flush` 조건에서 알트 4종 공동반등 **+0.13~0.17%/30min** (t=1.9~2.8). 가설했던 "청산 캐스케이드"는 정반대로 기각 — **청산 부재가 반등의 조건**.
4. **검증**: permutation p=0.041 (증분 정보 실재, 아슬아슬) / 실행 시뮬 maker net **+0.131%/trade**, 누적 +31.5%, maxDD 7.9%.
5. **결정적 함정**: 241개 이벤트 전부 2023-01~2024-12. **2024-12 이후 crowded funding 레짐 자체가 부재** (17개월 휴면). 상시 전략이 아닌 레짐 조건부 후보.
6. **Forward tracker 등록 + 첫 실행**: OOS 2개월(2026-05-08~07-04) DOGE down 147건 탐지, crowded 버킷 0.0% → **DORMANT, 신호 0** (설계대로).

---

## Part 1 — Informed-Trading 축: VPIN-proxy (step1~2)

**질문**: BH-FDR 통과한 `DOGE down → AVAX/ADA +30min` 공동반응을 누가 만드는가 — informed trader의 orchestrated cascade인가, noise herding인가?

**방법**: DOGE aggTrades 60개월(**581M trades**)에서 15분 버킷별 order-flow imbalance.
`is_buyer_maker`로 실제 taker aggressor side 사용 (BVC 근사 아님).
`toxicity = |buyVol − sellVol| / totalVol`, `vpin_50` = 50버킷 이동평균.

**결과**: headline 정확 재현 (AVAX t=3.42 / ADA t=3.24) 후 toxicity 3분위 분해 —

| | low (noise) | high (informed) |
|---|---|---|
| DOGE→AVAX | +0.072%, t=2.44 | +0.035%, t=1.39 |
| DOGE→ADA | +0.054%, t=2.15 | +0.025%, t=1.22 |

4개 조합(2 target × 2 지표) 모두 monotonic으로 **low-toxicity에 집중**.

**판정**: informed cascade **기각**. 공동반응은 noise-driven 공동 mean reversion — 선행연구의 "herding은 정보흐름 약할 때의 consensus" 서사와 정합.

## Part 2 — Vol × Toxicity 결합 (step3)

step6(고변동성에서 강함)과 step2(저toxicity에서 강함)의 긴장 해소 시도.

- **교락 없음**: spearman −0.13(전체)/−0.09(이벤트 내), 분포 균등 → 독립 축
- **"high-vol × low-tox 집중" 가설 기각**: 해당 셀 t=1.5/1.0. toxicity 기울기는 low-vol에서만 유지
- 셀 최대 delta +0.09~0.10% — 수수료 장벽(taker 왕복 ~0.10~0.15%) 미달

**판정**: 같은 데이터 내 조합 탐색 중단. **새 정보축으로 이동.**

## Part 3 — 레버리지 축: funding/OI (step4~7)

**데이터** (Binance futures 공개 아카이브): fundingRate 8h + metrics OI 5분(2021-12~, 결손 0일).
15분 그리드, look-ahead 명시 구분: `funding_pre`·`d_oi_24h_pre`(ex-ante) / `d_oi_event`(버킷 종료 시점 관측 완료).

**step5 단변량**: `funding_pre`가 **유일한 ex-ante 단조 기울기** (low t≈0 → high t=2.0~2.6).
OI flush ↔ toxicity spearman −0.01 → 두 축 직교 (Part 1과 모순 아님이 해명됨).

**step6 상호작용 반전**: funding을 경제 카테고리(negative / default / crowded>0.01%)로 나누자 —

| 셀 | AVAX | ADA |
|---|---|---|
| crowded × flush (청산 캐스케이드 가설) | t=0.33 ❌ | t=1.18 ❌ |
| **crowded × no_flush** | **+0.156%, t=2.25** | **+0.165%, t=2.38** |

해석: long 과열에서 DOGE가 빠졌는데 **OI가 버티면**(청산 미발생) = noise dip → 강한 공동반등.
OI가 실제 flush되면 = 디레버리징 개시 → 반등 불확실. **청산 캐스케이드가 아니라 청산 부재가 조건.**

**step7 robustness**: 알트 4종 전부 일반화 (SOL 최강 +0.172% t=2.80), 시간 반분할 부호 유지, BTC/ETH falsification 정합(효과 절반 이하).

## Part 4 — 검증 (step8~9)

**Permutation (step8)**: circular shift null(자기상관 보존, 1000회), 통계량은 알트 4종 basket delta(다중비교 제거).
- 관측 +0.157% vs null 평균 +0.040% / p95 +0.151% → **p=0.041** (one-sided)
- null 평균이 0이 아닌 이유: 기본 반등(t=3.4)은 어떤 부분집합에도 존재 — 이 검정은 **conditioning의 증분**을 잼. 증분 실재하나 z≈1.8로 강하지 않음.

**실행 시뮬 (step9)**: 30min hold, USDT-M 선물 수수료 —

| basket | gross | maker (0.04% RT) | taker (0.10% RT) |
|---|---|---|---|
| 알트 4종 | +0.156% | +0.116% | +0.056% |
| **+DOGE 5종** | +0.171% | **+0.131%** | +0.071% |

5종 maker 기준: win 55.6%, 누적 **+31.5%**, maxDD 7.9%. 연도별 일관 (2023 +0.168% / 2024 +0.151%).

**결정적 함정**: 241개 이벤트 전부 **2023-01-15 ~ 2024-12-07**. DOGE funding이 2024-12 이후 기본요율 초과 이력 없음 → **17개월 레짐 휴면**. step7의 "반분할 안정"은 실질 2023 vs 2024 비교였음.

## Part 5 — Forward Tracker (등록 + 첫 실행)

```bash
.venv/bin/python scripts/run_leverage_candidate_tracker.py --config configs/tick/leverage_candidate/tracker.yaml
```

- 이벤트 정의 = lead-lag matrix와 동일 파라미터 (rolling 5일 15%ile — 원래 ex-ante)
- 진입 필터 = funding_pre > 0.0001 AND d_oi_event > **−0.001146** (in-sample tercile 고정컷, OOS 순수성)
- OOS 경계 2026-05-08, append 로그 + 레짐 리포트(ACTIVE/DORMANT)

**첫 실행 (2026-07-06)**: OOS 2026-05-08~07-04, DOGE down 이벤트 147건, crowded 버킷 **0.0%** → **DORMANT, 신호 0건**. 휴면 진단이 신선한 OOS로 즉시 재확인. 파이프라인은 정상 (이벤트 탐지 작동).

---

## 갱신된 연구 스토리

> 광범위 herding 기각 (β₂=+4.53)
> → 살아남은 lead-lag은 CCF lag=0 **동시반응 beta**
> → informed cascade도 기각 (**noise-driven** 공동 mean reversion)
> → **단, crowded-long 레짐에서는 그 noise 반등이 ex-ante 조건으로 포착 가능하고 수수료를 넘는다** (p=0.041, maker net +0.13%)
> → 현재 레짐 휴면. tracker가 다음 crowded 레짐에서 최종 판정.

## 가설 장부 (오늘자 갱신)

| 가설 | 판정 |
|---|---|
| informed trader가 공동반응을 orchestrate | ❌ (low-toxicity 집중) |
| high-vol × low-tox에 효과 집중 | ❌ (교락은 없으나 집중도 없음) |
| 청산 캐스케이드(crowded × flush)가 메커니즘 | ❌ (t=0.3~1.2) |
| **crowded × no_flush = noise dip 반등** | ⭕ 추적 중 (p=0.041, 레짐 휴면) |

## 산출물 및 재현

- 상세 리포트: `experiments/informed_trading/informed_trading_report.md`
- 스크립트: `experiments/informed_trading/step1~9_*.py` (전 단계 재현 가능)
- Tracker: `scripts/run_leverage_candidate_tracker.py` + `configs/tick/leverage_candidate/tracker.yaml`
- 대용량 중간산출물(23MB VPIN, 21MB state)은 gitignore — step1/step4로 재생성

## 유지보수

- **월 1회 tracker 실행** — DORMANT→ACTIVE 전환 시 OOS 성적 축적 시작
- 성과 주장 금지: tracker가 ACTIVE 레짐에서 생존한 뒤에만
- 잔여 학술 과제: canonical VPIN(equal-volume bucket) + SUR(herding × informed × vol)
