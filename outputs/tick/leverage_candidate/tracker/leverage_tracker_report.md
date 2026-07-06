# Leverage-state 후보 forward tracker

실행: 2026-07-06 10:35 UTC
OOS 구간: 2026-05-08 ~ 2026-07-04

## 규칙
- DOGEUSDT down micro-herding event (15m, rolling 5d 15%ile)
- funding_pre > 0.0001 (crowded long)
- d_oi_event > -0.001146 (OI no-flush, in-sample tercile 고정컷)
- basket: AVAX, ADA, XRP, SOL, DOGE 동일가중 +30min

## 레짐 상태
- OOS 구간 crowded funding 버킷 비중: **0.0%**
- 판정: **DORMANT** (기준: 5% 초과 시 ACTIVE)

## 신규 관측
- DOGEUSDT down 이벤트: 147
- 진입 신호 (전체 조건 충족): **0**

## 누적 성과
- 아직 기록된 신호 없음 (레짐 휴면 지속 중이면 정상)

## 참고
- 근거 분석: `experiments/informed_trading/informed_trading_report.md`
- 판정 기준: `outputs/tracker_decision_criteria_2026-04-11.md`
- 이 후보는 crowded-long 레짐 조건부 — DORMANT 기간의 신호 부재는 후보 결함이 아님