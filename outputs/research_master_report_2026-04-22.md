# 암호화폐 허딩 연구 최신 통합 보고서

작성일: 2026-04-22
프로젝트 위치: `/home/jonghan/findalpha/herding`

이 보고서는 2026-04-11 버전 이후 약 열흘간의 변화를 반영한 스냅샷이다. 이전 버전은 `outputs/research_master_report_2026-04-11.md`에 그대로 보존되어 있다.

## 1. 이 연구를 왜 하고 있나

단순하게 말하면 두 가지 질문이다.

`암호화폐 시장에서 사람들이 서로를 따라 움직이는 허딩 비슷한 현상이 실제로 있는가?`

그리고,

`그런 현상이 있다면 아주 짧은 시간 안에 반복되는 가격 반응이 있는가?`

프로젝트는 처음부터 자동매매 시스템이 아니라 재현 가능한 연구 파이프라인을 목표로 했다. 지금까지 연구는 네 단계로 진행됐다.

1. 최근 2년 1분봉 baseline 연구
2. 논문 유사 저빈도 비교 연구
3. XRP 중심 tick microstructure 연구
4. 여러 심볼로 넓힌 일반화와 후보 tracker 구축
5. (신규) 뉴스/Reddit 보조 텍스트 감성 레이어

이번 보고서는 4월 11일 이후 변화의 핵심인 Reddit sentiment 확장 결과를 포함한다.

## 2. Baseline과 Paper-like: 변화 없음

2년 1분봉 baseline CSAD 회귀의 결론은 4월 11일 이후 바뀌지 않았다.

- 관측치 수: 약 `105만`
- `beta2 = 4.534877`, `t = 559.160`

즉 전체 평균에서는 허딩이 강하게 보이지 않는다. 반면 주간 paper-like 저빈도 회귀에서는 논문 방향과 부합하는 음(-)의 beta2가 다시 확인된다.

- `standard_csad = -0.340271`
- `no_intercept_csad = -0.847877`
- `scsad = -0.583287`

이 구분은 중요하다: `학술적 허딩 존재 여부`와 `짧은 시간 안에 쓸 수 있는 패턴`은 같은 질문이 아니다.

## 3. Tick 메인라인과 Tracker 최신 상태

기본 시간 구조는 그대로다.

`15분 micro-herding 이벤트 → 다음 30분 반응`

### 3-1. XRP 5년 dual tracker (as of 2026-04-17)

- `time_17_18`
  - 전체 `171건`, 누적 25.47%, 승률 58.48%, 평균 순수익 0.1356%
  - 최근 60일: 거래 7건, 평균 0.2368%, 승률 71.43%, 누적 1.66%
  - 최근 30일: 거래 1건, 평균 0.1846%
- `time_17_18_prior_drop_q4`
  - 전체 `44건`, 누적 16.44%, 승률 72.73%, 평균 순수익 0.3512%
  - 최근 60일: 거래 2건, 평균 0.6668%
  - 최근 30일: 거래 0건 (고신호 규칙의 희소성 여전)

두 규칙 모두 4월 6일 이후 새 신호가 없다 (기준 시각 기준). 희소 규칙은 재차 `track, not promote` 원칙을 따라야 한다.

### 3-2. 다심볼 후보 basket (as of 2026-04-16)

| 후보 | 전체 건수 | 전체 평균 순수익 | 누적 | 최근 30일 | 최근 60일 | 최근 90일 |
|---|---|---|---|---|---|---|
| AVAX down 21 | 95 | +0.1215% | 11.89% | +0.5934% (3건) | +0.2463% (7건) | +0.0476% (16건) |
| DOGE down 21-22 | 162 | +0.2183% | 40.67% | +0.1725% (13건) | −0.0751% (30건) | −0.0728% (37건) |
| XRP up 21-22 | 171 | +0.1611% | 29.19% | +0.0957% (8건) | −0.1121% (20건) | +0.0131% (41건) |

해석 업데이트:

- 전체 성과는 여전히 `DOGE down 21-22`가 1위이나, **최근 60일·90일에서 음수로 돌아섰다**. 기준 시각인 4월 16일까지 -0.0751%/-0.0728%. 주시 필요.
- `AVAX down 21`은 전체 숫자는 작지만 최근 30일 +0.5934%로 가장 안정적으로 유지되고 있다.
- `XRP up 21-22`도 최근 60일 -0.1121%로 약함. 핵심 mainline이 단기 흐름을 잃고 있다는 것은 민감한 신호.

## 4. 뉴스 Sentiment 확장 (상태 요약)

뉴스 headline 수집·점수화 파이프라인은 그대로 운영 중이다.

- Google News RSS: 안정적
- GDELT DOC API: **여전히 `HTTP 429 Too Many Requests`가 주기적으로 발생** (최근 오류 2026-04-12)
- 저속 수집기(`run_gdelt_slow_collector.py`)로 보완 중이지만 누적 headline 수는 890건 수준

뉴스 sentiment의 이벤트 결합 결과 자체는 4월 11일 이후 구조 변화 없음. 강한 feature group은 여전히 샘플이 적어 단독 알파로는 취급하지 않는다.

## 5. Reddit Sentiment 확장 (신규 / 이번 보고서의 핵심 delta)

### 5-1. 구현 상태

- 구현 완료: 2026-04-20 (`src/reddit_sentiment.py`, `scripts/run_reddit_sentiment_extension.py`)
- 1차 어휘 튜닝: 2026-04-22 (Reddit 슬랭 moon/hodl/rekt/ngmi/wagmi/dump/rug/bagholder/bloodbath/cope 등 추가)
- 전체 산출물: `outputs/baseline/reddit_sentiment_extension/`

구조는 뉴스 extension을 그대로 미러링한다. `src/event_sentiment.py`의 범용 attach/engineer/summary 헬퍼를 재사용하고, Reddit 전용은 loader 래퍼와 report 라벨 치환 정도에 그친다. 뉴스 DEFAULT 어휘 사전은 건드리지 않고, Reddit 어휘는 `configs/baseline/config.yaml`의 `reddit_sentiment_extension.reddit.scoring`에서만 정의된다. 트랙 분리 원칙 유지.

### 5-2. 두 번의 런 비교

| 지표 | Run 1 (뉴스 사전) | Run 2 (슬랭 추가) |
|---|---|---|
| scored posts | 1,946 | 1,946 |
| 구간 | 2025-04-16 ~ 2026-04-20 | 동일 |
| neutral 비율 | 87.72% | 85.41% |
| negative 비율 | 4.98% | 6.47% |
| event-level negative group | 1,428 | 2,457 (+72%) |
| `panic_herd_strong` 건수 | 314 | 409 |
| `negative_shock_strong` 건수 | 264 | 347 |

슬랭 추가의 주된 효과는 **negative 시그널 감지의 개선**이다. positive는 큰 변화가 없다 (Reddit 제목에서 positive 슬랭은 상대적으로 분산·이중해석이 많아 보임).

### 5-3. Event study 결과

`negative_shock` 5분 horizon은 Run 1에서 t=2.05(marginal)까지 나왔으나 Run 2에서는 t=0.75로 희석됐다. 이유는 라벨 재분배 — 고신호 샘플이 새로 만들어진 `negative_shock_strong` 그룹으로 옮겨가면서 원본 `negative_shock` 그룹이 평균화됐다.

Run 2 기준으로 가장 큰 후보는:

- `negative_shock_strong` 6h: 평균 +0.1247%, t=1.05, n=347 (marginal)
- `positive_shock` 6h: 평균 +0.0423%, t=1.97, n=5,018 (marginal)

두 수치 모두 t<2에 머무는 약한 신호다. 다른 그룹들은 대부분 t≈0.

### 5-4. 정직한 해석

Reddit-only 단독 예측력은 어휘를 튜닝한 후에도 약하다. 이 결과 자체가 의미 있는 발견이다.

1. 프로젝트의 가정과 일치한다: 뉴스·Reddit은 **보조 feature**이지 단독 예측자가 아니다.
2. negative 이벤트 감지 개선(+72%)은 "텍스트 데이터의 coverage 자체는 확장 가능하다"는 것을 보여준다 — 신호 품질은 아직 부족하지만 구조적 여지가 있다.
3. 뉴스 대비 Reddit의 `label counts/event`가 여전히 낮다. Reddit public JSON의 샘플 규모와 시간 커버리지 한계가 병목이다.

## 6. 지금 시점에서 가장 정직한 결론

한 줄로:

`암호화폐 허딩 현상은 전체 2년 1분봉 평균으로는 강하게 안 보이지만, tick 기반 short-horizon에서는 특정 심볼과 특정 방향에서 반복되는 미시구조 패턴이 나타난다. 텍스트(뉴스·Reddit)는 이벤트 보조 feature 이상의 단독 예측력을 아직 보여주지 않는다.`

요약:

1. 최근 2년 1분봉 baseline CSAD는 허딩을 강하게 지지하지 않는다
2. weekly 저빈도 비교는 논문 방향과 부합한다
3. tick `15분 → 30분` 구조가 여전히 mainline이다
4. XRP up, DOGE/AVAX down 후보는 살아 있지만 **최근 60~90일 drift가 약해지는 중**이다
5. 뉴스 sentiment는 small-sample 강한 그룹 외에는 예측력 약함
6. Reddit sentiment는 슬랭 튜닝 후에도 단독 예측력 약함, 다만 negative 이벤트 감지는 +72% 개선됨

## 7. 다음에 확인할 것

우선순위:

1. **Tracker drift 모니터링** — XRP up과 DOGE down의 최근 60일 음수가 일시적 변동인지 구조적 변화인지 한 달 더 지켜봐야 판단 가능
2. **뉴스·Reddit 비교 리포트 (claude.md Step 3)** — 두 extension이 모두 존재하므로 feature_overview와 event_study_summary를 나란히 비교하는 리포트 작성이 가능해짐. 단, 결합은 아직 금지.
3. **무료 뉴스 커버리지 확장 (claude.md Step 2)** — GDELT 429가 지속되므로 CoinDesk/Cointelegraph/Decrypt RSS 추가로 총량 확보
4. **Reddit 어휘 2차 튜닝** — negative 개선이 있었던 만큼 positive 어휘도 tune (pump 양면성, hold vs hodl 구분 등)

## 8. 지금 어디를 보면 되나

전체 맥락:

- `outputs/research_master_report_2026-04-22.md` (이 문서)

XRP 메인라인:

- `outputs/tick/xrp_5y/trackers/dual_tracker/tick_dual_tracker_report.md`

다심볼 후보:

- `outputs/tick/multi_asset_365d/trackers/candidate_basket/tick_candidate_basket_tracker_report.md`

뉴스 sentiment:

- `outputs/baseline/sentiment_extension/sentiment_extension_report.md`

Reddit sentiment (신규):

- `outputs/baseline/reddit_sentiment_extension/reddit_sentiment_extension_report.md`

## 9. 확인 리듬

매일 들여다볼 필요는 없다. 다음 정도면 충분하다.

- 내일 한 번
- 1주일 뒤 한 번
- 1개월 뒤 본격 판단

확인 포인트:

- 최근 30일 평균 순수익 유지 여부
- 최근 60일·90일 붕괴 여부 (현재 DOGE down/XRP up이 이 구간에서 약함 — 주시)
- 신호 개수 자체가 너무 얇지 않은지

지금은 새 규칙을 더 찾는 단계보다, 찾은 규칙이 진짜 살아 있는지 지켜보는 단계에 가깝다. 텍스트 감성은 단독 알파가 아니라 **이벤트 품질을 해석하는 보조 층**으로 다루면 된다.
