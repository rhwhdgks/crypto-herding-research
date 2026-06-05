# 암호화폐 허딩 연구 통합 보고서

작성일: 2026-04-09  
프로젝트 위치: `/home/jonghan/바탕화면/파알/herding`

## 1. 이 연구가 무엇을 하려는가

이 프로젝트의 출발점은 단순합니다.  
암호화폐 시장에서 투자자들이 서로를 따라 움직이는 `허딩(herding)` 유사 현상이 실제로 관측되는지, 그리고 그런 현상이 있다면 그 직후의 가격 반응이 일정한 패턴을 보이는지를 검증하는 것입니다.

이 연구는 처음부터 자동매매 시스템을 만드는 것이 목적은 아니었습니다.  
우선은 재현 가능한 연구 파이프라인을 만들고, 실제 시장 데이터에서 허딩 유사 현상이 통계적으로 보이는지, 보인다면 어느 시간축에서 가장 의미가 있는지를 차근차근 확인하는 것이 목표였습니다.

연구는 크게 세 축으로 진행했습니다.

1. baseline 연구  
최근 2년 Binance 1분봉과 14개 주요 USDT 페어를 사용해 `CSAD`, `herding/shock event`, `event study`를 수행했습니다.

2. 논문 유사 비교 연구  
논문에 더 가까운 저빈도 환경을 만들기 위해 `daily/weekly` 자료, `Newey-West HAC`, `standard / no-intercept / SCSAD` 회귀를 별도로 돌렸습니다.

3. tick microstructure 연구  
1분봉에서는 잘 안 보이는 단기 반응이 tick 구조에서는 더 분명할 수 있다는 가설 아래, `XRPUSDT`의 `aggTrades`를 이용해 15분 이벤트와 이후 30분 반응을 연구했습니다.

---

## 2. 데이터와 기본 설계

### 2.1 Baseline 데이터

- 거래소: Binance Spot
- 주기: 1분봉
- 분석 창: 2024-04-08 ~ 2026-04-08
- 유니버스: 14개 주요 USDT 페어  
  `BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, SOLUSDT, DOGEUSDT, LINKUSDT, AVAXUSDT, DOTUSDT, LTCUSDT, TRXUSDT, ATOMUSDT, NEARUSDT`

### 2.2 핵심 지표

시장 수익률은 동일가중 방식으로 계산했습니다.

허딩 측정의 기본식은 다음과 같습니다.

`CSAD_t = (1/N) * Σ_i |R_{i,t} - R_{m,t}|`

그리고 baseline 회귀식은 다음과 같습니다.

`CSAD_t = alpha + beta1 * |R_m,t| + beta2 * R_m,t^2 + epsilon_t`

여기서 보통 `beta2 < 0`이면 시장이 크게 움직일수록 개별 자산들이 서로 더 비슷하게 움직인다고 해석할 수 있어, 허딩 가능성을 시사합니다.

### 2.3 이벤트 정의

Baseline에서는 두 종류의 이벤트를 정의했습니다.

- `herding-like event`
  - CSAD가 낮다
  - 시장수익률은 극단적이지 않다
- `shock-like event`
  - 시장수익률 절대값이 매우 크거나
  - 변동성 스파이크가 있다

이후 각 이벤트가 발생한 뒤 `5m, 15m, 30m, 1h, 2h, 6h, 1d` 선행 수익률을 비교했습니다.

---

## 3. Baseline 연구 결과

Baseline 핵심 결과는 [`outputs/baseline/report_summary.md`](/home/jonghan/바탕화면/파알/herding/outputs/baseline/report_summary.md)에 정리되어 있습니다.

### 3.1 가장 중요한 결과

- 분석 관측치: 약 105만 개
- `beta2 = 4.534877`
- `t = 559.160`
- herding 이벤트: `24,948`건
- shock 이벤트: `22,045`건

이 숫자의 의미는 명확합니다.  
우리가 기대한 것은 `beta2 < 0`인데, 실제로는 매우 강한 `양수`가 나왔습니다. 따라서 최근 2년 Binance 1분봉 14종목 전체를 한 번에 보면, baseline CSAD 회귀는 허딩을 지지하지 않습니다.

### 3.2 Event Study 결과

보유기간 하이라이트를 보면:

- shock / 6h: `0.0093%`, `t=0.74`
- shock / 5m: `0.0017%`, `t=0.90`
- shock / 30m: `0.0009%`, `t=0.23`
- herding / 15m: `0.0005%`, `t=0.25`

즉 baseline 전체 표본에서는 어느 구간도 아주 강한 예측력을 보이지 않았습니다.  
짧은 구간이 그나마 낫긴 하지만, full-sample 평균만 보면 “바로 쓸 수 있는 신호”라고 부르기 어렵습니다.

---

## 4. Baseline Robustness: 왜 전체 결과가 약하게 보였는가

이 부분은 [`outputs/baseline/robustness/baseline_robustness_report.md`](/home/jonghan/바탕화면/파알/herding/outputs/baseline/robustness/baseline_robustness_report.md)에 있습니다.

### 4.1 표본을 나눠도 beta2는 거의 계속 양수였다

반기, 분기, rolling 30일 창으로 나눠봐도 대부분 `beta2 > 0`였습니다.

- 4분할 구간 1: `beta2 = 4.5302`
- 4분할 구간 2: `beta2 = 3.2094`
- 4분할 구간 3: `beta2 = 6.8957`
- 4분할 구간 4: `beta2 = 4.1767`

즉 full-sample 결과가 단순한 평균 착시만은 아닙니다.  
최근 2년 Binance 1분봉 메이저 코인 시장에서는 baseline CSAD가 구조적으로 허딩을 잘 포착하지 못하거나, 실제로 그런 형태의 허딩이 강하지 않을 수 있습니다.

### 4.2 하지만 수익률 반응은 구간마다 꽤 달랐다

가장 흥미로운 부분은 forward return이 구간에 따라 뒤집힌다는 점입니다.

- 4분할 구간 3 / herding / 1d: `+0.3475%`, `t=9.19`
- 4분할 구간 4 / herding / 1d: `-0.4233%`, `t=-9.97`

즉 “아무 효과가 없다”기보다, 서로 다른 시장 상태가 한 표본 안에 섞이면서 평균이 상쇄되는 `regime mixing` 가능성이 컸습니다.

### 4.3 고변동성 구간이 특히 중요했다

변동성 상태별 비교에서는:

- herding / 1d / 고변동성: `+0.2728%`
- herding / 1d / 저변동성: `-0.1494%`
- shock / 1d / 고변동성: `+0.2208%`
- shock / 1d / 저변동성: `-0.1415%`

이 결과는 이후 연구 방향을 결정하는 데 중요했습니다.  
즉 baseline 전체 표본에서는 약하지만, 특정 상태에서는 반응이 살아날 수 있으니 더 짧은 horizon과 더 미시적인 자료로 내려가 볼 가치가 있다는 뜻이었습니다.

---

## 5. 논문 유사 비교 결과

논문 유사 비교는 [`outputs/paper_like/weekly/paper_like_summary.md`](/home/jonghan/바탕화면/파알/herding/outputs/paper_like/weekly/paper_like_summary.md)와 [`outputs/paper_like/paper_alignment_report.md`](/home/jonghan/바탕화면/파알/herding/outputs/paper_like/paper_alignment_report.md)에 정리돼 있습니다.

### 5.1 왜 별도 비교를 했는가

최근 2년 1분봉 결과만 보면 “허딩이 없다”는 결론처럼 보일 수 있습니다.  
하지만 논문들은 더 긴 기간, 더 낮은 빈도, 더 넓은 유니버스에서 허딩을 다루는 경우가 많습니다.

그래서 Binance OHLCV로 완전 복제는 아니더라도, 논문에 가까운 환경을 별도로 구성해 비교했습니다.

### 5.2 Weekly 결과는 꽤 괜찮았다

Weekly full sample 결과:

- `standard_csad = -0.340271`, `p=0.00075`
- `no_intercept_csad = -0.847877`, `p=3.76e-05`
- `scsad = -0.583287`, `p=0.0678`

해석은 다음과 같습니다.

- standard CSAD: 음수, 유의
- no-intercept: 음수, 유의
- SCSAD: 방향은 음수지만 5% 유의수준은 못 넘김

즉 weekly 저빈도 환경에서는 “허딩 유사 행동이 있다”는 논문 방향과 어느 정도 정렬됩니다.

### 5.3 논문과 얼마나 맞았는가

추적한 주장 7개 중 4개가 일치했습니다.

핵심 해석:

- daily보다 weekly가 논문 방향에 더 가깝다
- no-intercept 모델은 꽤 안정적으로 음수다
- SCSAD도 방향은 음수인데 유의성이 조금 부족하다

이건 중요한 의미가 있습니다.  
즉 “우리 데이터로는 허딩이 전혀 안 보인다”가 아니라,  
`저빈도 장기 환경에서는 보이고, 최근 1분봉 실용 환경에서는 약하다`는 식으로 보는 편이 더 정확합니다.

---

## 6. Tick Microstructure 연구로 왜 내려갔는가

Baseline 1분봉 연구에서는 긴 horizon보다 `15분 ~ 30분` 같은 짧은 구간이 상대적으로 더 나았습니다.  
그래서 질문을 바꿨습니다.

“혹시 허딩 유사 반응은 1분봉 전체 평균보다, tick run 구조에서 더 잘 보이는 것 아닐까?”

이 질문으로 내려간 연구가 tick microstructure 축입니다.

현재 메인라인은:

- 종목: `XRPUSDT`
- 데이터: Binance public `aggTrades`
- 세션: `UTC 16-23`
- 이벤트 버킷: `15분`
- 목표 반응: 이후 `30분`

---

## 7. Tick Short-Horizon 결과

이 부분은 [`outputs/tick/xrp_365d/short_horizon/tick_short_horizon_report.md`](/home/jonghan/바탕화면/파알/herding/outputs/tick/xrp_365d/short_horizon/tick_short_horizon_report.md)에 있습니다.

핵심 결과:

- all / 30m 차이: `+0.0195%`, `t=0.80`
- up / 30m 차이: `+0.0501%`, `t=1.23`
- down / 30m 차이: `-0.0067%`, `t=-0.24`

즉 15분 버킷에서 `up micro-herding`으로 분류된 이벤트가, 이후 30분 반응에서 가장 좋았습니다.

이 단계에서 중요한 결론은 하나였습니다.

`15분 이벤트를 정의하고, 그 뒤 30분 반응을 본다`는 구조가 지금까지 본 short-horizon 설계 중 가장 낫다.

---

## 8. 고정 룰 검증: 최근 30일과 60일에서 재현되는가

30일 holdout 결과는 [`outputs/tick/xrp_365d/fixed_rule/oos/tick_fixed_rule_oos_report.md`](/home/jonghan/바탕화면/파알/herding/outputs/tick/xrp_365d/fixed_rule/oos/tick_fixed_rule_oos_report.md),  
60일 holdout 결과는 [`outputs/tick/xrp_365d/fixed_rule/oos_60d/tick_fixed_rule_oos_report.md`](/home/jonghan/바탕화면/파알/herding/outputs/tick/xrp_365d/fixed_rule/oos_60d/tick_fixed_rule_oos_report.md)에 있습니다.

고정한 룰은:

- 15분 버킷
- up 이벤트
- 다음 30분 반응
- XRPUSDT

결과:

- full sample 차이: `+0.0501%`, `t=1.23`
- holdout 30d 차이: `+0.0778%`, `t=1.30`
- holdout 60d 차이: `+0.0061%`, `t=0.14`

이 해석은 매우 중요합니다.

- 최근 30일만 보면 좋아 보인다
- 하지만 60일로 넓히면 거의 사라진다

즉 “최근에만 좋았던 우연한 신호일 수 있다”는 경고를 동시에 줍니다.

---

## 9. Regime와 Walk-Forward: 그래도 완전히 우연만은 아닌가

### 9.1 Rolling regime

[`outputs/tick/xrp_365d/fixed_rule/regime/tick_fixed_rule_regime_report.md`](/home/jonghan/바탕화면/파알/herding/outputs/tick/xrp_365d/fixed_rule/regime/tick_fixed_rule_regime_report.md) 기준:

- 10일 rolling 양수 비중: `66.85%`
- 20일 rolling 양수 비중: `69.65%`
- 30일 rolling 양수 비중: `75.60%`

완전히 한쪽 방향으로 안정적인 것은 아니지만, 그래도 `양수 구간이 더 많다`는 점은 남았습니다.

### 9.2 30일 walk-forward

[`outputs/tick/xrp_365d/fixed_rule/walkforward/tick_fixed_rule_walkforward_report.md`](/home/jonghan/바탕화면/파알/herding/outputs/tick/xrp_365d/fixed_rule/walkforward/tick_fixed_rule_walkforward_report.md) 기준:

- 12개 블록 중 양(+) 블록 비중: `83.33%`

이건 꽤 좋은 신호였습니다.  
즉 “좋은 한두 달만 있었다”기보다는, 여러 30일 블록에서 같은 방향이 반복되었다는 뜻입니다.

하지만 블록별 크기는 여전히 들쭉날쭉합니다.  
그래서 재현성은 어느 정도 보였지만, 아직 바로 실전 자동매매로 넘기기엔 조심스러웠습니다.

---

## 10. 비용을 넣어 보니 무슨 일이 벌어졌는가

이 부분은 [`outputs/tick/xrp_365d/cost_sanity/tick_cost_sanity_report.md`](/home/jonghan/바탕화면/파알/herding/outputs/tick/xrp_365d/cost_sanity/tick_cost_sanity_report.md)에 있습니다.

핵심 결과:

- full sample 평균 총수익: `0.0244%`
- 손익분기 비용: `2.44 bps`
- holdout 30d 손익분기 비용: `0.94 bps`
- holdout 60d 손익분기 비용: `0.99 bps`

그리고 4bps round-trip 비용을 넣으면:

- full sample 평균 순수익: `-0.0156%`
- holdout 30d 평균 순수익: `-0.0306%`
- holdout 60d 평균 순수익: `-0.0301%`

이 결과가 의미하는 바는 분명합니다.

`연구적으로는 흥미롭지만, 현재 규칙 그대로는 비용을 감당하지 못한다.`

즉 이벤트-대조군 차이가 양수라는 사실과, 실제 매매 후 비용 차감 순수익이 양수라는 사실은 전혀 다릅니다.  
이 단계에서 우리는 “더 강한 subset이 필요하다”는 결론으로 갔습니다.

---

## 11. Subset 후보 비교: 무엇을 남겨야 하는가

이 부분은 [`outputs/tick/xrp_365d/subset_candidates/tick_subset_candidates_report.md`](/home/jonghan/바탕화면/파알/herding/outputs/tick/xrp_365d/subset_candidates/tick_subset_candidates_report.md)에 있습니다.

기준 규칙은:

`XRPUSDT / UTC 16-23 / 15분 up micro-herding / 다음 30분 long`

여기에 해석 가능한 필터를 추가해서 subset을 비교했습니다.

### 11.1 균형형 후보: `prev_neg`

정의:

- 직전 15분 버킷 수익률이 음수일 때만 진입

결과:

- full sample 순수익: `+0.0122%`
- holdout 60d 순수익: `+0.0238%`
- 거래 수: `515건`
- 30일 블록 양수 비중: `54.55%`

해석:

- 거래 수가 비교적 충분하다
- 성과는 아주 강하지 않지만 비용 후에도 간신히 양수
- “균형형 실전 후보”라는 표현이 적절하다

### 11.2 고확신 저회전 후보: `ratio_1_40_16_18`

정의:

- event strength ratio가 `1.40` 이상
- UTC `16-18` 시각대만 사용

결과:

- full sample 순수익: `+0.1493%`
- holdout 60d 순수익: `+0.1083%`
- 거래 수: `84건`
- 30일 블록 양수 비중: `72.73%`

해석:

- 거래 수는 적지만 성과가 가장 좋다
- 매우 강한 이벤트만 남겨서 비용 내구성이 커졌다
- “고확신 저회전 후보”라고 보는 게 적절하다

---

## 12. 후보 두 개를 paper simulation으로 추적해 본 결과

이 부분은 [`outputs/tick/xrp_365d/candidate_paper_sim/tick_candidate_paper_sim_report.md`](/home/jonghan/바탕화면/파알/herding/outputs/tick/xrp_365d/candidate_paper_sim/tick_candidate_paper_sim_report.md)에 있습니다.

비교한 후보는 두 개입니다.

- `prev_neg`
- `ratio_1_40_16_18`

### 12.1 `prev_neg`

- full sample: 거래 `515건`, 평균 순수익 `0.0122%`, 누적 `0.51%`, 최대낙폭 `-22.98%`
- recent 60d: 평균 순수익 `0.0238%`, 누적 `1.44%`

해석:

- 무난하지만 강한 전략은 아니다
- 계속 추적할 가치는 있으나, 단독 메인 후보라고 보긴 어렵다

### 12.2 `ratio_1_40_16_18`

- full sample: 거래 `84건`, 평균 순수익 `0.1493%`, 누적 `13.03%`, 최대낙폭 `-3.58%`
- recent 60d: 평균 순수익 `0.1083%`, 누적 `1.73%`
- recent 90d: 평균 순수익 `0.3335%`, 누적 `9.67%`
- 양(+) 월 비중: `76.92%`

해석:

- 지금까지 실험한 후보 중 가장 강하다
- 저회전이지만 성과와 안정성의 균형이 가장 좋다
- 현재 메인 후보는 이쪽이다

---

## 13. 지금까지 연구를 한 문장으로 요약하면

최근 2년 Binance 1분봉 14종목 전체를 대상으로 한 baseline CSAD 회귀는 허딩을 강하게 지지하지 않았지만, 더 짧은 시간축과 더 미시적인 tick 구조로 내려가면 `XRPUSDT / UTC 16-23 / 15분 up micro-herding / 다음 30분`에서 제한적이지만 일관된 short-horizon alpha 후보가 관찰되었고, 그 안에서도 `ratio_1_40_16_18`가 가장 유망한 고확신 저회전 후보로 남았습니다.

---

## 14. 우리가 지금 확실히 말할 수 있는 것

1. `최근 2년 Binance 1분봉 전체 시장` 기준으로는 baseline CSAD herding 증거가 약하다.  
즉 broad market level herding을 바로 주장하긴 어렵다.

2. `논문 유사 저빈도 weekly`에서는 herding 방향이 더 잘 보인다.  
즉 허딩 현상 자체가 완전히 없다고 보긴 어렵다.

3. `짧은 horizon`, 특히 `15분 이벤트 정의 + 이후 30분 반응`이 가장 유망하다.  
이건 baseline, session, tick 연구가 모두 비슷한 방향을 가리켰다.

4. 단순 up-only 규칙은 비용을 못 버틴다.  
그래서 subset 필터링이 필수다.

5. subset 중에서는 `ratio_1_40_16_18`가 가장 유망하다.  
현재까지는 이 규칙이 가장 “발표할 만하고”, 동시에 “실전 후보로 더 추적할 가치가 있는” 결과다.

---

## 15. 아직 조심해야 하는 점

1. 아직 실시간 자동매매 전략이라고 부르면 안 됩니다.  
연구 후보이지, 집행 가능한 완성 전략은 아닙니다.

2. XRP 단일 종목 의존성이 큽니다.  
따라서 범용 시장 현상인지, XRP 고유 현상인지는 더 확인해야 합니다.

3. 최근 30일이 특히 좋았던 영향이 일부 남아 있습니다.  
60일 holdout에서 성과가 약해지는 구간도 있었기 때문에, 더 긴 미래 구간 검증이 필요합니다.

4. 비용은 생각보다 치명적입니다.  
평균 총수익이 작기 때문에, 체결 품질이 나쁘면 신호는 바로 무너집니다.

---

## 16. 다음 단계 제안

지금 상태에서 가장 자연스러운 다음 단계는 아래 순서입니다.

1. `ratio_1_40_16_18`와 `prev_neg`를 paper-tracking 대상으로 계속 누적 기록
2. 실제 거래 가능한 비용 가정을 더 보수적으로 반영
3. XRP 외에 비슷한 구조가 보이는 단일 심볼이 있는지 점검
4. 그 뒤에야 sentiment, liquidation 같은 보조 특성을 추가

즉 순서는:

`기존 short-horizon alpha 후보를 더 검증 -> 비용과 재현성 확인 -> 그 다음 확장 변수 추가`

가 맞습니다.

---

## 17. 결론

이 프로젝트는 처음에는 broad herding 연구로 시작했지만, 실제 데이터를 계속 파면서 결론이 더 구체화됐습니다.

- 장기/저빈도에서는 허딩 유사 현상이 어느 정도 보인다
- 최근 2년 메이저 코인 1분봉 전체에서는 그 신호가 약하다
- 대신 tick 수준의 짧은 구간으로 내려가면, 제한적이지만 더 실용적인 후보가 생긴다

현재까지 가장 중요한 실무적 결론은 다음과 같습니다.

`우리가 찾은 가장 의미 있는 구조는 "15분 이벤트를 정의하고, 그 뒤 30분 반응을 본다"는 틀이다. 그리고 그 안에서 XRP의 강한 up micro-herding subset, 특히 ratio_1_40_16_18가 가장 유망한 연구 후보다.`

이 문장은 지금까지의 전체 연구 결과를 가장 잘 요약합니다.
