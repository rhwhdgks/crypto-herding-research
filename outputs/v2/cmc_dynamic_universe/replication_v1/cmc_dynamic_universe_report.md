# CMC 동적 Universe Herding 재현 보고서

## 한 문장 결론

> 일부 corrected specification에서만 herding 계수가 통과해 method-sensitive evidence로 해석합니다.

## 재현 설계

- CMC snapshot: 2017-12-31 ~ 2024-04-09
- 분석: 2018-01-01 ~ 2024-04-09
- universe: 전월 말 Top-200, 월간 고정, $100M market-cap floor
- 제외: stablecoin, wrapped token, liquid-staking derivative, point-in-time peg rule
- market return: 전일 시가총액 가중
- CSAD: 활성 자산 단순 평균 절대편차
- inference: Newey-West HAC, period별 6-test BH-FDR

## 데이터 품질

- snapshot: 2,292일, 1,145,791행
- 500행 미만 partial snapshot: 22일; 최소 318행
- 최소 양수 USD quote 비율: 99.20%
- current metadata: 5,000자산
- 월별 universe: 최소 38, 중앙 164, 최대 191
- 월별 교체: 평균 진입 12.9, 평균 이탈 12.5
- daily 적격일: 2,290/2,291 (99.96%)
- weekly 적격주: 328/328 (100.00%)

### Quality gates

| check | observed | required | passes |
|---|---:|---:|---|
| snapshot_completion | 1.0000 | 1.0000 | True |
| minimum_monthly_universe | 38.0000 | 20.0000 | True |
| eligible_daily_share | 0.9996 | 0.9500 | True |

## Full-sample 핵심 결과

| frequency | model | target | coefficient | HAC t | p | BH q | herding support |
|---|---|---|---:|---:|---:|---:|---|
| daily | no_intercept_csad | market_return_sq | -1.966743 | -3.811 | 0.0001385 | 0.0002769 | True |
| daily | scsad | market_return_cu | -3.086357 | -4.489 | 7.149e-06 | 2.145e-05 | True |
| daily | standard_csad | market_return_sq | -0.136354 | -1.008 | 0.3135 | 0.3762 | False |
| weekly | no_intercept_csad | market_return_sq | -1.989704 | -5.697 | 1.218e-08 | 7.305e-08 | True |
| weekly | scsad | market_return_cu | -2.758694 | -3.076 | 0.002101 | 0.003152 | True |
| weekly | standard_csad | market_return_sq | -0.027837 | -0.181 | 0.8566 | 0.8566 | False |

## 선행논문 수치 대조

- 일별 평균 CSAD: 우리 3.10%, 논문 2.94%
- 주별 평균 CSAD: 우리 8.53%, 논문 7.94%
- 논문은 raw Newey-West 유의성, 우리는 period별 6-test BH-FDR까지 적용한 더 엄격한 판정입니다.

| period | frequency | model | paper coef | our coef | paper t | our t | sign match |
|---|---|---|---:|---:|---:|---:|---|
| full_sample | daily | scsad | -2.924 | -3.086 | -4.471 | -4.489 | True |
| pre_covid | daily | scsad | -14.892 | -13.255 | -3.209 | -2.902 | True |
| covid | daily | scsad | -4.807 | -4.275 | -6.006 | -4.796 | True |
| post_covid | daily | scsad | -30.813 | -28.338 | -7.546 | -6.684 | True |
| full_sample | weekly | standard_csad | -0.115 | -0.028 | -0.405 | -0.181 | True |
| full_sample | weekly | no_intercept_csad | -2.096 | -1.990 | -5.904 | -5.697 | True |
| full_sample | weekly | scsad | -3.359 | -2.759 | -3.632 | -3.076 | True |

## Subperiod 진단

- pre_covid: 4/6 통과 (daily:no_intercept_csad, daily:scsad, weekly:no_intercept_csad, weekly:scsad)
- covid: 4/6 통과 (daily:no_intercept_csad, daily:scsad, weekly:no_intercept_csad, weekly:scsad)
- post_covid: 4/6 통과 (daily:no_intercept_csad, daily:scsad, weekly:no_intercept_csad, weekly:scsad)

## 해석 제약

- 이 분석은 CMC point-in-time dynamic extension이며 원논문 fixed 62-coin sample의 완전 복제가 아닙니다.
- Current metadata tag로 과거 자산 유형을 분류하는 후행적 한계가 있어 point-in-time peg rule을 병행했습니다.
- CMC public endpoint의 22개 partial snapshot을 임의 보간하지 않았고, 행 수·누락 rank를 manifest에 보존했습니다.
- 음의 계수는 허딩의 필요조건으로만 해석하며 intentional imitation을 직접 증명하지 않습니다.
- 이 연구는 수익률 alpha 백테스트가 아닙니다.

## 그림

- `outputs/v2/cmc_dynamic_universe/replication_v1/plots/universe_turnover.png`
- `outputs/v2/cmc_dynamic_universe/replication_v1/plots/daily_csad_vs_market.png`
- `outputs/v2/cmc_dynamic_universe/replication_v1/plots/target_coefficients.png`
