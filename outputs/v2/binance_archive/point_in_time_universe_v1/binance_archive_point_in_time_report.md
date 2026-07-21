# Binance Archive Point-in-Time Universe 외부검증

## 한눈에 보는 결론

- 동일가중 corrected 필수 셀: 2/4 통과
- 전기 quote-volume 가중 필수 셀: 0/4 통과
- 동일가중 strict criterion: 미통과
- 전기 quote-volume 가중 strict criterion: 미통과
- 이 결과는 동시적 횡단면 수렴 관계의 외부검증이며 미래수익률 alpha, 인과효과 또는 의도적 모방의 검정이 아닙니다.

## 왜 이 연구를 다시 했나

현재까지 살아남은 코인만 고르면 과거에 상장폐지된 실패 자산이 빠집니다. 이를 survivor bias라고 하며, 과거의 시장이 실제보다 안정적이거나 동질적이었던 것처럼 보이게 할 수 있습니다. 이번 검증은 현재 목록이 아니라 Binance 공식 아카이브에 당시 일봉이 실제로 남은 종목을 다시 열거해 이 편향을 줄였습니다.

핵심 질문은 단순합니다. 상장폐지 종목까지 포함하고, 당시에 알 수 있었던 전월 정보로만 종목을 골라도 corrected CSAD 관계가 daily와 weekly에서 모두 재현되는지를 묻습니다.

## 누가 다시 실행해도 같은 표본을 만드는 방법

1. Binance Vision spot monthly `1d` bucket의 symbol prefix를 페이지 끝까지 전수 열거합니다.
2. 결과를 보기 전에 고정한 USDT·leveraged token·stable/fiat·wrapped/staked 필터만 적용합니다.
3. 같은 ticker에 7일을 넘는 공백이 있으면 새 listing episode로 분리해 LUNA처럼 이름이 재사용된 자산의 수익률을 연결하지 않습니다.
4. 각 달의 universe는 직전 달 20일 이상 거래된 episode 중 quote volume 상위 50개로 고정합니다. 당월 신규상장은 다음 달부터만 들어옵니다.
5. daily·weekly와 동일가중·정확한 전기 quote-volume 가중을 나눠 standard, no-intercept, SCSAD를 실행하고 각 6개 검정군을 BH-FDR로 보정합니다.

사전 strict 기준은 daily·weekly no-intercept·SCSAD 네 셀의 계수가 모두 음수이고 BH q-value가 0.05 이하일 것입니다. 일부 셀만 좋은 결과를 전체 재현으로 포장하지 않기 위한 기준입니다.

## 데이터와 생존편향 통제

- 공식 archive root에서 발견한 전체 symbol prefix: 3,643개
- 사전 필터를 통과한 USDT 현물 후보: 611개
- 분석 범위 monthly ZIP: 21,479개
- listing episode: 584개, 종료일 이전 archive가 끊긴 episode: 156개
- 월별 universe: 전월 20일 이상 관측 episode 중 quote volume 상위 50개
- membership 월: 61개, 실제 편입 episode: 375개
- 공식 checksum 검증: 3,801/3,801개
- 품질 gate: 15/15 통과
- 원시 ZIP 전수 감사에서 완전 동일 행 1개만 축약했고 값이 다른 충돌 timestamp는 0개였습니다.
- point_in_time_equal_weight: daily 1,826개, weekly 262개, daily 활성 episode 48~50개
- point_in_time_lagged_liquidity: daily 1,826개, weekly 262개, daily 활성 episode 48~50개
- weekly panel은 경계일 가격 정렬을 위한 부분 시작주까지 262개를 보존하고, 사전 5년 회귀 구간은 완전히 들어오는 261개 주를 사용합니다.

## 가중치 농축 감사

- point_in_time_equal_weight / daily: 기간별 최대 종목비중 중앙값 2.0%, 95백분위 2.0%, 최대 2.1%, 유효 종목 수 중앙값 50.0개
- point_in_time_equal_weight / weekly: 기간별 최대 종목비중 중앙값 2.0%, 95백분위 2.0%, 최대 2.0%, 유효 종목 수 중앙값 50.0개
- point_in_time_lagged_liquidity / daily: 기간별 최대 종목비중 중앙값 29.8%, 95백분위 70.0%, 최대 79.0%, 유효 종목 수 중앙값 6.9개
- point_in_time_lagged_liquidity / weekly: 기간별 최대 종목비중 중앙값 29.4%, 95백분위 68.3%, 최대 75.8%, 유효 종목 수 중앙값 7.1개

## 5년 전체표본 회귀

- point_in_time_equal_weight / daily / standard_csad: 계수 0.1964, 표준화 0.115, t=0.53, BH q=0.599, 지지=False
- point_in_time_equal_weight / daily / no_intercept_csad: 계수 -1.2550, 표준화 -0.736, t=-4.08, BH q=0.000137, 지지=True
- point_in_time_equal_weight / daily / scsad: 계수 -2.4598, 표준화 -0.226, t=-7.80, BH q=3.84e-14, 지지=True
- point_in_time_equal_weight / weekly / standard_csad: 계수 0.4073, 표준화 0.443, t=0.80, BH q=0.599, 지지=False
- point_in_time_equal_weight / weekly / no_intercept_csad: 계수 -0.4676, 표준화 -0.508, t=-1.20, BH q=0.457, 지지=False
- point_in_time_equal_weight / weekly / scsad: 계수 -0.3627, 표준화 -0.117, t=-0.62, BH q=0.599, 지지=False
- point_in_time_lagged_liquidity / daily / standard_csad: 계수 0.3808, 표준화 0.726, t=14.43, BH q=1.93e-46, 지지=False
- point_in_time_lagged_liquidity / daily / no_intercept_csad: 계수 0.2123, 표준화 0.405, t=10.73, BH q=2.31e-26, 지지=False
- point_in_time_lagged_liquidity / daily / scsad: 계수 0.2155, 표준화 0.715, t=3.17, BH q=0.00225, 지지=False
- point_in_time_lagged_liquidity / weekly / standard_csad: 계수 0.8428, 표준화 0.814, t=4.33, BH q=2.96e-05, 지지=False
- point_in_time_lagged_liquidity / weekly / no_intercept_csad: 계수 -0.0068, 표준화 -0.007, t=-0.02, BH q=0.982, 지지=False
- point_in_time_lagged_liquidity / weekly / scsad: 계수 0.9534, 표준화 0.379, t=1.69, BH q=0.109, 지지=False

## 전반·후반 안정성

- full_5y / point_in_time_equal_weight: corrected 2/4 통과
- full_5y / point_in_time_lagged_liquidity: corrected 0/4 통과
- early_half / point_in_time_equal_weight: corrected 2/4 통과
- early_half / point_in_time_lagged_liquidity: corrected 0/4 통과
- late_half / point_in_time_equal_weight: corrected 4/4 통과
- late_half / point_in_time_lagged_liquidity: corrected 4/4 통과

동일가중은 5년 전체에서 daily 두 셀만 통과했고 weekly 두 셀은 통과하지 못했습니다. 전기 유동성가중은 5년 전체에서 네 셀 모두 미통과였습니다. 후반기에는 두 가중법 모두 4/4로 바뀌지만, 이는 full-sample 사전 판정을 대체하지 않으며 관계가 시기에 따라 크게 달라진다는 진단으로만 사용합니다.

## Descriptive pooled evidence

- daily / no_intercept_csad: random-effects 표준화 계수 -0.516 (95% CI -1.037~0.006), BH q=0.0525, I-squared=96.8%
- daily / scsad: random-effects 표준화 계수 -0.147 (95% CI -0.236~-0.058), BH q=0.00164, I-squared=82.2%
- weekly / no_intercept_csad: random-effects 표준화 계수 -0.804 (95% CI -1.064~-0.545), BH q=5.08e-09, I-squared=81.9%
- weekly / scsad: random-effects 표준화 계수 -0.201 (95% CI -0.318~-0.085), BH q=0.0014, I-squared=69.6%

CMC·Binance·OKX의 8개 사양을 표준화해 합치면 random-effects에서 4셀 중 3셀이 음수·BH-FDR를 통과합니다. 그러나 I-squared가 69.6%~96.8%로 매우 높아, 평균 계수 하나보다 공급자·universe·가중법에 따라 결과가 달라진다는 사실이 더 중요합니다. 특히 archive point-in-time 유동성가중은 daily corrected 계수가 양수로 바뀌어 보편적 herding 관계라는 해석을 약화시킵니다.

## 결국 무슨 의미인가

이번 결과는 corrected CSAD 관계가 완전히 사라졌다는 뜻도, 암호화폐 시장 전체에서 항상 허딩이 있다는 뜻도 아닙니다. 생존 종목 고정 표본에서 비교적 넓게 보이던 관계가 상장폐지를 포함한 동적 universe와 실제 전기 유동성 가중에서는 크게 약해졌다는 뜻입니다.

따라서 현재 가장 안전한 결론은 `corrected CSAD는 특정 표본·시기·가중법에서 횡단면 수렴을 보여주지만, 보편적이거나 투자 가능한 alpha로 식별되지 않았다`입니다.

## 해석 제한

- Binance Vision archive 존재는 당시 거래 가능성의 감사 가능한 proxy지만 완전한 historical exchangeInfo master는 아닙니다.
- 상장폐지 시각은 마지막 complete UTC daily candle로 근사하며 공식 공지 시각과 다를 수 있습니다.
- ticker 공백 7일 초과를 새 episode로 분리했지만 모든 토큰 migration을 완전하게 식별하지 못할 수 있습니다.
- 정적 stablecoin·wrapped token 제외는 point-in-time taxonomy의 완전한 복원이 아닙니다.
- 결과 확인 후 taxonomy 감사에서 사전 제외목록에 없던 USD 이름 episode 5개가 일부 후반 membership에 포함된 것을 찾았습니다. primary를 post-hoc로 바꾸지 않고 감사표로 보존합니다.
- 전기 quote-volume 가중은 특정 날 단일 종목 비중이 크게 올라가므로 동일가중 결과와 함께 해석해야 합니다.
- pooled estimate는 기간과 표본이 겹치는 의존적 결과의 기술적 합성이며 독립 연구 메타분석으로 해석하지 않습니다.
- 유의한 음의 corrected coefficient가 있어도 미래수익률 예측력이나 거래전략 수익성을 뜻하지 않습니다.

## 그림

- `outputs/v2/binance_archive/point_in_time_universe_v1/plots/corrected_csad_meta_forest.png`
