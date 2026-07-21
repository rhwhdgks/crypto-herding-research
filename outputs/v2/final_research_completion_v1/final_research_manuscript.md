# From Replication to Falsification: Specification Mechanics, Universe Dependence, and Tick-Level Clustering in Cryptocurrency Herding Research

## Abstract

This project evaluates cryptocurrency herding across classical cross-sectional absolute deviation (CSAD) regressions, paper-aligned CoinMarketCap replications, exchange and universe external validations, factor-adjusted convergence designs, and two years of raw Binance aggregate trades. The published fixed-62 CoinMarketCap corrected-CSAD coefficients were closely reproduced and persisted in a non-overlapping temporal holdout. However, corrected relations weakened across Binance, OKX, and point-in-time universes. A preregistered synthetic-null audit showed that no-intercept CSAD and SCSAD generated negative significance in 97-100% of non-herding simulations, while no empirical coefficient was more negative than its matched null after false-discovery control. Tick-level winner labels failed semantic validation and had no economically meaningful out-of-sample directional response. Zero-tick runs were far more clustered than an exact count-conditioned random arrangement, but their five contemporaneous microstructure associations did not survive the corresponding mechanical-null family test, and they did not improve 5-30 minute absolute-return or realized-volatility forecasts. The evidence therefore supports numerical replication and descriptive market-state clustering, not intentional imitation, universal herding, or short-horizon return predictability.

## 한국어 제목

# 복제에서 반증까지: 암호화폐 herding 통계의 사양 기계성, universe 의존성, tick 배열 clustering

## 초록

본 연구는 암호화폐 herding을 한 개 회귀식의 유의성으로 판단하지 않고, 수치 복제·외부타당성·비허딩 귀무모형·변수 의미·시간 외 표본·미시구조 기계성을 순차적으로 감사했다. 선행논문의 CoinMarketCap fixed-62 corrected CSAD 수치는 거의 정확히 복제됐고 같은 공급자의 비중첩 시간 holdout에서도 유지됐다. 그러나 Binance, OKX, 상장폐지를 포함한 point-in-time universe로 이동하면 엄격한 재현 기준을 통과하지 못했다. 더 중요하게, no-intercept와 SCSAD는 herding 규칙이 없는 synthetic data에서도 97~100%의 거짓양성을 만들었고 empirical 계수는 대응 null보다 더 극단적이지 않았다. Raw tick 연구에서는 방향 label을 폐기했고, zero-run 배열 자체는 거래 수와 zero-tick 수를 보존한 exact null보다 강하게 clustered됐지만 기존 동시점 미시구조 5개 관계는 null 대비 0/5였다. 미래 절대수익률과 실현변동성 예측 family도 0/2였다. 따라서 이 프로젝트는 논문 수치의 복제와 시장상태의 기술적 clustering은 지지하지만, 의도적 모방·보편적 herding·단기 alpha는 지지하지 않는다.

## 1. 연구 질문과 판정 원칙

연구 질문은 세 층으로 분리했다. 첫째, 기존 논문의 숫자를 재현할 수 있는가. 둘째, 그 통계가 herding이 없는 반사실적과 구분되는가. 셋째, 그 변수가 미래 움직임에 증분 정보를 주는가. 첫째가 성공해도 둘째와 셋째를 자동으로 의미하지 않는다.

모든 핵심 연구는 가능한 경우 결과 확인 전 protocol, universe, 기간, family, 다중검정, 효과크기를 고정했다. 결과가 실패하면 같은 표본에서 threshold·자산·기간을 다시 골라 살리지 않았다. 현재 자동매매, tracker, 방향성 alpha는 연구 범위에서 제외한다.

## 2. 데이터와 연구 축

| 축 | 자료 | 기간·크기 | 핵심 역할 |
|---|---|---|---|
| Binance baseline | 14개 USDT 1분봉 | 2024-04-08~2026-04-08, 1,051,199분 | Classical CSAD |
| CMC dynamic | 월별 point-in-time Top-200 | 2018-01-01~2024-04-09 | 논문형 확장·구조변화 |
| CMC fixed-62 | 논문 Table 1 legacy ID | historical 2,291일 + holdout 830일 | 직접 복제·시간 외 검증 |
| Binance·OKX | 14종목 5년 | daily 1,826일, weekly 261주 | 공급자 외부검증 |
| Binance archive | PIT Top-50 | 636,506 asset-day, 584 listing episode | survivor bias 완화 |
| Raw aggTrades | 7개 자산 | 490,560개 15분 bucket | tick 의미·zero-run |
| Exact mechanical null | OOS 층화표본 | 9,044개, 999회 | n·zero-count 조건부 배열 감사 |

## 3. Classical CSAD: Binance에서는 음의 곡률이 없었다

기본식은 `CSAD_t = alpha + beta1 |Rm,t| + beta2 Rm,t^2 + error_t`이다. 고정 14종목 2년 1분봉에서 `beta2=+4.534759`, HAC `t=11.25`였다. Classical herding의 필요조건으로 쓰이는 음의 곡률과 반대다. 1분·5분·15분·1시간·4시간·1일과 EW14·BTC/ETH 제외 EW12를 결합한 12개 셀도 모두 음의 herding 판정을 지지하지 않았다.

이 결과는 암호화폐 전체에 herding이 없다는 뜻이 아니다. 특정 거래소, 고정 survivor universe, 특정 기간에서 standard CSAD 필요조건이 관찰되지 않았다는 뜻이다.

## 4. 선행논문 복제: 숫자는 재현됐다

CMC fixed-62 historical 표본에서 daily SCSAD는 `-2.902 (t=-4.621)`로 논문의 `-2.924 (t=-4.471)`와 근접했고, no-intercept·SCSAD daily·weekly 4개 셀이 모두 통과했다. Weekly 계수도 가까웠지만 공개 규칙으로는 논문 307주가 아니라 달력상 328주가 만들어져 21주 차이를 숨기지 않았다.

논문 이후 2024-04-10~2026-07-18의 비중첩 holdout에서도 current와 lagged-size corrected가 각각 4/4였다. 따라서 '같은 CMC 공급자와 같은 fixed-62에서 corrected 수치가 시간적으로 반복된다'는 명제는 지지된다. 그러나 fixed survivor 목록과 동일 provider이므로 완전한 외부타당성은 아니다.

## 5. 외부타당성: 현실적인 universe로 갈수록 약해졌다

| 표본 | Equal/current | Lagged liquidity | 엄격 판정 |
|---|---:|---:|---|
| CMC fixed-62 historical | 4/4 | 4/4 | 수치 복제 |
| CMC fixed-62 temporal holdout | 4/4 | 4/4 | 동일 provider 시간 재현 |
| Binance fixed-14 | 3/4 | 3/4 | 미통과 |
| OKX listing-aware 14 | 3/4 | 1/4 | 미통과 |
| Binance archive PIT Top-50 | 2/4 | 0/4 | 미통과 |

Binance archive는 21,479개 monthly ZIP과 636,506 asset-day를 열거해 종료 episode 156개를 포함했다. 동일가중도 weekly에서 약해졌고, 전기 quote-volume 가중은 0/4였다. 통합 이질성 `I²=69.3%~96.9%`는 pooled 음수보다 provider·universe·가중법 차이가 핵심임을 보여준다.

## 6. Specification null audit: corrected 계수의 행동 해석은 무효다

4개 비허딩 DGP, 16개 N·기간·가중 시나리오, 시나리오당 300회에서 standard CSAD의 BH 거짓양성 중앙값은 약 1%였지만 no-intercept는 99~100%, SCSAD는 97~100%였다. No-intercept 설명변수에 절편만 복원하자 기존 음의 지지 16/16이 사라지거나 표준화 절대크기가 절반 이상 감소했다.

Empirical 10개 사양×2빈도×3모형×4DGP의 240개 직접 비교에서 대응 null보다 더 음수인 셀은 BH-FDR 후 `0/240`이었다. 구조적 강건성 기준도 `0/10`이었다. 따라서 논문 숫자의 복제 성공과 behavioral herding 식별 실패를 동시에 기록해야 한다.

## 7. Factor-adjusted convergence: 조건부 기술 증거

365일 leave-one-out 단일시장요인 반사실적에서 평균 초과 수렴은 28.22%, 극단시장 `delta2=6.772 (t=4.42, q=2.91e-05)`였다. MKT·SIZE·LIQ·MOM 다요인 normal은 `6.600`, empirical residual은 `6.885`로 전체표본에서 유지됐다.

그러나 730일 창 전체는 미통과였고 regime 4는 단일·다요인·empirical 모두 반대였다. 이 결과는 알려진 공통요인 이상으로 수익률이 모이는 시점이 있다는 기술적 근거지만, 누락 요인과 의도적 모방을 분리하지 못한다.

## 8. Tick 의미 감사: 방향 winner를 폐기했다

구형 `run_clustering_side`는 조건부 run-z가 가장 낮은 category일 뿐 가격 방향이 아니었다. 2년 raw 표본에서 가격 방향 일치율은 49.23%, buyer-maker aggressor 방향 일치율은 48.13%였다. Winner event의 91.82%가 zero-run이었고, up/down/zero 30분 시장중립 계수는 모두 BH `q=0.9336`이었다.

Winner를 없애고 up/down/zero intensity를 동시에 넣은 개발·OOS 회귀에서도 사전 기준 통과는 0/9였다. 미래 방향 계수의 신뢰구간은 ±5bp 실용 경계 안에 있었다. 따라서 과거 winner 기반 전략·tracker는 연구 증거에서 제외한다.

## 9. Zero-run: 배열 clustering은 실재하지만 경제 해석은 제한된다

초기 사전등록 OOS 분석에서 zero-run intensity는 Amihud 비유동성 `-0.207 SD`, 평균 거래간격 `-0.537 SD`, 거래대금 `+0.503 SD`, 절대 aggressor 불균형 `-0.120 SD`와 연결돼 5/5였다. 동시에 미래 절대수익률과 실현변동성 family는 0/2였고 개발→OOS RMSE 개선은 `-0.0073%~+0.0004%`였다.

후속 기계성 감사는 결과 확인 전 n과 zero-tick 수를 고정한 exact binary-arrangement null을 봉인했다. 233,201개 OOS 모집단에서 141개 층, 9,044개 표본, 999회 조합분포를 사용했다. Null FPR은 2.537%로 보정됐지만 실제 `z<=-1.96` share는 87.347%였고 7/7 자산이 두 excess 기준을 통과했다. 즉 zero/non-zero 순서는 단순 무작위 배열이 아니다.

반면 같은 null draw로 5개 동시점 계수 분포를 만들자 BH 기준 통과는 0/5였다. 최종 분류는 `clustering_beyond_counts_but_mechanism_not_distinct`다. 배열의 비무작위성은 남지만, aggTrades만으로 기존 유동성·균형 상태 해석을 독립적으로 확정할 수 없다.

## 10. 통합 결론

1. **수치 복제:** 성공했다. CMC fixed-62 corrected 계수와 시간 holdout은 재현된다.
2. **행동적 herding 식별:** 실패했다. 핵심 corrected 통계가 비허딩 null을 구분하지 못한다.
3. **외부 보편성:** 실패했다. 거래소·point-in-time universe·가중법에 따라 엄격 기준이 무너진다.
4. **Tick 방향 의미:** 반증됐다. Winner side는 가격·aggressor 방향이 아니다.
5. **배열 구조:** 지지된다. Zero-run 순서는 count-conditioned 무작위 배열보다 훨씬 clustered다.
6. **동시점 경제 메커니즘:** 기술적 관계만 남는다. Exact-null family는 0/5다.
7. **미래 예측:** 지지되지 않는다. 방향성, 절대수익률, 실현변동성 모두 사전 기준을 통과하지 못했다.

따라서 이 프로젝트의 최종 기여는 alpha 발견이 아니라, 논문 수치를 재현하면서도 그 행동 해석을 반증하고, tick-level 비무작위 배열과 예측 불가능성을 분리한 재현 가능한 방법론 감사다.

## 11. 가설 종료 현황

총 19개 질문 중 supported 3개, partially supported 2개, descriptive only 1개, falsified 9개, methodologically invalid 1개, 새 외부자료 필요 3개다.

상세 근거·파일·재개 조건은 `hypothesis_closure_status.csv`에 있다. 현재 자료로 닫힌 가설은 같은 표본에서 재최적화하지 않는다.

## 12. 한계와 다음 연구의 최소 조건

- CSAD는 동시적 횡단면 분산이므로 관찰만으로 의도적 모방을 식별하지 못한다.
- Fixed-62와 현재 상장목록은 survivor bias가 있으며, CMC·provider·universe 효과는 완전히 분리되지 않는다.
- Break와 regime 일부는 같은 표본에서 선택된 기술적 결과다.
- AggTrades는 taker order의 fill 집계이며 당시 bid·ask spread, depth, queue를 제공하지 않는다.
- Zero-run exact null은 n과 zero count만 보존하고 duration, size, aggressor sequence는 보존하지 않는다.
- News·Reddit은 `first_seen_at_utc`가 있는 충분한 과거 archive와 검증된 base event 없이는 confirmatory feature가 아니다.
- 다음 연구는 완전히 새로운 거래소·기간의 quote/order-book 자료, 결과 전 protocol, 고정 family, 외부표본을 필요로 한다.

## 13. 재현 자료

최종 패키지의 `research_evidence_manifest.csv`는 본문 근거 파일의 크기와 SHA-256을 보존한다. `artifact_manifest.csv`는 이 폴더 전체 산출물을 봉인하고, `scripts/verify_final_research_completion.py`는 이를 읽기 전용으로 재검증한다. 실행 순서는 `REPRODUCIBILITY.md`에 정리했다.
