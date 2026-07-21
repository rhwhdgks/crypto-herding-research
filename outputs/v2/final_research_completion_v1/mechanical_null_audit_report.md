# Zero-Run 조건부 배열 Null 기계성 감사

## 한눈에 보는 결론

최종 사전등록 판정은 **clustering_beyond_counts_but_mechanism_not_distinct**이며 전체 감사 기준은 **미통과**입니다.
이 판정은 거래 수와 zero-tick 수를 그대로 보존했을 때도 관찰된 run 배열과 동시점 미시구조 관계가 남는지를 묻습니다. 미래 가격 방향, 수익률 alpha, 의도적 모방 또는 인과관계를 검증하지 않습니다.

- 조건부 cutoff calibration: **통과**
- 실제 excess clustering: **통과** (7/7 자산)
- count-conditioned null을 넘는 mechanism family: **미통과** (0/5)
- 기존 미래 절대수익률·실현변동성 family 0/2 판정은 변경하지 않습니다.

## 사전등록과 표본

결과를 보기 전에 protocol, config, seed, 999회 반복, 23개 그룹, 5개 outcome과 모든 판정 기준을 SHA-256으로 봉인했습니다.

- OOS 적격 모집단: 233,201개 15분 bucket
- 층화 감사 표본: 9,044개, 141개 자산×거래수×zero-share 층
- 조건부 null: 각 row의 `n=transaction_count`, `k=zero_ticks` 고정, 정확한 조합분포에서 999회 추출
- 원본 run-z 재구성 최대 오차: 5.68e-14
- PMF 합 최대 오차: 8.88e-16

## 1. Cutoff calibration과 실제 배열

Pooled null FPR은 2.537%이며 사전 허용범위 1.5%~3.5%와 비교했습니다. 실제 clustering share는 87.347%입니다.

| 그룹 | 실제 평균 intensity | null 평균 | intensity BH q | 실제 tail | null FPR | tail BH q | 두 지표 지지 |
|---|---:|---:|---:|---:|---:|---:|---|
| all | 20.6239 | -0.0001 | 0.001 | 87.347% | 2.537% | 0.001 | 예 |
| ADAUSDT | 3.6038 | -0.0013 | 0.001 | 63.825% | 2.536% | 0.001 | 예 |
| AVAXUSDT | 3.2479 | 0.0028 | 0.001 | 69.112% | 2.649% | 0.001 | 예 |
| BTCUSDT | 49.8248 | -0.0012 | 0.001 | 100.000% | 2.494% | 0.001 | 예 |
| DOGEUSDT | 8.0748 | -0.0015 | 0.001 | 83.896% | 2.497% | 0.001 | 예 |
| ETHUSDT | 54.3442 | -0.0005 | 0.001 | 100.000% | 2.602% | 0.001 | 예 |
| SOLUSDT | 8.7380 | 0.0010 | 0.001 | 94.520% | 2.522% | 0.001 | 예 |
| XRPUSDT | 10.5630 | 0.0008 | 0.001 | 93.529% | 2.490% | 0.001 | 예 |

## 2. 동시점 미시구조 관계

실제 감사 표본 계수와 같은 row의 count-conditioned null 계수 999개를 비교했습니다. 계수는 기존 개발구간 scaler를 재사용한 가중 표준화 계수입니다.

| Outcome | Full OOS | 감사 표본 | null 95% | BH q | 크기비율 | 최종 |
|---|---:|---:|---:|---:|---:|---|
| Amihud형 비유동성 | -0.2071 | -0.2141 | [-0.8922, 0.9549] | 0.886 | 1.03 | 미통과 |
| 무가격변화 tick 비중 | -0.1083 | -0.0938 | [-0.5187, 0.5032] | 0.886 | 0.87 | 미통과 |
| 평균 거래 간격 | -0.5371 | -0.5595 | [-0.4753, 0.4546] | 0.1 | 1.04 | 미통과 |
| USDT 거래대금 | 0.5027 | 0.5376 | [-0.5321, 0.5324] | 0.115 | 1.07 | 미통과 |
| 절대 aggressor 불균형 | -0.1201 | -0.0882 | [-1.2622, 1.2002] | 0.886 | 0.73 | 미통과 |

## 3. 해석 경계

- 이 감사가 통과하면 zero/non-zero 배열 순서와 동시점 거래구조의 관계가 단순한 거래 수와 zero-tick 수 이상의 정보를 가진다는 뜻입니다.
- 이 감사가 미통과하면 기존 5/5 동시점 관계의 일부 또는 전부가 run-z 산식과 구성변수의 기계적 연결로 설명될 수 있다는 뜻입니다.
- 어느 경우에도 같은 15분 안의 동시성만으로 투자자의 의도적 따라하기나 인과를 식별할 수 없습니다.
- AggTrades에는 호가 spread와 depth가 없으므로 quote/order-book 자료가 있어야 다음 메커니즘을 직접 검증할 수 있습니다.
- 기존 OOS 미래 family 0/2, 방향성 alpha 미검정, tracker 비활성 상태는 그대로 유지합니다.

## 그림

- `outputs/v2/final_research_completion_v1/plots/conditional_null_fpr.png`
- `outputs/v2/final_research_completion_v1/plots/empirical_vs_conditional_null.png`
- `outputs/v2/final_research_completion_v1/plots/mechanism_vs_conditional_null.png`

## 재현

```bash
python scripts/run_zero_run_mechanical_null_audit.py
python scripts/verify_final_research_completion.py
```

프로토콜과 설정은 null 결과를 보기 전에 봉인됐으며, 산출물의 해시와 판정식은 읽기 전용 verifier에서 다시 계산합니다.
