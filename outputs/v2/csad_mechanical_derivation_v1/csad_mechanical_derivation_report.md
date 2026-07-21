# CSAD 기계적 음의 계수 연구 보고서

## 1. 한 문장 결론

사전 고정한 모든 조건을 통과하지 못해 기계적 null을 확정하지 못했습니다.

이 결과는 기존 empirical 계수가 틀렸다는 뜻이 아니라, 그 계수만으로 행동적 herding을 식별할 수 없다는 뜻입니다.

## 2. 왜 음수가 되는가

독립 Gaussian 수익률에서는 시장평균 `M`과 각 코인의 평균 이탈분이 독립입니다. 따라서 시장이 어느 방향으로 얼마나 움직였는지와 무관하게 평균 CSAD는 양의 상수입니다.

No-intercept 식은 이 양의 상수를 표현할 절편이 없습니다. 그래서 `|M|` 항이 원점에서 빠르게 올라가고, `M^2` 항이 바깥쪽을 다시 아래로 굽혀 상수에 가깝게 만듭니다. 이때 제곱항은 수학적으로 반드시 음수가 됩니다.

SCSAD는 양의 시장수익률에서 `+CSAD`, 음의 시장수익률에서 `-CSAD`이므로 원점에서 부호가 바뀌는 계단 모양입니다. 이를 선형항과 세제곱항으로 근사하면 선형항이 계단을 따라 올라가고 세제곱항이 양 끝의 과도한 증가를 누르므로 세제곱항이 음수가 됩니다.

폐형식은 다음과 같습니다. `s=sigma/sqrt(N)`, `c=E[CSAD]`, `u=sqrt(2/pi)`입니다.

```text
No-intercept delta2* = (c/s^2) * (1 - 4/pi) / (3 - 8/pi) < 0
SCSAD gamma3*        = -c*u / (6*s^3) < 0
Standard beta2*      = 0
Intercept-restored   = 0
```

## 3. 수학식 독립 검증

자산 수 14·50·62의 세 셀에서 폐형식과 Gaussian 절대모멘트 정규방정식의 최대 차이는 `7.276e-11`였습니다.
사전 허용오차 이내 통과는 `3/3`입니다.

## 4. 대규모 Monte Carlo 수렴

각 자산 수에서 12,000개 시점과 200회 반복을 사용했습니다. 아래 값은 Monte Carlo 평균과 이론값의 차이입니다.

| 모형 | N | 이론 계수 | MC 평균 | 상대오차 | 음수 비율 | raw FPR | BH3 FPR | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| No-intercept | 14 | -259.407 | -260.078 | 0.26% | 100.0% | 100.0% | 100.0% | 통과 |
| No-intercept | 50 | -951.764 | -950.604 | 0.12% | 100.0% | 100.0% | 100.0% | 통과 |
| No-intercept | 62 | -1182.52 | -1186.86 | 0.37% | 100.0% | 100.0% | 100.0% | 통과 |
| SCSAD | 14 | -8569.36 | -8639.12 | 0.81% | 100.0% | 100.0% | 100.0% | 통과 |
| SCSAD | 50 | -59417.8 | -59385.7 | 0.05% | 100.0% | 100.0% | 100.0% | 통과 |
| SCSAD | 62 | -82206.4 | -83044.9 | 1.02% | 100.0% | 100.0% | 100.0% | 실패 |

절편이 있는 control의 raw 음의 거짓양성률은 다음과 같습니다.

| 모형 | raw FPR 최소 | 중앙 | 최대 | 7.5% gate 통과 |
|---|---:|---:|---:|---:|
| Intercept-restored | 1.5% | 2.0% | 3.0% | 3/3 |
| Standard | 1.5% | 2.0% | 3.0% | 3/3 |

## 5. 더 현실적인 비허딩 과정

정규성만의 우연인지 확인하기 위해 공통요인, 확률변동성, Student-t fat-tail, 양의 왜도, jump, 시변 상관, 음의 충격 비대칭을 포함했습니다. 어느 과정에도 다른 자산의 행동을 보고 따라가는 규칙은 없습니다.

| 모형 | raw FPR 최소 | 중앙 | 최대 |
|---|---:|---:|---:|
| Intercept-restored | 0.0% | 2.5% | 70.7% |
| No-intercept | 95.7% | 100.0% | 100.0% |
| SCSAD | 0.3% | 100.0% | 100.0% |
| Standard | 0.0% | 0.3% | 42.7% |

사전 정의한 대칭 DGP에서 두 기계적 모형의 음수 비율이 모두 80% 이상인 셀은 `45/45`였습니다. 통과 비중은 `100.0%`이고 75% gate는 `통과`했습니다.

Standard CSAD도 이분산성·fat-tail·비대칭처럼 `E[CSAD|M]`가 상수가 아닌 과정에서는 명목 오탐률을 벗어날 수 있습니다. 따라서 절편을 넣는 것만으로 행동적 herding이 자동 식별되는 것은 아닙니다.

## 6. 네 가지 최종 판정

### 1. 기존 음의 CSAD 계수가 실제 herding 증거인가?

아닙니다. 음의 계수는 관찰된 수치 관계이지만 intentional imitation의 충분조건이 아닙니다. 특히 no-intercept와 SCSAD target은 비허딩 Gaussian null에서도 population 값 자체가 음수입니다.

### 2. 모형 구조가 만든 통계적 착시인가?

Gaussian null에 대해서는 `확정 불가`입니다. 최종 사전등록 판정은 `mechanical_null_not_confirmed`입니다. 절편 없는 양의 수준 근사와 sign-step의 cubic 근사가 음의 곡률을 강제합니다.

### 3. 어떤 CSAD 사양까지 연구에 사용할 수 있는가?

Standard CSAD는 절편을 포함하고 해당 universe·가중·분포에 맞춘 simulation 또는 bootstrap null로 크기를 교정할 때 기술적 조건부 수렴 검정에 사용할 수 있습니다. No-intercept와 현재 SCSAD의 일반 HAC p-value는 herding 식별 검정으로 사용하지 않습니다. 두 모형은 선행논문 수치 재현과 방법론 진단용으로만 보존합니다.

### 4. 후속 실증연구의 식별 기준은 무엇인가?

1. 절편을 임의로 제거하지 않는다.
2. 선택한 universe, 가중법, 표본 길이와 동일한 비허딩 null에서 검정 크기를 먼저 교정한다.
3. 공통요인·시변변동성·fat-tail·jump를 포함한 반사실적보다 empirical 계수가 더 극단적인지 확인한다.
4. provider, 기간, point-in-time universe가 다른 외부 표본에서 방향과 효과크기를 재현한다.
5. 음의 계수를 투자자 의도나 alpha로 바로 번역하지 않는다.

## 7. 가정과 한계

- 정확한 폐형식은 독립·동분산 Gaussian 자산과 동일가중 시장에 한정됩니다.
- 강건성 simulation은 사전 고정한 여덟 DGP를 다루지만 가능한 모든 시장구조의 증명은 아닙니다.
- HAC 유의성과 BH-FDR은 모형이 잘못 지정됐을 때 경제적 null을 복구하지 못합니다.
- 이 연구는 뉴스, sentiment, 주문흐름, 미래수익률 또는 거래비용을 검정하지 않았습니다.

## 8. 재현성

- Protocol SHA-256: `e321e53712f86f767e2f8a9625958acfb39b334976fbc3f19fa603da32c07bb4`
- Config SHA-256: `2262fde4375feeb2c942ea54ad73c1ab94c8de16d96af8a5bb1dbbcf6492b78c`
- 최종 판정: `mechanical_null_not_confirmed`
- Gaussian equation gate: `3/3`
- Mechanical convergence gate: `5/6`
- Nominal control gate: `6/6`

실행:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_csad_mechanical_derivation.py
PYTHONPATH=src .venv/bin/python scripts/verify_csad_mechanical_derivation.py
```

## 9. 그림

- `plots/gaussian_theory_convergence.png`
- `plots/nonherding_false_positive_contrast.png`
- `plots/mechanical_negative_sign_robustness.png`
- `plots/projection_mechanism.png`

## 10. 구현 sanity check

독립 Gaussian 동일가중 셀의 평균 `corr(CSAD, M)` 절대값 중앙은 `0.0010`, `corr(CSAD, |M|)` 절대값 중앙은 `0.0007`였습니다.
