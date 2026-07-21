# CSAD Mechanical Derivation v1.1 유한표본 수렴 보충등록

## 1. 등록 시점과 amendment 사유

- 등록 시각: 2026-07-21T04:46:51Z
- 원 protocol: `research_protocols/csad_mechanical_derivation_v1.md`
- 원 결과: `outputs/v2/csad_mechanical_derivation_v1/`
- 보충 설정: `configs/research/csad_mechanical_convergence_supplement_v1_1.yaml`
- 보충 결과: `outputs/v2/csad_mechanical_derivation_v1/supplement_v1_1/`

원 v1은 결과 생성 전에 고정됐으며 다음 결과를 냈다.

- 폐형식-모멘트 검증: 3/3
- No-intercept 및 SCSAD Gaussian FPR gate: 6/6
- Standard 및 intercept-restored nominal control: 6/6
- 대칭 DGP 강건성: 45/45
- 대규모 수렴 composite gate: 5/6

유일한 실패는 N=62 SCSAD였다. 이론값 `-82,206.37`, Monte Carlo 평균 `-83,044.91`, 상대오차 1.02%였지만 200회 평균의 99% CI 상단 `-82,233.41`이 이론값을 27.04만큼 포함하지 못했다. 원 protocol의 `mechanical_null_not_confirmed` 판정과 산출물은 수정하지 않는다.

이 보충은 결과를 본 뒤 시작한 finite-sample 진단이다. 따라서 새 결과가 좋아도 원 v1을 사후 통과로 바꾸지 않는다. 목적은 population 폐형식이 틀린지, 유한 T의 OLS 비선형 투영 편향 또는 Monte Carlo 변동인지 구분하는 것이다.

## 2. 결과 확인 전 고정하는 새 simulation

- DGP: iid Gaussian, `sigma=0.025`
- 가중: equal
- 자산 수: 14, 50, 62 모두 사용
- 표본 길이 T: 3,000, 12,000, 48,000
- 반복: 각 셀 300회
- 독립 seed: `2026072103`
- 모형: Standard, no-intercept, intercept-restored, SCSAD
- HAC 및 BH3/BH4 정의는 원 v1과 동일

N=62만 다시 골라 돌리지 않고 원래 세 자산 수를 모두 사용한다. T=12,000 재실행도 새 seed를 사용하므로 원 결과의 단순 재사용이 아니다.

## 3. 보충 성공 기준

Primary supplement gate는 T=48,000의 12개 모형 셀만 사용한다.

No-intercept와 SCSAD의 N=14·50·62, 총 6개 셀은 다음을 모두 만족해야 한다.

1. 이론 target 대비 평균 상대오차 2.5% 이하
2. 이론 target이 300회 평균의 99% Monte Carlo CI 안에 포함
3. 음의 계수 비율 95% 이상
4. raw HAC 및 BH3 false-positive rate 95% 이상

Standard와 intercept-restored의 N=14·50·62, 총 6개 control 셀은 raw 음의 false-positive rate가 7.5% 이하여야 한다.

모든 12개 셀이 통과할 때 `finite_sample_convergence_supported`로 판정한다. T=3,000→12,000→48,000의 상대오차 경로는 방향을 강제하지 않고 기술적으로 모두 보고한다. Monte Carlo 평균의 CI는 유한표본 추정량의 정확한 불편성을 검정하며, population pseudo-true 계수의 수학적 성립과 같은 명제가 아님을 분리한다.

## 4. 최종 해석 규칙

- 원 v1 판정은 항상 `mechanical_null_not_confirmed`와 5/6로 보존한다.
- 보충 gate가 통과하면 `analytic mechanism established; finite-sample convergence supported`라고 표현할 수 있다.
- 보충 gate가 실패하면 정확한 폐형식은 가정 아래 성립하지만 선택한 finite-sample simulation에서 수렴 증거가 충분하지 않다고 표현한다.
- 어느 경우에도 no-intercept 또는 SCSAD 음의 계수를 intentional herding이나 alpha로 해석하지 않는다.
- 새 threshold, seed, 반복, N, T는 보충 결과를 본 뒤 변경하지 않는다.

