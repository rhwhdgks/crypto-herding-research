# CSAD Mechanical Coefficient Derivation v1 사전등록

## 1. 등록 시점과 연구 범위

- 등록 시각: 2026-07-21T04:31:43Z
- 설정: `configs/research/csad_mechanical_derivation_v1.yaml`
- 결과: `outputs/v2/csad_mechanical_derivation_v1/`
- 이 문서와 설정은 새 simulation 결과를 생성하기 전에 고정한다.
- 기존 `csad_specification_audit_v1`의 결과와 파일은 읽기 전용 선행 증거로 취급하고 덮어쓰지 않는다.
- 목적은 intentional herding이 없는 자료에서도 no-intercept CSAD와 SCSAD의 음의 비선형 계수가 발생하는 수학적 이유를 규명하는 것이다.
- 이 연구는 투자자 모방, 미래수익률 alpha 또는 거래전략을 검정하지 않는다.

## 2. 모형과 target

### Standard CSAD

\[
CSAD_t=\alpha+\beta_1|M_t|+\beta_2M_t^2+u_t
\]

### No-intercept CSAD

\[
CSAD_t=\delta_0M_t+\delta_1|M_t|+\delta_2M_t^2+u_t
\]

### Intercept-restored control

\[
CSAD_t=\alpha+\delta_0M_t+\delta_1|M_t|+\delta_2M_t^2+u_t
\]

### SCSAD

\[
SCSAD_t=\operatorname{sign}(M_t)CSAD_t
=\alpha+\gamma_1M_t+\gamma_2M_t^2+\gamma_3M_t^3+u_t
\]

Target은 앞의 세 모형에서 제곱항, SCSAD에서 세제곱항이다. 모든 유의성 검정은 양측 HAC p-value를 사용하되 false positive는 target 계수가 음수이고 `p<=0.05`인 경우로 정의한다.

## 3. Gaussian null의 사전 고정 수학적 명제

자산수익률이 서로 독립인 `R_it ~ N(0, sigma^2)`이고 시장수익률이 동일가중 평균이라고 가정한다. 자산 수는 `N>=3`이다.

표본평균 `M_t`와 Gaussian 잔차벡터 `R_it-M_t`는 독립이다. 따라서 `CSAD_t`와 `M_t`도 독립이고 다음이 성립한다.

\[
s_M=\frac{\sigma}{\sqrt{N}},\qquad
c_N=E[CSAD_t]=\sigma\sqrt{1-\frac{1}{N}}\sqrt{\frac{2}{\pi}}
\]

`X=|M|`, `u=sqrt(2/pi)`라 두고 양의 상수 `c_N`을 절편 없이 `aX+bX^2`로 투영하면 정규방정식은 다음과 같다.

\[
\begin{bmatrix}E[X^2]&E[X^3]\\E[X^3]&E[X^4]\end{bmatrix}
\begin{bmatrix}a\\b\end{bmatrix}
=c_N\begin{bmatrix}E[X]\\E[X^2]\end{bmatrix}
\]

Gaussian 절대모멘트를 대입한 사전 이론값은 다음과 같다.

\[
\delta_1^*=\frac{c_N}{s_M}\frac{u}{3-8/\pi},\qquad
\delta_2^*=\frac{c_N}{s_M^2}\frac{1-4/\pi}{3-8/\pi}<0
\]

대칭성 때문에 signed `M` 계수 `delta_0^*`는 0이다. 음의 `delta_2^*`는 양의 절편을 원점 통과 포물선으로 근사하는 투영계수이며 intentional herding을 필요로 하지 않는다.

SCSAD에서는 양의 반축의 상수 `c_N`을 `aX+bX^3`로 투영한다.

\[
\begin{bmatrix}E[X^2]&E[X^4]\\E[X^4]&E[X^6]\end{bmatrix}
\begin{bmatrix}a\\b\end{bmatrix}
=c_N\begin{bmatrix}E[X]\\E[X^3]\end{bmatrix}
\]

사전 이론값은 다음과 같다.

\[
\gamma_1^*=\frac{3c_Nu}{2s_M},\qquad
\gamma_3^*=-\frac{c_Nu}{6s_M^3}<0
\]

이는 `sign(M)`의 불연속 계단을 저차 홀수 다항식으로 근사하면서 생기는 음의 cubic curvature다. 대칭성 때문에 SCSAD의 절편과 제곱항 pseudo-true 계수는 0이다.

Standard와 intercept-restored 모형은 상수 `E[CSAD|M]=c_N`를 절편이 흡수하므로 이상적 Gaussian null에서 target pseudo-true 계수가 0이다.

## 4. 폐형식 검증

코드는 다음을 독립적으로 계산한다.

1. Gaussian 절대모멘트의 폐형식
2. 모멘트 정규방정식의 수치해
3. 위에 적은 단순화된 폐형식

두 결과의 절대차는 `1e-10` 이하여야 한다. 이 검증이 실패하면 simulation 결과와 관계없이 수학적 구현은 실패로 판정한다.

## 5. 대규모 Monte Carlo 수렴 검증

- DGP: independent Gaussian
- 가중: equal
- `N`: 14, 50, 62
- `T`: 12,000
- 반복: 각 200회
- sigma: 0.025

No-intercept 제곱항과 SCSAD 세제곱항 각각에 대해 다음을 모두 요구한다.

1. replicate 평균의 이론값 대비 상대오차가 7.5% 이하
2. 이론값이 replicate 평균의 99% Monte Carlo 신뢰구간 안에 포함
3. 음의 계수 비율이 95% 이상
4. raw HAC false-positive rate와 기존 3모형 BH false-positive rate가 각각 90% 이상

Standard와 intercept-restored target은 참값 0으로 둔다. 각 `N` 셀에서 raw HAC false-positive rate가 7.5% 이하여야 한다.

## 6. 강건성 simulation

모든 DGP에는 자산 간 모방 규칙이 없다. Seed, 반복, 모수는 config에 고정한다.

1. `independent_gaussian`: 독립 정규 자산수익률
2. `common_factor`: 이질적 beta를 가진 Gaussian 공통요인
3. `stochastic_volatility_factor`: 공통 AR(1) 확률변동성
4. `student_t_factor`: 분산을 정규화한 자유도 5 fat-tail 요인
5. `skewed_factor`: 평균·분산을 정규화한 centered lognormal 충격
6. `jump_diffusion_factor`: 확산과 희소 공통·고유 jump
7. `time_varying_correlation`: 과거 상태만으로 움직이는 상관 AR(1)
8. `asymmetric_common_shock`: 음의 공통충격에서 변동성만 확대되는 leverage형 과정

표본 길이는 250, 1,000, 2,500, 자산 수는 14, 50, 62를 포함한다. 가중은 equal, 수익률과 독립인 전기 lognormal liquidity, 고정 집중가중을 포함한다. 강건성 반복은 DGP·scenario당 300회다.

## 7. 다중검정과 저장 통계량

- `raw_false_positive`: 음의 target 및 HAC `p<=0.05`
- `bh3_false_positive`: 같은 replicate의 standard, no-intercept, SCSAD 세 p-value에 BH-FDR
- `bh4_false_positive`: intercept-restored까지 포함한 네 p-value에 BH-FDR
- 각 DGP·scenario·model별 계수 평균, bias, 표준편차, 음수 비율, raw/BH FPR, Wilson 95% CI를 저장한다.
- exact theory가 존재하지 않는 DGP의 coefficient bias는 계산하지 않고 결측으로 명시한다.
- 대칭 DGP는 independent Gaussian, common factor, stochastic volatility, Student-t, time-varying correlation이다.

기계적 부호의 분포 강건성은 대칭 DGP × 모든 scenario 셀 중 no-intercept와 SCSAD 모두 음수 비율 80% 이상인 셀이 75% 이상일 때만 지지한다. 이는 Gaussian 정리의 보편성 증명이 아니라 제한된 simulation 강건성이다.

## 8. 진단과 해석 경계

각 simulation 셀에서 시장수익률 평균·표준편차·왜도·초과첨도, CSAD 평균·표준편차, `corr(CSAD, M)`, `corr(CSAD, |M|)`, 평균 HHI를 저장한다.

- Gaussian 정리는 독립·동분산·동일가중·정규성 아래의 population projection 결과다.
- 공통요인, 이분산성, fat-tail, 비대칭, jump, 시변 상관과 집중가중에서는 조건부 CSAD가 상수가 아닐 수 있다.
- 이 경우 standard 모형의 nominal FPR도 깨질 수 있으며, 이를 herding으로 재명명하지 않는다.
- simulation에서 음의 계수가 반복돼도 intentional imitation의 충분조건이 아니다.
- 결과를 본 뒤 threshold, DGP 모수, 제외 셀을 변경하지 않는다. 오류 수정은 별도 amendment와 새 artifact version으로 보존한다.

## 9. 사전 최종 판정

`mechanical_null_confirmed`는 다음이 모두 참일 때만 부여한다.

1. 폐형식과 모멘트 정규방정식 검증 3/3 통과
2. 대규모 Gaussian 수렴 셀에서 no-intercept 3/3 및 SCSAD 3/3 통과
3. Gaussian 셀의 standard 및 intercept-restored nominal raw FPR gate 6/6 통과
4. No-intercept와 SCSAD의 Gaussian raw/BH FPR gate 6/6 통과

분포 강건성 gate는 별도 판정하며 `mechanical_null_confirmed`를 사후로 구제하거나 무효화하지 않는다.

최종 보고서는 다음 네 질문에 답한다.

1. 기존 음의 계수가 실제 herding 증거인가?
2. 모형 구조가 만든 통계적 착시인가?
3. 어떤 CSAD 사양을 연구에 사용할 수 있는가?
4. 후속 실증연구가 지켜야 할 식별 기준은 무엇인가?

