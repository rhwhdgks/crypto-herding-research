# CSAD audit v1 비수치 보완 기록

- 작성일: 2026-07-20
- 동결된 protocol, config, empirical 계수, Monte Carlo 반복과 판정 기준은 변경하지 않았습니다.
- 기존 전체 moderator 동시 meta-regression은 provider와 universe가 교락되어 3/3 모형에서 rank deficient였습니다.
- 해당 계수를 인과적으로 해석하지 않도록 moderator별 HC3 단변량 회귀와 model별 BH-FDR 표를 추가했습니다.
- 기존 corrected 기준(no-intercept와 SCSAD daily·weekly 4셀)을 별도 열로 명시했지만, 사전등록된 구조적 강건성 최종 판정은 변경하지 않았습니다.
- 시스템에 설치된 Noto Sans CJK KR 폰트로 두 그림만 새 파일에 다시 렌더링했습니다. 기존 그림은 보존했습니다.
- 보완 보고서: `csad_specification_audit_report_v1_1.md`
