# Research outputs

`outputs/`에는 실행 결과가 저장됩니다. 공개 저장소는 결과 전체를 복제하는 데이터 저장소가 아니며, 검토한 보고서·판정표·manifest·그림만 명시적으로 포함합니다.

## 현재 구조

- `baseline/`: Binance 14자산, 정확한 2년 1분봉 baseline
- `paper_like/`: 초기 선행논문 비교 결과
- `tick/`: 초기 tick 탐색과 tracker 기록
- `legacy/`: 교정 전에 생성된 탐색 결과
- `v2/`: 현재 해석에 사용하는 corrected·preregistered 연구

## 외부 공개 시 우선 확인할 v2 결과

- `v2/final_research_completion_v1/`
- `v2/csad_specification_audit_v1/`
- `v2/csad_mechanical_derivation_v1/`
- `v2/zero_run_microstructure_v1/`
- `v2/cmc_fixed_62/`
- `v2/binance_14/`
- `v2/okx_14/`
- `v2/binance_archive/`

상세 안내는 [외부용 통합 연구 보고서](../docs/EXTERNAL_RESEARCH_REPORT_KO.md)를 참고하세요.

## 보존 원칙

- 기존 `v2` 결과, 판정, manifest는 덮어쓰거나 재선택하지 않습니다.
- Manifest가 참조하는 파일의 경로와 이름을 바꾸지 않습니다.
- `intermediate/`, 대형 parquet, raw simulation row는 로컬에 보존하고 Git에는 기본 포함하지 않습니다.
- 교정 전 결과는 삭제하지 않고 `legacy/` 또는 무효화 기록과 함께 보존합니다.
- 새 결과는 별도 versioned directory에 생성합니다.
