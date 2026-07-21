# Manifest amendment v1.1

원 보충 실행은 갱신 전 master report를 input으로 읽은 뒤 같은 경로에 amendment를 발행했습니다.
그 결과 원 `input_manifest.csv`의 크기와 SHA-256은 정확하지만, 검증 시점의 현재 경로 내용은 갱신본이 되어 경로 검증이 실패했습니다.

- 원 manifest와 실패 verifier는 보존합니다.
- 기록된 input과 byte-for-byte 일치하는 `research_master_report_2026-07-21_pre_supplement.md` 백업이 존재합니다.
- `input_manifest_v1_1.csv`는 그 보존 백업 경로만 교정합니다.
- simulation, summary, gate, decision, report 내용은 변경하지 않았습니다.
- 보존 input SHA-256: `45b1d9e8b0f0e1a64507ad2a46e93bac30aaee2ed73be627fa66ee5c9b7a1787`
