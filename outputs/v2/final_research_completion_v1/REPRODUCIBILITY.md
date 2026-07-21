# 최종 연구 재현 안내

## 빠른 무결성 검증

기존 결과를 변경하지 않고 해시, 행 수, FDR, 판정식, protocol seal, 보고서 필수 섹션을 확인합니다.

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_final_research_completion.py
```

## 기계성 감사 재실행

새 복제 작업공간에서 입력 산출물을 먼저 준비한 뒤 실행합니다. 관찰된 결과를 덮어쓰지 않도록 runner는 기존 판정 파일이 있으면 중단합니다.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_zero_run_microstructure.py
PYTHONPATH=src .venv/bin/python scripts/verify_zero_run_microstructure.py
PYTHONPATH=src .venv/bin/python scripts/run_zero_run_mechanical_null_audit.py
PYTHONPATH=src .venv/bin/python scripts/build_final_research_completion.py
PYTHONPATH=src .venv/bin/python scripts/verify_final_research_completion.py
```

## 전체 테스트

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

## 재현 보장 범위

- Protocol과 config의 사전 봉인 SHA-256을 검사합니다.
- 원시·분석 frame의 key, row count, run-z 재구성 오차를 검사합니다.
- Exact conditional run PMF는 작은 binary sequence 전수열거 테스트와 비교합니다.
- 23개 그룹의 두 BH family, 5개 mechanism BH family, 최종 판정식을 독립 재계산합니다.
- 최종 evidence와 artifact manifest의 파일 크기·SHA-256을 전수 검사합니다.

이 패키지는 방향성 alpha, tracker 또는 자동매매를 재현 대상으로 선언하지 않습니다.
