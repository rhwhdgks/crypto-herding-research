# Public Release Checklist

이 문서는 새 공개 커밋을 만들기 전 확인할 최소 절차입니다. 원자료와 전체 산출물은 로컬에 두고, 검토한 코드·문서·경량 결과만 명시적으로 추가합니다.

## 1. 보안 점검

```bash
git status --short
git diff --check

# 실제 값이 들어간 자격증명과 개인 절대경로를 점검합니다.
rg -n --hidden -g '!.git/**' -g '!.venv/**' -g '!data/**' -g '!outputs/**' \
  '(PASSWORD|API_KEY|API_SECRET|ACCESS_TOKEN).*[=:]|/home/[^/]+' .
```

`.env`, DB 접속정보, API key, 개인 절대경로는 커밋하지 않습니다. 예제 설정은 환경변수 또는 placeholder만 사용합니다.

## 2. 공개 대상

- 프로젝트 문서: `README.md`, `docs/`, `research_protocols/`
- 재현 코드: `src/`, `scripts/`, `tests/`
- 설정 예시: `configs/`, `.env.example`
- 경량 핵심 결과: 검토한 보고서, 판정 CSV, manifest, 그림

다음은 기본 제외합니다.

- `data/`의 raw·normalized market data
- `.env`, credentials, DB dump
- `outputs/`의 intermediate·대형 parquet·전체 로컬 결과
- `dist/`, cache, staging, virtual environment
- 저작권이 있는 `references/` 원문

## 3. 검증

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python scripts/verify_csad_specification_audit.py
PYTHONPATH=src .venv/bin/python scripts/verify_csad_mechanical_derivation_v1_1_amended.py
PYTHONPATH=src .venv/bin/python scripts/verify_final_research_completion.py
git diff --check
```

## 4. Staging 감사

```bash
git diff --cached --name-status
git diff --cached --stat

# GitHub 일반 파일 제한보다 큰 신규 blob이 없는지 확인합니다.
git diff --cached --name-only -z | xargs -0 -r stat -c '%s %n' | sort -nr | head

# staged 파일에 민감정보가 없는지 재검사합니다.
git diff --cached | rg '(PASSWORD|API_KEY|API_SECRET|ACCESS_TOKEN).*[=:]|/home/[^/]+'
```

`outputs/`는 `.gitignore`에서 opt-in입니다. 필요한 경량 결과만 파일 단위로 검토한 뒤 `git add -f <path>`를 사용합니다.

## 5. Commit과 Push

```bash
git commit -m "docs: prepare crypto herding research for public release"
git show --stat --oneline HEAD
```

Push는 커밋과 별도 단계입니다. 최종 diff, 원격 저장소, 브랜치를 다시 확인한 뒤 일반 `git push`만 사용하며 force push와 history rewrite는 하지 않습니다.

## 6. 기존 이력 주의

현재 저장소의 과거 이력에는 오래된 대형 intermediate blob이 존재할 수 있습니다. 이번 공개 정리에서는 새 대형 파일을 추가하지 않으며, 기존 이력을 정리해야 한다면 별도의 백업·합의·마이그레이션 계획으로 수행합니다. 이 작업에서 `commit --amend`, force push, history rewrite를 사용하지 않습니다.
