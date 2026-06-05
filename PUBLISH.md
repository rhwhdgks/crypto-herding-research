# GitHub Publishing Guide

이 파일은 본인이 GitHub에 올릴 때 실행할 명령어 모음입니다. 공개 후 삭제 또는 `.gitignore`에 추가하세요.

## 0. 사전 점검 (보안)

```bash
# DB password가 더 이상 평문으로 남아있지 않은지 확인
grep -rn "208307\|password:" configs/ src/ scripts/ | grep -v "\${" | grep -v "환경변수"
# (결과가 비어있어야 안전)

# 기타 자격증명 흔적 점검
grep -rIn -E "(secret|api[_-]?key|bearer|token)" configs/ src/ scripts/ | head
```

## 1. Git init + first commit

```bash
cd /home/jonghan/findalpha/herding

git init -b main
git add .gitignore LICENSE README.md requirements.txt .env.example
git add agents.md claude.md       # 빼고 싶다면 .gitignore에 이미 들어있음
git add configs/ src/ scripts/ experiments/ data/README.md
git add outputs/*.md outputs/*.pdf outputs/README.md
git add outputs/tick/multi_asset_5y/lead_lag_matrix/tick_lead_lag_matrix_report.md
git add outputs/tick/multi_asset_5y/lead_lag_matrix/lead_lag_matrix_summary.csv
git add outputs/tick/multi_asset_5y/lead_lag_matrix/plots/
git add experiments/lead_lag_robustness/outputs/*.md
git add experiments/lead_lag_robustness/outputs/*.csv
git add experiments/lead_lag_robustness/outputs/*.png
git add experiments/cointegration_lead_lag/outputs/*.csv
git add references/  # (논문 PDF 저작권 주의 — 제외 권장. .gitignore 이미 포함)

git status                         # 의도하지 않은 파일이 staging되지 않았는지 확인
git diff --cached --stat | tail -20

git config user.name "Jonghan Ko"
git config user.email "your_email@example.com"

git commit -m "Initial release: crypto herding research pipeline + lead-lag matrix"
```

## 2. GitHub remote 추가

```bash
# GitHub에서 빈 repo 생성 후 (Add README, gitignore, license 모두 체크해제)
git remote add origin git@github.com:rhwhdgks/cryptomarket_herding.git
git push -u origin main
```

## 3. 권장 후속 작업

- [ ] GitHub repo 설정에서 **Topics** 추가: `quantitative-finance`, `cryptocurrency`, `herding`, `event-study`, `microstructure`, `python`
- [ ] **About** 섹션에 한 줄 설명 입력
- [ ] `outputs/presentation_short_2026-05-09.pdf`를 GitHub Release로 첨부하면 채용 담당자가 바로 다운로드 가능
- [ ] 핵심 그림 (`experiments/lead_lag_robustness/outputs/tick_level_lead_time_5y.png`)을 README 상단에 임베드
- [ ] GitHub Actions로 lint 또는 import test 추가 (선택)

## 4. 제외 권장 파일 (사이즈 / 민감성)

| 경로 | 사이즈 | 처리 |
|---|---|---|
| `.venv/` | ~수백 MB | `.gitignore` ✓ |
| `data/` | ~106 GB | `.gitignore` ✓ |
| `outputs/legacy/` | ~6.4 GB | `.gitignore` ✓ |
| `outputs/baseline/*.csv` | ~수백 MB | `.gitignore` ✓ |
| `outputs/tick/**/intermediate/` | ~수 GB | `.gitignore` ✓ |
| `agents.md`, `claude.md`, `.claude/` | dev internal | `.gitignore` (선택) |
| `logs/`, `*.log` | 크지 않지만 노이즈 | `.gitignore` ✓ |
| `references/*.pdf` | 저작권 이슈 | `.gitignore` ✓ |

## 5. 최종 확인

```bash
# repo 크기 확인 (수십 MB 이하여야 정상)
git count-objects -vH

# 무엇이 트래킹되고 있는지 한번에 검토
git ls-files | head -50
git ls-files | wc -l
```

## 6. 문제 발생 시 (실수로 큰 파일 commit)

```bash
# 마지막 commit에서 특정 파일만 제거
git rm --cached path/to/large_file
git commit --amend

# 이미 push했다면 git-filter-repo 또는 BFG Repo-Cleaner 권장
# https://rtyley.github.io/bfg-repo-cleaner/
```
