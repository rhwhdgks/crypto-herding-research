# Crypto Herding Research

암호화폐 시장의 CSAD 집단추종 신호를 재현하고, 그 음의 계수가 행동적 herding인지 모형 구조가 만든 결과인지 반증 중심으로 검증한 재현 가능한 연구 저장소입니다.

> **핵심 결론:** 선행논문의 no-intercept·SCSAD 계수는 정밀하게 재현됐지만, 두 모형은 herding이 없는 합성자료에서도 거의 항상 음의 유의한 계수를 만듭니다. 현재 표본에서 거래 가능한 단기 alpha는 확인되지 않았습니다.

상세한 연구 과정과 수치는 [외부용 통합 연구 보고서](docs/EXTERNAL_RESEARCH_REPORT_KO.md)에서 확인할 수 있습니다.

## 연구 질문과 결과

| 질문 | 결과 | 현재 해석 |
|---|---|---|
| Binance 14자산 2년 1분봉에서 classical CSAD herding이 나타나는가? | Standard CSAD의 `beta2 > 0` | 지지하지 않음 |
| 선행논문의 CMC fixed-62 결과를 재현할 수 있는가? | Daily no-intercept `-1.837`, SCSAD `-2.902` | 수치 재현 성공 |
| corrected 모형이 기간·거래소·point-in-time universe를 넘어 유지되는가? | CMC holdout 4/4, Binance 3/4, OKX 3/4·1/4, Binance PIT 2/4·0/4 | 보편적 강건성 없음 |
| 음의 계수가 intentional herding을 식별하는가? | 비허딩 null에서 no-intercept 99~100%, SCSAD 97~100% false positive | 식별하지 못함 |
| 기계적 음의 계수를 수학적으로 설명할 수 있는가? | Gaussian 식 3/3, 원 v1 수렴 5/6, 독립 v1.1 보충 12/12 | 수학식·유한표본 수렴 지지, 원 실패는 보존 |
| Tick run-clustering이 단기 방향을 예측하는가? | 사전등록 OOS 0/9 | 방향성 alpha 없음 |
| Zero-run이 시장상태와 미래 변동을 설명하는가? | 동시점 5/5, 미래 family 0/2 | 상태 기술 가능, 미래 예측력 없음 |

논문 수치를 재현했다는 것과 투자자의 의도적 모방을 식별했다는 것은 서로 다른 주장입니다. 이 저장소는 두 주장을 분리합니다.

## CSAD

횡단면 절대편차(CSAD)는 시점 `t`에서 각 자산 수익률이 시장수익률에서 얼마나 떨어져 있는지 측정합니다.

```text
CSAD_t = mean_i |R_i,t - R_m,t|

CSAD_t = alpha + beta1 * |R_m,t| + beta2 * R_m,t^2 + error_t
```

Standard 모형에서 `beta2 < 0`은 시장 움직임이 커질수록 횡단면 분산의 증가세가 둔화되는 현상을 뜻합니다. 그러나 이것만으로 투자자의 의도적 집단추종이나 미래수익률 alpha를 입증하지는 않습니다.

## 저장소 구조

```text
.
├── configs/               # baseline, replication, external-validation 설정
├── data/                  # 로컬 원자료 안내; 실제 데이터는 Git 제외
├── docs/                  # 외부 독자를 위한 통합 문서
├── experiments/           # 과거 탐색 연구와 무효화 기록
├── outputs/v2/            # 수정된 연구 결과; 공개 파일은 명시적 opt-in
├── research_protocols/    # 결과 확인 전 고정한 연구 프로토콜
├── scripts/               # 실행기, 보고서 생성기, 읽기 전용 verifier
├── src/                   # 분석·simulation·reporting 모듈
└── tests/                 # 단위·통합·재현성 테스트
```

## 설치

Python 3.11 이상을 권장합니다.

```bash
git clone https://github.com/rhwhdgks/crypto-herding-research.git
cd crypto-herding-research
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## 최소 재현

먼저 코드와 공개 결과의 무결성을 확인합니다.

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/verify_csad_mechanical_derivation_v1_1_amended.py
```

`verify_final_research_completion.py`는 로컬의 전체 tick intermediate와 대형 입력까지 해시로 검사하는 strict verifier입니다. 공개 저장소에 제외된 대형 입력을 별도로 복원한 환경에서 실행하세요.

원자료가 준비된 환경에서는 개별 연구를 다시 실행할 수 있습니다.

```bash
# Binance 14자산, 정확한 2년 1분봉 baseline
PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/baseline/config.yaml

# CSAD specification audit
PYTHONPATH=src python scripts/run_csad_specification_audit.py \
  --config configs/research/csad_specification_audit_v1.yaml

# 기계적 음의 계수 simulation
PYTHONPATH=src python scripts/run_csad_mechanical_derivation.py \
  --config configs/research/csad_mechanical_derivation_v1.yaml
```

전체 simulation과 tick 연구는 시간이 오래 걸리고 로컬 raw archive를 요구합니다. 각 프로토콜과 config의 기간·seed·판정 기준을 변경하면 원 사전등록 결과의 재현으로 간주할 수 없습니다.

## 핵심 문서

- [외부용 통합 연구 보고서](docs/EXTERNAL_RESEARCH_REPORT_KO.md)
- [최종 논문형 원고](outputs/v2/final_research_completion_v1/final_research_manuscript.md)
- [최종 재현 안내](outputs/v2/final_research_completion_v1/REPRODUCIBILITY.md)
- [CSAD specification audit](outputs/v2/csad_specification_audit_v1/csad_specification_audit_report_v1_1.md)
- [기계적 음의 계수 보고서 v1](outputs/v2/csad_mechanical_derivation_v1/csad_mechanical_derivation_report.md)
- [독립 수렴 보충 v1.1](outputs/v2/csad_mechanical_derivation_v1/supplement_v1_1/csad_mechanical_convergence_supplement_report.md)
- [Zero-run 미시구조 보고서](outputs/v2/zero_run_microstructure_v1/zero_run_microstructure_report.md)

## 데이터와 결과물 정책

Raw OHLCV, aggTrades, CoinMarketCap·OKX 응답, 뉴스·Reddit archive, DB dump는 크기·라이선스·보안 문제로 Git에 포함하지 않습니다. [data/README.md](data/README.md)에 출처와 생성 방법을 정리했습니다.

`outputs/`도 기본적으로 Git에서 제외됩니다. 공개 저장소에는 결론을 확인하는 데 필요한 보고서, 판정표, manifest, 그림만 파일 단위로 검토해 포함합니다. 대형 parquet와 intermediate는 로컬에 보존하며, immutable 결과의 경로나 해시는 바꾸지 않습니다.

## 한계

- CSAD는 수익률 동조를 측정할 뿐 투자자의 의도나 정보전파 경로를 직접 관찰하지 않습니다.
- Fixed universe에는 survivorship·listing bias가, 시가총액 가중에는 동시성 문제가 남을 수 있습니다.
- Standard CSAD도 공통요인, 시변변동성, fat tail 아래에서 null 크기 교정이 필요합니다.
- Tick 결과는 aggTrades 기반이며 bid-ask spread, depth, 주문 취소를 직접 보지 못합니다.
- News·Reddit sentiment는 검증된 point-in-time archive가 부족해 confirmatory 결론에 사용하지 않았습니다.

이 프로젝트는 자동매매 봇, 투자 권유, 수익 보장 프로젝트가 아닙니다. 사전 기준을 통과한 미래수익률 alpha가 없으므로 tracker와 paper-sim은 활성 연구 결론에서 제외합니다.

## License

코드와 저장소 문서는 [MIT License](LICENSE)로 배포됩니다. 외부 데이터와 참고 논문에는 각 제공자의 별도 이용조건이 적용됩니다.
