# 암호화폐 집단추종 지표 검증

여러 코인이 함께 오르내리는 현상을 투자자의 집단추종(herding)으로 해석할 수 있는지 검증했습니다. [Natashekara & Sampath (2026)](https://doi.org/10.1016/j.jbef.2026.101167)의 CoinMarketCap 62종목 CSAD 분석을 재현한 뒤, 다른 기간·거래소·종목 구성에서 반복하고, 집단추종이 없는 가상시장에서도 같은 신호가 나오는지 실험했습니다.

## 결론

- 선행논문의 집단추종 계수를 거의 같은 값으로 재현
- 집단추종이 없도록 만든 가상시장에서도 같은 종류의 음의 계수가 97~100% 빈도로 나타남
- 따라서 논문의 계산은 재현되지만, 그 계수만으로 투자자가 서로 따라 거래했다고 볼 수 없음
- 1분봉과 개별 체결 자료에서도 단기 수익률을 안정적으로 예측하는 신호는 찾지 못함

## 검증 결과

| 검증 | 결과 | 해석 |
|---|---|---|
| Binance 주요 14종목, 2년 1분봉 | 전통적 CSAD 집단추종 신호 없음 | 최근 Binance 표본에서는 뚜렷한 집단추종 없음 |
| 선행논문 CoinMarketCap 62종목 | 논문 계수를 매우 가깝게 재현 | 논문의 계산 자체는 재현 가능 |
| 논문 표본 이후 기간 | 같은 종류의 음의 계수 재등장 | 특정 기간에만 생긴 값은 아님 |
| Binance·OKX, 시점별 종목 구성 | 거래소·구성 방식을 바꾸면 결과가 약해짐 | 종목 선택 방식에 민감 |
| 집단추종 없는 가상시장 | 기존 보정 모형이 거의 항상 음의 계수 생성 | 음의 계수만으로 집단추종을 판정할 수 없음 |
| 개별 체결의 연속성 | 단기 가격 방향 예측 실패 | 매매 신호로 확인되지 않음 |
| 동일 가격 체결 반복(zero-run) | 현재 거래 상태는 설명하나 미래 변동은 예측하지 못함 | 상태 설명 변수이지 예측 변수는 아님 |

## 음의 계수가 생기는 원인

CSAD는 같은 시점에 각 코인 수익률이 시장 평균에서 떨어진 거리의 평균입니다. 시장이 크게 움직일 때 이 거리가 예상보다 덜 벌어지면(제곱항 계수가 음수이면) 집단추종의 근거로 봅니다.

선행연구의 일부 회귀식은 절편을 제거하거나 CSAD의 부호를 바꿔 추정합니다. CSAD는 집단추종이 없어도 0보다 큰 기본 분산 수준을 갖는데, 절편이 없으면 회귀선이 원점에서 출발해 이 수준을 따라가려고 올라갔다가 휘어지면서 음의 제곱항이 생깁니다. 이 과정을 폐형식 유도와 Monte Carlo 시뮬레이션으로 확인했습니다.

## 진행 순서

1. 기초 분석: Binance 14종목 2년 1분봉으로 시장수익률과 CSAD 계산
2. 논문 재현: CoinMarketCap 자료와 논문의 62종목 목록으로 주요 계수 재현
3. 외부 검증: 새 기간, Binance, OKX, 시점별 Top-50 구성에서 반복
4. 모형 점검: 집단추종이 없는 가상시장 여러 개로 거짓양성률 측정
5. 수학적 검증: 음의 계수가 생기는 원인을 폐형식과 시뮬레이션으로 확인
6. 미시구조: 개별 체결 자료로 단기 예측력 검정

판정 기준은 결과를 보기 전에 [research_protocols/](research_protocols)에 고정했고, 전체 보고서는 [docs/EXTERNAL_RESEARCH_REPORT_KO.md](docs/EXTERNAL_RESEARCH_REPORT_KO.md)에 있습니다.

## 실행

Python 3.11 이상

```bash
git clone https://github.com/rhwhdgks/crypto-herding-research.git
cd crypto-herding-research
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

테스트와 공개 결과 검증

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/verify_csad_mechanical_derivation_v1_1_amended.py
```

Binance 2년 1분봉 기본 분석

```bash
PYTHONPATH=src python scripts/run_pipeline.py --config configs/baseline/config.yaml
```

전체 재실행에는 Git에 포함하지 않은 원자료가 필요합니다. 데이터 출처와 생성 방법은 [data/README.md](data/README.md)에 있습니다.

## 구조

| 경로 | 내용 |
|---|---|
| `src/` | 데이터 처리, CSAD, 시뮬레이션, 체결 분석 |
| `scripts/` | 연구 실행기와 결과 검증 스크립트 |
| `configs/` | 기간·종목·통계 기준 설정 |
| `research_protocols/` | 결과 관찰 전에 작성한 연구 계획 |
| `outputs/v2/` | 핵심 보고서와 판정 결과 |
| `tests/` | 단위·통합·재현성 테스트 |
| `docs/` | 연구 보고서 |

대용량 OHLCV·체결 원자료, 데이터베이스, API 자격증명은 저장소에 포함하지 않았습니다.

## 한계

- CSAD는 수익률이 비슷하게 움직였는지를 측정할 뿐 투자자의 의도는 관찰하지 못함
- 고정 종목 목록에는 상장폐지·생존편향이 남을 수 있음
- 개별 체결 자료에 호가, 주문장 깊이, 주문 취소 정보가 없음

## License

코드와 문서는 [MIT License](LICENSE)를 따릅니다. 외부 데이터에는 각 제공자의 이용조건이 적용됩니다.
