# `data/` — Raw market data

이 디렉토리의 대용량 파일들은 git에서 제외되어 있습니다 (`.gitignore` 참조).

## 재현 방법

| 데이터 | 생성 명령어 | 크기 (대략) |
|---|---|---|
| Binance 1분봉 OHLCV (14 자산 × 2년) | `python scripts/run_pipeline.py --config configs/baseline/config.yaml` | ~10 GB |
| Binance tick aggTrades (7 자산 × 1년) | `python scripts/run_tick_lead_lag_matrix.py --config configs/tick/multi_asset_365d/lead_lag_matrix.yaml` | ~25 GB |
| Binance tick aggTrades (7 자산 × 5년) | `python scripts/run_tick_lead_lag_matrix.py --config configs/tick/multi_asset_5y/lead_lag_matrix.yaml` | ~100 GB |
| 뉴스 헤드라인 (Google News + GDELT) | `python scripts/collect_news_headlines.py --config configs/baseline/config.yaml` | ~1 MB |
| Reddit posts (CryptoCurrency 등) | `python scripts/collect_reddit_posts.py --config configs/baseline/config.yaml` | ~1 MB |

## 출처

- **OHLCV / aggTrades**: [Binance Public Data Archive](https://data.binance.vision/)
- **뉴스 헤드라인**: Google News RSS + [GDELT DOC 2.0 API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)
- **Reddit**: public JSON search API (r/CryptoCurrency, r/CryptoMarkets)

## 디렉토리 구조 (재현 후)

```
data/
├── *.parquet              # OHLCV 1m/1d/1w per symbol
├── tick_archive/
│   ├── BTCUSDT/aggTrades/
│   ├── ETHUSDT/aggTrades/
│   └── ...
├── news/news_headlines.csv
└── reddit/reddit_posts.csv
```
