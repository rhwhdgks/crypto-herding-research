# 뉴스 Sentiment 확장 리포트

## 뉴스 데이터
- scored headline 수: 1786
- 구간: 2025-07-11 07:00:00+00:00 ~ 2026-04-20 05:30:00+00:00
- positive 20.60%, negative 5.99%, neutral 73.40%

## 이벤트 결합
- sentiment feature가 붙은 event 수: 46993
- 15분 기준 sentiment 분포: {'neutral': 32744, 'positive': 13441, 'negative': 808}

## 허딩 feature 레이어
- 뉴스 제목 데이터는 `sentiment_mean_15m`, `news_count_15m`, `news_attention_15m`, `sentiment_strength_15m`, `sentiment_pressure_15m` 형태로 event feature에 직접 들어갑니다.
- dense news 기준: news_count_15m >= 27.00
- strong sentiment 기준: |sentiment_mean_15m| >= 0.2727
- feature group count:
  - herding_other: 24948건, 평균 news 6.06, 평균 pressure 0.1512
  - shock_other: 22045건, 평균 news 6.06, 평균 pressure 0.1531

## sentiment split event study
- bullish_herd: 최고 horizon 15m, 평균 -0.0032%, t -0.96, count 7140
- negative_shock: 최고 horizon 1d, 평균 0.5492%, t 3.01, count 358
- neutral_herd: 최고 horizon 1h, 평균 0.0025%, t 0.49, count 17357
- neutral_shock: 최고 horizon 1d, 평균 0.0830%, t 2.86, count 15387
- panic_herd: 최고 horizon 1d, 평균 0.4854%, t 3.56, count 450
- positive_shock: 최고 horizon 15m, 평균 0.0047%, t 1.01, count 6300

## feature group event study
- herding_other: 최고 horizon 15m, 평균 0.0005%, t 0.25, count 24947
- shock_other: 최고 horizon 6h, 평균 0.0093%, t 0.74, count 22039

## 출력물
- /home/jonghan/findalpha/herding/outputs/baseline/sentiment_extension/news_sentiment_scored.csv
- /home/jonghan/findalpha/herding/outputs/baseline/sentiment_extension/event_sentiment_features.csv
- /home/jonghan/findalpha/herding/outputs/baseline/sentiment_extension/sentiment_event_study_summary.csv
- /home/jonghan/findalpha/herding/outputs/baseline/sentiment_extension/sentiment_feature_event_study_summary.csv
- /home/jonghan/findalpha/herding/outputs/baseline/sentiment_extension/sentiment_feature_overview.csv
- /home/jonghan/findalpha/herding/outputs/baseline/sentiment_extension/sentiment_feature_thresholds.csv
- /home/jonghan/findalpha/herding/outputs/baseline/sentiment_extension/plots/sentiment_split_plots.png
- /home/jonghan/findalpha/herding/outputs/baseline/sentiment_extension/plots/sentiment_feature_group_plots.png
