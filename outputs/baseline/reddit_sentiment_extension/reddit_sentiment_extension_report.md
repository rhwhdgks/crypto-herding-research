# Reddit Sentiment 확장 리포트

## Reddit 데이터
- scored Reddit 제목 수: 1946
- 구간: 2025-04-16 01:40:29+00:00 ~ 2026-04-20 05:10:47+00:00
- positive 8.12%, negative 6.47%, neutral 85.41%

## 이벤트 결합
- sentiment feature가 붙은 event 수: 46993
- 15분 기준 sentiment 분포: {'neutral': 33905, 'positive': 10631, 'negative': 2457}

## 허딩 feature 레이어
- Reddit 제목 데이터는 `sentiment_mean_15m`, `news_count_15m`, `news_attention_15m`, `sentiment_strength_15m`, `sentiment_pressure_15m` 형태로 event feature에 직접 들어갑니다.
- dense news 기준: news_count_15m >= 23.00
- strong sentiment 기준: |sentiment_mean_15m| >= 0.0577
- feature group count:
  - herding_other: 22468건, 평균 news 19.01, 평균 pressure 0.0504
  - shock_other: 19910건, 평균 news 19.14, 평균 pressure 0.0526
  - bullish_herd_strong: 2071건, 평균 news 38.03, 평균 pressure 0.3751
  - positive_shock_strong: 1788건, 평균 news 37.84, 평균 pressure 0.3731
  - panic_herd_strong: 409건, 평균 news 39.41, 평균 pressure -0.3035
  - negative_shock_strong: 347건, 평균 news 39.02, 평균 pressure -0.3070

## sentiment split event study
- bullish_herd: 최고 horizon 2h, 평균 0.0067%, t 0.62, count 5613
- negative_shock: 최고 horizon 2h, 평균 0.0238%, t 0.75, count 1140
- neutral_herd: 최고 horizon 5m, 평균 0.0004%, t 0.34, count 18018
- neutral_shock: 최고 horizon 5m, 평균 0.0002%, t 0.08, count 15887
- panic_herd: 최고 horizon 6h, 평균 0.0064%, t 0.13, count 1317
- positive_shock: 최고 horizon 6h, 평균 0.0423%, t 1.97, count 5018

## feature group event study
- bullish_herd_strong: 최고 horizon 5m, 평균 -0.0003%, t -0.11, count 2071
- herding_other: 최고 horizon 1h, 평균 0.0014%, t 0.32, count 22466
- negative_shock_strong: 최고 horizon 6h, 평균 0.1247%, t 1.05, count 347
- panic_herd_strong: 최고 horizon 15m, 평균 -0.0064%, t -0.38, count 409
- positive_shock_strong: 최고 horizon 2h, 평균 0.0250%, t 1.13, count 1788
- shock_other: 최고 horizon 6h, 평균 0.0072%, t 0.54, count 19904

## 출력물
- /home/jonghan/findalpha/herding/outputs/baseline/reddit_sentiment_extension/reddit_sentiment_scored.csv
- /home/jonghan/findalpha/herding/outputs/baseline/reddit_sentiment_extension/event_reddit_sentiment_features.csv
- /home/jonghan/findalpha/herding/outputs/baseline/reddit_sentiment_extension/reddit_sentiment_event_study_summary.csv
- /home/jonghan/findalpha/herding/outputs/baseline/reddit_sentiment_extension/reddit_sentiment_feature_event_study_summary.csv
- /home/jonghan/findalpha/herding/outputs/baseline/reddit_sentiment_extension/reddit_sentiment_feature_overview.csv
- /home/jonghan/findalpha/herding/outputs/baseline/reddit_sentiment_extension/reddit_sentiment_feature_thresholds.csv
- /home/jonghan/findalpha/herding/outputs/baseline/reddit_sentiment_extension/plots/reddit_sentiment_split_plots.png
- /home/jonghan/findalpha/herding/outputs/baseline/reddit_sentiment_extension/plots/reddit_sentiment_feature_group_plots.png
