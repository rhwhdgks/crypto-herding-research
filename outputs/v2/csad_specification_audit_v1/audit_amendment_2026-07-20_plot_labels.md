# CSAD audit v1 그림 라벨 보완

- 기존 `plots/empirical_vs_null.png`은 shared y-axis 처리 때문에 행 라벨이 표시되지 않았고, null 구간을 empirical 점 중심 error bar로 표현해 구간 밖 관측의 시각적 해석이 불명확했습니다.
- 수치와 판정은 변경하지 않고 null 최소·최대 95% 범위를 수평선으로, empirical 표준화 계수를 별도 다이아몬드로 표시한 `plots/empirical_vs_null_labels_fixed.png`를 추가했습니다.
- False-positive heatmap의 높은 값은 흰색 글씨로 바꾼 `plots/false_positive_rates_contrast_fixed.png`를 추가했습니다.
- 이후 문서와 발표에서는 위 두 교정 그림을 우선합니다. 기존 그림은 감사 추적을 위해 보존합니다.
