# CSAD audit v1 렌더링 기록 정정

- 최초 `audit_amendment_2026-07-20.md`에는 Noto Sans CJK KR로 다시 렌더링했다고 적었으나, 격리된 Matplotlib 캐시에서는 해당 family를 찾지 못해 DejaVu Sans fallback을 사용했습니다.
- 보완 그림의 제목과 축은 영문이라 글리프 손실은 없습니다.
- 계수, p/q-value, simulation, 판정 기준과 보고서 수치는 전혀 변경하지 않았습니다.
- 향후 재실행 코드는 특정 폰트를 강제하지 않고 영문 그림 제목을 사용합니다.
