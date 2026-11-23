# 진행 기록

커밋·푸시할 때마다 맨 아래에 한 줄 추가한다. (규칙: [레포 README](../README.md#협업-규칙-사람ai-에이전트-공통))

| 날짜 | 담당 | 가이드 단계 | 한 일 | 막힌 곳 / 다음 할 일 |
|---|---|---|---|---|
| 2026-11-19 | 🟩 모델 | - | 레포 구조, 협업 규칙, 담당별 가이드(`data/` `model/` `phone/` README), `shared/` 약속 세팅 | 다음: 각자 자기 폴더 README 1단계부터 시작 |
| 2026-11-19 | 🟩 모델 | - | README 7-1절(예상 성능·과적합 주의) 추가, 모델 가이드 3단계에 은닉 노드 항목 추가 | 다음: 모델 1~2단계 |
| 2026-11-23 | 🟦 데이터 | 1~2단계 | `data/data.py`(로더 `load_signals`·펼치기 `flatten`·사람 단위 검증 분할 `load_train_val`·정규화 `normalize`), `shared/norm_stats.npz` 업로드, 검증용 사람 번호 `shared/README.md`에 기록. `data/standing_gravity.py`: UCI 서 있을 때 total_acc 평균 x=+1.00g, y=−0.16g, z=−0.03g (폰 담당 방향 확인용) | 다음: 샘플 그래프 확인, 실험 2 예상표 확정, 모델 담당 `train.py` 기다림 |
