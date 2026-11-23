# 진행 기록

커밋·푸시할 때마다 맨 아래에 한 줄 추가한다. (규칙: [레포 README](../README.md#협업-규칙-사람ai-에이전트-공통))

| 날짜 | 담당 | 가이드 단계 | 한 일 | 막힌 곳 / 다음 할 일 |
|---|---|---|---|---|
| 2026-11-19 | 🟩 모델 | - | 레포 구조, 협업 규칙, 담당별 가이드(`data/` `model/` `phone/` README), `shared/` 약속 세팅 | 다음: 각자 자기 폴더 README 1단계부터 시작 |
| 2026-11-19 | 🟩 모델 | - | README 7-1절(예상 성능·과적합 주의) 추가, 모델 가이드 3단계에 은닉 노드 항목 추가 | 다음: 모델 1~2단계 |
| 2026-11-23 | 🟦 데이터 | 1~2단계 | `data/data.py`(로더 `load_signals`·펼치기 `flatten`·사람 단위 검증 분할 `load_train_val`·정규화 `normalize`), `shared/norm_stats.npz` 업로드, 검증용 사람 번호 `shared/README.md`에 기록. `data/standing_gravity.py`: UCI 서 있을 때 total_acc 평균 x=+1.00g, y=−0.16g, z=−0.03g (폰 담당 방향 확인용) | 다음: 샘플 그래프 확인, 실험 2 예상표 확정, 모델 담당 `train.py` 기다림 |
| 2026-11-23 | 🟩 모델 | 1~3단계 | `model/layers.py`·`model.py`(교재 ch05 기반 2층 신경망), 기울기 확인(최대 상대오차 5.8e-8)·100개 과적합 확인 통과, `model/train.py`로 학습 설정 확정(lr 0.1, He, 은닉 100, 23 epoch, 배치 100 → `train.py` 기본값), 테스트 87.0% ± 0.5%p(시드 3개), `shared/weights.npz`(조건 ⑤) 업로드, 결과 정리 `model/REPORT.md` | 다음: 데이터 담당과 결과 파일 형식 확인 후 실험 2 실행(`python model/train.py --cond k --seed s --test --out ...`). 눕기 오분류 27개가 모두 테스트 대상자 10번 → 실험 2·5단계에서 확인. 5단계는 폰 데이터·PCA 함수 대기 |
