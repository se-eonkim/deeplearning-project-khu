# 🟦 데이터·실험 담당: 데이터 파이프라인 + 핵심 실험 + 결과 정리

**역할:** 모든 실험이 같은 데이터 위에서 돌게 만들고, 핵심 질문인 "어떤 센서가 어떤 행동을 가르는가"(실험 2)에 답한다. 마지막에 팀 결과를 정리한다.

**원칙:** 이 파트의 실수는 에러 없이 점수만 이상하게 만든다. 코드가 돌아가는 것과 맞는 것은 다르다.

> 시작 전에 [레포 README](../README.md)의 협업 규칙과 [`shared/README.md`](../shared/README.md)의 약속(채널 순서, 정답 형식)을 먼저 읽는다.

---

## 1단계. 데이터 파이프라인

**자료:** [UCI HAR 페이지](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) · [zip 다운로드](https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip) · [README 원문](https://huggingface.co/datasets/udayl/UCI_HAR/blob/main/README.txt) (처음에 끝까지 읽기) · [로딩 참고: Chevalier `load_X`](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition)

| 파일 | 내용 |
|---|---|
| `Inertial Signals/{total_acc, body_acc, body_gyro}_{x,y,z}_train.txt` | 원신호 9개, 각 `(7352, 128)` |
| `y_train.txt` / `subject_train.txt` / `activity_labels.txt` | 행동 1~6 / 사람 번호 / 번호↔이름 |
| ~~`X_train.txt`~~ | 가공 특징 561개, 안 씀 |

| 할 일 | 신경 쓸 것 |
|---|---|
| **로더** `load_signals(split, channels)` → `(N,128,C)` ([`np.loadtxt`](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html), [`np.stack`](https://numpy.org/doc/stable/reference/generated/numpy.stack.html)), 정답은 `y-1` | 채널은 이름으로 고르고 **[`shared/README.md`](../shared/README.md)의 순서**를 따른다. 펼치기(`reshape(N,-1)`)는 **이 함수 하나로만** (폰 담당도 같은 함수 사용) |
| **확인** shape, NaN, 행동별 샘플 수, 샘플 하나 그래프 | 걷기는 출렁이고 눕기는 평평한지 눈으로 확인 |
| **검증 분할** 학습 21명 중 5명을 통째로 | **무작위 샘플 분할 금지** (구간이 50% 겹쳐서 점수가 부풂). 뺀 사람 번호를 `shared/README.md`에 기록하고 고정. 6행동이 다 있는지 확인 |
| **정규화** 학습 데이터로 **채널별** 평균·표준편차 → 전부 변환 → `shared/norm_stats.npz` (mean, std, channels) | **학습 데이터로만** 통계를 구한다. **샘플별 정규화 금지** (중력·기울기 정보가 지워져 눕기·서기를 못 가름). 채널 일부만 쓸 때는 전체 통계에서 잘라 쓰기 |

- 테스트 데이터는 **마지막 평가에만** 쓴다. 설정 고르기는 검증 데이터로 한다.
- UCI 원본은 레포에 올리지 않는다 (`.gitignore` 처리됨).

## 2단계. 넘기기 (가장 먼저)
- 로더, 펼치기 함수, 검증 분할, `norm_stats.npz`를 **`shared/`와 `data/`에 올린다** → 모델 담당이 UCI 학습을 시작할 수 있다.

> ⚠️ **나를 기다리는 사람:** 모델 담당은 이게 와야 UCI 학습을 시작한다. 폰 담당은 다음 단계(정규화·펼치기)에서 필요하다. 채널 순서를 바꾸면 둘 다 다시 작업해야 한다.

## 3단계. 실험 2: 센서 비교 ★ 핵심

**설계:** 결과를 보기 전에 [레포 README 6절](../README.md#6-실험)의 예상표를 팀과 확정한다. 배경 자료는 [Shoaib 2014](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4118351/)(센서 단독·조합 비교)와 [Chevalier 결과](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition)(앉기↔서기, 걷기 3종 사이에서 혼동).

| ① gyro | ② body_acc | ③ total_acc | ④ body_acc+gyro | ⑤ total_acc+gyro |
|---|---|---|---|---|

| 할 일 | 신경 쓸 것 |
|---|---|
| **실행** 5조건 × 시드 3개 = 15회 (모델 담당의 `train.py`에 채널·시드만 바꿔서) | 바꾸는 건 **채널뿐**. 시드는 조건 간에 **같은 값**(0,1,2). 실행마다 설정·곡선·**테스트 예측값**을 저장 (`data/results/E2_c5_s1.npz`). 곡선이 이상하면 모델 담당에게 먼저 공유 |
| **결과표** 정확도 평균±표준편차, 행동별 정확도 | 조건 간 차이가 시드 간 편차보다 작으면 "차이 없음" |
| **혼동행렬 5장** 행=정답, 열=예측 ([`np.add.at`](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html), [`plt.imshow`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)) | **행별 비율**로 그린다 (샘플 수 차이 보정). 행동 순서는 5장 모두 동일 |
| **예상 vs 결과** 조건별 맞음/틀림/이유 | 숫자보다 **"어떤 혼동이 생기고, 어떤 센서를 넣으면 사라지나"**를 본다 |

> ⚠️ **내가 기다리는 것:** 모델 담당의 `train.py`와 확정된 학습 설정.

## 4단계. 실험 3: 입력 공간 vs 은닉 공간
- 조건 ⑤ 모델로 테스트 데이터의 `X`와 `H = ReLU(XW₁+b₁)`를 각각 PCA 2D로 그려 나란히 둔다 (행동별 색). 여유가 있으면 조건 ①도 그린다.
- PCA 직접 구현: 평균 빼고 [`np.linalg.svd`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) → 앞의 두 성분에 투영
- 참고 그림: [COCOA 2022](https://arxiv.org/abs/2208.00467) (원 입력 vs 학습 표현), [Saeed 2019](https://arxiv.org/abs/1907.11879) (은닉층 군집)

**신경 쓸 것:** PCA 함수는 **`fit`과 `transform`을 분리**한다 (모델 담당이 폰 샘플 투영에 재사용). 두 그림은 축이 달라서 **군집 모양만 비교**한다. 3단계 혼동행렬과 그림이 일치하는지 대조한다.

## 5단계. 결과 정리
- [`docs/trial_log.md`](../docs/trial_log.md)(팀 시행착오)와 [`docs/weekly.md`](../docs/weekly.md)를 바탕으로 발표에 넣을 시행착오를 고른다.
- 결과표, 혼동행렬, PCA 그림, 예상 vs 결과 표를 모으고 발표 자료를 구성한다: 질문 → 데이터 → 모델 → 예상 → 결과 → 은닉공간 → 폰 데모 → 한계

**신경 쓸 것:** 그림마다 **"여기서 볼 것" 한 줄**을 붙인다. 예상이 틀린 결과와 실패도 넣는다 (시행착오가 평가 대상).

---

## 협업 인터페이스

| 순서 | 주는 것 | 받는 사람 |
|---|---|---|
| 1 | 로더, 펼치기 함수, 검증 분할, `shared/norm_stats.npz` | 모델, 폰 |
| 2 | PCA 함수 (`fit`/`transform`) | 모델 |
| 3 | 결과표, 그림, 발표 자료 | 팀 |

**받는 것:** `train.py`, `shared/weights.npz` (모델) · 폰 혼동행렬, 영상 (폰)

## 발표 담당 파트
1. 데이터와 전처리: 왜 사람 단위 분할, 왜 채널별 정규화인가
2. **센서 비교 결과 (핵심 질문):** 예상 vs 결과, 혼동행렬
3. 입력 공간 vs 은닉 공간
