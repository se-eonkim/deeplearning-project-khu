# 스마트폰 센서 기반 인간 행동 분류 프로젝트

> 상태: 팀 논의용 개요(확정 아님). 날짜는 아직 정하지 않았다.

## 레포 구조

| 폴더 | 담당 | 내용 |
|---|---|---|
| [`data/`](data/) | 🟦 데이터 | UCI 로더·정규화, 실험 2·3 결과 |
| [`model/`](model/) | 🟩 모델 | 모델 구현·검증·학습, 은닉층 분석 |
| [`phone/`](phone/) | 🟧 폰 | 폰 데이터 수집·변환·예측, 데모 영상 |
| [`shared/`](shared/) | 다 같이 | 주고받는 파일(`norm_stats.npz`, `weights.npz`), 채널 순서·정답 형식 약속 |
| [`docs/`](docs/) | 다 같이 | [`weekly.md`](docs/weekly.md) 진행 기록, [`trial_log.md`](docs/trial_log.md) 시행착오 |

각 폴더의 `README.md` = 그 담당의 가이드. **작업 전에 자기 폴더 README부터 읽는다.**

## 협업 규칙 (사람·AI 에이전트 공통)

커밋·푸시할 때마다 지킨다. AI 에이전트(Claude Code, Codex 등)로 작업하면 에이전트가 자동으로 수행한다.

1. **진행 기록:** [`docs/weekly.md`](docs/weekly.md) 맨 아래에 한 줄 추가
   `| 날짜 | 담당 | 가이드 단계 | 한 일 | 막힌 곳 / 다음 할 일 |`
2. **시행착오:** 문제를 해결했으면 [`docs/trial_log.md`](docs/trial_log.md)에 한 줄 추가
   `| 날짜 | 담당 | 증상 | 원인 | 해결 |`
3. **자기 폴더만 수정한다.** `shared/`에는 약속된 파일만 올린다.
4. **실행해 본 코드만** 올린다.
5. **올리지 않는 것:** 영상, 얼굴 사진, UCI 원본 데이터
6. 커밋 메시지 앞에 `[data]` `[model]` `[phone]` `[docs]`
7. 올리기 전에 `git pull`

---

## 0. 프로젝트 한 줄 설명

스마트폰의 가속도계와 자이로스코프 숫자만 보고 사람이 **걷는지·앉았는지·누웠는지** 맞히는 신경망을 **NumPy로 직접 구현**한다.
그리고 **"어떤 센서가 어떤 행동을 알아보게 해 주는가?"**, **"신경망은 그 정보를 은닉층에서 어떻게 다시 배치하는가?"**를 실험으로 확인한다.

이 프로젝트의 목적은 가장 높은 정확도가 아니다.

1. 센서값이 신경망 입력이 되는 과정을 이해한다.
2. 순전파와 역전파를 직접 구현한다. (수업 취지: 라이브러리 없이 `forward → loss → gradient → update → 평가`를 설명할 수 있는 것)
3. 가속도·중력·회전 정보가 각각 어떤 행동 구분에 기여하는지 비교한다.
4. 결과와 함께 구현 과정, 선택 이유, 시행착오를 기록한다.

---

## 1. 데이터

**UCI HAR (Human Activity Recognition Using Smartphones)**

- 📄 [데이터셋 페이지 (UCI)](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- ⬇️ [바로 다운로드 (zip, 58MB)](https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip)
- 📘 [README 원문 (Hugging Face 미러)](https://huggingface.co/datasets/udayl/UCI_HAR/blob/main/README.txt): 파일 구조·단위 설명
- 📑 [원 논문: Anguita et al., ESANN 2013 (PDF)](https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf)
- 🔁 [Kaggle 버전](https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones): 561개 특징 버전. 원신호는 UCI zip에서 받는다. (링크 직접 열람은 미확인)
- 라이선스: CC BY 4.0 (출처 표기 후 사용 가능)
- 확장판(이번엔 사용 안 함): [HAPT 데이터셋](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions) — 자세 전환 포함

데이터 구성:

- 30명(19~48세), 허리에 Samsung Galaxy S II 착용
- 50Hz 측정, 2.56초(128시점) 구간, **구간끼리 50% 겹침**
- 학습 7,352개 / 테스트 2,947개, **사람 단위 70:30 분리**
- 사용하는 부분: 가공 특징 561개가 아니라 `train/Inertial Signals/`, `test/Inertial Signals/`의 **원신호**

## 2. 입력

센서 파일 9개 (`센서_축_train.txt`, 축 = x, y, z):

| 센서 | 의미 | 단위 |
|---|---|---|
| `total_acc` | 움직임 + 중력 (폰의 기울기가 담김) | g (= 9.8 m/s²) |
| `body_acc` | 몸의 움직임 (total에서 0.3Hz 저역통과 필터로 중력을 뺀 것) | g |
| `body_gyro` | 회전 | rad/s |

- `subject_train.txt`: 각 샘플이 몇 번 사람(1~30)인지 → **검증 데이터를 사람 단위로 뗄 때** 사용
- 입력 형태: (샘플, 128, 채널) → 한 줄로 펼쳐 `128 × 채널 수` (6채널이면 768)
- 정규화: **학습 데이터 통계로만** 채널별 평균 0, 표준편차 1

## 3. 출력

6개 행동 (`y_train.txt`, 1~6): 걷기 `WALKING` / 계단 오르기 `WALKING_UPSTAIRS` / 계단 내려가기 `WALKING_DOWNSTAIRS` / 앉기 `SITTING` / 서기 `STANDING` / 눕기 `LAYING`

```text
WALKING               0.78
WALKING_UPSTAIRS      0.11
...
예측: WALKING
```

## 4. 모델

- 2층 신경망(MLP), **NumPy로 직접 구현**
- `입력 → Affine → ReLU → Affine → Softmax`, 은닉층 100개(기본값)

$$
H=\operatorname{ReLU}(XW_1+b_1),\qquad Y=\operatorname{Softmax}(HW_2+b_2)
$$

- 출발 코드: [밑바닥부터 시작하는 딥러닝 공식 코드](https://github.com/oreilly-japan/deep-learning-from-scratch)의 `ch05/two_layer_net.py`, `common/layers.py` ([원본 파일](https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch05/two_layer_net.py))
- 직접 구현: Affine, ReLU, SoftmaxWithLoss, 역전파, 미니배치 학습, SGD(시간 되면 Adam), 정확도 계산
- PyTorch/TensorFlow 자동미분으로 모델을 완성하지 않는다.

## 5. 학습 설정

- 손실: Cross-entropy
- 기울기: 역전파. 수치미분은 작은 샘플 검증용으로만 (교수님이 "수치미분은 오래 걸리니 하지 말라"고 한 범위가 학습에만 해당하는지는 확인 필요)
- 최적화: SGD, 미니배치
- 검증 데이터: 학습 데이터에서 **사람 몇 명을 통째로** 떼어 냄 (구간이 50% 겹쳐 무작위로 떼면 점수가 부풂)
- 공정한 비교: 학습률·epoch·은닉층 크기 고정, 시드 3개로 반복

## 6. 실험

| 실험 | 내용 | 등급 |
|---|---|---|
| **실험 0** | 구현 검증: 작은 데이터에 과적합되는지, 기울기가 맞는지 | 필수 |
| **실험 1** | 기본 모델: 6채널로 학습, 하이퍼파라미터 고정 | 필수 |
| **실험 2** | **센서 비교 (핵심)**: 아래 5가지 조건 | 필수 |
| 실험 3 | 은닉층 들여다보기: 입력과 은닉층을 PCA 2D 그림으로 비교 | 시간 되면 (1순위) |
| 실험 4 | 모델 크기: 은닉 20/50/100, 층 1개/2개 | 시간 되면 |
| 실험 5 | 실제 폰 데이터 넣어 보기 ([phyphox](https://phyphox.org/), [원격 인터페이스](https://phyphox.org/docs/remote-interface/)) | 맨 뒤 |

### 실험 2: 조건과 예상 (실험 전에 적어 두는 가설)

| 조건 | 센서 | 예상 |
|---|---|---|
| ① | 회전만 (`body_gyro`) | 앉기·서기·눕기를 구분 못 함 (정지 상태라 회전 ≈ 0) |
| ② | 움직임만 (`body_acc`) | ①과 비슷 (중력 정보 없음) |
| ③ | 움직임 + 중력 (`total_acc`) | 눕기가 분리됨 (중력 방향 = 폰 기울기) |
| ④ | 움직임 + 회전 (`body_acc` + `body_gyro`) | 걷기·계단 구분이 좋아짐 |
| ⑤ | 움직임 + 중력 + 회전 (`total_acc` + `body_gyro`) | 가장 좋음. 앉기↔서기 혼동은 남을 수 있음 |

> 예상의 근거(참고): [Chevalier의 LSTM 프로젝트](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition)는 눕기 537/537 정답, 주된 혼동은 **앉기↔서기**와 **걷기 3종 사이**였고, 자이로를 빼면 정확도가 약 91.7% → 약 87%로 떨어졌다고 보고. 다만 LSTM 결과이므로 우리 MLP에서 같은 패턴이 나올지는 실험으로 확인한다.

### 실험 5: 실제 폰 데이터 (확장)

UCI로 학습한 모델에 폰 센서값을 넣어 본다(재학습 없음). 폰을 허리에 고정 → phyphox로 측정 → 50Hz·128시점으로 맞춤 → 학습 때와 같은 정규화 → 예측.
기종·단위(UCI는 g, 폰은 보통 m/s²)·중력 분리 방식·착용 방향이 달라 정확도가 낮을 수 있다. 필수 성공 조건이 아니며, 실패하면 **"왜 안 됐는가"(분포 차이)**를 기록하는 것으로 충분하다. 실시간보다 **녹화한 CSV 한 번 넣기**부터.

## 7. 평가 방법

- 정확도 (전체, 행동별)
- **혼동행렬**: 어떤 행동을 어떤 행동으로 헷갈렸는지
- loss·정확도 곡선 (학습 / 검증)
- 실험 2는 **예상 vs 결과**를 나란히 놓고 비교
- 조건 간 차이가 시드 간 차이보다 작으면 "차이 없음"으로 판단

---

## 8. 진행 단계 (날짜 미정)

원칙: **앞 단계가 틀리면 뒤 단계가 전부 무의미해지는 순서**로 쌓는다.

```text
0 이해 → 1 데이터 → 2 구현 검증 → 3 기본 모델 → 4 센서 비교(핵심) → 5 은닉층 → 6 여유 → 7 발표
```

### 0단계. 같이 이해하기
**핵심:** 셋이 같은 그림을 갖는다.
- [ ] 이 문서 + 수업 [`two_layer_net.py`](https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch05/two_layer_net.py) 같이 읽기
- [ ] 샘플 하나를 손으로 그리기: `(128시점 × 채널)` → 펼쳐서 `768` → `100` → `6`
- [ ] 실험 2의 예상표를 각자 생각해 오기
- **산출:** 없음 (공통 이해)
- **이유:** 동료 평가·질의응답 대비. 처음 맞추는 게 가장 싸다.

### 1단계. 데이터 준비
**핵심:** 원신호를 `(샘플, 768)` 행렬로 만들되, **누수 없이**.
- [ ] `load_signals(split, channels)` → `(N, 128, C)` (파일 9개 중 고른 채널만 쌓기)
- [ ] 정답 `y` 1~6 → 0~5로 바꾸기
- [ ] 검증 분할: 학습자 21명 중 **약 5명을 통째로** 검증으로 (`subject_train.txt` 사용)
- [ ] 정규화: **학습 데이터로만** 채널별 평균·표준편차 계산 → 학습/검증/테스트 모두 그 값으로 변환
- [ ] 펼치기: `(N, 128, C)` → `(N, 128·C)`
- [ ] 확인: shape, 채널별 평균≈0·표준편차≈1, 행동별 샘플 수, 검증·학습 사람이 겹치지 않음
- **산출:** `data_loader.py`
- **이유:** 이 단계의 실수는 에러 없이 점수만 부풀린다 → 나중에 못 찾는다.

### 2단계. 모델 구현과 검증 (실험 0)
**핵심:** 역전파가 **맞다는 증거**를 먼저 만든다.
- [ ] `Affine`, `Relu`, `SoftmaxWithLoss` 직접 작성 (교재 `common/layers.py`와 대조)
- [ ] `TwoLayerNet(input=768, hidden=100, output=6)`, 가중치 초기화 작게
- [ ] **기울기 검증:** 샘플 3개·작은 은닉층에서 역전파 vs 수치미분 → 상대오차가 아주 작은지
- [ ] **과적합 검사:** 샘플 100개만으로 학습 → 정확도 100% 나오는지
- **산출:** `layers.py`, `model.py`, 검증 로그
- **이유:** 버그가 있어도 학습은 "되는 척" 한다. 이 상태로 비교하면 센서 차이인지 버그인지 모른다.

### 3단계. 기본 모델 (실험 1)
**핵심:** 6채널로 학습하고 **설정을 확정**한다.
- [ ] 미니배치 100, SGD
- [ ] 학습률 2~3개 시도 (예: 0.01 / 0.1 / 0.5) → 검증 정확도 기준으로 하나 고르기
- [ ] epoch 수: 검증 정확도가 멈추는 지점으로
- [ ] 매 epoch 학습/검증 loss·정확도 기록, 가중치 `np.savez`로 저장
- [ ] 테스트 혼동행렬 (6×6)
- [ ] **여기서 고른 학습률·epoch·은닉층 크기는 이후 고정**
- **산출:** `train.py`, `evaluate.py`, 곡선 그림, 혼동행렬 1장
- **이유:** 실험 2는 "센서만 바꾸고 나머지는 같게"여야 한다. 그 나머지를 정하는 단계.

### 4단계. 센서 비교 (실험 2) ★ 핵심
**핵심:** 예상을 **먼저 적고**, 5조건을 같은 설정으로 돌린다.
- [ ] 6절 예상표를 팀이 확정 (결과 보기 전에)
- [ ] 조건 5개 × 시드 3개 = 학습 15회 (입력 크기만 384 또는 768로 달라짐)
- [ ] 결과표: 테스트 정확도 평균±편차, **행동별 정확도**
- [ ] 조건별 혼동행렬 5장 → "어떤 혼동이 생기고, 어떤 센서를 넣으면 사라지나"
- [ ] 예상 vs 결과: 맞음 / 틀림 / 왜
- **산출:** `sensor_comparison.png`, 혼동행렬 5장, 비교 표
- **이유:** 코드 변경이 채널 선택뿐이라 싸고, 결과가 곧 해석이다. 예상을 먼저 적어야 끼워 맞추기가 안 된다.

### 5단계. 은닉층 들여다보기 (실험 3)
**핵심:** 같은 테스트 데이터를 **입력 공간 vs 은닉공간**에서 비교한다.
- [ ] 4단계 모델(조건 ⑤)로 은닉층 출력 `H = ReLU(XW₁+b₁)` 계산
- [ ] `X`와 `H` 각각 NumPy로 PCA (평균 빼고 SVD) → 2D
- [ ] 두 그림을 나란히, 행동별 색
- [ ] (선택) 조건 ①의 은닉층도 그려서 ⑤와 비교 → 정적 행동이 섞이는지
- [ ] 4단계 혼동행렬과 그림이 맞는지 확인
- **산출:** `pca_input_vs_hidden.png`
- **이유:** 4단계는 "무엇이 구분되나", 5단계는 "안에서 어떻게 구분되나". 학습된 모델 재사용이라 추가 학습 없음.

### 6단계. 여유 실험
**핵심:** 핵심이 끝났을 때만.
- [ ] 실험 4: 은닉 20/50/100, 층 1/2 → 학습/테스트 간격
- [ ] Adam vs SGD → 수렴 속도
- [ ] 실험 5: 폰 녹화 CSV 1회 → 단위(m/s² → g)·50Hz·128시점 맞추고 예측
- **이유:** 중심 질문과 거리가 있거나(크기·optimizer) 불확실성이 크다(폰).

### 7단계. 정리·발표
**핵심:** 셋 다 처음부터 끝까지 설명할 수 있게.
- [ ] `docs/시행착오_기록.md` 정리
- [ ] 예상 질문 10개와 답 (예: dW는 왜 Xᵀ·dY? 검증은 왜 사람 단위? 자이로만으로는 왜 앉기·서기를 못 가르나?)
- [ ] 발표 흐름: 질문 → 데이터 → 모델 → 예상 → 결과 → 은닉층 → 한계
- **이유:** 질의응답 적극성이 평가에 들어간다.

---

## 9. 팀 역할

| 역할 | 맡는 단계 | 가이드 |
|---|---|---|
| 🟦 데이터·실험 | 1단계, 4단계(실험 2), 5단계(실험 3), 결과 정리 | [`data/README.md`](data/README.md) |
| 🟩 모델 | 2~3단계, 학습 설정 확정, 폰 데이터 은닉층 분석 | [`model/README.md`](model/README.md) |
| 🟧 폰 | 폰 데이터 수집·변환·예측, 데모 영상 (전 기간 병행) | [`phone/README.md`](phone/README.md) |

각자 코드를 맡더라도 **전체 흐름은 모두 설명할 수 있어야 한다.** 주고받는 파일과 약속은 [`shared/README.md`](shared/README.md)에 있다.

---

## 10. 참고 자료

### 비슷한 프로젝트 (코드)

코드를 그대로 가져오지 않고, 데이터 구조·실험 구성·해석 방식을 참고한다.

| 프로젝트 | 무엇을 했나 | 우리에게 쓸모 |
|---|---|---|
| [guillaume-chevalier/LSTM-Human-Activity-Recognition](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition) | 원신호 9채널(128×9) LSTM, TensorFlow, 약 91.7% | **데이터 로딩**(파일 9개 쌓기), **혼동행렬 해석**. 가장 가까운 참고 |
| [ASR1999/HumanActivityRecognition](https://github.com/ASR1999/HumanActivityRecognition) | 데이터 탐색, 561특징 MLP, 원신호 CNN-LSTM (PyTorch) | 시각화 방식 |
| [aiff22/har](https://github.com/aiff22/har) | 128샘플 구간 CNN, 약 97% (TensorFlow) | 성능 상한선 참고 |
| [rajneeshbabu/human-activity-recognition](https://github.com/rajneeshbabu/human-activity-recognition) | 폰 브라우저 센서로 실시간 예측 (SVM) | 실험 5 참고 |
| [austinjtaylor/activity-recognition-with-sensor-data](https://github.com/austinjtaylor/activity-recognition-with-sensor-data) | 가속도·자이로 시계열 LSTM | 시계열 입력 구조 |

> NumPy만으로 UCI HAR 원신호를 MLP로 학습한 공개 저장소는 검색에서 찾지 못했다(2026-11 기준). 직접 구현 자체가 차별점이다.

### 관련 연구 (논문)

| 논문 | 핵심 | 연결 |
|---|---|---|
| [Shoaib et al., *Sensors* 2014 — Fusion of Smartphone Motion Sensors for Physical Activity Recognition](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4118351/) | 가속도계·자이로·지자기를 단독/조합으로 비교, "가속도와 자이로는 서로 보완" | **실험 2 선행연구** |
| [Ronao & Cho, *Expert Systems with Applications* 2016](https://doi.org/10.1016/j.eswa.2016.04.032) | 원신호 1D CNN 약 94.8% (30명 데이터, UCI HAR로 보이나 미확정) | 성능 비교 기준 |
| [Saeed et al. 2019 — Multi-task Self-Supervised Learning for HAR (arXiv)](https://arxiv.org/abs/1907.11879) | UCI HAR 은닉층 활성화 t-SNE, 행동별 군집 | **실험 3** |
| [COCOA 2022 (arXiv)](https://arxiv.org/abs/2208.00467) | UCI HAR **원 입력 vs 학습된 표현** t-SNE 비교 | **실험 3과 같은 구도** |
| [Explaining 1D CNN Embeddings in HAR, *Sensors* 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10181687/) | 분류 직전 층 시각화로 판단 설명 (다른 데이터셋) | 해석 방법 |
| [nanoML for HAR 2025 (arXiv)](https://arxiv.org/abs/2502.12173) | 전처리 없이 원신호 128샘플로 작은 모델 | 설정이 비슷 |
| [Wang et al. 2019 — Deep Learning for Sensor-based Activity Recognition: A Survey (arXiv)](https://arxiv.org/abs/1707.03502) | 센서 행동 인식 딥러닝 총정리 | 배경 설명 |

센서 비교 연구(Shoaib)와 은닉표현 시각화 연구(Saeed, COCOA)는 각각 있지만, **둘을 직접 구현한 MLP 하나로 연결해 보는 것**이 우리 구도다.

---

## 11. 최종 발표 메시지 (초안)

> 스마트폰은 움직임을 숫자 신호로 측정한다. 우리는 이 신호로 행동을 분류하는 신경망을 직접 구현하고, 움직임·중력·회전 정보가 각각 어떤 행동 구분을 가능하게 하는지 예상하고 확인했다. 그리고 신경망의 은닉층이 그 정보를 어떻게 다시 배치하는지 들여다보았다.
