# 🟩 모델 담당: 학습 엔진 설계·구현·검증 + 은닉 표현 분석

**역할:** 프로젝트의 학습 엔진을 책임진다. NumPy로 순전파와 역전파를 직접 구현해 정확성을 검증하고, 모든 실험의 기준이 되는 학습 설정을 확정한다. 마지막에 은닉 표현 분석으로 폰 데이터에서 모델이 **왜** 틀렸는지 설명한다.

**원칙:** 역전파가 틀려도 학습은 "되는 척" 한다. 여기서 검증되지 않은 엔진은 실험 2와 폰 결과를 전부 무의미하게 만든다. **돌아가는 것이 아니라 맞다는 증거**를 남긴다.

> **이 파트의 의미:** 수업의 핵심인 "직접 구현"을 맡는 파트다. 동시에 학습 과정(loss 곡선)에서 은닉 표현, 예측 실패까지 이어지는 흐름을 한 사람이 처음부터 끝까지 본다.

> 시작 전에 [레포 README](../README.md)의 협업 규칙과 [`shared/README.md`](../shared/README.md)의 약속(채널 순서, 정답 형식)을 먼저 읽는다.

---

## 1단계. 모델 아키텍처 구현

**자료:** [교재 `ch05/two_layer_net.py`](https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch05/two_layer_net.py) · [`common/layers.py`](https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/layers.py)

| 할 일 | 신경 쓸 것 |
|---|---|
| `Affine → ReLU → Affine → SoftmaxWithLoss` 계층형 MLP 구성 (입력 768, 은닉 100, 출력 6) | 새로 짜지 않고 교재 구조를 기반으로 한다. **입력 크기는 인자로** 받는다 (조건 ①~③은 384) |
| Affine backward: `dx = dout·Wᵀ`, `dW = xᵀ·dout`, `db = Σ dout` (배치 축) | **기울기 shape = 파라미터 shape**인지 assert로 확인한다 |
| Softmax | 지수 계산 전에 **최댓값을 빼서** overflow를 막는다 (교재 방식) |
| 정답 형식 | **인덱스 0~5로 통일**한다 ([`shared/README.md`](../shared/README.md)). one-hot과 섞이면 loss가 조용히 틀린다 |

- **산출:** `model/layers.py`, `model/model.py`

## 2단계. 구현 정확성 검증

**자료:** [교재 `ch05/gradient_check.py`](https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch05/gradient_check.py) · [CS231n: Gradient checks](https://cs231n.github.io/neural-networks-3/#gradcheck)

| 할 일 | 신경 쓸 것 |
|---|---|
| **Gradient check:** 역전파 vs 수치미분(중앙차분)을 W1, b1, W2, b2별로 비교하고 상대오차 표 작성 | **작은 네트워크**(입력 20, 은닉 10, 샘플 3)로 한다. 수치미분은 느리다 (교수님 언급). `float64`를 쓴다. ReLU가 0 근처에 걸리면 오차가 조금 튈 수 있다 → 대부분의 원소에서 오차가 작으면 정상 |
| **Overfitting sanity check:** 샘플 100개로 정확도 100% 도달 확인 | 정규화 기법 없이 한다. 도달하지 못하면 **학습률 → loss → backward 순서로** 의심한다 |
| 랜덤 입력으로 진행 | UCI 데이터 없이 가능 → 데이터 담당과 **병렬로 시작** |

- **산출:** 기울기 검증 표, 과적합 곡선

## 3단계. 학습 설정 확정 (Baseline = 조건 ⑤)

**자료:** [교재 `ch06/weight_init_compare.py`](https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch06/weight_init_compare.py) · [He et al. 2015 (He 초기화)](https://arxiv.org/abs/1502.01852)

| 할 일 | 신경 쓸 것 |
|---|---|
| 조건 ⑤(`total_acc` + `body_gyro`, 768)로 학습 | 이 모델이 그대로 **폰용 모델**이 된다. 따로 학습할 필요가 없다 |
| 학습률 0.01 / 0.1 / 0.5 비교 | **검증 정확도로** 고른다. 테스트 점수는 보지 않는다 |
| 초기화 비교: std 0.01 vs He (`√(2/n)`) | ReLU에는 He가 이론적으로 맞다. 차이가 곡선에 어떻게 나타나는지 본다 (ch06 연결) |
| Early stopping: 검증 정확도가 정체되는 epoch | 학습 정확도만 오르고 검증이 멈추면 과적합 신호 |
| 시드 관리 | 초기화와 배치 셔플에 **같은 시드**를 쓴다 ([`np.random.default_rng`](https://numpy.org/doc/stable/reference/random/generator.html)). 그래야 실험 2의 시드 3개 비교가 성립한다 |
| 확정값을 **전 실험 공통 설정으로 고정** | 이후에 바꾸면 실험 2 전체를 다시 돌려야 한다 |

- **산출:** 학습/검증 loss·accuracy 곡선, 설정 확정표, baseline 혼동행렬

> ⚠️ **내가 기다리는 것:** 데이터 담당의 로더, 검증 분할, `shared/norm_stats.npz`. 1~2단계는 기다리지 않고 먼저 한다.

> **이 단계에서 보는 것:** 같은 모델이라도 학습률과 초기화에 따라 학습 경로가 달라진다. loss 곡선의 모양으로 그 차이를 확인한다.

## 4단계. 학습 파이프라인 배포

| 할 일 | 신경 쓸 것 |
|---|---|
| `model/train.py`: 채널 조합, 시드, 은닉 크기를 인자로 | 데이터 담당이 코드를 고치지 않고 15회를 돌릴 수 있어야 한다. 결과 저장 형식(설정, 곡선, **테스트 예측값**)은 데이터 담당과 맞춘다 |
| `shared/weights.npz` 저장 (3단계 조건 ⑤ 모델) | 가중치와 함께 **설정(채널 목록, 은닉 크기, 학습률)도 같이 저장**한다. 파일만 봐도 어떤 모델인지 알게 한다 |

- **산출:** `model/train.py`, `shared/weights.npz`

> ⚠️ **나를 기다리는 사람:** 데이터 담당(실험 2 실행), 폰 담당(폰 예측).

## 5단계. 은닉 표현 분석: 폰 데이터는 왜 틀렸나

| 할 일 | 신경 쓸 것 |
|---|---|
| 조건 ⑤ 모델로 `H = ReLU(XW₁+b₁)` 추출 (UCI 테스트, 폰 데이터) | 두 데이터 모두 **같은 `norm_stats.npz`**로 정규화된 상태인지 확인 |
| 데이터 담당의 PCA 함수로 **UCI 테스트 `H`에만 fit** → 폰 `H`는 transform만 | **둘을 합쳐서 fit하면 안 된다.** 축이 폰 쪽으로 휘어서 "UCI로 배운 공간에서 폰이 어디에 있나"를 볼 수 없다 |
| 그림: UCI는 행동별 색의 점, 폰은 같은 색의 × | 질문: 폰 샘플이 자기 군집에 들어가나, 다른 군집으로 밀리나? 어떤 행동에서 가장 크게 밀리나? **폰 혼동행렬과 일치하나?** |
| 보조 지표: 행동별 ReLU 활성 비율 (`(H>0).mean`) | UCI와 폰에서 **켜지는 뉴런 패턴**이 다른지 본다. UCI에서 한 번도 안 켜지는 뉴런(dead ReLU)도 세어 둔다 |

- **산출:** 은닉공간 투영 그림, 분석 요약 3~4줄

> ⚠️ **내가 기다리는 것:** 폰 담당의 변환된 폰 데이터 `(N, 768)` + 정답, 데이터 담당의 PCA 함수.

> **이 단계에서 보는 것:** 입력 분포가 바뀌면 은닉 표현이 이동하고, 그 결과 예측이 틀린다. 정확도 숫자만으로는 보이지 않는 실패 원인을 은닉층 그림으로 설명한다.

---

## 협업 인터페이스

| 순서 | 주는 것 | 받는 사람 |
|---|---|---|
| 1 | `model/layers.py`, `model/model.py` (검증 완료) | 팀 |
| 2 | `model/train.py`, 설정 확정표 | 데이터·실험 |
| 3 | `shared/weights.npz` (조건 ⑤) | 데이터·실험, 폰 |
| 4 | 은닉공간 그림, 분석 요약 | 데이터·실험 (발표 정리용) |

**받는 것:** UCI 로더, 펼치기 함수, `shared/norm_stats.npz`, PCA 함수 (데이터·실험) · 변환된 폰 데이터 `(N, 768)` + 정답 (폰)

## 발표 담당 파트
1. **모델 구조와 학습 원리:** 순전파 → loss → 역전파 → 갱신
2. **구현 검증과 시행착오:** gradient check, 학습률·초기화에 따른 곡선 차이
3. **폰 실패 원인 분석:** 은닉공간에서 본 분포 이동 (폰 데모 직후)
