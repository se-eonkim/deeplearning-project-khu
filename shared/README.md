# 🔁 shared: 약속과 주고받는 파일

이 폴더에는 **약속된 파일만** 올린다. 여기에 올리면 전달 완료다.

## 약속 (바꾸려면 팀 전체 합의)

### 채널 순서 (전체 9채널)

| 인덱스 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| 채널 | `total_acc_x` | `total_acc_y` | `total_acc_z` | `body_acc_x` | `body_acc_y` | `body_acc_z` | `body_gyro_x` | `body_gyro_y` | `body_gyro_z` |

- 조건별로 채널 일부만 쓸 때도 **이 상대 순서를 유지**한다.
- 조건 ⑤(기본 모델, 폰용) = `total_acc_x,y,z → body_gyro_x,y,z` (6채널, 768)

### 펼치기 순서
- `(N, 128, C)` → `(N, 128·C)`는 **데이터 담당의 펼치기 함수 하나로만** 한다 (NumPy 기본 `reshape`, 시점 우선).

### 정답 형식
- **인덱스 0~5** (UCI 라벨 − 1). one-hot 쓰지 않음.

| 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| WALKING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | SITTING | STANDING | LAYING |

### 검증 분할
- 검증용 사람 번호: _(데이터 담당이 정하면 여기에 기록)_

## 주고받는 파일

| 파일 | 만드는 사람 | 쓰는 사람 | 내용 |
|---|---|---|---|
| `norm_stats.npz` | 🟦 데이터 | 🟩 모델, 🟧 폰 | `mean`, `std` (채널별, 학습 데이터 기준), `channels` (이름 목록) |
| `weights.npz` | 🟩 모델 | 🟦 데이터, 🟧 폰 | `W1`, `b1`, `W2`, `b2` + 설정 (`channels`, `hidden_size`, `lr`) |

> ⚠️ 위 파일을 다시 올리면(덮어쓰면) `docs/weekly.md`에 **꼭 기록**한다. 다른 사람의 결과가 바뀐다.
