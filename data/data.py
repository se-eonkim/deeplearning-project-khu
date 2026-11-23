"""UCI HAR 원신호 로더와 펼치기 함수.

팀 공통 약속 (shared/README.md, 노션 '코드 · 데이터 공통 약속'):
- 정답은 인덱스 0~5 (y - 1). one-hot과 섞지 않는다.
- 채널 순서는 아래 CHANNELS / CONDITIONS 로만 정한다. 바꾸지 않는다.
- 펼치기는 flatten() 하나로만 한다. 직접 reshape 하지 않는다.
- 정규화는 norm_stats.npz + normalize() 하나로만 한다. 샘플별·폰 데이터 자체 정규화 금지.

준비: UCI zip을 풀어 레포 루트에 `UCI HAR Dataset/`가 오게 둔다 (.gitignore 처리됨)
처음 한 번: python data/data.py   (data/cache/ 와 shared/norm_stats.npz 생성)

사용 예:
    import sys; sys.path.insert(0, "data")   # 레포 루트에서 실행할 때
    from data import load_signals, load_train_val, CONDITIONS
    X, y, subj = load_signals("train", CONDITIONS[5])   # 정규화 + 펼침, X: (7352, 768)
    raw, y, subj = load_signals("train", CONDITIONS[5], flat=False)  # (7352, 128, 6)
    (X_tr, y_tr, s_tr), (X_val, y_val, s_val) = load_train_val(CONDITIONS[5])  # 사람 단위 분할

폰 데이터 (g 단위, (N, 128, C), 채널 순서는 CONDITIONS[k]와 같게):
    X_phone = flatten(normalize(raw_phone, CONDITIONS[5]))

펼친 모양 (시점 우선, NumPy 기본 reshape. shared/README.md 약속):
    [t0의 C채널 | t1의 C채널 | ... ]   ->   x[t * C + c] = raw[t, c]
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # 레포 루트
DATA_DIR = ROOT / "UCI HAR Dataset"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
NORM_PATH = ROOT / "shared" / "norm_stats.npz"

T = 128  # 한 샘플 = 2.56초 x 50Hz
ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

# 센서 그룹 -> 채널 이름 (파일 이름 앞부분). 축 순서는 항상 x, y, z.
SENSORS = {
    "total_acc": ["total_acc_x", "total_acc_y", "total_acc_z"],   # 움직임 + 중력 [g]
    "body_acc": ["body_acc_x", "body_acc_y", "body_acc_z"],       # 몸의 움직임 [g]
    "body_gyro": ["body_gyro_x", "body_gyro_y", "body_gyro_z"],   # 회전 [rad/s]
}
CHANNELS = SENSORS["total_acc"] + SENSORS["body_acc"] + SENSORS["body_gyro"]  # 9채널 전체

# 실험 2의 5가지 조건. 6채널 기본(⑤)은 total_acc x,y,z -> body_gyro x,y,z
CONDITIONS = {
    1: SENSORS["body_gyro"],                         # ① 회전만
    2: SENSORS["body_acc"],                          # ② 움직임만
    3: SENSORS["total_acc"],                         # ③ 움직임 + 중력
    4: SENSORS["body_acc"] + SENSORS["body_gyro"],   # ④ 움직임 + 회전
    5: SENSORS["total_acc"] + SENSORS["body_gyro"],  # ⑤ 움직임 + 중력 + 회전
}


# 검증용 사람 (학습 21명 중 5명, 고정). 바꾸면 모든 실험을 다시 돌려야 한다.
# np.random.default_rng(0).choice(학습자 21명, 5, replace=False)로 뽑았고,
# 행동별로 학습 데이터의 22~24%씩 들어가 비율이 고르다.
VAL_SUBJECTS = (8, 11, 16, 19, 23)


def _load_split(split):
    """9채널 원신호 전체를 (N, 128, 9)로 읽는다. 처음 한 번만 txt를 읽고 cache/에 저장."""
    if split not in ("train", "test"):
        raise ValueError(f"split은 'train' 또는 'test': {split!r}")
    cache = CACHE_DIR / f"uci_{split}.npz"
    if cache.exists():
        d = np.load(cache)
        if list(d["channels"]) == CHANNELS:
            return d["raw"], d["y"], d["subject"]

    sig_dir = DATA_DIR / split / "Inertial Signals"
    raw = np.stack([np.loadtxt(sig_dir / f"{ch}_{split}.txt", dtype=np.float32) for ch in CHANNELS], axis=-1)
    y = np.loadtxt(DATA_DIR / split / f"y_{split}.txt", dtype=np.int64) - 1  # 1~6 -> 0~5
    subject = np.loadtxt(DATA_DIR / split / f"subject_{split}.txt", dtype=np.int64)
    assert raw.shape[1:] == (T, len(CHANNELS)) and len(raw) == len(y) == len(subject)
    assert y.min() == 0 and y.max() == 5

    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(cache, raw=raw, y=y, subject=subject, channels=np.array(CHANNELS))
    return raw, y, subject


def _channel_index(channels):
    unknown = [ch for ch in channels if ch not in CHANNELS]
    if unknown:
        raise ValueError(f"모르는 채널: {unknown}. 가능한 채널: {CHANNELS}")
    return [CHANNELS.index(ch) for ch in channels]


def load_signals(split, channels=CONDITIONS[5], flat=True, norm=True):
    """원신호를 읽어 고른 채널만 돌려준다.

    split    : "train" 또는 "test"
    channels : 채널 이름 리스트 (보통 CONDITIONS[k]). 이 순서 그대로 쌓인다.
    flat     : True면 (N, 128*C), False면 (N, 128, C)
    norm     : True면 norm_stats.npz로 채널별 정규화 (False는 원래 단위 g, rad/s)

    반환: X, y (0~5, int), subject (1~30, int)
    """
    idx = _channel_index(channels)
    raw, y, subject = _load_split(split)
    raw = raw[:, :, idx]
    if norm:
        raw = normalize(raw, channels)
    return (flatten(raw) if flat else raw), y, subject


def load_train_val(channels=CONDITIONS[5], flat=True, norm=True):
    """UCI 학습 데이터를 사람 단위로 학습 / 검증으로 나눈다 (VAL_SUBJECTS 고정).

    구간이 50%씩 겹쳐 있어서 샘플 단위로 무작위로 떼면 점수가 부풀므로 사람 단위로 뗀다.
    반환: (X_tr, y_tr, subj_tr), (X_val, y_val, subj_val)
    """
    X, y, subject = load_signals("train", channels, flat, norm)
    val = np.isin(subject, VAL_SUBJECTS)
    return (X[~val], y[~val], subject[~val]), (X[val], y[val], subject[val])


def make_norm_stats():
    """채널별 평균·표준편차를 학습 16명(VAL_SUBJECTS 제외)으로만 구해 norm_stats.npz에 저장.

    9채널 전부 저장하므로 어떤 조건이든 같은 파일을 쓴다.
    통계는 (샘플, 시간) 전체에 대해 채널마다 하나씩: mean[c], std[c].
    """
    raw, _, subject = _load_split("train")
    tr = raw[~np.isin(subject, VAL_SUBJECTS)].astype(np.float64)
    mean = tr.mean(axis=(0, 1))
    std = tr.std(axis=(0, 1))
    np.savez(NORM_PATH, mean=mean, std=std, channels=np.array(CHANNELS),
             val_subjects=np.array(VAL_SUBJECTS), n_samples=len(tr))
    return mean, std


_norm_cache = None


def normalize(raw, channels):
    """(…, 128, C) 원신호를 norm_stats.npz로 채널별 정규화: (x - mean[c]) / std[c].

    UCI든 폰이든 이 함수만 쓴다. channels는 raw의 채널 순서 그대로.
    """
    global _norm_cache
    if _norm_cache is None:
        if not NORM_PATH.exists():
            raise FileNotFoundError(f"{NORM_PATH.name} 없음. 먼저 `python data/data.py` 실행")
        d = np.load(NORM_PATH)
        if list(d["channels"]) != CHANNELS:
            raise ValueError(f"{NORM_PATH.name}의 채널 순서가 CHANNELS와 다름")
        _norm_cache = (d["mean"], d["std"])
    raw = np.asarray(raw)
    if raw.shape[-1] != len(channels):
        raise ValueError(f"raw 채널 수 {raw.shape[-1]} != channels {len(channels)}개")
    idx = _channel_index(channels)
    mean, std = _norm_cache
    return ((raw - mean[idx]) / std[idx]).astype(np.float32)


def flatten(raw):
    """(N, 128, C) -> (N, 128*C), 시점 우선: x[t*C + c] = raw[t, c] (NumPy 기본 reshape).

    폰 데이터도 반드시 이 함수로 펼친다. 한 샘플 (128, C)도 받는다 -> (128*C,).
    """
    raw = np.asarray(raw)
    if raw.shape[-2] != T:
        raise ValueError(f"시간축 길이는 {T}이어야 함: shape={raw.shape}")
    return raw.reshape(*raw.shape[:-2], -1)


def unflatten(x, n_channels):
    """flatten의 역변환: (N, 128*C) -> (N, 128, C)."""
    x = np.asarray(x)
    if x.shape[-1] != T * n_channels:
        raise ValueError(f"마지막 축은 {T}*{n_channels}이어야 함: shape={x.shape}")
    return x.reshape(*x.shape[:-1], T, n_channels)


if __name__ == "__main__":
    mean, std = make_norm_stats()
    print(f"{NORM_PATH.name} 저장 (학습 16명, 검증 {VAL_SUBJECTS} 제외)")
    for ch, m, s in zip(CHANNELS, mean, std):
        print(f"  {ch:<12} mean {m:>8.4f}  std {s:.4f}")
    for split in ("train", "test"):
        X, y, subj = load_signals(split)
        print(f"{split}: X {X.shape} {X.dtype}, y {y.shape} (0~5: {np.bincount(y)}), 사람 {len(np.unique(subj))}명")
    (X_tr, y_tr, s_tr), (X_val, y_val, s_val) = load_train_val()
    print(f"학습: {X_tr.shape}, 사람 {len(np.unique(s_tr))}명 / 검증: {X_val.shape}, 사람 {sorted(set(s_val.tolist()))}")
    for k, chs in CONDITIONS.items():
        X, _, _ = load_signals("train", chs)
        print(f"조건 {k}: {X.shape[1]:>4}차원  {chs}")
