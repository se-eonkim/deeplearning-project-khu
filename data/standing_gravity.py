"""UCI HAR: 행동별 total_acc(x, y, z) 평균 (단위 g).

폰 담당이 폰 방향을 확인할 때 쓰는 기준값이다.
가만히 서 있으면 total_acc에는 거의 중력만 찍히므로,
STANDING 평균이 "UCI 폰이 허리에 차여 있던 방향"을 알려 준다.

실행: python data/standing_gravity.py
"""
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "UCI HAR Dataset"  # 레포 루트
ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]


def load_total_acc(split):
    sig_dir = DATA_DIR / split / "Inertial Signals"
    # (N, 128, 3): 축 순서 x, y, z
    acc = np.stack([np.loadtxt(sig_dir / f"total_acc_{ax}_{split}.txt") for ax in "xyz"], axis=-1)
    y = np.loadtxt(DATA_DIR / split / f"y_{split}.txt", dtype=int) - 1  # 0~5
    subj = np.loadtxt(DATA_DIR / split / f"subject_{split}.txt", dtype=int)
    return acc, y, subj


def main():
    acc, y, subj = load_total_acc("train")
    win_mean = acc.mean(axis=1)  # 샘플(2.56초)별 평균 (N, 3)

    print("UCI HAR train, total_acc 평균 [g]  (x, y, z, |g|)")
    print(f"{'activity':<20}{'n':>6}{'x':>9}{'y':>9}{'z':>9}{'|g|':>8}")
    for k, name in enumerate(ACTIVITIES):
        m = win_mean[y == k].mean(axis=0)
        print(f"{name:<20}{(y == k).sum():>6}{m[0]:>9.3f}{m[1]:>9.3f}{m[2]:>9.3f}{np.linalg.norm(m):>8.3f}")

    # STANDING: 사람마다 얼마나 다른지 (폰 값이 이 범위 안이면 방향이 맞다고 본다)
    st = y == ACTIVITIES.index("STANDING")
    per_subj = np.array([win_mean[st & (subj == s)].mean(axis=0) for s in np.unique(subj[st])])
    print("\nSTANDING, 사람별 평균의 분포 (학습자", len(per_subj), "명)")
    print(f"{'':<8}{'x':>9}{'y':>9}{'z':>9}")
    print(f"{'mean':<8}" + "".join(f"{v:>9.3f}" for v in per_subj.mean(axis=0)))
    print(f"{'std':<8}" + "".join(f"{v:>9.3f}" for v in per_subj.std(axis=0)))
    print(f"{'min':<8}" + "".join(f"{v:>9.3f}" for v in per_subj.min(axis=0)))
    print(f"{'max':<8}" + "".join(f"{v:>9.3f}" for v in per_subj.max(axis=0)))


if __name__ == "__main__":
    main()
