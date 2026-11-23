"""3단계 그림: model/results/*.npz (train.py 결과)를 읽어 설정 비교 그림을 그린다.

실행 (레포 루트에서): python model/plots/plot_settings.py
산출 (model/figures/)
  settings_lr_init.png   학습률 3개 x 초기화 2개: 학습 loss, 검증 정확도
  settings_hidden.png    은닉 50 vs 100: 학습/검증 정확도 (과적합 간격)
  settings_seeds.png     확정 설정, 시드 0/1/2: 검증 정확도 (결과 파일이 있을 때만)
  confusion_baseline.png 확정 설정 시드 0의 테스트 혼동행렬, 행별 비율 (--test 결과가 있을 때만)
"""
import sys
from pathlib import Path

import numpy as np

import plot_style
import matplotlib.pyplot as plt

MODEL_DIR = Path(__file__).resolve().parent.parent   # model/
RES = MODEL_DIR / "results"
FIG = MODEL_DIR / "figures"
FIG.mkdir(exist_ok=True)

SMOOTH = 5  # 검증 정확도 이동평균 창 (epoch). epoch마다 ±2%p씩 흔들려서 정점을 이동평균으로 읽는다


def load(name):
    p = RES / f"{name}.npz"
    return np.load(p) if p.exists() else None


def moving_average(x, w=SMOOTH):
    """앞뒤 w//2 epoch 평균 (가장자리는 있는 만큼만)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    for i in range(len(x)):
        lo, hi = max(0, i - w // 2), min(len(x), i + w // 2 + 1)
        out[i] = np.nanmean(x[lo:hi])
    return out


def fig_lr_init():
    lrs, inits = (0.01, 0.1, 0.5), ("std", "he")
    titles = {"std": "Init: std 0.01", "he": "Init: He"}
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for j, init in enumerate(inits):
        for lr in lrs:
            d = load(f"c5_h100_lr{lr:g}_{init}_e100_s0")
            if d is None:
                continue
            ep = np.arange(len(d["val_acc"]))
            ax[0, j].plot(ep, d["train_loss"], label=f"lr {lr:g}")
            ax[1, j].plot(ep, d["val_acc"], lw=0.8, alpha=0.35, color=ax[0, j].lines[-1].get_color())
            ax[1, j].plot(ep, moving_average(d["val_acc"]), label=f"lr {lr:g}", color=ax[0, j].lines[-1].get_color())
        ax[0, j].set(title=titles[init], ylabel="train loss", yscale="log")
        ax[1, j].set(xlabel="epoch", ylabel="validation accuracy", ylim=(0.4, 0.95))
        ax[0, j].legend()
        ax[1, j].legend(loc="lower right")
    fig.suptitle(f"Condition 5 (768 inputs), hidden 100, seed 0. Validation: thin = raw, thick = {SMOOTH}-epoch moving average")
    fig.tight_layout()
    fig.savefig(FIG / "settings_lr_init.png")
    plt.close(fig)


def fig_hidden():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for j, h in enumerate((50, 100)):
        d = load(f"c5_h{h}_lr0.1_he_e100_s0")
        if d is None:
            continue
        ep = np.arange(len(d["val_acc"]))
        ax[j].plot(ep, d["train_acc"], label="train")
        ax[j].plot(ep, d["val_acc"], lw=0.8, alpha=0.35, color="C1")
        ax[j].plot(ep, moving_average(d["val_acc"]), color="C1", label=f"validation ({SMOOTH}-ep avg)")
        ax[j].set(title=f"Hidden {h}", xlabel="epoch", ylim=(0.6, 1.0))
        ax[j].legend(loc="lower right")
    ax[0].set_ylabel("accuracy")
    fig.suptitle("lr 0.1, He init, seed 0: gap between train and validation = overfitting")
    fig.tight_layout()
    fig.savefig(FIG / "settings_hidden.png")
    plt.close(fig)


def fig_seeds(name_fmt):
    runs = [load(name_fmt.format(s=s)) for s in (0, 1, 2)]
    if any(r is None for r in runs):
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for s, d in enumerate(runs):
        ax.plot(np.arange(len(d["val_acc"])), d["val_acc"], label=f"seed {s}")
    ax.set(xlabel="epoch", ylabel="validation accuracy", ylim=(0.6, 1.0),
           title="Final setting, 3 seeds: " + name_fmt.format(s="*"))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "settings_seeds.png")
    plt.close(fig)


def fig_confusion(name):
    d = load(name)
    if d is None or "test_pred" not in d.files:
        return
    cm = np.zeros((6, 6))
    np.add.at(cm, (d["test_true"], d["test_pred"]), 1)
    rate = cm / cm.sum(axis=1, keepdims=True)  # 행별 비율 (행 = 정답)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for k in range(6):
            ax.text(k, i, f"{rate[i, k]:.2f}", ha="center", va="center",
                    color="white" if rate[i, k] > 0.5 else "black", fontsize=10)
    ax.set_xticks(range(6), plot_style.ACTIVITY_SHORT)
    ax.set_yticks(range(6), plot_style.ACTIVITY_SHORT)
    ax.set(xlabel="predicted", ylabel="true", title=f"Test confusion (row-normalized), acc {float(d['test_acc']):.3f}")
    ax.spines[:].set_visible(True)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_baseline.png")
    plt.close(fig)


def summary():
    print(f"{'run':<32}{'best val (raw)':>16}{'best val (avg)':>18}{'final val':>11}{'train@end':>11}")
    for p in sorted(RES.glob("c5_*.npz")):
        d = np.load(p)
        va, sm = d["val_acc"], moving_average(d["val_acc"])
        b, bs = int(np.nanargmax(va)), int(np.nanargmax(sm))
        print(f"{p.stem:<32}{va[b]:>10.4f} @{b:<4}{sm[bs]:>12.4f} @{bs:<4}{va[-1]:>11.4f}{d['train_acc'][-1]:>11.4f}")


if __name__ == "__main__":
    fig_lr_init()
    fig_hidden()
    if len(sys.argv) > 1:  # 확정 설정 이름 형식, 예: c5_h100_lr0.1_he_e30_s{s}
        fig_seeds(sys.argv[1])
        fig_confusion(sys.argv[1].format(s=0))
    summary()
