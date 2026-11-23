"""2-2단계: 작은 데이터로 100% 확인 (overfitting sanity check).

UCI 학습 데이터(조건 ⑤, 768차원) 100개만으로 학습해서 학습 정확도 100%에 닿는지 본다.
100개는 행동별로 고르게(약 17개씩) 시드 0으로 뽑는다. 앞 100개를 그냥 쓰면 데이터가 사람·시간 순서라
계단 오르기·내려가기가 0개가 되어 확인이 너무 쉬워진다.
파라미터(약 7.8만)가 샘플(100)보다 훨씬 많으므로, 학습 루프가 맞다면 100개를 다 외울 수 있어야 한다.
닿지 못하면 학습률 -> loss -> backward 순서로 의심한다 (model/README.md 2단계).

- 설정: 은닉 100, He 초기화, 학습률 0.1, 100개 전체를 한 배치로, 정규화 기법 없음
- 검증 정확도는 참고로만 출력한다 (100개만 외운 모델이라 낮은 게 정상)

실행 (레포 루트에서): python model/check_overfit.py
산출: model/figures/overfit_100.png
"""
import sys
from pathlib import Path

import numpy as np

import plot_style  # noqa: F401  (Times New Roman)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
from data import CONDITIONS, load_train_val  # noqa: E402

from model import PARAM_KEYS, TwoLayerNet  # noqa: E402

N_SAMPLES = 100
LR = 0.1
MAX_ITERS = 1000


def balanced_subset(y, n, seed=0):
    """행동별로 n/6개씩(나머지는 앞 행동부터 1개씩 더) 무작위로 뽑은 인덱스."""
    rng = np.random.default_rng(seed)
    per_class = [n // 6 + (k < n % 6) for k in range(6)]
    idx = np.concatenate([rng.choice(np.flatnonzero(y == k), m, replace=False) for k, m in enumerate(per_class)])
    return rng.permutation(idx)


if __name__ == "__main__":
    (X_tr, y_tr, _), (X_val, y_val, _) = load_train_val(CONDITIONS[5])
    idx = balanced_subset(y_tr, N_SAMPLES)
    x, t = X_tr[idx], y_tr[idx]
    print(f"data: x {x.shape}, labels per class {np.bincount(t, minlength=6)}")

    net = TwoLayerNet(input_size=x.shape[1], hidden_size=100, output_size=6, init="he", seed=0)
    losses, accs = [], []
    reached = None
    for i in range(MAX_ITERS):
        grads = net.gradient(x, t)
        for key in PARAM_KEYS:
            net.params[key] -= LR * grads[key]
        losses.append(net.loss(x, t))
        accs.append(net.accuracy(x, t))
        if reached is None and accs[-1] == 1.0:
            reached = i + 1
        if i + 1 in (1, 10, 50, 100, 200, 500, MAX_ITERS):
            print(f"iter {i + 1:>5}: loss {losses[-1]:.4f}, train acc {accs[-1]:.3f}")

    print(f"\ntrain acc 100% first reached at iter {reached}" if reached else "\ntrain acc 100% NOT reached")
    print(f"final loss {losses[-1]:.5f}, final train acc {accs[-1]:.3f}")
    print(f"(reference) val acc of this 100-sample model: {net.accuracy(X_val, y_val):.3f}")
    print(f"=> {'PASS' if accs[-1] == 1.0 else 'FAIL'}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(losses)
    ax[0].set(title=f"Loss on {N_SAMPLES} samples", xlabel="iteration (full batch)", ylabel="loss", yscale="log")
    ax[1].plot(accs)
    if reached:
        ax[1].axvline(reached, ls="--", c="gray")
        ax[1].text(reached, 0.5, f" 100% at iter {reached}", color="gray")
    ax[1].set(title=f"Train accuracy on {N_SAMPLES} samples", xlabel="iteration (full batch)", ylabel="accuracy", ylim=(0, 1.05))
    fig.suptitle(f"Overfit check: cond 5 (768), hidden 100, He init, lr {LR}")
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "figures" / "overfit_100.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved {out.relative_to(ROOT)}")
