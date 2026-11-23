"""학습 스크립트: 조건(채널) 하나로 TwoLayerNet을 학습하고 결과를 .npz로 저장한다.

3단계(학습 설정 확정)와 실험 2(센서 비교 15회)가 모두 이 파일로 돈다.
설정은 명령줄 인자로만 바꾼다. 코드를 고치지 않는다.

교재 ch05/train_neuralnet.py와 다른 점
  - 배치: 복원추출 대신 epoch마다 섞어서 한 바퀴 (epoch 단위로 곡선을 읽기 위해)
  - 기록: 매 epoch 학습·검증 loss/정확도. 테스트는 --test를 줄 때만 마지막에 한 번
  - 시드: 초기화와 셔플에 같은 시드 값

예시 (레포 루트에서)
  python model/train.py                                   # 조건 5, 기본 설정
  python model/train.py --lr 0.01 --init std              # 설정 탐색 (테스트 안 봄)
  python model/train.py --cond 1 --seed 2 --test --out data/results/E2_c1_s2.npz   # 실험 2

결과 파일 (.npz)
  설정: cond, channels, seed, hidden_size, lr, init, epochs, batch
  곡선 (epoch별, 길이 epochs+1, 0번째는 학습 전): train_loss, train_acc, val_loss, val_acc
  --test일 때: test_pred, test_true, test_acc
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
from data import CONDITIONS, load_signals, load_train_val  # noqa: E402

from model import PARAM_KEYS, TwoLayerNet, save_model  # noqa: E402

# 3단계에서 확정한 공통 설정 (11/23). 바꾸면 실험 2 전체를 다시 돌려야 한다.
#   lr 0.1 / He / hidden 100 / epochs 23 / batch 100. 근거: model/README.md 3단계 결과, figures/settings_*.png
DEFAULTS = dict(cond=5, seed=0, hidden=100, lr=0.1, init="he", epochs=23, batch=100)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cond", type=int, default=DEFAULTS["cond"], choices=sorted(CONDITIONS), help="data.CONDITIONS 번호")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--hidden", type=int, default=DEFAULTS["hidden"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--init", default=DEFAULTS["init"], choices=["he", "std"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    p.add_argument("--test", action="store_true", help="마지막에 테스트 예측값 저장 (설정 탐색 중에는 쓰지 않는다)")
    p.add_argument("--out", default=None, help="결과 .npz 경로 (기본: model/results/<설정>.npz)")
    p.add_argument("--save-weights", default=None, help="가중치+설정 저장 경로 (예: shared/weights.npz)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def run_name(a):
    return f"c{a.cond}_h{a.hidden}_lr{a.lr:g}_{a.init}_e{a.epochs}_s{a.seed}"


def evaluate(net, x, t):
    return net.loss(x, t), net.accuracy(x, t)


def train(a):
    channels = CONDITIONS[a.cond]
    (X_tr, y_tr, _), (X_val, y_val, _) = load_train_val(channels)

    net = TwoLayerNet(input_size=X_tr.shape[1], hidden_size=a.hidden, output_size=6, init=a.init, seed=a.seed)
    rng = np.random.default_rng(a.seed)  # 셔플용 (초기화와 같은 시드 값)

    curves = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}

    def record():
        for split, (x, t) in (("train", (X_tr, y_tr)), ("val", (X_val, y_val))):
            loss, acc = evaluate(net, x, t)
            curves[f"{split}_loss"].append(loss)
            curves[f"{split}_acc"].append(acc)

    record()  # epoch 0 = 학습 전
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        order = rng.permutation(len(X_tr))
        for start in range(0, len(order), a.batch):
            idx = order[start:start + a.batch]
            grads = net.gradient(X_tr[idx], y_tr[idx])
            for key in PARAM_KEYS:
                net.params[key] -= a.lr * grads[key]
        record()
        if not np.isfinite(curves["train_loss"][-1]):
            print(f"epoch {epoch}: loss가 {curves['train_loss'][-1]} -> 발산, 중단")
            break
        if not a.quiet and (epoch % 10 == 0 or epoch == 1):
            print(f"epoch {epoch:>4}  train loss {curves['train_loss'][-1]:.4f} acc {curves['train_acc'][-1]:.4f}"
                  f" | val loss {curves['val_loss'][-1]:.4f} acc {curves['val_acc'][-1]:.4f}")

    elapsed = time.time() - t0
    val_acc = np.array(curves["val_acc"])
    best = int(np.nanargmax(val_acc))
    print(f"[{run_name(a)}] {elapsed:.1f}s | best val acc {val_acc[best]:.4f} at epoch {best}"
          f" | final val acc {val_acc[-1]:.4f}")

    result = dict(cond=a.cond, channels=np.array(channels), seed=a.seed, hidden_size=a.hidden, lr=a.lr,
                  init=a.init, epochs=a.epochs, batch=a.batch, elapsed=elapsed,
                  **{k: np.array(v) for k, v in curves.items()})

    if a.test:
        X_te, y_te, _ = load_signals("test", channels)
        pred = np.argmax(net.predict(X_te), axis=1)
        result.update(test_pred=pred, test_true=y_te, test_acc=float(np.mean(pred == y_te)))
        print(f"test acc {result['test_acc']:.4f}")

    out = Path(a.out) if a.out else ROOT / "model" / "results" / f"{run_name(a)}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **result)
    print(f"saved {out}")

    if a.save_weights:
        save_model(a.save_weights, net, dict(channels=list(channels), hidden_size=a.hidden, lr=a.lr, cond=a.cond,
                                             init=a.init, epochs=a.epochs, batch=a.batch, seed=a.seed))
        print(f"saved weights {a.save_weights}")
    return net, result


if __name__ == "__main__":
    train(parse_args())
