"""2-1단계: 기울기 확인 (gradient check).

역전파로 구한 기울기(net.gradient)와 수치미분(중앙차분)으로 구한 기울기를 비교한다.
- 네트워크: 입력 20, 은닉 10, 출력 6 (수치미분은 파라미터마다 loss를 2번 계산하므로 작게)
- 데이터: 랜덤 입력 3개, 랜덤 정답 3개 (float64). UCI 데이터는 쓰지 않는다
- 학습(갱신)은 하지 않는다. 초기 가중치 상태에서 기울기를 한 번 비교할 뿐이다
- 대조: Affine의 db를 일부러 2배로 틀리게 만들면 이 확인이 잡아내는지도 본다

loss의 +1e-7에 대해:
  교재 cross_entropy_error는 log(0)을 막으려고 -log(y + 1e-7)을 쓴다.
  그런데 역전파 공식 (y - t)/N 은 -log(y)의 미분이다. 두 loss는 기울기가 원소마다 약 1e-7/y 만큼 다르다.
  그래서 수치미분을 교재 loss로 하면, 역전파가 맞아도 상대오차가 1e-6~1e-5 수준으로 나온다.
  -> 역전파 공식을 확인할 때는 1e-7을 뺀 loss로 수치미분한다 ([1]).
     교재 loss 그대로 한 결과는 참고로 함께 출력한다 ([0]). 학습에는 교재 loss를 그대로 쓴다.

실행 (레포 루트에서): python model/checks/check_gradient.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # model/ (layers.py, model.py)
import layers  # noqa: E402
from model import PARAM_KEYS, TwoLayerNet  # noqa: E402

H = 1e-4        # 중앙차분 간격 (교재 common/gradient.py와 같음)
TOL = 1e-5      # 통과 기준: 파라미터별 최대 상대오차


def numerical_gradient(f, x):
    """중앙차분: (f(x+h) - f(x-h)) / 2h 를 원소마다. 교재 common/gradient.py와 같은 방식."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp = x[idx]
        x[idx] = tmp + H
        fxh1 = f()
        x[idx] = tmp - H
        fxh2 = f()
        grad[idx] = (fxh1 - fxh2) / (2 * H)
        x[idx] = tmp  # 값 복원
        it.iternext()
    return grad


def relative_error(a, b):
    """원소별 |a-b| / max(|a|, |b|). 둘 다 0에 가까우면(1e-8 미만) 분모를 1e-8로 둔다."""
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-8)


def compare(net, x, t):
    """파라미터별 (최대 상대오차, 평균 상대오차, 기준 초과 원소 수, 원소 수)."""
    grad_bp = net.gradient(x, t)
    rows = {}
    for key in PARAM_KEYS:
        # net.params[key]를 제자리에서 흔들면 계층(Affine)도 같은 배열을 보므로 loss에 반영된다
        grad_num = numerical_gradient(lambda: net.loss(x, t), net.params[key])
        err = relative_error(grad_bp[key], grad_num)
        rows[key] = (err.max(), err.mean(), int((err > TOL).sum()), err.size)
    return rows


def cross_entropy_exact(y, t):
    """-log(y)의 배치 평균 (1e-7 없음). 역전파 공식 (y - t)/N 과 정확히 짝이 맞는 loss."""
    return -np.sum(np.log(y[np.arange(y.shape[0]), t])) / y.shape[0]


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'param':<6}{'shape':>10}{'max rel err':>14}{'mean rel err':>14}{'> tol':>8}")
    for key, (mx, mean, n_bad, size) in rows.items():
        print(f"{key:<6}{str(SHAPES[key]):>10}{mx:>14.2e}{mean:>14.2e}{n_bad:>5}/{size}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 20))   # 샘플 3개, 입력 20
    t = rng.integers(0, 6, size=3)     # 정답 인덱스 0~5

    net = TwoLayerNet(input_size=20, hidden_size=10, output_size=6, init="he", seed=0)
    SHAPES = {k: v.shape for k, v in net.params.items()}

    # [0] 참고: 교재 loss(-log(y + 1e-7)) 그대로 수치미분
    print_table("[0] reference: numerical grad of textbook loss -log(y + 1e-7)", compare(net, x, t))

    # 이후 [1], [2]는 1e-7을 뺀 loss로 수치미분 (SoftmaxWithLoss가 layers 모듈의 함수를 부르므로 바꿔 끼운다)
    textbook_ce = layers.cross_entropy_error
    layers.cross_entropy_error = cross_entropy_exact

    rows = compare(net, x, t)
    print_table("[1] backprop vs numerical grad of -log(y)", rows)
    ok = all(mx < TOL for mx, *_ in rows.values())
    print(f"=> {'PASS' if ok else 'FAIL'} (max rel err < {TOL:g} for all params)")

    # 대조: db를 일부러 2배로 틀리게 -> b1, b2에서만 오차가 커져야 한다
    original_backward = layers.Affine.backward

    def wrong_backward(self, dout):
        dx = original_backward(self, dout)
        self.db = self.db * 2
        return dx

    layers.Affine.backward = wrong_backward
    rows_bug = compare(net, x, t)
    layers.Affine.backward = original_backward
    print_table("[2] control: db deliberately doubled (bug injected)", rows_bug)
    layers.cross_entropy_error = textbook_ce
    caught = rows_bug["b1"][0] > TOL and rows_bug["b2"][0] > TOL and rows_bug["W1"][0] < TOL and rows_bug["W2"][0] < TOL
    print(f"=> bug {'caught' if caught else 'NOT caught'}: only b1, b2 should exceed tol")
