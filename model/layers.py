"""신경망 부품: Affine, Relu, SoftmaxWithLoss.

출처: 『밑바닥부터 시작하는 딥러닝』 공식 코드
  - softmax, cross_entropy_error : common/functions.py
  - Relu, Affine, SoftmaxWithLoss : common/layers.py
교재 layers.py는 CNN용 common.util을 import하므로 통째로 쓰지 않고 필요한 것만 옮겼다.
교재와 다른 점: Affine.backward에 기울기 shape 확인(assert)을 추가했다.

정답 t는 인덱스 0~5 (shared/README.md 약속). one-hot은 쓰지 않는다.
"""
import numpy as np


def softmax(x):
    """점수 (N, K) -> 확률 (N, K). 지수 계산 전에 행별 최댓값을 빼서 overflow를 막는다."""
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def cross_entropy_error(y, t):
    """확률 y (N, K), 정답 t -> 배치 평균 loss. log(0)을 막으려고 1e-7을 더한다."""
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:  # one-hot이 들어오면 인덱스로 바꾼다 (교재 호환)
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)  # 0 이하였던 자리를 기억해 둔다
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0  # forward에서 0이 된 자리는 기울기도 0
        return dout


class Affine:
    def __init__(self, W, b):
        self.W = W  # params['W1'] 등과 같은 배열을 가리킨다 (복사 아님)
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)        # (N, out) @ (out, in) -> (N, in)
        self.dW = np.dot(self.x.T, dout)   # (in, N) @ (N, out) -> (in, out)
        self.db = np.sum(dout, axis=0)     # 배치 방향으로 합 -> (out,)
        assert self.dW.shape == self.W.shape, (self.dW.shape, self.W.shape)
        assert self.db.shape == self.b.shape, (self.db.shape, self.b.shape)
        assert dx.shape == self.x.shape, (dx.shape, self.x.shape)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax 출력 (확률)
        self.t = None  # 정답 (인덱스 0~5)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # one-hot (교재 호환)
            dx = (self.y - self.t) / batch_size
        else:                           # 인덱스: 정답 칸에서만 1을 뺀다
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx
