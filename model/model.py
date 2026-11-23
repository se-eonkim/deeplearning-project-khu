"""2층 신경망: Affine -> ReLU -> Affine -> SoftmaxWithLoss.

출처: 『밑바닥부터 시작하는 딥러닝』 공식 코드 ch05/two_layer_net.py
교재와 다른 점
  - 난수: np.random.randn -> np.random.default_rng(seed). 같은 시드면 같은 초기 가중치
  - 초기화 선택: init="he" (std = sqrt(2/n_in)) 또는 "std" (weight_std 고정)
  - hidden(x): 은닉층 H = ReLU(XW1+b1)를 꺼낸다 (5단계 은닉 표현 분석용)
  - numerical_gradient 메서드는 model/checks/check_gradient.py로 뺐다 (학습에는 쓰지 않음)
  - save_model / load_model 추가 (shared/weights.npz)

사용 예 (model/ 폴더의 다른 스크립트에서):
    from model import TwoLayerNet, save_model, load_model
    net = TwoLayerNet(input_size=768, hidden_size=100, output_size=6, init="he", seed=0)
    grads = net.gradient(x_batch, t_batch)        # t: 인덱스 0~5
    for key in ("W1", "b1", "W2", "b2"):
        net.params[key] -= lr * grads[key]        # 반드시 -= (제자리 갱신)
"""
from collections import OrderedDict

import numpy as np

from layers import Affine, Relu, SoftmaxWithLoss

PARAM_KEYS = ("W1", "b1", "W2", "b2")


class TwoLayerNet:
    def __init__(self, input_size, hidden_size=100, output_size=6, init="he", weight_std=0.01, seed=0):
        rng = np.random.default_rng(seed)
        std1, std2 = self._init_std(init, weight_std, input_size, hidden_size)

        self.params = {
            "W1": std1 * rng.standard_normal((input_size, hidden_size)),
            "b1": np.zeros(hidden_size),
            "W2": std2 * rng.standard_normal((hidden_size, output_size)),
            "b2": np.zeros(output_size),
        }
        self._build_layers()

    @staticmethod
    def _init_std(init, weight_std, input_size, hidden_size):
        if init == "he":    # ReLU 앞 계층에 맞는 초기화 (교재 6.2.3)
            return np.sqrt(2.0 / input_size), np.sqrt(2.0 / hidden_size)
        if init == "std":   # 교재 기본값: 모든 층 같은 표준편차
            return weight_std, weight_std
        raise ValueError(f"init은 'he' 또는 'std': {init!r}")

    def _build_layers(self):
        """계층을 만든다. Affine은 params의 배열을 복사하지 않고 그대로 가리킨다.

        그래서 params[key] -= ... (제자리 갱신)는 계층에 바로 반영되지만,
        params[key] = 새 배열 로 바꾸면 계층은 옛 배열을 계속 쓴다 -> 그때는 이 함수를 다시 호출한다.
        """
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        """점수 (N, 6). softmax 전 값이다 (argmax만 하면 되므로 softmax는 생략)."""
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def hidden(self, x):
        """은닉층 출력 H = ReLU(XW1+b1), (N, hidden_size)."""
        h = self.layers["Affine1"].forward(x)
        return self.layers["Relu1"].forward(h)

    def loss(self, x, t):
        return self.lastLayer.forward(self.predict(x), t)

    def accuracy(self, x, t):
        y = np.argmax(self.predict(x), axis=1)
        if t.ndim != 1:  # one-hot이 들어오면 인덱스로 (교재 호환)
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def gradient(self, x, t):
        """역전파로 W1, b1, W2, b2의 기울기를 구한다. 갱신은 하지 않는다."""
        self.loss(x, t)  # forward: 각 계층이 backward에 필요한 값을 저장

        dout = self.lastLayer.backward(1)
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        return {
            "W1": self.layers["Affine1"].dW, "b1": self.layers["Affine1"].db,
            "W2": self.layers["Affine2"].dW, "b2": self.layers["Affine2"].db,
        }


def save_model(path, net, config):
    """가중치 + 설정을 한 파일에 저장한다.

    config: 최소한 channels(채널 이름 목록), hidden_size, lr. 그 밖의 설정(init, epochs, seed 등)도 같이 넣는다.
    """
    for key in ("channels", "hidden_size", "lr"):
        if key not in config:
            raise ValueError(f"config에 {key!r}가 없음 (shared/README.md 약속)")
    np.savez(path, **{k: net.params[k] for k in PARAM_KEYS}, **{k: np.asarray(v) for k, v in config.items()})


def load_model(path):
    """save_model로 저장한 파일 -> (net, config)."""
    d = np.load(path)
    W1, W2 = d["W1"], d["W2"]
    net = TwoLayerNet(input_size=W1.shape[0], hidden_size=W1.shape[1], output_size=W2.shape[1])
    for key in PARAM_KEYS:
        net.params[key] = d[key].astype(np.float64)
    net._build_layers()  # params를 새 배열로 바꿨으므로 계층을 다시 연결
    config = {k: (d[k].tolist() if d[k].ndim else d[k].item()) for k in d.files if k not in PARAM_KEYS}
    return net, config
