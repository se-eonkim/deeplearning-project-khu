"""모델 구조 그림: 조건 ⑤ 2층 신경망 (768 -> 100 -> 6).

계층 이름은 model/model.py의 _build_layers와 같다: Affine1, Relu1, Affine2, lastLayer(SoftmaxWithLoss).
입력 768 = 128시점 x 6채널 (data.flatten, 시점 우선: x[t*6 + c]).
선 하나 = 가중치 하나. 강조한 두 선은 W1[0, 0], W2[0, 0] (코드의 0부터 세는 인덱스).

실행 (레포 루트에서): python model/plot_network.py
산출: model/figures/network.png
"""
from pathlib import Path

import plot_style
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

INK, GRAY, LIGHT = "#222222", "#8a8a8a", "#d4d4d4"
ACC, GYRO = "#a9c8ec", "#b7e0c2"   # 파스텔: total_acc, body_gyro
ACCENT = "#36454f"                 # 강조 선: 차콜
X_IN, X_HID, X_OUT, X_SM = 2.2, 5.6, 8.6, 11.4
R_IN, R = 0.11, 0.27

# 입력: 시점 묶음 3개 (t0, t1, t127) x 채널 6개
CH_LABELS = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]
GROUPS = [("$t_0$", 6.75), ("$t_1$", 4.95), ("$t_{127}$", 2.25)]   # (라벨, 묶음 맨 위 y)
STEP = 0.27
inputs = [(X_IN, top - c * STEP, c) for _, top in GROUPS for c in range(6)]
hidden = [(X_HID, y) for y in (6.1, 5.1, 4.1, 1.7)]
outputs = [(X_OUT, 6.1 - k * 0.88) for k in range(6)]
ACTIVITIES = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]

fig, ax = plt.subplots(figsize=(11.5, 8.2))
ax.set_xlim(-0.2, 13.0)
ax.set_ylim(-3.25, 8.4)
ax.set_aspect("equal")
ax.axis("off")

# 연결선 (모든 입력-은닉, 은닉-출력)
for xi, yi, _ in inputs:
    for xh, yh in hidden:
        ax.plot([xi + R_IN, xh - R], [yi, yh], color=LIGHT, lw=0.4, zorder=1)
for xh, yh in hidden:
    for xo, yo in outputs:
        ax.plot([xh + R, xo - R], [yh, yo], color=LIGHT, lw=0.6, zorder=1)

# 강조 선: W1[0, 0] (t0 acc x -> h1), W2[0, 0] (h1 -> Walking)
(xi, yi, _), (xh, yh), (xo, yo) = inputs[0], hidden[0], outputs[0]
ax.plot([xi + R_IN, xh - R], [yi, yh], color=ACCENT, lw=2.6, zorder=2, solid_capstyle="round")
ax.plot([xh + R, xo - R], [yh, yo], color=ACCENT, lw=2.6, zorder=2, solid_capstyle="round")
ax.text((xi + xh) / 2, (yi + yh) / 2 + 0.28, "$W_1[0,0]$", ha="center", fontsize=11, color=INK,
        bbox=dict(fc="white", ec="none", pad=1.5), zorder=4)
ax.text((xh + xo) / 2, (yh + yo) / 2 + 0.28, "$W_2[0,0]$", ha="center", fontsize=11, color=INK,
        bbox=dict(fc="white", ec="none", pad=1.5), zorder=4)

# 입력 노드
for xi, yi, c in inputs:
    ax.add_patch(Circle((xi, yi), R_IN, fc=ACC if c < 3 else GYRO, ec=INK, lw=0.7, zorder=3))
for c, label in enumerate(CH_LABELS):   # 채널 이름은 t0 묶음에만
    ax.text(X_IN - 0.25, GROUPS[0][1] - c * STEP, label, ha="right", va="center", fontsize=8.5, color=GRAY)
for label, top in GROUPS:               # 시점 묶음 괄호
    y0, y1 = top + 0.12, top - 5 * STEP - 0.12
    ax.plot([0.75, 0.65, 0.65, 0.75], [y0, y0, y1, y1], color=INK, lw=0.8)
    ax.text(0.5, (y0 + y1) / 2, label, ha="right", va="center", fontsize=12, color=INK)
ax.text(X_IN, 2.93, "$\\vdots$", ha="center", va="center", fontsize=16, color=INK)

# 은닉 노드
for (xh, yh), label in zip(hidden, ["$h_1$", "$h_2$", "$h_3$", "$h_{100}$"]):
    ax.add_patch(Circle((xh, yh), R, fc="white", ec=INK, lw=0.9, zorder=3))
    ax.text(xh, yh, label, ha="center", va="center", fontsize=9.5, zorder=4)
ax.text(X_HID, 2.95, "$\\vdots$", ha="center", va="center", fontsize=16, color=INK)

# 출력 노드
for k, ((xo, yo), name) in enumerate(zip(outputs, ACTIVITIES)):
    ax.add_patch(Circle((xo, yo), R, fc="white", ec=INK, lw=0.9, zorder=3))
    ax.text(xo, yo, str(k), ha="center", va="center", fontsize=10, zorder=4)
    ax.text(xo + 0.38, yo, name, ha="left", va="center", fontsize=10.5, color=INK)
    ax.annotate("", xy=(X_SM - 0.55, yo), xytext=(xo + 1.65, yo),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.6, mutation_scale=8))

# SoftmaxWithLoss 상자
ax.add_patch(FancyBboxPatch((X_SM - 0.5, 1.35), 1.0, 5.1, boxstyle="round,pad=0.02,rounding_size=0.12",
                            fc="white", ec=INK, lw=0.9, zorder=3))
ax.text(X_SM, 3.9, "softmax\n\n$\\downarrow$\n\ncross-\nentropy", ha="center", va="center", fontsize=10.5, zorder=4)
ax.annotate("", xy=(X_SM + 1.15, 3.9), xytext=(X_SM + 0.52, 3.9),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=10))
ax.text(X_SM + 1.25, 3.9, "$L$", ha="left", va="center", fontsize=13)

# 위: 계층 이름 (model.py의 이름 그대로)
TOP = 7.75
for x, name, formula in [((X_IN + X_HID) / 2, "Affine1", "$XW_1 + b_1$"),
                         (X_HID, "Relu1", "$\\max(0, \\cdot)$"),
                         ((X_HID + X_OUT) / 2 + 0.2, "Affine2", "$HW_2 + b_2$"),
                         (X_SM, "SoftmaxWithLoss", "lastLayer")]:
    ax.text(x, TOP + 0.3, name, ha="center", va="center", fontsize=12.5, fontweight="bold", color=INK)
    ax.text(x, TOP - 0.15, formula, ha="center", va="center", fontsize=11, color=GRAY)

# 아래: 행렬 모양
SHAPE_Y = 0.55
for x, txt in [(X_IN, "$X\\ (N, 768)$\n$768 = 128 \\times 6$"), ((X_IN + X_HID) / 2, "$W_1\\ (768, 100)$\n$b_1\\ (100,)$"),
               (X_HID, "$H\\ (N, 100)$"), ((X_HID + X_OUT) / 2 + 0.2, "$W_2\\ (100, 6)$\n$b_2\\ (6,)$"),
               (X_OUT + 0.6, "scores $(N, 6)$"), (X_SM + 0.3, "$P\\ (N, 6)$, $L$")]:
    ax.text(x, SHAPE_Y, txt, ha="center", va="top", fontsize=10.5, color=INK, linespacing=1.4)

# 맨 아래: 메서드가 계산하는 범위
for y, x_end, label in [(-0.95, X_OUT + 1.4, "predict(x): Affine1 → Relu1 → Affine2  (prediction = argmax of scores)"),
                        (-1.6, X_SM + 0.5, "loss(x, t): predict(x) → SoftmaxWithLoss")]:
    ax.plot([X_IN - 0.3, x_end], [y, y], color=INK, lw=0.8)
    for xe in (X_IN - 0.3, x_end):
        ax.plot([xe, xe], [y - 0.1, y + 0.1], color=INK, lw=0.8)
    ax.text(X_IN - 0.3, y + 0.17, label, ha="left", va="bottom", fontsize=10.5, color=INK)

# gradient: loss까지 순전파한 뒤, 거꾸로 backward (왼쪽 화살표)
y = -2.25
ax.annotate("", xy=(X_IN - 0.3, y), xytext=(X_SM + 0.5, y),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=10))
ax.plot([X_SM + 0.5, X_SM + 0.5], [y - 0.1, y + 0.1], color=INK, lw=0.8)
ax.text(X_IN - 0.3, y + 0.17,
        "gradient(x, t): loss(x, t), then backward from SoftmaxWithLoss to Affine1  →  $dW_1, db_1, dW_2, db_2$",
        ha="left", va="bottom", fontsize=10.5, color=INK)

# 범례
lx, ly = 0.1, -2.95
for i, (fc, txt) in enumerate([(ACC, "total_acc (x, y, z)"), (GYRO, "body_gyro (x, y, z)")]):
    ax.add_patch(Circle((lx + 0.12 + i * 2.9, ly), 0.1, fc=fc, ec=INK, lw=0.7))
    ax.text(lx + 0.32 + i * 2.9, ly, txt, va="center", fontsize=10)
ax.plot([lx + 5.75, lx + 6.3], [ly, ly], color=ACCENT, lw=2.6, solid_capstyle="round")
ax.text(lx + 6.45, ly, "one line = one weight", va="center", fontsize=10)

out = Path(__file__).resolve().parent / "figures" / "network.png"
fig.savefig(out)
print(f"saved {out}")
