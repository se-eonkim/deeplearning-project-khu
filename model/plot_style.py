"""모델 파트 그림 공통 스타일: Times New Roman, 수식은 STIX (Times 계열).

사용: import plot_style  (matplotlib.pyplot보다 먼저 또는 직후, 그림 그리기 전에)
Times New Roman이 없는 컴퓨터에서는 serif 기본 글꼴로 대체된다.
그림 안 글자는 영어로 쓴다 (팀원 컴퓨터에 한글 글꼴이 없을 수 있음).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

ACTIVITY_SHORT = ["Walk", "Up", "Down", "Sit", "Stand", "Lay"]  # data.ACTIVITIES 순서 (0~5)
