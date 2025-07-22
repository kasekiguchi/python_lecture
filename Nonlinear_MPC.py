import numpy as np
import matplotlib.pyplot as plt
from casadi import *

# 時間設定
dt = 0.1
N = 20  # ホライズン長

# 状態と入力の定義
x = MX.sym("x", 2)  # [x1, x2]
u = MX.sym("u")  # スカラー入力

# 非線形力学モデル
xdot = vertcat(x[1], u - x[0] ** 2)

# 離散化（Euler法）
x_next = x + dt * xdot
f = Function("f", [x, u], [x_next])

# 最適化問題のセットアップ
opti = Opti()

X = opti.variable(2, N + 1)
U = opti.variable(1, N)
x0 = opti.parameter(2)  # 初期状態
xref = np.array([0, 0])

# コスト関数
Q = np.diag([100, 0.1])
R = 0.01

cost = 0
for k in range(N):
    cost += mtimes((X[:, k] - xref).T, Q @ (X[:, k] - xref)) + R * U[:, k] ** 2
opti.minimize(cost)

# 初期状態制約
opti.subject_to(X[:, 0] == x0)

# ダイナミクス制約
for k in range(N):
    opti.subject_to(X[:, k + 1] == f(X[:, k], U[:, k]))

# 入力制限
# opti.subject_to(opti.bounded(-1, U, 1))

# ソルバ設定
opti.solver("ipopt")

# シミュレーション実行
x_current = np.array([2.0, 0.0])
simX = [x_current]
simU = []

for _ in range(5):
    opti.set_value(x0, x_current)
    sol = opti.solve()
    u_opt = sol.value(U[:, 0])
    x_current = f(x_current, u_opt).full().flatten()

    simX.append(x_current)
    simU.append(u_opt)

    # 初期値更新
    # opti.set_initial(X, sol.value(X))
    # opti.set_initial(U, sol.value(U))

# 結果プロット
simX = np.array(simX)
simU = np.array(simU)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(simX)
plt.ylabel("States x1, x2")

plt.subplot(2, 1, 2)
plt.plot(simU)
plt.ylabel("Input u")
plt.xlabel("Time step")
plt.tight_layout()
plt.show()
