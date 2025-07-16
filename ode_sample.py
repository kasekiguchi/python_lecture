import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 行列 A, ベクトル b
A = np.array([[0, 1],
              [-6, -5]])
b = np.array([0, 1])

# 外部入力
def u(t):
    return np.sin(t)

# 微分方程式
def dxdt(t, x):
    return A @ x + b * u(t)

# 初期条件
x0 = [1, 0]

# シミュレーション時間
t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)

# 数値積分
# sol = solve_ivp(dxdt, t_span, x0, t_eval=t_eval)
sol = solve_ivp(lambda t,x : A@x+b*u(t), t_span, x0, t_eval=t_eval)

# プロット
plt.figure(figsize=(10,5))
plt.plot(sol.t, sol.y[0], label='x1(t)')
plt.plot(sol.t, sol.y[1], label='x2(t)')
plt.xlabel('Time t')
plt.ylabel('States')
plt.legend()
plt.grid(True)
plt.title('State Response')
plt.show()
