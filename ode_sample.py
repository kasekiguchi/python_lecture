import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from plot_sol import plot_sol 

# 行列 A, ベクトル b
A = np.array([[-6, 18],
              [-2, 6]])
b = np.array([0, 1])

# 外部入力
def u(t):
    return np.sin(t)*0

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

plot_sol(sol)