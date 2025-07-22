# model.py
import casadi as ca
import numpy as np

nx = 4
nu = 1

# 状態・入力シンボル
x = ca.MX.sym("x", nx)
u = ca.MX.sym("u", nu)

# パラメータ
m = 1.0
M = 5.0
l = 2.0
g = 9.81
b = 0.1

# 状態展開
pos, theta, dpos, dtheta = x[0], x[1], x[2], x[3]

# 動力学モデル
ddtheta = (g * ca.sin(theta) - b * dtheta + u[0]) / l
ddpos = (u[0] - b * dpos - m * l * ddtheta * ca.cos(theta)) / (M + m)
xdot = ca.vertcat(dpos, dtheta, ddpos, ddtheta)

# CasADi関数
f_continuous = ca.Function("f", [x, u], [xdot])


# RK4 離散化器
def rk4_integrator(f, dt):
    def step(xk, uk):
        k1 = f(xk, uk)
        k2 = f(xk + dt / 2 * k1, uk)
        k3 = f(xk + dt / 2 * k2, uk)
        k4 = f(xk + dt * k3, uk)
        return xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return step


# 離散時間関数
def get_discrete_dynamics(dt=0.1):
    return rk4_integrator(f_continuous, dt)
