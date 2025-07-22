# main.py
import numpy as np
from model_casadi import get_discrete_dynamics, nx
from mpc_casadi import build_mpc
from plot_result import plot_result
import casadi as ca

# 時間設定
dt = 0.1
T_sim = 5.0
steps = int(T_sim / dt)

# 初期状態
x0 = np.array([0.5, 0.2, 0.0, 0.0])
x = x0.copy()

# MPC初期化
f_disc = get_discrete_dynamics(dt)
opti, X0, Xopt, Uopt = build_mpc(f_disc, N=20, dt=dt)

# ログ用
T = [0]
X_log = [x0]
U_log = []
x = ca.DM(x)
for t in range(steps):
    opti.set_value(X0, x)

    x_np = np.array(x.full()).reshape(-1, 1)  # CasADi → NumPy
    # x_np = np.arrayx.reshape(-1, 1)
    X_init = np.tile(x_np, (1, 21))  # NumPy でタイル
    opti.set_initial(Xopt, X_init)
    opti.set_initial(Uopt, np.zeros((1, 20)))

    try:
        sol = opti.solve()
    except:
        print("MPC solver failed at time step", t)
        break

    u = sol.value(Uopt[:, 0])
    x = f_disc(x, u)

    T.append(T[-1] + dt)
    X_log.append(x.full().flatten())
    U_log.append(u)

X_log = np.array(X_log)
U_log = np.array(U_log)

# Plot
plot_result(T, X_log, U_log)
