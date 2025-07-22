# main.py
import numpy as np
from model_qp import get_linear_model, nx, nu

# from mpc_qp import build_mpc_qp
from mpc_qp_xu import build_mpc_qp
from qpsolvers import solve_qp
from plot_result import plot_result
import time
# パラメータ
dt = 0.1
N = 10
T_sim = 5.0
steps = int(T_sim / dt)

Q = np.diag([100, 100, 1, 1])
R = np.diag([0.1,0.1])
x_ref = np.array([0, 0, 0, 0])

umin = np.array([-1.0,-1.0])
umax = np.array([1.0,1.0])

# 離散時間線形モデルの取得
A, B = get_linear_model(dt)
nx, nu = A.shape[0], B.shape[1]

x = np.array([0.5, 0.1, 0.0, 0.0])

X_log = [x]
U_log = []
T = [0]

S = time.time()
for t in range(steps):
    # QP構築
    P, q, G, h, bA, bB = build_mpc_qp(A, B, N, Q, R, x_ref, x, umin, umax)

    # 解く
    u_seq = solve_qp(P, q, G, h, bA, bB, solver="osqp")
    if u_seq is None:
        print(f"QP solve failed at step {t}")
        break

    u = u_seq[N*nx:N*nx+nu] # Extract the first control input from the sequence
    x = A @ x + B @ u
    X_log.append(x)
    U_log.append(u)
    T.append(T[-1] + dt)
E = time.time()
print(f"Total time: {E - S:.2f} seconds")
X_log = np.array(X_log)
U_log = np.array(U_log)
plot_result(T, X_log, U_log)
