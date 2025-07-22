import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.signal import cont2discrete, ss2tf
from control import lqr, dlqr
from CART_PENDULUM import CART_PENDULUM
from linear_cart_pendulum_model import Ac_CartPendulum, Bc_CartPendulum  # 別ファイルでモデルを定義
from scipy.integrate import solve_ivp
from qpsolvers import solve_qp
from mpc_qp import build_mpc_qp

# simulation parameters
dt = 0.01
te = 10
tspan = np.arange(0, te+dt, dt)

# control target
init = np.array([1,0,0,0])
cart = CART_PENDULUM(init)
param = cart.param
cart = CART_PENDULUM(init,plant_param=param,sys_noise=0.0, measure_noise=np.array([0.0, 0.0]), dead_zone=0.0)

# get nominal parameter
param = cart.param
# get system matrices
Ac = np.array(Ac_CartPendulum(param))
Bc = np.array(Bc_CartPendulum(param))
Cc = np.array([[1,0,0,0],[0,1,0,0]])
Dc = np.array([[0],[0]])

# discretize
Ad, Bd, Cd, Dd, dt = cont2discrete((Ac, Bc, Cc, Dc), dt)

# ========== MPC problem ==========
N = 20             # prediction horizon
nx = 4             # state: x, theta, dx, dtheta
nu = 1             # control: f

Q = np.diag([100.0, 1000.0, 1.0, 1.0])   # state cost
R = np.diag([0.001])         # input cost

# Reference
x_ref = np.array([0.0, 0.0, 0.0, 0.0]) # target state

# Constraints on inputs
f_max = 1.0



# %% 
# EKF parameters
P = np.eye(4)
Qd = np.diag([1.0,1.0,1.0,100.0])
Rd = 0.01*np.diag([0.02,0.05])

# logging
T = []
Y = []
X = []
U = []
PX = []

y = cart.measure()
xh = np.concatenate([y, [0,0]])
u = 0

for i in range(len(tspan)-1):
    xh_pre = Ad @ xh + Bd.flatten() * u
    P_pre = Ad @ P @ Ad.T + Bd @ Bd.T @ Qd
    Gd = P_pre @ Cd.T @ linalg.inv(Cd @ P_pre @ Cd.T + Rd)
    P = (np.eye(4) - Gd @ Cd) @ P_pre
    y = cart.measure()
    xh = xh_pre + Gd @ (y - Cd @ xh_pre)


    # QP構築
    H, f, G, h = build_mpc_qp(Ac, Bc, N, Q, R, x_ref, xh, -10, 10)

    # 解く
    u_seq = solve_qp(H, f, G, h, solver="osqp")
    if u_seq is None:
        print(f"QP solve failed at step {t}")
        break
    u = u_seq[0:nu]

    cart.apply_input(u, dt)

    T.extend([tspan[i], tspan[i+1]])
    X.extend([xh, xh])
    Y.extend([y, y])
    U.extend([u, u])
    PX.append(cart.state.copy())

T = np.array(T)
X = np.array(X)
Y = np.array(Y)
U = np.array(U)
PX = np.array(PX)

# plot
plt.figure()
plt.subplot(2,1,1)
plt.plot(T,Y)
plt.ylabel("y=[p;th]")
plt.xlim(0, te)

plt.subplot(2,1,2)
plt.plot(T,U)
plt.ylabel("u")
plt.xlabel("time [s]")
plt.xlim(0, te)

plt.show()

