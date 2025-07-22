import numpy as np
import casadi as ca
from casadi import MX, vertcat
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.signal import cont2discrete, ss2tf
from control import lqr, dlqr
from CART_PENDULUM import CART_PENDULUM
from linear_cart_pendulum_model import Ac_CartPendulum, Bc_CartPendulum  # 別ファイルでモデルを定義
from scipy.integrate import solve_ivp
from model_casadi import get_discrete_dynamics, nx
from mpc_casadi import build_mpc

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

Q = np.diag([10.0, 100.0, 1.0, 1.0])   # state cost
R = np.diag([0.001])         # input cost

# Reference
x_ref = np.array([0.0, 0.0, 0.0, 0.0]) # target state

# Instanse of CasADi Optimization problem
opti = ca.Opti()

# Decision variables
Xopt = opti.variable(nx, N+1)
Uopt = opti.variable(nu, N)

# Parameter for initial condition
X0 = opti.parameter(nx)

# model_ode =lambda x, u: (Ac @ x + Bc.flatten() * u)
model_ode = lambda x, u: ode(x,u, param)

# Dynamics constraints
for k in range(N):
    xk = Xopt[:, k]
    uk = Uopt[:, k]
    x_next = Xopt[:, k+1]
    xk1 = xk + dt*(model_ode(xk,uk))
    opti.subject_to(x_next == xk1)

# Objective function
cost = 0
for k in range(N):
    state_err = Xopt[:,k] - x_ref.reshape(-1, 1)
    input_use = Uopt[:,k]
    cost += ca.mtimes([state_err.T, Q, state_err]) + ca.mtimes([input_use.T, R, input_use])
state_err = Xopt[:,N] - x_ref.reshape(-1, 1)
cost += ca.mtimes([state_err.T, Q, state_err])

opti.minimize(cost)

# Constraints on inputs
f_max = 1.0
opti.subject_to(opti.bounded(-f_max, Uopt[0,:], f_max))

# Initial condition
opti.subject_to(Xopt[:,0] == X0)

# Solver settings
opts = {"ipopt.print_level":0, "print_time":0}
opti.solver("ipopt", opts)



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

    opti.set_value(X0, xh)
    # opti.set_initial(Xopt, np.tile(xh.reshape(-1, 1), (1,N+1)))
    # opti.set_initial(Uopt, np.zeros((nu,N)))
    if i == 0:
        # 初回は tile で埋める
        opti.set_initial(Xopt, np.tile(xh.reshape(-1, 1), (1, N+1)))
        opti.set_initial(Uopt, np.zeros((nu,N)))
    else:
        # 前回の最適化解を取得してずらす（warm start）
        X_prev = sol.value(Xopt)  # shape: (4, N+1)
        U_prev = sol.value(Uopt)  # shape: (1, N)

        # 状態は1ステップ先をずらして使う（最後はコピー）
        X_warm = np.hstack([X_prev[:,1:], X_prev[:,-1:]])  # shape: (4, N+1)

        # 入力も同様にシフト（最後はコピー or 0）
        U_warm = np.hstack([U_prev[1:], U_prev[-1:]])  # shape: (1, N)

        # セット
        opti.set_initial(Xopt, X_warm)
        opti.set_initial(Uopt, U_warm)
    try:
        sol = opti.solve()
        u = sol.value(Uopt[:,0])
    except RuntimeError:
        print("Solver failed at step", tspan[i])
        break

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

