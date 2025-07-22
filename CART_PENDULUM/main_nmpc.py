import numpy as np
from model_casadi import get_discrete_dynamics
from mpc_casadi import build_mpc
from plot_result import plot_result
import casadi as ca
from CART_PENDULUM import CART_PENDULUM
from linear_cart_pendulum_model import Ac_CartPendulum, Bc_CartPendulum  # 別ファイルでモデルを定義
from scipy import linalg
from scipy.signal import cont2discrete, ss2tf

# simulation parameters
dt = 0.01
te =3
tspan = np.arange(0, te+dt, dt)

# control target
x0 = np.array([1,0,0,0])
cart = CART_PENDULUM(x0)
# param = cart.param
# cart = CART_PENDULUM(x0,plant_param=param,sys_noise=0.0, measure_noise=np.array([0.0, 0.0]), dead_zone=0.0)

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

# MPC初期化

# Reference
x_ref = np.array([0.0, 0.0, 0.0, 0.0]) # target state

Q = np.diag([10.0, 100.0, 1.0, 1.0])   # state cost
R = np.diag([0.001])         # input cost
f_disc = get_discrete_dynamics(param,dt)
opti, X0, Xopt, Uopt = build_mpc(f_disc, N=20, dt=dt,Q=Q,R=R,x_ref=x_ref)

# %% 
# EKF parameters
P = np.eye(4)
Qd = np.diag([1.0,1.0,1.0,100.0])
Rd = 0.01*np.diag([0.02,0.05])

#
x = ca.DM(x0)

y = cart.measure()
xh = np.concatenate([y, [0,0]])
u = 0
# ログ用
T = [0]
X_log = [x0]
PX_log = [x0]
Y_log = [y]
U_log = []


for i in range(len(tspan)-1):
    xh_pre = Ad @ xh + Bd.flatten() * u
    P_pre = Ad @ P @ Ad.T + Bd @ Bd.T @ Qd
    Gd = P_pre @ Cd.T @ linalg.inv(Cd @ P_pre @ Cd.T + Rd)
    P = (np.eye(4) - Gd @ Cd) @ P_pre
    y = cart.measure()
    xh = xh_pre + Gd @ (y - Cd @ xh_pre)

    opti.set_value(X0, ca.DM(xh))

    X_init = np.tile(xh.reshape(-1,1), (1, 21))  # NumPy でタイル
    opti.set_initial(Xopt, X_init)
    opti.set_initial(Uopt, np.zeros((1, 20)))

    try:
        sol = opti.solve()
    except:
        print("MPC solver failed at time step", tspan[i])
        break

    u = sol.value(Uopt[:, 0])
    print(u,xh)
    cart.apply_input(u, dt)

    T.append(T[-1] + dt)
    X_log.append(xh)
    U_log.append(u)

    Y_log.extend([y, y])
    PX_log.append(cart.state.copy())

T = np.array(T)
X = np.array(X_log)
Y = np.array(Y_log)
U = np.array(U_log)
PX = np.array(PX_log)

# Plot
plot_result(T, X, U)
