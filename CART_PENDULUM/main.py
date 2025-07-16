import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.signal import cont2discrete, ss2tf
from control import lqr, dlqr
from CART_PENDULUM import CART_PENDULUM
from linear_cart_pendulum_model import Ac_CartPendulum, Bc_CartPendulum  # 別ファイルでモデルを定義

# flags
flag = {
    "time": "discrete",  # or "continuous"
    "estimator": "EKF",  # or "observer"
}

# simulation parameters
dt = 0.3
te = 30
tspan = np.arange(0, te+dt, dt)

# control target
init = np.array([1,0,0,0])
cart = CART_PENDULUM(init)

# get nominal parameter
param = cart.param
print("nominal parameter:", param)
# get system matrices
Ac = np.array(Ac_CartPendulum(param))
Bc = np.array(Bc_CartPendulum(param))
Cc = np.array([[1,0,0,0],[0,1,0,0]])
Dc = np.array([[0],[0]])

# discretize
Ad, Bd, Cd, Dd, dt = cont2discrete((Ac, Bc, Cc, Dc), dt)

# design controllers
Fd, _, Ed = dlqr(Ad, Bd, np.diag([1,100,1,1]), np.array([[1]]))
Fod, _, Edo = dlqr(Ad.T, Cd.T, np.diag([1,100,1,1]), 0.1*np.eye(2))
Fod = Fod.T
print(np.linalg.eig(Ad-Bd*Fd))
print(np.abs(Edo))
Fc, _, _ = lqr(Ac, Bc, np.diag([1,100,1,1]), np.array([[1]]))
Foc, _, _ = lqr(Ac.T, Cc.T, np.diag([1,100,1,1]), 0.1*np.eye(2))
Foc = Foc.T
# %% 
# EKF parameters
if flag["estimator"] == "EKF":
    P = np.eye(4)
    Qd = 1
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
    if flag["time"] == "discrete":
        if flag["estimator"] == "EKF":
            xh_pre = Ad @ xh + Bd.flatten() * u
            P_pre = Ad @ P @ Ad.T + Bd @ Bd.T * Qd
            Gd = P_pre @ Cd.T @ linalg.inv(Cd @ P_pre @ Cd.T + Rd)
            P = (np.eye(4) - Gd @ Cd) @ P_pre
            y = cart.measure()
            xh = xh_pre + Gd @ (y - Cd @ xh_pre)
        else:
            xh = Ad @ xh + Bd.flatten() * u - Fod @ (Cd @ xh - y)
            y = cart.measure()
    else:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(lambda t, x: (Ac @ x + Bc.flatten() * u - Foc @ (Cc @ xh - y)),
                        [tspan[i], tspan[i+1]], xh)
        xh = sol.y[:, -1]
        y = cart.measure()

    if flag["time"] == "discrete":
        u = -Fd @ xh
    else:
        u = -Fc @ xh

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

plt.subplot(2,1,2)
plt.plot(T,U)
plt.ylabel("u")
plt.xlabel("time [s]")
plt.show()
