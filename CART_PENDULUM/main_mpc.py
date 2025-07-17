import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.signal import cont2discrete, ss2tf
from control import lqr, dlqr
from CART_PENDULUM import CART_PENDULUM
from linear_cart_pendulum_model import Ac_CartPendulum, Bc_CartPendulum  # 別ファイルでモデルを定義

# simulation parameters
dt = 0.3
te = 10
tspan = np.arange(0, te+dt, dt)

# control target
init = np.array([1,0,0,0])
# cart = CART_PENDULUM(init)
cart = CART_PENDULUM(init,sys_noise=0.0, measure_noise=np.array([0.0, 0.0]), dead_zone=0.0)

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

Q = np.diag([1.0, 100.0, 1.0, 1.0])   # state cost
R = np.diag([1.0])         # input cost

# Reference
x_ref = np.array([0.0, 0.0, 0.0, 0.0]) # target state

# Instanse of CasADi Optimization problem
opti = ca.Opti()

# Decision variables
Xopt = opti.variable(nx, N+1)
Uopt = opti.variable(nu, N)

# Parameter for initial condition
X0 = opti.parameter(nx)

model_ode =lambda x, u: (Ac @ x + Bc.flatten() * u)

# RK4 integrator for dynamics
def rk4_integrator(model, x, u, dt):
    k1 = model(x, u)
    k2 = model(x + dt/2 * k1, u)
    k3 = model(x + dt/2 * k2, u)
    k4 = model(x + dt * k3, u)
    x_next = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return x_next

# Dynamics constraints with RK4
for k in range(N):
    xk = Xopt[:, k]
    uk = Uopt[:, k]
    x_next = Xopt[:, k+1]
    xk1 = rk4_integrator(model_ode, xk, uk, dt)
    opti.subject_to(x_next == xk1)

# x <= 3 の制約を追加
# for k in range(N+1):
#     opti.subject_to(Xopt[0, k] <= 3)

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
    xh_pre = Ad @ xh + Bd.flatten() * u
    P_pre = Ad @ P @ Ad.T + Bd @ Bd.T * Qd
    Gd = P_pre @ Cd.T @ linalg.inv(Cd @ P_pre @ Cd.T + Rd)
    P = (np.eye(4) - Gd @ Cd) @ P_pre
    y = cart.measure()
    xh = xh_pre + Gd @ (y - Cd @ xh_pre)

    opti.set_value(X0, xh)
    opti.set_initial(Xopt, np.tile(xh.reshape(-1, 1), (1,N+1)))
    opti.set_initial(Uopt, np.zeros((nu,N)))

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
