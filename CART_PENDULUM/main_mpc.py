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

def ode(x, input, p):
    """
    ODE of cart pendulum
    """
    Dp, Dth, J, M, a = p[4], p[5], p[2], p[1], p[7]
    dp, dth = x[2], x[3]
    gravity, lg, m, th = p[6], p[3], p[0], x[1]

    t2 = np.cos(th)
    t3 = np.sin(th)
    t4 = J * m
    t5 = dth**2
    t6 = lg**2
    t7 = m**2
    t8 = J * M
    t9 = t2**2
    t10 = M * m * t6
    t11 = t6 * t7
    t12 = t9 * t11
    t13 = -t12
    t14 = t4 + t8 + t10 + t11 + t13
    t15 = 1.0 / t14
    dxdt = x
    dxdt[0] = dp
    dxdt[1] = dth
    dxdt[2] = -t15 * (Dp * J * dp + gravity * t2 * t3 * t11 - lg * t3 * t4 * t5
                        - lg**3 * t3 * t5 * t7 + Dp * dp * m * t6 - Dth * dth * lg * m * t2) \
                + a * input * t15 * (J + m * t6)
    dxdt[3] = -t15 * (Dth * M * dth + Dth * dth * m - gravity * lg * t3 * t7
                        + t2 * t3 * t5 * t11 - Dp * dp * lg * m * t2 - M * gravity * lg * m * t3) \
                - a * input * lg * m * t2 * t15

    return dxdt

# simulation parameters
dt = 0.01
te = 3
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

Q = np.diag([1000.0, 100.0, 1.0, 1.0])   # state cost
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
    # xk1 = rk4_integrator(model_ode, xk, uk, dt)    
    # xk1 = rk4_integrator(model_ode, xk, uk, dt)    
    # xk1 = xk + dt*(cart.ode(xk,uk,param))
    xk1 = xk + dt*(model_ode(xk,uk))
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

