# mpc.py
import casadi as ca
import numpy as np


def build_mpc(f_disc, N=20, dt=0.1, Q=None, R=None, x_ref=None):
    opti = ca.Opti()

    nx = 4
    nu = 1
    X = opti.variable(nx, N + 1)
    U = opti.variable(nu, N)
    X0 = opti.parameter(nx)

    if Q is None:
        Q = np.diag([100, 100, 1, 1])
    if R is None:
        R = np.diag([0.01])
    if x_ref is None:
        x_ref = np.zeros(nx)

    # Dynamics
    opti.subject_to(X[:, 0] == X0)
    for k in range(N):
        x_next = f_disc(X[:, k], U[:, k])
        opti.subject_to(X[:, k + 1] == x_next)

    # Cost
    cost = 0
    for k in range(N):
        dx = X[:, k] - x_ref
        du = U[:, k]
        cost += ca.mtimes([dx.T, Q, dx]) + ca.mtimes([du.T, R, du])
    dx = X[:, N] - x_ref
    cost += ca.mtimes([dx.T, Q, dx])

    opti.minimize(cost)

    # Input bounds
    # u_max = 10.0
    # opti.subject_to(opti.bounded(-u_max, U, u_max))

    opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0})

    return opti, X0, X, U
