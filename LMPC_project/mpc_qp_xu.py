# mpc.py
import numpy as np
from qpsolvers import solve_qp
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix

def build_mpc_qp(A, B,dt,N, Q, R, x_ref, x0, umin, umax, xmin, xmax):
    nx, nu = A.shape[0], B.shape[1]

    bQ = np.kron(np.eye(N), Q)
    bR = np.kron(np.eye(N), R)
    bA = np.block(
        [[np.zeros((nx, N * nx))], [np.kron(np.eye(N-1), A), np.zeros(((N - 1) * nx, nx))]]
    )
    bB = np.kron(np.eye(N), B)
    A0x0 = np.block([[A @ x0.reshape(-1,1)], [np.zeros(((N - 1) * nx, 1))]])

    G = np.hstack([np.zeros((2*N*nu,N*nx)),np.vstack([np.eye(N * nu), -np.eye(N * nu)])])
    h = np.hstack([np.tile(umax, N), -np.tile(umin, N)])
    # Xr = np.tile(x_ref,N)
    Xr = np.hstack([np.hstack([x0[0] + x_ref[2]*dt*(i+1),x_ref[1:]]) for i in range(N)])
    Ur = np.zeros(nu*N)
    return (
        csc_matrix(block_diag(bQ, bR)),
        # np.zeros((nx * N + nu * N, 1)),
        -block_diag(bQ, bR).T @ np.hstack([Xr,Ur]),
        csc_matrix(G),
        h,
        csc_matrix(np.block([np.eye(bA.shape[0]) - bA, -bB])),
        A0x0,
        np.hstack([np.tile(xmin, N),np.tile(umin, N)]),
        np.hstack([np.tile(xmax, N),np.tile(umax, N)])
    )
