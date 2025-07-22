# mpc.py
import numpy as np
from qpsolvers import solve_qp
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix

def build_mpc_qp(A, B, N, Q, R, x_ref, x0, umin, umax):
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

    return (
        csc_matrix(block_diag(bQ, bR)),
        np.zeros((nx * N + nu * N, 1)),
        csc_matrix(G),
        h,
        csc_matrix(np.block([np.eye(bA.shape[0]) - bA, -bB])),
        A0x0,
    )
