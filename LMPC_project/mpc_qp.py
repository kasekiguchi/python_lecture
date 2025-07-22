# mpc.py
import numpy as np
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix

def build_mpc_qp(A, B, N, Q, R, x_ref, x0, umin, umax):
    nx, nu = A.shape[0], B.shape[1]

    H = np.kron(np.eye(N), R)  # 入力のコスト行列

    Gamma = np.zeros((N * nx, N * nu)) 
    Phi = np.zeros((N * nx, nx))  

    A_power = A

    # 状態遷移行列と入力遷移行列の構築
    for i in range(N):
        Phi[i * nx : (i + 1) * nx, :] = A_power
        for j in range(i + 1):
            Gamma[i * nx : (i + 1) * nx, j * nu : (j + 1) * nu] = (
                np.linalg.matrix_power(A, i - j) @ B
            )
        A_power = A @ A_power

    x_ref_stack = np.tile(x_ref.reshape(-1, 1), (N, 1))
    dx_ref = Phi @ x0.reshape(-1,1) - x_ref_stack
    f = (Gamma.T @ Q @ dx_ref)
    H_qp = Gamma.T @ Q @ Gamma + H

    G = np.vstack([np.eye(N * nu), -np.eye(N * nu)])
    h = np.hstack([np.tile(umax, N), -np.tile(umin, N)])

    return csc_matrix(H_qp), f, csc_matrix(G), h
