# model.py
import numpy as np

# システム次元
nx = 4
nu = 2


# 離散時間線形モデル: x[k+1] = A x[k] + B u[k]
def get_linear_model(dt=0.1):
    A = np.eye(nx) + dt * np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    B = dt * np.array([[0,0], [0,0], [1,0], [0,1]])
    return A, B
