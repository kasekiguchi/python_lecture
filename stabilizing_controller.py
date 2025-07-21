import numpy as np
from control import place
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from plot_sol import plot_sol

B = np.array([[0], [1]])

A = np.array([[0, 1], [-6, 5]])
desired_poles = np.array([-1 + 1.0j, -1 - 1.0j])
F = place(A, B, desired_poles)
E, V = np.linalg.eig(A - B @ F)
print("Quiz 1(2) Eigenvalues:", E)


A = np.array([[0, 1], [-1, 0]])
desired_poles = np.array([-1, -2])
F = place(A, B, desired_poles)
E, V = np.linalg.eig(A - B @ F)
print("(3) Eigenvalues:", E)

A = np.array([[-6, 18], [-2, 6]])
desired_poles = np.array([-0.1, -0.2])
F = place(A, B, desired_poles)
E, V = np.linalg.eig(A - B @ F)
print("(4) Eigenvalues:", E)

# %%
# シミュレーション時間
t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)

B = np.array([[0], [1]])
A = np.array([[0, 1], [-6, 5]])
x0 = np.array([1, 0])
# 数値積分
sol = solve_ivp(lambda t, x: A @ x, t_span, x0, t_eval=t_eval)
plot_sol(sol)

# (1) Faster convergence
desired_poles = np.array([-9, -8])
F = place(A, B, desired_poles)
E, V = np.linalg.eig(A - B @ F)
print("Quiz 1(1) Eigenvalues:", E)

# 数値積分
u = lambda x: -F @ x
sol = solve_ivp(lambda t, x: A @ x + B @ u(x), t_span, x0, t_eval=t_eval)
plot_sol(sol)

# (2) Oscillatory response
desired_poles = np.array([-1 + 3.0j, -1 - 3.0j])
F = place(A, B, desired_poles)
E, V = np.linalg.eig(A - B @ F)
print("Quiz 1(2) Eigenvalues:", E)
# 数値積分
u = lambda x: -F @ x
sol = solve_ivp(lambda t, x: A @ x + B @ u(x), t_span, x0, t_eval=t_eval)
plot_sol(sol)

# %% Quiz 2
B = np.array([[1], [2]])
A = np.array([[0, 1], [-2, 3]])
x0 = np.array([1, 0])

desired_poles = np.array([-1, -2])
F = place(A, B, desired_poles)
E, V = np.linalg.eig(A - B @ F)
print("Quiz 2 Eigenvalues:", E)
