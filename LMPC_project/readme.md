# Linear MPC

mpc_qp.py : OSQP 用 QP問題（min 1/2 u^T H u + f^T u s.t. G u <= h）を構築
mpc_qp_xu.py  : OSQP 用 QP問題（min 1/2 [[x],[u]]^T Q [[x],[u]] s.t. Aqp [[x],[u]] == A0 x0,   G u <= h）を構築

## Prepare

cd linear_mpc_project
pip install cvxopt matplotlib numpy
pip install "qpsolvers[open_source_solvers]"

## Run

python main.py
