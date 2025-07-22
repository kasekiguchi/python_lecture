# model.py
import casadi as ca

# RK4 離散化器
def rk4_integrator(f, dt):
    def step(xk, uk):
        k1 = f(xk, uk)
        k2 = f(xk + dt / 2 * k1, uk)
        k3 = f(xk + dt / 2 * k2, uk)
        k4 = f(xk + dt * k3, uk)
        return xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return step


# 離散時間関数
def get_discrete_dynamics(p, dt=0.1):
        
    nx = 4
    nu = 1

    # 状態・入力シンボル
    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)

    # パラメータ
    Dp, Dth, J, M, a = p[4], p[5], p[2], p[1], p[7]
    gravity, lg, m= p[6], p[3], p[0]

    # 状態展開
    th, dp, dth =x[1], x[2], x[3]

    # 動力学モデル
    t2 = ca.cos(th)
    t3 = ca.sin(th)
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
    
    ddpos = -t15 * (Dp * J * dp + gravity * t2 * t3 * t11 - lg * t3 * t4 * t5
                        - lg**3 * t3 * t5 * t7 + Dp * dp * m * t6 - Dth * dth * lg * m * t2) \
                + a * u * t15 * (J + m * t6)
    ddtheta = -t15 * (Dth * M * dth + Dth * dth * m - gravity * lg * t3 * t7
                        + t2 * t3 * t5 * t11 - Dp * dp * lg * m * t2 - M * gravity * lg * m * t3) \
                - a * u * lg * m * t2 * t15


    xdot = ca.vertcat(dp, dth, ddpos, ddtheta)

    # CasADi関数
    f_continuous = ca.Function("f", [x, u], [xdot])
    return rk4_integrator(f_continuous,  dt)
