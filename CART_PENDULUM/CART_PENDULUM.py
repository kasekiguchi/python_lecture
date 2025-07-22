import numpy as np
from scipy.integrate import solve_ivp

class CART_PENDULUM:
    """
    cart pendulum model (python version)
    """

    def __init__(self, initial, 
                 sys_noise=0.0001,
                 measure_noise=np.array([0.005, 2*np.pi/(2*360)]),
                 param=np.array([0.1, 0.7, 0.01, 0.3, 0.2, 0.002, 9.81, 3]),
                 plant_param=np.array([0.125, 0.740, 9.1e-3, 0.354, 0.26, 1.98e-3, 9.813, 3.61]),
                 dead_zone=0.01):
        """
        [input]
            initial : initial state [p, th, dp, dth]
        """
        self.sys_noise = sys_noise
        self.measure_noise = measure_noise
        self.param = param
        self.plant_param = plant_param
        self.dead_zone = dead_zone
        self.state = initial
        self.t = 0.0

        self.TT = np.array([0.0])
        self.XX = np.array([initial])
        self.output = None
        self.h = lambda x: np.array([x[0], x[1]])  # output function y = h(x)

    def apply_input(self, u, dt):
        """
        Apply the input u to the system for duration dt
        """
        if abs(u) < self.dead_zone:
            u = 0.0
        u = u + self.sys_noise * np.random.randn() / dt

        sol = solve_ivp(
            lambda t, x: self.ode(x, u, self.plant_param),
            [self.t, self.t + dt],
            self.state,
            method='RK45', t_eval=[self.t + dt]
        )
        self.state = sol.y[:, -1]
        self.t = sol.t[-1]

        self.TT = np.append(self.TT, sol.t[-1])
        self.XX = np.vstack([self.XX, self.state])

    def measure(self, t=None):
        """
        Return noisy measurement [p, th]
        """
        if t is None:
            t = self.t
        idx = np.argmin(np.abs(self.TT - t))
        self.state = self.XX[idx]
        self.output = self.h(self.state)
        return self.output + self.measure_noise * np.random.randn(2)

    def ode(self, x, input, p):
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
        dxdt = np.zeros(4)
        # dxdt = x
        dxdt[0] = dp
        dxdt[1] = dth
        dxdt[2] = -t15 * (Dp * J * dp + gravity * t2 * t3 * t11 - lg * t3 * t4 * t5
                          - lg**3 * t3 * t5 * t7 + Dp * dp * m * t6 - Dth * dth * lg * m * t2) \
                  + a * input * t15 * (J + m * t6)
        dxdt[3] = -t15 * (Dth * M * dth + Dth * dth * m - gravity * lg * t3 * t7
                          + t2 * t3 * t5 * t11 - Dp * dp * lg * m * t2 - M * gravity * lg * m * t3) \
                  - a * input * lg * m * t2 * t15

        return dxdt
