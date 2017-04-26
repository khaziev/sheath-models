import numpy as np
from scipy.integrate import ode


class Zimmerman:

    def __init__(self, omega, alpha_deg, tau, delta):

        self.omega = omega
        self.delta = delta
        self.tau = tau
        self.alpha_def = alpha_deg

        self.alpha = np.pi *alpha_deg/180

        self.omega_tau = tau*omega

        self.__initial_step__()




    def __gradient__(self, t, x):
        '''

        :param t: float, dummy variable require for ode integration, distance
        :param x: sequence of 4 floats, [vx, vy, vz, phi]
        :param params:dictionary of plasma parameters
        :return:
        '''
        result = np.zeros(x.shape)

        alpha = self.alpha
        delta = self.delta

        omega_tau = self.omega_tau

        # setting up v_x derivative
        result[0] = (omega_tau * np.cos(alpha) * x[0] * x[2] + delta + x[0] ** 2) / (1 - x[0] ** 2)

        # v_y derivative
        result[1] = (omega_tau * np.sin(alpha) * x[2] - x[1]) / x[0]

        # v_Z derivative
        result[2] = omega_tau * np.cos(alpha) - omega_tau * np.sin(alpha) * x[1] / x[0] - x[2] / x[0]

        # potential derivative
        result[3] = (result[0] - delta) / x[0]

        return result

    def __dynamic_stepping__(self, y, y_prev, dx_prev, dx_min=1e-10, dx_max=10, dy=5e-3):

        '''

        :param y: sequence of 4 floats, [vx, vy, vz, phi], values at x
        :param y_prev: sequence of 4 floats, [vx, vy, vz, phi], values at x-dx
        :param dx_prev: float, dx, value at x-dx
        :param dx_min: float, min value of dx
        :param dx_max: float, max value of dx
        :param dy: float, target change in y
        :return:
        '''
        #grad = np.mean(np.abs(y - y_prev) / dx_prev)
        grad = np.max(np.abs(y - y_prev) / dx_prev)
        dx = dy / grad

        if dx < dx_min:
            dx = dx_min
        elif dx > dx_max:
            dx = dx_max

        dx = np.sqrt(dx_prev * dx)

        return dx

    def __initial_step__(self, ys=[1e-2, 1e-6], deltas=[0, 1]):

        '''
        Zimmerman solver is sensitive for values of y0 for small delta.
        This initial condition provides an empirical values of the initial conditions of the Zimmerman's solver
        :return:
        '''

        if deltas[0] <= self.delta <= deltas[1]:
            y0_log = np.log10(ys[0]) + (np.log10(ys[1]) - np.log10(ys[0])) * (self.delta-deltas[0])/(deltas[1] - deltas[0])
            y0 = 10 **y0_log
        elif deltas[1] < self.delta:
            y0 = ys[1]
        else:
            y0 = ys[0]
            raise Exception('delta is negative')

        self.y0 = y0 *np.ones(4)

        return self.y0

    def get_smoothness(self):

        diff = np.diff(self.y, axis=0, n=1)
        self.smoothness = np.std(diff)
        return self.smoothness

    def solve(self, x0=0, vx_max=1, dx=1e-8, nsteps=10000, nmax=10000):

        integrator = ode(self.__gradient__).set_integrator('dop853', nsteps=nsteps)
        integrator.set_initial_value(self.y0, x0)

        if self.omega_tau >= 100:
            dx = 1/self.omega_tau / 1e4
        else:
            dx = 1e-6


        #print(dx)

        x = [x0]
        y = [self.y0]

        soln = self.y0

        self.steps = [dx]


        while soln[0] <= vx_max and integrator.successful():

            time = integrator.t + dx

            soln = integrator.integrate(time)

            x.append(time)
            y.append(soln)

            dx = self.__dynamic_stepping__(y[-1], y[-2], dx)

            self.steps.append(dx)

        # time = 0
        # n = 0
        # while soln[0] <= vx_max and n < nmax:
        #     time += dx
        #
        #     soln = integrator.integrate(time)
        #
        #     if integrator.successful():
        #
        #         x.append(time)
        #         y.append(soln)
        #
        #         dx = self.__dynamic_stepping__(y[-1], y[-2], dx)
        #
        #         self.steps.append(dx)
        #     else:
        #         integrator = ode(self.__gradient__).set_integrator('dop853', nsteps=nsteps)
        #         integrator.set_initial_value(self.y0, x0)
        #
        #     n += 1
        #
        #     #print(n)
        #
        self.x = np.array(x)
        self.y = np.array(y)

        return x, y

if __name__ == '__main__':

    omega = 1
    tau = 100
    delta = 10
    alpha = 20

    model = Zimmerman(omega, alpha, tau, delta)
    model.solve()

