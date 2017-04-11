import numpy as np
import scipy.constants as const
from scipy import optimize
from scipy import integrate

class Stangeby:
    #plasma_params = {'T_e': 1., 'T_i': 1., 'm_i': 2e-3 / const.N_A, 'gamma': 1, 'c': 1., 'alpha': np.pi / 180 * 1}

    def __init__(self, T_e, T_i, m_i, alpha, gamma=1, c=1, n = 100, log_u = True):

        '''

        :param T_e: electron temperature in eV
        :param T_i: ion temperature in eV
        :param m_i: ion mass in amu
        :param alpha: angle in degrees
        :param gamma: isothermal constant
        :param c: supersonic criterion
        :param n: integer, number of point in definition of u
        :param log_u: use log space for u
        '''

        #write initial parameters to the class
        self.T_e = T_e
        self.T_i = T_i
        self.m_i_amu = m_i
        self.alpha_deg = alpha
        self.gamma = gamma
        self.c = c

        #initialize drift velocty parameters:
        self.w = None
        self.v = None
        self.u = None
        self.potential = None
        self.density = None

        #calcualte alpha and m_i in physical units
        self.m_i = self.m_i_amu *1e-3/const.N_A
        self.alpha = self.alpha_deg * np.pi/180

        #calculate parameters required for Stangeby's model

        #initial drift velocity
        self.u0 = self.c *np.sin(self.alpha)

        #variable used for calculation of the critical angle and mach number
        temp = 2 * np.pi * const.m_e / self.m_i * (1. + self.T_i / self.T_e)

        # critical angle in degrees and radians
        self.alpha_critical = np.arcsin(np.sqrt(temp))
        self.alpha_critical_deg = self.alpha_critical * 180 /np.pi

        #calculate mach number at the wall
        self.mach_critical = np.sin(self.alpha) / np.sqrt(temp)

        #calculate min value of u
        self.__min_u__()

        #setup space for u
        if log_u:
            self.u = np.logspace(np.log10(self.u_min), np.log10(self.mach_critical), n)
        else:
            self.u = np.linspace(self.u_min, self.mach_critical, n)

        #execute the model
        self.execute()


    def __min_u__(self):
        '''
        Finds minumal resolvable values of u
        :return:
        '''

        self.u_min = 1.001 *optimize.newton(self.__func_f__, 0.1)



    def __func_f__(self, u):
        '''

        Energy balance functional used to find U as function of Zeta
        ---------------------------------------------

        :param u: float
        :return:
        '''

        # find and calculte the result of te function
        w = (self.c + 1. / self.c) / np.cos(self.alpha) - np.tan(self.alpha) * (u + 1. / u)
        return self.c ** 2 + 2. * np.log(u / self.u0) - u ** 2 - w ** 2

    def __integrand_u__(self, u):
        '''
        Integrand of the integral equation connectin U and Zeta
        -------------------------------------------

        parameters:
        u - float like
        '''

        #get w at the sheath entrance
        w = (self.c + 1. / self.c) / np.cos(self.alpha) - np.tan(self.alpha) * (u + 1. / u)

        #calcalate energy balance function
        energy_functional =  self.c ** 2 + 2. * np.log(u / self.u0) - u ** 2 - w ** 2

        return (1. - u ** 2) / u / np.sqrt(energy_functional)

    def get_zeta(self, stangeby = True):
        '''

        Find values of zeta for a given list of the u values
        -----------------------------------------------------

        :params stangeby: boolean, if True use classical model
        '''

        self.max_u = self.mach_critical if stangeby else 1
        self.zeta = np.array([integrate.fixed_quad(self.__integrand_u__, u, self.max_u, n=500)[0] for u in self.u])

        return self.zeta

    def get_w(self):
        '''

        Finds the values of the drift velocity in ExB planes in the direction parallel to the wall
        -----------------------------------------------------

        u_space - sequence (ex. list, np.array), contains values between 0 and 1
        plasma_params - dictionary like
        '''

        # conversion function of w
        f_w = lambda u: (2. - (u + 1. / u) * np.sin(self.alpha)) / np.cos(self.alpha)
        self.w = np.array([f_w(u) for u in self.u])

        return self.w

    def get_v(self):
        '''

        Finds the values of the drift velocity in the direction of ExB drift
        -----------------------------------------------------
        '''

        # conversion function for v
        f_v = lambda u, w: np.sqrt(2 * np.log(u / np.sin(self.alpha)) + 1 - u ** 2 - w ** 2)

        if self.w is None:
            self.get_w()

        self.v = np.array([f_v(u, w) for u, w in zip(self.u, self.w)])
        return self.v

    def get_potential(self):
        '''

        Calculates plasma potential in physical units
        '''

        # scale for plasma potential
        scale = -self.T_e

        # evaulate potential
        self.potential = [scale * np.log(u) for u in self.u]

        return self.potential

    def get_density(self):
        '''

        Calculates plasma potential in relative units
        '''

        # scale for electrostatic potential
        scale = self.T_e

        if self.potential is None:
            self.get_potential()

        # evaulate potential
        self.density = np.array([np.exp(potential / scale) for potential in self.potential])
        self.density /= self.density[0]

        return self.density

    def execute(self):

        self.get_zeta()
        self.get_v()
        self.get_density()


if __name__ == '__main__':

    #TODO remove
    # test output
    T_e = 1
    T_i = 1
    m_i = 2
    alpha = 2
    model = Stangeby(T_e, T_i, m_i, alpha)
    print(model.zeta)
