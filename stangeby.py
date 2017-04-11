import numpy as np
import scipy.constants as const
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


        #setup space for u
        if log_u:
            self.u = np.logspace(1e-3, self.mach_critical, n)
        else:
            self.u = np.linspace(0, self.mach_critical, n)

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

        max_u = self.mach_critical if stangeby else 1
        result = [integrate.fixed_quad(self.__integrand_u__, u, max_u, n=500)[0] for u in self.u]
        return result


if __name__ == '__main__':


    T_e = 1
    T_i = 1
    m_i = 2
    alpha = 2
    model = Stangeby(T_e, T_i, m_i, alpha)
    result = model.get_zeta()
    print(result)
