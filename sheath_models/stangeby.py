import numpy as np
import scipy.constants as const
from scipy import optimize
from scipy import integrate
import pandas as pd

class Stangeby:
    #plasma_params = {'T_e': 1., 'T_i': 1., 'm_i': 2e-3 / const.N_A, 'gamma': 1, 'c': 1., 'alpha': np.pi / 180 * 1}

    def __init__(self, T_e, T_i, m_i, B, alpha=None, gamma=1, c=1, n = 100, log_u = True):

        '''

        :param T_e: electron temperature in eV
        :param T_i: ion temperature in eV
        :param m_i: ion mass in amu
        :param alpha: angle in degrees
        :param gamma: isothermal constant
        :param c: supersonic criterion
        :param n: integer, number of point in definition of u
        :param log_u: use log space for u
        :param B: magnetic field, sequence of size 3
        '''

        #write initial parameters to the class
        self.T_e = T_e
        self.T_i = T_i
        self.m_i_amu = m_i
        self.gamma = gamma
        self.c = c

        if len(B) != 3:
            raise Exception("B's dimensionality is not 3: {0}".format(B))
        else:
            self.B = np.array(B)

            #self scheck for B
            alpha_B = np.arctan2(B[2], B[0]) *180/np.pi

            if alpha is None:
                self.alpha_deg = alpha_B
            elif abs(alpha_B - alpha) > 1e-2:
                raise Exception("Alpha does not agree with B: alpha={0}, alpha_b={1}".format(alpha, alpha_B))
            else:
                self.alpha_deg = alpha

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
        self.potential = np.array([scale * np.log(u) for u in self.u])
        self.potential -= self.potential[-1]

        return self.potential

    def __get_scales__(self):

        #cyclotron frequency vector
        self.omega = self.B *const.e /self.m_i

        #Bohm velocity
        self.acoustic_velocity = np.sqrt(const.e * (self.T_e + self.gamma *self.T_i) /self.m_i)

        #find the scale for zeta -> z
        self.zeta_scale = self.acoustic_velocity / self.omega[0]

        self.v_scale = self.w_scale = self.u_scale = self.acoustic_velocity

    def get_physical_value(self):

        self.__get_scales__()

        #get physical zeta
        self.z = self.zeta * self.zeta_scale

        #get scale velocity
        self.w_physical = self.w *self.w_scale
        self.v_physical = self.v *self.v_scale
        self.u_physical = self.u *self.u_scale

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

    def get_sheath_df(self, geometry = 'khaziev', save=False, path='sheath-report.csv'):

        '''
        Write the result of the sheath model to the dataframe and write the file
        :param geometry: khaziev or stangeby
        :param save: boolean
        :param path: path to the report file
        :return:
        '''

        if geometry == 'khaziev':
            self.df_sheath = self.__get_sheath_df_khaziev__()
        elif geometry == 'stangeby':
            self.df_sheath = self.__get_sheath_df_stangeby__()
        else:
            raise Exception('Provided model is incorrect')

        if save:
            self.df_sheath.to_csv(path, index=False)


    def __get_sheath_df_stangeby__(self):
        df_sheath = pd.DataFrame({'z': self.z,
                                  'V_z': self.u_physical,
                                  'V_x': self.w_physical,
                                  'V_y': self.v_physical,
                                  'potential': self.potential,
                                  'density': self.density})

        columns = ['z', 'V_x', 'V_y', 'V_z', 'potential', 'density']
        self.df_sheath_stangeby = df_sheath[columns]

        return self.df_sheath_stangeby

    def __get_sheath_df_khaziev__(self):
        df_sheath = pd.DataFrame({'z': self.z,
                                  'V_z': self.u_physical,
                                  'V_x': self.w_physical,
                                  'V_y': self.v_physical,
                                  'potential': self.potential,
                                  'density': self.density})

        columns = ['z', 'V_x', 'V_y', 'V_z', 'potential', 'density']
        self.df_sheath_khaziev = df_sheath[columns]
        return self.df_sheath_khaziev



    def execute(self):

        self.get_zeta()
        self.get_v()
        self.get_density()
        self.get_physical_value()


if __name__ == '__main__':

    #TODO remove
    # test output
    T_e = 1
    T_i = 1
    m_i = 2
    alpha = 2
    model = Stangeby(T_e, T_i, m_i, alpha)
    print(model.zeta)
