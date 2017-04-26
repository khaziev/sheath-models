import matplotlib.pyplot as plt
import pandas as pd

from sheath_models.zimmerman import Zimmerman



import numpy as np
from multiprocessing import Pool
import time
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def test_model(params):
    '''

    Tester for Zimemrmans model
    '''

    omega = 1
    alpha = params[0]
    tau = params[1]
    delta = params[2]
    y0 = params[3]

    # progress_bar.update()

    model = Zimmerman(omega, alpha, tau, delta)
    model.y0 = y0 *np.ones(4)
    model.solve()
    smoothness = model.get_smoothness()

    return len(model.x), smoothness


#setup test parameters
omega = 1

taus = [0, 1e-2, 1, 1e2, 1e3, 1e4]
alphas = [0, 5, 10, 20, 40]
deltas = [0, 1e-2, 0.1, 0.2, 0.5, 1, 5, 10]

y0_vals = [1e-7, 1e-6, 1e-5, 1e-3, 1e-2, 1e-1]

tau, alpha, delta = np.meshgrid(taus, alphas, deltas)

tau = list(tau.ravel())
alpha = list(alpha.ravel())
delta = list(delta.ravel())

y0_par = [y0_vals[0]] *len(tau)
for y in y0_vals[1:]:
    y0_par = y0_par + [y]*len(tau)

tau = tau *len(y0_vals)
alpha = alpha *len(y0_vals)
delta = delta *len(y0_vals)

def parallel_eval(n_jobs, chunksize=1):

    # chunksize = np.int(len(alpha)/n_jobs/4)
    # chunksize = 1

    time_start = time.time()
    pool = Pool(n_jobs)
    args = zip(alpha, tau, delta, y0_par)
    result = pool.map(test_model, args, chunksize=chunksize)
    pool.close()
    time_end = time.time()
    print('proc: {0}, time: {1:.2e}'.format(n_jobs, time_end-time_start))
    return result


result = parallel_eval(40)

print(result)

points = [x[0] for x in result]
smooth = [x[1] for x in result]

df_smooth = pd.DataFrame({'points': points,
                          'smooth': smooth,
                          'alpha': alpha,
                          'delta': delta,
                          'omega_tau': tau,
                          'y0': y0_par})
df_smooth.to_csv('smoothness.csv', index=False)

# jobs = [1, 4, 8, 16, 30, 40]
#
# times = [parallel_eval(job) for job in jobs]
#
# df_times = pd.DataFrame({'time': times, 'procs': jobs})
#
# df_times.to_csv('times.csv', index=False)

#chunksize test

# n_jobs = 20
# chunks  =  np.linspace(1, np.int(len(alpha)/n_jobs), 3)
#
# chunks = [np.int(x) for x in chunks]
#
# print(chunks)
#
# times = [parallel_eval(n_jobs, chunk) for chunk in chunks]
#
# df_times = pd.DataFrame({'time': times, 'chunks': chunks})
#
# df_times.to_csv('time-chunk-size.csv', index=False)
