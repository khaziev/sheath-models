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

    global progress_bar

    omega = 1
    alpha = params[0]
    tau = params[1]
    delta = params[2]

    # progress_bar.update()

    model = Zimmerman(omega, alpha, tau, delta)
    model.solve()
    smoothness = model.get_smoothness()

    return len(model.x), smoothness


#setup test parameters
omega = 1

n_points = 10
taus = np.logspace(-5, 4, n_points)
taus[0] = 0
alphas = np.linspace(0, 40, n_points)
deltas = np.logspace(-3, 1, n_points)
deltas[0] = 0

tau, alpha, delta = np.meshgrid(taus, alphas, deltas)
tau = tau.ravel()
alpha = alpha.ravel()
delta = delta.ravel()

def parallel_eval(n_jobs, chunksize=1):

    # chunksize = np.int(len(alpha)/n_jobs/4)
    # chunksize = 1

    time_start = time.time()
    pool = Pool(n_jobs)
    args = zip(alpha, tau, delta)
    result = pool.map(test_model, args, chunksize=chunksize)
    pool.close()
    time_end = time.time()
    print('proc: {0}, time: {1:.2e}'.format(n_jobs, time_end-time_start))
    return result, time_end-time_start


result, time = parallel_eval(40)

points = [x[0] for x in result]
smooth = [x[1] for x in result]

df_smooth = pd.DataFrame({'points': points, 'smooth': smooth, 'alpha': alpha, 'delta': delta, 'omega_tau': tau})
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
