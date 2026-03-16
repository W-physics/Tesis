import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd

from save_plot.save_files import SaveCSV

def FindMaximumOfGenerations(ndata, repetitions):

    timesteps = 1000

    x = np.linspace(0,2,1000)

    h = 0.01

    list_c = [0,0+h]
    
    distros = [
        [
            pd.read_csv(f'data/n={ndata}_c={c}_m={m}.csv', header=None).to_numpy()
            for c in list_c
        ]
        for m in range(repetitions)
    ]    

    xmax = np.zeros((repetitions, 2, timesteps))

    for m, distros_n in enumerate(distros):

        for n, d in enumerate(distros_n):

            for t in range(timesteps):

                kernel = gaussian_kde(d[t], bw_method=0.3)
                y = kernel(x)
                xmax[m, n, t] = x[np.argmax(y)]

    average_maximums = np.average(xmax, axis=0)
    standard_error_maximums = np.std(xmax, axis=0) / np.sqrt(repetitions)

    SaveCSV(average_maximums, f'AM_n={ndata}')
    SaveCSV(standard_error_maximums, f'SE_n={ndata}')