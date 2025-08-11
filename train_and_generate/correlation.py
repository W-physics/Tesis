import numpy as np 

import pandas as pd

def Get_correlations(array):

    corr = np.zeros(len(array))
    ndata = len(array[0])
    timesteps = len(array)

    for i in range(timesteps):
        corr[i] = np.sum(array[i]**2) / ndata

    return corr
