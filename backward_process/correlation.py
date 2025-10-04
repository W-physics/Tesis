import numpy as np 

import pandas as pd

def GetCorrelations(array):
    
    array = array.T

    filtered = array[array[:,-1] < 0].T

    timesteps = len(filtered)
    corr = np.zeros(timesteps)
    ndata = len(filtered[0])

    for i in range(timesteps):
        corr[i] = np.sum(filtered[i]**2) / ndata

    return corr
