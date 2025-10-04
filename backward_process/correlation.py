import numpy as np 

import pandas as pd

def GetCorrelations(array):
    
    array = array.T

    filtered = array[array[:,-1] < 0].T

    timesteps = len(filtered)
    ndata = len(filtered[0])

    corr = (filtered**2).mean(axis=1) - (filtered.mean(axis=1))**2

    return corr
