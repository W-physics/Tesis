import numpy as np

def GenerateTwoUnequalDeltas(ndata, c):

    print(f"Generating two unequal deltas distribution with {ndata} data points...")

    w1 = np.exp(c) / (np.exp(c) + np.exp(-c) )
    w2 = np.exp(-c) / (np.exp(c) + np.exp(-c) )
    n_ones = round(w1*ndata)
    n_negative_ones = ndata - n_ones

    array = np.concatenate([np.ones(n_ones),-np.ones(n_negative_ones)])

    return array