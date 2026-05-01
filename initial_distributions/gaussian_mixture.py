import numpy as np

def GaussianMixture(ndata, dimension, c):

    w1 = np.exp(c) / (np.exp(c) + np.exp(-c) )
    w2 = np.exp(-c) / (np.exp(c) + np.exp(-c) )
    n_ones = round(w1*ndata)
    n_negative_ones = ndata - n_ones
    
    first_gaussian = np.random.normal(loc=10, scale=1, size=(dimension,n_ones))
    second_gaussian = np.random.normal(loc=-10, scale=1, size=(dimension, n_negative_ones))

    full_distro = np.concatenate([first_gaussian, second_gaussian], axis=1).T

    return full_distro

