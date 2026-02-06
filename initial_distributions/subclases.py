from scipy.signal import square
import numpy as np

from random_sampling.sampler import MakeSampler

def SquaredSignal(x):
    amplitude=0.5
    frecuency=5

    square_wave = square(x - np.pi/2, duty = 1/2) + 1
    cos_wave = amplitude * np.cos(frecuency * x)**2 + 2

    return square_wave*cos_wave

def GenerateSubclasses(ndata): 

    generator = MakeSampler(SquaredSignal)[1]

    return generator(ndata)

