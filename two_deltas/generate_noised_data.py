import numpy as np


def GenerateTwoDeltas(Ndata): 

    return np.concatenate([np.ones(Ndata//2),-np.ones(Ndata//2)])

def ForwardProcess(initial_data, drift_term, noise_level, timestep):

    Ndata = len(initial_data)
    random_noise = np.random.normal(loc=0.0, scale=noise_level, size=Ndata)
    temperature_list = (1 - drift_term)**[np.arange(timestep - 1)]

    return initial_data * (1 - drift_term)**(timestep) + np.sqrt(2*noise_level)*random_noise * np.sum(temperature_list)


def GenerateNoisedData(data):

    data = GenerateTwoDeltas()
    
    return ForwardProcess(data)