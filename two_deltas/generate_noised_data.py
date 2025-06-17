import numpy as np

def GenerateTwoDeltas(Ndata): 

    return np.concatenate([np.ones(Ndata//2),-np.ones(Ndata//2)])

def ForwardProcess(initial_data, drift_term, noise_level):

    Ndata = len(initial_data)

    noised_data = np.zeros(Ndata)
    noise = np.zeros(Ndata)

    for i in range(Ndata):

        noise[i] = np.random.normal(loc=0.0, scale=1)
        temperature_list = (1 - drift_term)**[np.arange(i - 1)]

        noised_data[i] = initial_data * (1 - drift_term)**(i) + np.sqrt(2*noise_level)*noise[i]*np.sum(temperature_list)

    return  noised_data, noise

def GenerateNoisedData(drift_term, noise_level):

    data = GenerateTwoDeltas(10000)
    
    return ForwardProcess(data, drift_term, noise_level)