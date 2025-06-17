import numpy as np

def GenerateTwoDeltas(Ndata): 

    return np.concatenate([np.ones(Ndata//2),-np.ones(Ndata//2)])

def ForwardProcess(initial_data, drift_term, noise_level):

    ndata = len(initial_data)
    noised_data = np.zeros(ndata)
    noise = np.zeros(ndata)

    for i in range(ndata):

        time = np.random.randint(1000)
        noise[i] = np.random.normal(0,1)
        temperature_list = [(1-drift_term)**(j) for j in range(time+1)]

        noised_data[i] = initial_data[i] * (1 - drift_term)**(time) + np.sqrt(2*noise_level)*noise[i]*np.sum(temperature_list)

    return  noised_data, noise

def GenerateNoisedData(drift_term, noise_level):

    data = GenerateTwoDeltas(10000)
    
    return ForwardProcess(data, drift_term, noise_level)