import numpy as np

def BetaSchedule(n_steps, start=1e-4, end=0.02):
    """
    Generates a beta schedule for the forward process.
    """
    return np.linspace(start, end, n_steps)

def ForwardProcess(timesteps, initial_data):
    
    """
    Generates noised data using the forward process. Takes one random timestep
    generate a 1D noised data and store it in the noised_data array. This is repeated
    for each data point in the initial_data array.
    """
    samples = 30
    ndata = len(initial_data)
    noised_data = np.zeros((ndata,samples))
    noises = np.random.normal(0,1,size=ndata)
    beta = BetaSchedule(timesteps)
    alpha = 1 - beta
    times = np.random.randint(low=0, high=timesteps-samples, size=ndata)

    for i in range(ndata):

        noise = noises[i]

        for j in range(samples):
            time = times[i] + j
            noised_data[i,j] = initial_data[i]*np.sqrt(np.prod(alpha[:time])) + (1 - np.prod(alpha[:time]))*noise

    return  noised_data, noises

def GenerateNoisedData(timesteps, ndata, initial_distribution):

    data = initial_distribution(ndata)
    
    return ForwardProcess(timesteps, data)