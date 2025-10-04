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
    ndata = len(initial_data)
    features = np.zeros((ndata, timesteps, 2))
    noises = np.random.normal(0,1,size=(ndata,timesteps))
    beta = BetaSchedule(timesteps)
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)
    times = np.arange(timesteps)
    for i in range(ndata):

        # usar np.prod(alpha[:time]) genera un bug pues el primer elemento es 1
        # en vez alpha[0] 
        # es mejor usar cumprod

        for j in range(timesteps):
            noise = noises[i,j]
            time = times[j]
            noised_data = initial_data[i]*np.sqrt(alpha_bar[time]) + np.sqrt((1 - alpha_bar[time]))*noise
            # Error inicial: la desviación estándar del ruido es sqrt(1 - alpha_bar y no 1-alpha_bar
            # noised_data = initial_data[i]*np.sqrt(alpha_bar[time]) + (1 - alpha_bar[time])*noise
            features[i,j] = noised_data, time


    return  features, noises

def GenerateNoisedData(timesteps, ndata, initial_distribution):

    data = initial_distribution(ndata)
    
    return ForwardProcess(timesteps, data)