import numpy as np
from save_plot.save_files import SaveCSV

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
    features = np.zeros((ndata, timesteps,  2))
    noises = np.random.normal(0,1,size=(ndata,timesteps))
    beta = BetaSchedule(timesteps)
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)
    alpha_bar = np.cumprod(alpha)
    times = np.arange(timesteps)

    # Versión optimizada con broadcast de numpy en vez de hacer loops i, j
    noised_data = initial_data[:, None]*np.sqrt(alpha_bar) + np.sqrt((1 - alpha_bar))*noises
    features[:,:,0] = noised_data
    features[:,:,1] = times

    features = features.reshape(-1,2)
    noise = noise.reshape(-1)

    return  features, noises

def GenerateNoisedData(timesteps, ndata, initial_distribution):

    data = initial_distribution(ndata)
    
    SaveCSV(ForwardProcess(timesteps, data), "noised_data")