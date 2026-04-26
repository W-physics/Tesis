import numpy as np

def BetaSchedule(n_steps, start=1e-4, end=0.02):
    """
    Generates a beta schedule for the forward process.
    """
    return np.linspace(start, end, n_steps)


def ForwardProcess(timesteps, ndata, dimension, initial_data):

    distros = np.zeros((ndata, dimension, timesteps))

    # Ruido independiente para cada dato
    noises = np.random.normal(0, 1, size=(ndata, dimension, timesteps))

    beta = BetaSchedule(timesteps)
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)

    times = np.arange(timesteps)

    # Expandimos dimensiones para broadcasting correcto
    alpha_bar = alpha_bar[None, None, :]          # (1,1,T)
    initial_data = initial_data[:, :, None]       # (N,D,1)

    noised_data = (
        initial_data * np.sqrt(alpha_bar) +
        np.sqrt(1 - alpha_bar) * noises
    )

    distros[:, :, :] = noised_data

    return distros, times, noises

def GenerateNoisedData(timesteps, ndata, dimension, initial_distribution, c):

    data = initial_distribution(ndata, dimension, c)
    
    return ForwardProcess(timesteps, ndata, dimension, data)