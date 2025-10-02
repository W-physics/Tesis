from scipy.integrate import simpson
import numpy as np

from forward_process.generate_noised_data import BetaSchedule


def Theta(time, beta):
    integral = simpson(beta[:time+1], dx=1)
    return np.exp(-0.5*integral)

def TrueDynamics(timesteps, ndata):

    distros = np.zeros((timesteps, ndata))
    betas = BetaSchedule(timesteps)

    distros[0] = np.random.normal(0, 1, ndata)
    for t in range(1, timesteps):
        beta = betas[t]
        theta = Theta(t, betas)
        delta = 1 - theta**2
        fraction = theta/delta 
        previous_distro = distros[t-1]
        distros[t] = previous_distro*(beta/2 * theta**2/delta + 1) - beta/2 * fraction * np.tanh(fraction * previous_distro)
    
    return distros