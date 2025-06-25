from forward_process.generate_noised_data import BetaSchedule
import numpy as np

def Theta(time,timesteps):

    beta = BetaSchedule(n_steps=timesteps)
    slope = (beta[-1] - beta[0]) / timesteps

    return np.exp(-0.5 * beta[0] * time) * np.exp(-0.25 * slope * time**2)

