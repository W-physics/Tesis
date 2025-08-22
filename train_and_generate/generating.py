import numpy as np
import torch

from forward_process.neural_network import FeedForward
from forward_process.generate_noised_data import BetaSchedule
from train_and_generate.training_nn import TrainModel

from save_plot.save_files import SaveCSV

def Generate(initial_distribution, timesteps, ndata):

    """
    Emulates the backward process of a diffusion model to generate data. The 
    dynamics of the data for each timestep is stored in an array called distros
    (shape=(timesteps, ndata)). The first row is filled with random noise.
    """
    
    beta = BetaSchedule(timesteps)
    alpha = 1 - beta

    distros = np.zeros((timesteps,ndata))
    distros[0] = np.random.normal(0, 1, ndata)

    model = TrainModel(timesteps, ndata, initial_distribution)

    print("Backward process started...")

    for t in range(1,timesteps):
        
        previous_distro = torch.tensor(distros[t-1],dtype=torch.float32).reshape(-1,1)

        guessed_noise = model(previous_distro).detach().numpy().reshape(-1,1).flatten()

        beta_hat = beta[t] * (1 - np.prod(alpha[:t-1]))/(1 - np.prod(alpha[:t]))
        noise =  np.sqrt(beta_hat) * np.random.normal(0,1,ndata)
        
        distros[t] = 1/np.sqrt(alpha[t]) * (distros[t-1] - guessed_noise* beta[t]/(np.sqrt(1 - np.prod(alpha[:t]))) ) + noise

    SaveCSV(distros, "generated_data")


    
