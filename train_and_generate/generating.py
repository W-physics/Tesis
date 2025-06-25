import numpy as np
import torch

from forward_process.neural_network import FeedForward
from forward_process.generate_noised_data import BetaSchedule
from train_and_generate.training_nn import TrainModel

from save_plot.save_files import SaveCSV

def Generate(initial_distribution, ndata, timesteps):
    
    beta = BetaSchedule(timesteps)

    distros = np.zeros((timesteps,ndata))
    distros[0] = np.random.normal(0, 1, ndata)

    model = TrainModel(ndata, initial_distribution)

    print("Backward process started...")

    for t in range(1,timesteps):

        previous_t = torch.tensor(distros[t-1],dtype=torch.float32).reshape(-1,1)
        noise =  np.sqrt(beta[t]) * np.random.normal(0,1,ndata)
        deterministic = np.sqrt(beta[t]) * model(previous_t) + previous_t * (1 - 0.5* beta[t])
        deterministic_np = deterministic.detach().numpy().reshape(noise.shape)
        distros[t] = deterministic_np + noise

    SaveCSV(distros, "generated_data")


    
