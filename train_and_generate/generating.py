import numpy as np
import torch

from two_deltas.neural_network import FeedForward
from two_deltas.generate_noised_data import BetaSchedule

from save_plot.save_files import SaveCSV

def GetModel():

    model = FeedForward(input_size=1,output_size=1,n_hidden_layers=2,depht=5)
    checkpoint = torch.load("data/checkpoint.pth.tar")
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    
    return model

def Generate():

    ndata = 10000
    timesteps = 300
    
    beta = BetaSchedule(timesteps)

    distros = np.zeros((timesteps,ndata))
    distros[0] = np.random.normal(0, 1, ndata)

    model = GetModel()

    for t in range(1,timesteps):

        previous_t = torch.tensor(distros[t-1],dtype=torch.float32).reshape(-1,1)
        noise =  np.sqrt(beta[t]) * np.random.normal(0,1,ndata)
        deterministic = np.sqrt(beta[t]) * model(previous_t) + previous_t * (1 - 0.5* beta[t])
        deterministic_np = deterministic.detach().numpy().reshape(noise.shape)
        distros[t] = deterministic_np + noise

    SaveCSV(distros, "generated_data")


    
