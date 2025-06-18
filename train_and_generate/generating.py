import numpy as np
import torch

from two_deltas.neural_network import FeedForward

from save_plot.save_files import SaveCSV

def GetModel():

    model = FeedForward(input_size=1,output_size=1,n_hidden_layers=2,depht=5)
    checkpoint = torch.load("data/checkpoint.pth.tar")
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    
    return model

def Generate(drift_term, noise_level):

    ndata = 10000
    timesteps = 1000
    pure_noise = np.random.normal(0,1,ndata)

    distros = np.zeros((timesteps,ndata))
    distros[0] = pure_noise

    model = GetModel()

    for t in range(1,timesteps):

        previous_t = torch.tensor(distros[t-1],dtype=torch.float32).reshape(-1,1)
        noise = np.sqrt(2*noise_level) * np.random.normal(0,1,ndata)
        deterministic = (1 - drift_term) * previous_t - 2 * noise_level * model(previous_t)
        deterministic_np = deterministic.detach().numpy().reshape(noise.shape)
        distros[t] = deterministic_np + noise

    SaveCSV(distros, "generated_data")


    
