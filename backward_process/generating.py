import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from forward_process.generate_noised_data import BetaSchedule
from neural_network.neural_network import FeedForward

from save_plot.save_files import SaveCSV

device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'

def Generate(timesteps, ndata):

    """
    Emulates the backward process of a diffusion model to generate data. The 
    dynamics of the data for each timestep is stored in an array called distros
    (shape=(timesteps, ndata)). The first row is filled with random noise.
    """
    
    beta = BetaSchedule(timesteps)
    alpha = 1 - beta

    distros = np.zeros((timesteps,ndata))
    distros[0] = np.random.normal(0, 1, ndata)

    model = FeedForward(input_size=2,output_size=1,n_hidden_layers=2,depht=200).to(device)
    model.load_state_dict(torch.load('model/FeedForward.pth',weights_only=True))

    scaler = StandardScaler()

    print("Backward process started...")
 
    for t in range(1,timesteps):
        
        s = timesteps - t

        times = np.array(s).repeat(ndata)
        
        feat = np.vstack((distros[t-1], times)).T
        scaled_feat = scaler.fit_transform(feat)

        features = torch.tensor(scaled_feat, dtype=torch.float32)

        guessed_noise = model(features).detach().numpy().flatten()

        beta_hat = beta[s] * (1 - np.prod(alpha[:s-1]))/(1 - np.prod(alpha[:s]))
        noise =  np.sqrt(beta_hat) * np.random.normal(0,1,ndata)
        
        distros[t] = 1/np.sqrt(alpha[s]) * (distros[t-1] - guessed_noise* beta[s]/(np.sqrt(1 - np.prod(alpha[:s]))) ) + noise

    SaveCSV(distros, "generated_data")


