from backward_process.generating import Generate
from initial_distributions.two_unequal_deltas import GenerateTwoUnequalDeltas
from neural_network.training_nn import TrainModel
from neural_network.neural_network import FeedForward
from initial_distributions import GaussianMixture   

import torch   

def TrainAndGenerateDatasets(ndata, dimension):

    timesteps = 1000

    h = 0.01

    initial_distribution=GaussianMixture

    list_c = [-h,h]

    for exponent in list_c:

        test_loss, scaler = TrainModel(timesteps, ndata, dimension, initial_distribution, exponent)
        
        model = FeedForward(input_size=dimension+1, output_size=dimension, n_hidden_layers=2, depht=200)
        state_dict = torch.load(f'models/c={exponent}-d={dimension}.pth')
        model.load_state_dict(state_dict)
        model.eval();
        
        distros = Generate(timesteps, ndata, dimension, model=model, scaler=scaler, exponent=exponent)