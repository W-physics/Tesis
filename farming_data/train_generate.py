from backward_process.generating import Generate
from initial_distributions.two_unequal_deltas import GenerateTwoUnequalDeltas
from neural_network.training_nn import TrainModel
from neural_network.neural_network import FeedForward

import pickle
import torch   

def TrainAndGenerateDatasets(ndata, repetitions, train):

    timesteps = 1000

    h = 0.01

    initial_distribution=GenerateTwoUnequalDeltas

    list_c = [-h,h]

    for exponent in list_c:

        if train:
            TrainModel(timesteps, ndata, initial_distribution, exponent)


        with open('models/scaler_file.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        model = FeedForward(input_size=2,output_size=1,n_hidden_layers=2,depht=200)
        state_dict = torch.load('models/c='+str(exponent)+'.pth')
        model.load_state_dict(state_dict)
        model.eval();

        for m in range(repetitions):

            Generate(timesteps, ndata, model=model, scaler=scaler, exponent=exponent, repetition=m)