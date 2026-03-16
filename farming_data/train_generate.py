from backward_process.generating import Generate
from initial_distributions.two_unequal_deltas import GenerateTwoUnequalDeltas

from neural_network.training_nn import TrainModel
from neural_network.neural_network import FeedForward

import torch 

def TrainAndGenerateDatasets(ndata, repetitions):

    timesteps = 1000

    h = 0.01

    initial_distribution=GenerateTwoUnequalDeltas

    list_c = [0,0+h]

    for exponent in list_c:

        loss_hist_train, val_hist_train, scaler, test_loss = TrainModel(timesteps, 
                                                                ndata,
                                                                initial_distribution,
                                                                exponent)
        
        print(f'test loss = {test_loss}')

        model = FeedForward(input_size=2,output_size=1,n_hidden_layers=2,depht=200)
        state_dict = torch.load('models/n='+str(ndata)+'_c='+str(exponent)+'.pth')
        model.load_state_dict(state_dict)
        model.eval();

        for m in range(repetitions):

            Generate(timesteps, ndata, model=model, scaler=scaler, exponent=exponent, repetition=m)