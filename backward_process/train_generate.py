from backward_process.generating import Generate
from save_plot.plotter import PlotCritical
from initial_distributions.two_deltas import GenerateTwoDeltas
from neural_network.training_nn import TrainModel
#from initial_distributions.six_deltas import GenerateSixDeltas

import matplotlib.pyplot as plt

def TrainGenerate():

    ndata = 20000
    timesteps = 300
    plt.style.use('bmh')

    #model, loss_train, loss_val = TrainModel(timesteps, ndata, initial_distribution=GenerateTwoDeltas)

    #plt.plot(loss_train, label='train')
    #plt.plot(loss_val, label='valid')
    #plt.legend();
    #plt.show()  
    Generate(GenerateTwoDeltas, timesteps, ndata)
    PlotCritical(timesteps, ndata)
 
