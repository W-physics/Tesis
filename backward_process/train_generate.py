from backward_process.generating import Generate
from save_plot.plotter import PlotCritical
from initial_distributions.two_deltas import GenerateTwoDeltas
from neural_network.training_nn import TrainModel
#from initial_distributions.six_deltas import GenerateSixDeltas

import matplotlib.pyplot as plt

def TrainGenerate():

    ndata = 200
    timesteps = 300
    plt.style.use('bmh')

    #Generate(GenerateTwoDeltas, timesteps, ndata)
    PlotCritical(timesteps, ndata)
 
