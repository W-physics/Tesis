from backward_process.generating import Generate
from save_plot.plotter import PlotCritical
from initial_distributions.two_deltas import GenerateTwoDeltas
#from initial_distributions.six_deltas import GenerateSixDeltas

import matplotlib.pyplot as plt

def TrainGenerate():

    ndata = 20000
    timesteps = 300
    plt.style.use('bmh')

    Generate(GenerateTwoDeltas, timesteps, ndata)
    PlotCritical(timesteps, ndata)
 
