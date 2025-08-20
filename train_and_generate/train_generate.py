from train_and_generate.generating import Generate
from save_plot.plotter import PlotTrainval, PlotCritical
from initial_distributions.two_deltas import GenerateTwoDeltas
from initial_distributions.six_deltas import GenerateSixDeltas

import matplotlib.pyplot as plt

def TrainGenerate():

    ndata = [500,1000,5000,10000]
    timesteps = 300
    plt.style.use('bmh')

    for n in ndata:

        Generate(GenerateTwoDeltas,
             n,
             timesteps)
        PlotTrainval(n)
        PlotCritical(n)
 
