from train_and_generate.generating import Generate
from save_plot.plotter import PlotTrainval, PlotViolin
from save_plot.plotter import PlotConsistency
from initial_distributions.two_deltas import GenerateTwoDeltas
from initial_distributions.six_deltas import GenerateSixDeltas

import matplotlib.pyplot as plt

def TrainGenerate(iter):

    ndata = 1000
    timesteps = 300
    plt.style.use('bmh')

    #Generate(GenerateTwoDeltas,
#             ndata,
#             timesteps)
    #PlotTrainval(iter)
    #PlotViolin(iter)
    PlotConsistency(time=200, timesteps=timesteps)

