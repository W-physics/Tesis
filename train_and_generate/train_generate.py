from train_and_generate.generating import Generate
from save_plot.plotter import PlotTrainval, PlotViolin
from initial_distributions.two_deltas import GenerateTwoDeltas
from initial_distributions.six_deltas import GenerateSixDeltas
import matplotlib.pyplot as plt

def TrainGenerate(iter):

    plt.style.use('bmh')

    Generate(GenerateSixDeltas,
             ndata=1000,
             timesteps=300)
    PlotTrainval(iter)
    PlotViolin(iter)