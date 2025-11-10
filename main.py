from backward_process.generating import Generate
from save_plot.plotter import PlotCritical
from initial_distributions.two_deltas import GenerateTwoDeltas
from initial_distributions.two_inequal_deltas import GenerateTwoInequalDeltas
from neural_network.training_nn import TrainModel

import matplotlib.pyplot as plt
import torch 

device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'

torch.set_default_device(device)


def main():

    repetitions = 100
    ndata = 10000
    timesteps = 300
    plt.style.use('bmh')

    train_steps = 1000
    model, loss_hist_train, val_hist_train, scaler = TrainModel(train_steps, ndata, initial_distribution=GenerateTwoDeltas)
     
    Generate(repetitions, timesteps, ndata, model=model, scaler=scaler)
    PlotCritical(repetitions, timesteps, ndata)

main()

print("Ended Successfully :)")