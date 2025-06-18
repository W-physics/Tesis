from train_and_generate.train_generate import TrainGenerate
from save_plot.plotter import PlotTrainval, PlotViolin

def main():

    for i in range(10):
        print(f"Iteration {i+1} of 10")
        TrainGenerate(drift_term=0.01, noise_level=0.1**i)
        PlotTrainval(i)
        PlotViolin(i)

main()
