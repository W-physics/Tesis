from train_and_generate.train_generate import TrainGenerate
from save_plot.plotter import PlotTrainval, PlotViolin

def main():
        
    TrainGenerate(drift_term=0, noise_level=2)
    PlotTrainval(1)
    PlotViolin(1)

    TrainGenerate(drift_term=0, noise_level=1)
    PlotTrainval(2)
    PlotViolin(2)

    TrainGenerate(drift_term=0, noise_level=0.1)
    PlotTrainval(3)
    PlotViolin(3)



main()
