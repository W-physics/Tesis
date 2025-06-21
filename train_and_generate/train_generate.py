from train_and_generate.training_nn import TrainTwoDeltas
from train_and_generate.generating import Generate
from save_plot.plotter import PlotTrainval, PlotViolin

def TrainGenerate(iter):
    """
    Train the neural network and generate data.
    
    Parameters:
    drift_term (float): The drift term for the model.
    noise_level (float): The noise level for the model.
    """
    # Train the neural network
    TrainTwoDeltas()
    PlotTrainval(iter)

    
    # Generate data using the trained model
    Generate()
    PlotViolin(iter)