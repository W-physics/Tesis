from train_and_generate.training_nn import TrainTwoDeltas
from train_and_generate.generating import Generate

def TrainGenerate(drift_term, noise_level):
    """
    Train the neural network and generate data.
    
    Parameters:
    drift_term (float): The drift term for the model.
    noise_level (float): The noise level for the model.
    """
    # Train the neural network
    TrainTwoDeltas(drift_term, noise_level)
    
    # Generate data using the trained model
    Generate(drift_term, noise_level)