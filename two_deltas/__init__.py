import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt


from .neural_network import FeedForward
from .generate_noised_data import GenerateTwoDeltas, GenerateNoisedData
from .preprocessing import Preprocessing, CreateDataloader