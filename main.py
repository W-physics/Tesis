from farming_data.train_generate import TrainAndGenerateDatasets
from farming_data.find_maximums import FindMaximumOfGenerations

import numpy as np


def main():

    ndata=10000
    dimension = 1

    TrainAndGenerateDatasets(ndata, dimension)
    
main()

print("Ended Successfully :)")