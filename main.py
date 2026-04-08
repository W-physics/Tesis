from farming_data.train_generate import TrainAndGenerateDatasets
from farming_data.find_maximums import FindMaximumOfGenerations

import numpy as np


def main():

    ndata_list = np.arange(start=20000, stop=31000, step=1000)
    #ndata_list = np.arange(start=100, stop=200, step=10)

    for i in range(len(ndata_list)):

        repetitions = 10

        TrainAndGenerateDatasets(ndata_list[i], repetitions, False)

        FindMaximumOfGenerations(ndata_list[i], repetitions)
    
main()

print("Ended Successfully :)")