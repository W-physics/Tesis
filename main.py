from farming_data.train_generate import TrainAndGenerateDatasets
from farming_data.find_maximums import FindMaximumOfGenerations

import numpy as np


def main():

    ndata_list = np.arange(start=20000, stop=31000, step=1000)

    for i in range(len(ndata_list)):

        if i == 0:

            train = True

        else:
            
            train = False

        TrainAndGenerateDatasets(ndata_list[i], repetitions, train)

        repetitions = 10

        FindMaximumOfGenerations(ndata_list[i], repetitions)
    
main()

print("Ended Successfully :)")