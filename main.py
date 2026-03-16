from farming_data.train_generate import TrainAndGenerateDatasets
from farming_data.find_maximums import FindMaximumOfGenerations

import torch 


def main():

    ndata_list = [21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 30000]

    for i in range(len(ndata_list)):

        repetitions = 10

        TrainAndGenerateDatasets(ndata_list[i], repetitions)
        FindMaximumOfGenerations(ndata_list[i], repetitions)
    
main()

print("Ended Successfully :)")