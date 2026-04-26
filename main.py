from farming_data.train_generate import TrainAndGenerateDatasets
from farming_data.find_maximums import FindMaximumOfGenerations
from farming_data.projection import Project
from save_plot.save_files import SaveCSV

import numpy as np


def main():

    h = 0.01
    ndata = 10000
    list_c = [-h,0,h]
    dimensions = np.arange(start=1, stop=10, step=1)

    for d in dimensions:

        for c in list_c:
        
            distros = TrainAndGenerateDatasets(ndata, dimension=d, c=c)
            projections = Project(distros=distros,
                                  ndata=ndata,
                                  timesteps=1000,
                                  dimension=d)
            
            SaveCSV(projections, f'projetions_d={d}_c={c}')
            FindMaximumOfGenerations(projections, dimension=d, c=c, repetitions=1)
    
main()

print("Ended Successfully :)")