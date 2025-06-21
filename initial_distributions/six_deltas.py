import numpy as np

def GenerateSixDeltas(ndata):
     
    print(f"Generating six delta distribution with {ndata} data points...")
    
    data  = np.array([])

    for i in range(3):

        data = np.concatenate([data, np.ones(ndata//6) * (i + 1)])
        data = np.concatenate([data, np.ones(ndata//6) * -(i + 1)])

    return data
