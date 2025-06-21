import numpy as np

def GenerateTwoDeltas(ndata): 

    print(f"Generating two deltas distribution with {ndata} data points...")

    return np.concatenate([np.ones(ndata//2),-np.ones(ndata//2)])