import numpy as np

def GenerateTwoUnequalDeltas(ndata):

    print(f"Generating two unequal deltas distribution with {ndata} data points...")

    array = np.concatenate([np.ones(2 * ndata//3),-np.ones(ndata//3)])

    return array