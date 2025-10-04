import numpy as np

def GenerateTwoInequalDeltas(ndata):

    array = np.concatenate([np.ones(2 * ndata//3),-np.ones(ndata//3)])

    return array