import numpy as np

'''
def Project(distros, timesteps, ndata, dimension):

    positions = np.zeros((timesteps, ndata))

    normaliced_unity = 1 / np.sqrt(dimension) * np.ones(dimension)

    for t in range(timesteps):

        for n in range(ndata):

            vector = distros[t,n,:]

            projection = vector @ normaliced_unity * vector / np.linalg.norm(vector)

            positions[t,n] = np.linalg.norm(projection)

    return positions
'''

def Project(distros, timesteps, ndata, dimension):

    # vector fijo (no depende de t ni n)
    normalized_unity = np.ones(dimension) / np.sqrt(dimension)

    # producto punto vectorizado
    # distros shape: (timesteps, ndata, dimension)
    # resultado: (timesteps, ndata)
    projections = np.tensordot(distros, normalized_unity, axes=([2], [0]))

    return projections