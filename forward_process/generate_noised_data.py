import numpy as np

def BetaSchedule(n_steps, start=1e-4, end=0.02):
    """
    Generates a beta schedule for the forward process.
    """
    return np.linspace(start, end, n_steps)

def ForwardProcess(initial_data):

    ndata = len(initial_data)
    noised_data = np.zeros(ndata)
    noise = np.zeros(ndata)
    beta = BetaSchedule(n_steps=1000)

    for i in range(ndata):

        time = np.random.randint(len(beta))
        noise[i] = np.random.normal(0,1)
        factor = np.zeros(len(beta[:time]))
        for j in range(len(factor)):
            factor[j] = np.sqrt(beta[j]) * np.prod(1 - beta[j+1:time])
            
        noised_data[i] = initial_data[i] * np.prod(1 - beta) + noise[i] * np.sum(factor)

    return  noised_data, noise

def GenerateNoisedData(ndata, initial_distribution):

    data = initial_distribution(ndata)

    print("forward process started...")
    
    return ForwardProcess(data)