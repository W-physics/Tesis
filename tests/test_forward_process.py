import numpy as np
from forward_process.generate_noised_data import ForwardProcess, GenerateNoisedData

def test_ForwardProcess_shapes():
    """ Test that the output shapes are correct.
    """

    timesteps = 10
    ndata = 5
    initial_data = np.zeros(ndata)
    features, noises = ForwardProcess(timesteps, initial_data)
    assert features.shape == (ndata, timesteps,2)
    assert noises.shape == (ndata, timesteps)

def test_ForwardProcess_zero_average():
    """ Test that the average of the features is close to zero for initial 0 position.
    """
    timesteps = 2 
    ndata = 10_000
    initial_data = np.zeros(ndata) 
    features, noises = ForwardProcess(timesteps, initial_data)
    
    for t in range(timesteps):
        avg = np.mean(features[:, t, 0])
        assert np.isclose(avg, 0, atol=5e-3), f"{t=}, {avg=}, expected_avg=0"

def test_ForwardProcess_nonzero_average():
    """ Test that the average of the features is close to the alpha_bar for
    initial position 1.0.
    """
    timesteps = 5 
    ndata = 10_000
    initial_data = np.ones(ndata) 
    features, noises = ForwardProcess(timesteps, initial_data)
    
    avg = np.mean(features[:, -1, 0])
    beta_sched = np.linspace(start=1e-4, stop=0.02, num=timesteps)
    alpha = 1 - beta_sched
    alpha_bar = np.cumprod(alpha)
    for t in range(timesteps):
        avg = np.mean(features[:, t, 0])
        expected_avg = np.sqrt(alpha_bar[t])
        assert np.isclose(avg, expected_avg, atol=5e-3), f"{t=}, {avg=}, {expected_avg=}"

def test_ForwardProcess_variance():
    """ Test that the variance of the features is close to 1 - alpha_bar.
    """
    timesteps = 5 
    ndata = 10_000
    initial_data = np.ones(ndata) 
    features, noises = ForwardProcess(timesteps, initial_data)
    
    beta_sched = np.linspace(start=1e-4, stop=0.02, num=timesteps)
    alpha = 1 - beta_sched
    alpha_bar = np.cumprod(alpha)

    for t in range(timesteps):
        var = np.var(features[:, t, 0])
        expected_var = 1.0 - alpha_bar[t]
        assert np.isclose(var, expected_var, atol=5e-3), f"t={t}, var={var}, expected_var={expected_var}"

def test_ForwardProcess_gaussian():
    """ Test that the features are Gaussian at each timestep.
    """
    timesteps = 4 
    ndata = 100_000
    initial_data = np.ones(ndata) 
    features, noises = ForwardProcess(timesteps, initial_data)

    def gaussian(x, mu, sigma2):
        return 1/(np.sqrt(2 * np.pi * sigma2)) * np.exp(-0.5 * ((x - mu) ** 2) / sigma2)

    beta_sched = np.linspace(start=1e-4, stop=0.02, num=timesteps)
    alpha = 1 - beta_sched
    alpha_bar = np.cumprod(alpha)

    for t in range(timesteps):
        pass
    t = timesteps - 1
    expected_var = 1.0 - alpha_bar[t]
    expected_avg = np.sqrt(alpha_bar[t])
    hist, bin_edges = np.histogram(features[:, t, 0], bins=50, density=True)        
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    gauss = gaussian(bin_centers, expected_avg, expected_var)
    err = np.square((hist - gauss)).mean()
    assert np.isclose(err, 0, atol=1e-3), f"{t=}, {err=}"




def test_GenerateNoisedData_shapes():
    timesteps = 10
    ndata = 5
    def initial_distribution(ndata):
        return np.zeros(ndata)
    features, noises = GenerateNoisedData(timesteps, ndata, initial_distribution)
    assert features.shape == (ndata, timesteps,2)
    assert noises.shape == (ndata, timesteps)