import numpy as np
import matplotlib.pyplot as plt

from forward_process.generate_noised_data import BetaSchedule


from save_plot.save_files import SaveCSV

def Generate(repetitions, timesteps, ndata, model, scaler):

    """
    Emulates the backward process of a diffusion model to generate data. The 
    dynamics of the data for each timestep is stored in an array called distros
    (shape=(timesteps, ndata)). The first row is filled with random noise.
    """
    
    beta = BetaSchedule(timesteps)
    alpha = 1 - beta

    distros = np.zeros((timesteps,ndata))
    distros[0] = np.random.normal(0, 1, ndata)

    print("Backward process started...")
 
    distros = np.zeros((repetitions, timesteps, ndata))
    distros[:, 0, :] = np.random.normal(0, 1, (repetitions, ndata))  # condiciones iniciales

    for t in range(1, timesteps):
        s = timesteps - t
        times = np.full((repetitions, ndata), s)

        # Construimos las features en bloque
        feat = np.stack((distros[:, t-1, :], times), axis=-1).reshape(-1, 2)
        scaled_feat = scaler.transform(feat)

        features = torch.tensor(scaled_feat, dtype=torch.float32)
        guessed_noise = model(features).detach().numpy().flatten()
        guessed_noise = guessed_noise.reshape(repetitions, ndata)

        beta_hat = beta[s] * (1 - np.prod(alpha[:s-1])) / (1 - np.prod(alpha[:s]))
        noise = np.sqrt(beta_hat) * np.random.normal(0, 1, (repetitions, ndata))

        distros[:, t, :] = (
            1 / np.sqrt(alpha[s])
            * (distros[:, t-1, :] - guessed_noise * beta[s] / np.sqrt(1 - np.prod(alpha[:s])))
            + noise
        )

    SaveCSV(distros, "generated_data")

def PlotTrainval(ndata, loss_hist_train, loss_hist_valid):
    """
    Plots learning curves of training and validation sets
    """

    fig, ax = plt.subplots()

    ax.set_xlabel("epochs")
    ax.set_ylabel("MSE")
    ax.set_ylim(0, max(loss_hist_train.max(), loss_hist_valid.max())) 
    ax.set_title("Training and validation errors")

    ax.plot(loss_hist_train,label='train')
    ax.plot(loss_hist_valid,label='valid')
    ax.legend(fontsize='large')

