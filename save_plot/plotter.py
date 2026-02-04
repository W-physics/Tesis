import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def PlotCritical(timesteps, ndata):

    name = str(ndata)+" datapoints, "+str(timesteps)+" generating timesteps"

    critical_time = 158

    reduced_timesteps = np.arange(timesteps)/critical_time

    fig, ax = plt.subplots(ncols=3, width_ratios=[1, 4, 1], sharey=True)
    ax[0].set_ylabel(r"$x_{s}$")
    ax[1].set_xlabel(r"$s/s_c$")
    fig.suptitle("Trajectorie plot "+name)


    ncurves = 20
    distros = pd.read_csv("data/generated_data.csv", header=None).to_numpy()

    distros_c1 = distros[:, distros[-1,:] >= 0]
    distros_c2 = distros[:, distros[-1,:] < 0]


    PlotHistograms(ax, distros_c1)
    PlotHistograms(ax, distros_c2)

    PlotSim(distros, reduced_timesteps, ax[1], ncurves)

    PlotMean(ax[1], distros_c1, reduced_timesteps)
    PlotMean(ax[1], distros_c2, reduced_timesteps)
                               
    #fig.savefig("figures/trajectories/"+name+".svg")


def PlotSim(distros, reduced_timesteps, ax, ncurves):

    ndata = distros.shape[1]

    reduced_distros = distros[:,::ndata//ncurves]

    for i in range(reduced_distros.shape[1]):

        color = 'b' if reduced_distros[-1,i] >= 0 else 'g'

        ax.scatter(reduced_timesteps, reduced_distros[:,i], c=color, alpha = 0.5, s=0.3)

def PlotHistograms(ax, distros):

    hist0 = distros[0,:]
    hist2= distros[-1,:]

    color = 'b' if hist2[0] >= 0 else 'g'

    ax[0].hist(hist0, bins=50, orientation='horizontal', density=True, color=color, alpha=0.5)
    ax[2].hist(hist2, bins=5, orientation='horizontal', density=True, color=color)

    ax[0].invert_xaxis()

def PlotMean(ax, distro, reduced_timesteps):

    means = np.mean(distro, axis=1)
    color = 'b' if means[-1] >= 0 else 'g'
    ax.plot(reduced_timesteps,means, c=color, ls='dashed')