from backward_process.correlation import GetCorrelations
from test.true_dynamics import TrueDynamics
from save_plot.save_files import LoadCSV
#from backward_process.fitting import ExponentialFitting


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def PlotCritical(timesteps, ndata):

    critical_time = 158

    reduced_timesteps = np.arange(timesteps)/critical_time

    fig, ax = plt.subplots(ncols=3, width_ratios=[1, 4, 1], sharey=True)
    ax[0].set_ylabel(r"$x_{s}$")
    ax[1].set_xlabel(r"$s/s_c$")
    fig.suptitle("Trajectorie plot of critical time n = "+str(ndata))


    ncurves = 20
    distros = LoadCSV("data/generated_data.csv")
    distros_c1 = distros[:, distros[-1,:] >= 0]
    distros_c2 = distros[:, distros[-1,:] < 0]


    PlotHistograms(ax, distros_c1)
    PlotHistograms(ax, distros_c2)

    PlotSim(distros, reduced_timesteps, ax[1], ncurves)
    #PlotTest(ax[1], timesteps, ndata, ncurves, critical_time)

    fig2, ax2 = plt.subplots()

    ax2.set_ylabel(r"$D_{s}$")
    ax2.set_xlabel(r"$s/s_c$")
    ax2.set_title("Correlations with n = "+str(ndata))

    PlotCorrelations(ax2, distros_c1, reduced_timesteps, color='b')
    PlotCorrelations(ax2, distros_c2, reduced_timesteps, color='g')
    #ax.legend()

    #exponent_steps = 10
    #c1_curves = distros_c1.shape[1]
    #exponent_distros = distros_c1[critical_time:,:].reshape(-1)
    #exponent_timesteps = np.tile(np.arange(len(distros_c1[critical_time:,:])), c1_curves)

    #exponent, sigma_exponent = ExponentialFitting(exponent_timesteps, exponent_distros)
                                                  

    fig.savefig("figures/trajectories/n="+str(ndata)+".svg")
    print(f"Trajerctorie plot of critical time n = {ndata} plotted and saved to figures/trajectories/n={ndata}.pdf")

    fig2.savefig("figures/correlations/n="+str(ndata)+".svg")
    print(f"Correlations at critical time n = {ndata} plotted and saved to figures/correlations/n={ndata}.pdf")


def PlotSim(distros, reduced_timesteps, ax, ncurves):

    ndata = distros.shape[1]

    reduced_distros = distros[:,::ndata//ncurves]

    for i in range(reduced_distros.shape[1]):

        color = 'b' if reduced_distros[-1,i] >= 0 else 'g'

        ax.scatter(reduced_timesteps, reduced_distros[:,i], c=color, alpha = 0.5, s=0.3)


def PlotCorrelations(ax,distros, reduced_timesteps, color):

    ndata = distros.shape[1]

    correlations = GetCorrelations(distros)

    ax.plot(reduced_timesteps, correlations, color=color)

def PlotHistograms(ax, distros):

    hist0 = distros[0,:]
    hist2= distros[-1,:]

    color = 'b' if hist2[0] >= 0 else 'g'

    ax[0].hist(hist0, bins=50, orientation='horizontal', density=True, color=color, alpha=0.5)
    ax[2].hist(hist2, bins=50, orientation='horizontal', density=True, color=color, alpha=0.5)

    ax[0].invert_xaxis()


def PlotTest(ax, timesteps, ndata, ncurves, critical_time):

    distros = TrueDynamics(timesteps, ndata)
    reduced_distros = distros[::-1,::ndata//ncurves]

    reduced_timesteps = np.arange(timesteps)/critical_time

    PlotCorrelations(distros, reduced_timesteps)

    ax.plot(reduced_timesteps, reduced_distros, c="k", alpha = 0.5, label="true dynamics")