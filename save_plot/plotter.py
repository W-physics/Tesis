
from backward_process.correlation import GetCorrelations
from test.true_dynamics import TrueDynamics


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def PlotCritical(timesteps, ndata):

    critical_time = timesteps - 158
    ymin = -5
    ymax = 5


    fig, ax = plt.subplots()
    ax.set_ylabel(r"$x_{s}$")
    ax.set_xlabel(r"$s/s_c$")
    ax.set_title("Trajectorie plot of critical time n = "+str(ndata))
    #ax.set_xlim(75,175)
    #ax.set_ylim(ymin,ymax)

    ncurves = 20

    PlotSim(ax, ndata, ncurves, critical_time)
    #PlotTest(ax, timesteps, ndata, ncurves, critical_time)
    #ax.legend()

    fig.savefig("figures/trajectories/n="+str(ndata)+".svg")

    print(f"Violin plot of critical time n = {ndata} plotted and saved to figures/trajectories/n={ndata}.pdf")

def PlotSim(ax, ndata, ncurves, critical_time):

    distros = pd.read_csv("data/generated_data.csv", header=None).to_numpy()
    reduced_distros = distros[::-1,::ndata//ncurves]
    timesteps = distros.shape[0]
    reduced_timesteps = np.arange(timesteps)/critical_time

    ax.plot(reduced_timesteps,reduced_distros, c="b", alpha = 0.5, label="simulated")

def PlotTest(ax, timesteps, ndata, ncurves, critical_time):

    distros = TrueDynamics(timesteps, ndata)
    reduced_distros = distros[::-1,::ndata//ncurves]

    reduced_timesteps = np.arange(timesteps)/critical_time


    ax.plot(reduced_timesteps, reduced_distros, c="k", alpha = 0.5, label="true dynamics")
    ax.legend()

def PlotCorrelations(ndata):

    generated = pd.read_csv("data/generated_data.csv", header=None).to_numpy()
    correlations = GetCorrelations(generated)

    fig2, ax2 = plt.subplots()

    ax2.set_ylabel(r"$D_{s}$")
    ax2.set_xlabel(r"$s$")
    ax2.set_title("Correlations at n = "+str(ndata))
    ax2.plot(correlations)

    fig2.savefig("figures/correlations/n="+str(ndata)+".svg")

    print(f"Correlations at critical time n = {ndata} plotted and saved to figures/correlations/n={ndata}.pdf")

""""
def PlotViolin(iter):

    generated = pd.read_csv("data/generated_data.csv", header=None).to_numpy()
    separation = 50
#    means = np.mean(generated,axis=1)
    reduced_distros = generated[::separation]

    fig, ax = plt.subplots()
    ax.set_ylabel(r"$x_{s}$")
    ax.set_xlabel(r"$s$")
    ax.set_title("Violin plot of generated data via backward process")

    size = len(reduced_distros)

    ax.violinplot(reduced_distros.T,positions=separation*np.arange(size),widths=25)

    fig.savefig("figures/violin_plot"+str(iter)+".svg")
    

    print(f"Violin plot of generated data plotted and saved to figures/violin_plot{iter}.pdf")
"""