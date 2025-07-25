#from critical.control_parameter import Theta

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def PlotTrainval(iter):

    loss_hist_train = pd.read_csv("data/loss_hist_train.csv", header=None).to_numpy()
    loss_hist_valid = pd.read_csv("data/loss_hist_valid.csv", header=None).to_numpy()

    fig, ax = plt.subplots()

    ax.set_xlabel("epochs")
    ax.set_ylabel("MSE")
    ax.set_title("Training and validation errors")

    ax.plot(loss_hist_train,label='train')
    ax.plot(loss_hist_valid,label='valid')
    ax.legend(fontsize='large')

    fig.savefig('figures/losses'+str(iter)+'.pdf')

    print(f"Training and validation losses plotted and saved to figures/losses{iter}.pdf")

def PlotViolin(iter):

    generated = pd.read_csv("data/generated_data.csv", header=None).to_numpy()
    separation = 50
#    means = np.mean(generated,axis=1)
    reduced_distros = generated[::separation]

    fig, ax = plt.subplots()
    ax.set_ylabel(r"$x_{t}$")
    ax.set_xlabel(r"$t$")
    ax.set_title("Violin plot of generated data via backward process")

    size = len(reduced_distros)

    ax.violinplot(reduced_distros.T,positions=separation*np.arange(size),widths=25)

    fig.savefig("figures/violin_plot"+str(iter)+".pdf")
    

    print(f"Violin plot of generated data plotted and saved to figures/violin_plot{iter}.pdf")


def PlotCritical(ndata):

    critical_time = 121
    ymin = -20
    ymax = 20

    generated = pd.read_csv("data/generated_data.csv", header=None).to_numpy()
    separation = 25
    reduced_distros = generated[::separation]

    fig, ax = plt.subplots()
    ax.set_ylabel(r"$x_{t}$")
    ax.set_xlabel(r"$t$")
    ax.set_title("Violin plot of critical time n = "+str(ndata))
    #ax.set_xlim(75,175)
    ax.set_ylim(ymin,ymax)

    size = len(reduced_distros)
    ax.vlines(critical_time, ymin=ymin, ymax=ymax, colors="red", linestyles='dashed', label='critical time')

    ax.violinplot(reduced_distros.T,positions=separation*np.arange(size),widths=15)
    ax.legend(fontsize='large')

    fig.savefig("figures/critical_time/n="+str(ndata)+".pdf")

    print(f"Violin plot of critical time n = {ndata} plotted and saved to figures/critical_time/n={ndata}.pdf")

'''
def PlotConsistency(time, timesteps):

    theta = Theta(time, timesteps)

    x = np.linspace(-1, 1, 100)

    right_side = (1 + theta**2) / (1 - theta**2) * x
    left_side = 2 * theta * np.tanh(x* theta / (1 - theta**2))

    fig, ax = plt.subplots()
    ax.set_title(f"Plot of the consistency equation with theta = {theta:.2f}")

    ax.plot(x, right_side, label="right_side")
    ax.plot(x, left_side, label="left_side")
    ax.legend(fontsize='large')

    fig.savefig("figures/consistency_plot.pdf")
''' 