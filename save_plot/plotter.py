import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def PlotTrainval(iter):

    plt.style.use('bmh')

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

    print("Training and validation losses plotted and saved to figures/losses.pdf")

def PlotViolin(iter):

    generated = pd.read_csv("data/generated_data.csv", header=None).to_numpy()

#    means = np.mean(generated,axis=1)
    reduced_distros = generated[::100]

    fig, ax = plt.subplots()

    ax.violinplot(reduced_distros.T,positions=100*np.arange(10),widths=75)

    fig.savefig("figures/violin_plot"+str(iter)+".pdf")

    print("Violin plot of generated data plotted and saved to figures/violin_plot.pdf")