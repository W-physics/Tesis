import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('bmh')

loss_hist_train = pd.read_csv("data\d-loss_hist_train.csv", usecols=[1])
loss_hist_valid = pd.read_csv("data\d-loss_hist_valid.csv", usecols=[1])

fig, ax = plt.subplots()

ax.set_xlabel("epochs")
ax.set_ylabel("MSE")
ax.set_title("Training and validation errors")

ax.plot(loss_hist_train,label='train')
ax.plot(loss_hist_valid,label='valid')
ax.legend(fontsize='large')

plt.savefig("figures\losses.pdf")