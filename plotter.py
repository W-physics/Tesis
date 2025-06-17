import pandas as pd

loss_hist_train = pd.read_csv("data\d-loss_hist_train.csv")
loss_hist_valid = pd.read_csv("data\d-loss_hist_valid.csv")

plt.plot(loss_hist_train,label='train')
plt.plot(loss_hist_valid,label='valid')
plt.legend()
plt.savefig("/home/william/Github/Tesis/figures/train_valid.svg")