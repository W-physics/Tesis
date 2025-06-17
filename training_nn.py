import torch
import torch.nn as nn

from two_deltas import FeedForward
from two_deltas import GenerateTwoDeltas, GenerateNoisedData
from two_deltas import Preprocessing, CreateDataloader


def Train(model,num_epochs,train_dl,valid_dl):

    loss_hist_train = np.zeros(num_epochs)
    loss_hist_valid = np.zeros(num_epochs)

    for epoch in range(num_epochs):

        for x_batch, y_batch in train_dl:

          pred = model(x_batch)
          #Define loss function
          loss = loss_fn(pred, y_batch)
          #Backpropagation
          loss.backward()
          #Apply gradient to the weights
          optimizer.step()
          #Make gradients zero
          optimizer.zero_grad()
          loss_hist_train[epoch] += loss.item()*y_batch.size(0)

        for x_batch, y_batch in valid_dl:

          pred = model(x_batch)
          loss = loss_fn(pred, y_batch)

          loss_hist_valid[epoch] += loss.item()*y_batch.size(0)

        loss_hist_train[epoch] /= len(train_dl.dataset)

    return loss_hist_train, loss_hist_valid

model = FeedForward(input_size=1,output_size=1,n_hidden_layers=2,depht=5)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn = nn.MSELoss()

initial_data = GenerateTwoDeltas(10000)
noised_data, noise = GenerateNoisedData(initial_data, drift_term=0.1, noise_level=0.1)
train_dl, valid_dl, test_dl = Preprocessing(noised_data, noise)

loss_hist_train,loss_hist_valid = Train(model=model, num_epochs=50, train_dl=train_dl, valid_dl=valid_dl)

plt.plot(loss_hist_train,label='train')
plt.plot(loss_hist_valid,label='valid')
plt.legend()
plt.savefig("/home/william/Github/Tesis/figures/train_valid.svg")