import torch
import torch.nn as nn
import numpy as np

torch.cuda.is_available()

device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'

torch.set_default_device(device)

from neural_network.neural_network import FeedForward
from forward_process.generate_noised_data import GenerateNoisedData
from neural_network.preprocessing import Preprocessing
from save_plot.save_files import SaveCSV

def Train(learning_rate, model, num_epochs, train_dl, valid_dl):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    loss_hist_train = np.zeros(num_epochs)
    loss_hist_valid = np.zeros(num_epochs)

    for epoch in range(num_epochs):

      model.train()
      train_loss = 0.0

      for x_batch, y_batch in train_dl:

        x = x_batch.view(x_batch.size(0), -1).detach().clone().requires_grad_(True)
        y = y_batch.view(y_batch.size(0), -1).detach().clone().requires_grad_(True)

        pred = model(x)
            #Define loss function
        loss = loss_fn(pred, y)
            #Backpropagation
        loss.backward()
            #Apply gradient to the weights
        optimizer.step()
            #Make gradients zero
        optimizer.zero_grad()
        loss_hist_train[epoch] = loss.item()

      for x_batch, y_batch in valid_dl:

        x = x_batch.view(x_batch.size(0), -1).detach().clone().requires_grad_(True)
        y = y_batch.view(y_batch.size(0), -1).detach().clone().requires_grad_(True)

        pred = model(x)
        loss = loss_fn(pred, y)

        loss_hist_valid[epoch] = loss.item()

    return loss_hist_train, loss_hist_valid

def TrainModel(timesteps, ndata, initial_distribution):

    model = FeedForward(input_size=2,output_size=1,n_hidden_layers=3,depht=200).to(device)

    features, noise = GenerateNoisedData(timesteps, ndata, initial_distribution)
    train_dl, valid_dl, test_feature, test_target = Preprocessing(features, noise)



    loss_hist_train, loss_hist_valid = Train(learning_rate=0.01, model=model, num_epochs=50,
                                           train_dl=train_dl, valid_dl=valid_dl
                                           )

    pred = model(test_feature.view(test_feature.size(0), -1))

    loss_fn = nn.MSELoss()

    test_loss = loss_fn(pred, test_target.view(test_target.size(0), -1))

    print(f'test error:  {test_loss}')

    return model, loss_hist_train, loss_hist_valid

