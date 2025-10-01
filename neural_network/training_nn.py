import torch
import torch.nn as nn
import numpy as np

torch.cuda.is_available()

device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'

torch.set_default_device(device)

from neural_network.neural_network import FeedForward
#from neural_network.neural_network import ConvNet
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

      for x_batch, y_batch in train_dl:

        pred = model(x_batch).view(-1)
            #Define loss function
        loss = loss_fn(pred, y_batch)
            #Backpropagation
        loss.backward()
            #Apply gradient to the weights
        optimizer.step()
            #Make gradients zero
        optimizer.zero_grad()
        loss_hist_train[epoch] = loss.item()

      for x_batch, y_batch in valid_dl:

        pred = model(x_batch).view(-1)
        loss = loss_fn(pred, y_batch)

        loss_hist_valid[epoch] = loss.item()

    return loss_hist_train, loss_hist_valid

def TrainModel(timesteps, ndata, initial_distribution):

    model = FeedForward(input_size=30,output_size=1,n_hidden_layers=3,depht=200).to(device)

    features, noise = GenerateNoisedData(timesteps, ndata, initial_distribution)


    train_dl, valid_dl, test_feature, test_target = Preprocessing(features, noise)



    loss_hist_train, loss_hist_valid = Train(learning_rate=0.0005, model=model, num_epochs=40,
                                           train_dl=train_dl, valid_dl=valid_dl
                                           )

    pred = model(test_feature).view(-1)

    loss_fn = nn.MSELoss()

    test_loss = loss_fn(pred, test_target).item()

    print(f'test error:  {test_loss}')

    return model, loss_hist_train, loss_hist_valid

