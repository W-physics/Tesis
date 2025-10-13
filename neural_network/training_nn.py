import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

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
      
        t_batch_loss = 0
        v_batch_loss = 0

        for x_train_batch, y_train_batch in train_dl:

            t_batch_loss += TrainStep(model, x_train_batch, y_train_batch, loss_fn, optimizer)
    
        for x_val_batch, y_val_batch in valid_dl:
         
            v_batch_loss += ValStep(model, x_val_batch, y_val_batch, loss_fn)
        

        loss_hist_train[epoch] = t_batch_loss / len(train_dl)
        loss_hist_valid[epoch] = v_batch_loss / len(valid_dl)

    return loss_hist_train, loss_hist_valid


def TrainStep(model, x_batch, y_batch, loss_fn, optimizer):

    model.train()

    train_pred = model(x_batch)
    train_loss = loss_fn(train_pred, y_batch)
   
    train_loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    return train_loss.item()

def ValStep(model, x_batch, y_batch, loss_fn):
   
   model.eval()

   val_pred = model(x_batch)

   val_loss = loss_fn(val_pred, y_batch)

   return val_loss.item()

def TrainModel(timesteps, ndata, initial_distribution):

    model = FeedForward(input_size=2,output_size=1,n_hidden_layers=2,depht=200).to(device)

    features, noise = GenerateNoisedData(timesteps, ndata, initial_distribution)

    features = features.reshape(-1,2)
    noise = noise.reshape(-1)
    
    scaler = StandardScaler()

    features = scaler.fit_transform(features)

    train_dl, valid_dl, test_feature, test_target = Preprocessing(features, noise)



    loss_hist_train, loss_hist_valid = Train(learning_rate=0.01, model=model, num_epochs=30,
                                           train_dl=train_dl, valid_dl=valid_dl
                                           )

    pred = model(test_feature)

    loss_fn = nn.MSELoss()

    test_loss = loss_fn(pred, test_target).item()

    print(f'test error:  {test_loss}')

    return model, loss_hist_train, loss_hist_valid, scaler

