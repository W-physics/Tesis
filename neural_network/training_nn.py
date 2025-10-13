import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

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

@torch.no_grad
def ValStep(model, x_batch, y_batch, loss_fn):
   
   model.eval()

   val_pred = model(x_batch)

   val_loss = loss_fn(val_pred, y_batch)

   return val_loss.item()


