import torch
import torch.nn as nn
import numpy as np

from forward_process.neural_network import FeedForward
from forward_process.generate_noised_data import GenerateNoisedData
from forward_process.preprocessing import Preprocessing
from save_plot.save_files import SaveCSV

def Train(model,num_epochs,train_dl,valid_dl, patience=5, min_delta=0.001):
    
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    loss_fn = nn.MSELoss()

    loss_hist_train = np.zeros(num_epochs)
    loss_hist_valid = np.zeros(num_epochs)

    # Variables for early stopping

    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        
      model.train()  # Set model to training mode
      train_loss = 0.0

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
      loss_hist_valid[epoch] /= len(valid_dl.dataset)

      if loss_hist_valid[epoch] < best_loss - min_delta:
        best_loss = loss_hist_valid[epoch]
        epochs_no_improve = 0
        best_model_state = model.state_dict()  # Save the best model
      else:
        epochs_no_improve += 1

      if epochs_no_improve >= patience:
        print(f'Early stopping triggered after {epoch+1} epochs!')
        model.load_state_dict(best_model_state)  # Restore best model
        final_epoch = epoch
        break
    
    loss_hist_train = loss_hist_train[:final_epoch+1]
    loss_hist_valid = loss_hist_valid[:final_epoch+1]

    return loss_hist_train, loss_hist_valid

def TrainModel(timesteps, ndata, initial_distribution):

  model = FeedForward(input_size=1,output_size=1,n_hidden_layers=2,depht=5)

  noised_data, noise = GenerateNoisedData(timesteps, ndata, initial_distribution)
  train_dl, valid_dl, test_dl = Preprocessing(noised_data, noise)

  loss_hist_train,loss_hist_valid = Train(model=model, num_epochs=50,
                                           train_dl=train_dl, valid_dl=valid_dl)
  
  #Save the model parameters
  #checkpoint = {'model_state_dict': model.state_dict()}
  #'optimizer_state_dict': optimizer.state_dict()}
  
  #SaveCheckpoint(checkpoint)
  SaveCSV(loss_hist_train, "loss_hist_train")
  SaveCSV(loss_hist_valid, "loss_hist_valid")

  return model
#  return model, loss_hist_train, loss_hist_valid, test_dl

