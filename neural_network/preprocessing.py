import torch

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def CreateDataloader(data,target):

  g = torch.Generator(device=data.device)
  data_ds = TensorDataset(data,target)
  data_dl = DataLoader(data_ds, batch_size=2000, shuffle=True, generator=g)

  return data_dl

def Preprocessing(data, target):

  data = torch.tensor(data.reshape(-1,2), dtype=torch.float32)
  target = torch.tensor(target, dtype=torch.float32)

  train_data_, test_data, train_target_, test_target = train_test_split(data, target, test_size=0.2)

  train_data, valid_data, train_target, valid_target = train_test_split(train_data_, train_target_, test_size=0.2)
  train_dl = CreateDataloader(train_data, train_target)
  valid_dl = CreateDataloader(valid_data, valid_target)
  
  return train_dl, valid_dl, test_data, test_target