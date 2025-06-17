from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def CreateDataloader(data,target):

  data_tensor = torch.tensor(data, dtype=torch.float32).reshape(-1,1)
  target_tensor = torch.tensor(target, dtype=torch.float32).reshape(-1,1)

  data_ds = TensorDataset(data_tensor,target_tensor)
  data_dl = DataLoader(data_ds, batch_size=1, shuffle=True)

  return data_dl

def Preprocessing(data, target):

  train_data_, test_data, train_target_, test_target = train_test_split(data, target, test_size=0.2)

  train_data, valid_data, train_target, valid_target = train_test_split(train_data_, train_target_, test_size=0.2)

  train_dl = CreateDataloader(train_data, train_target)
  valid_dl = CreateDataloader(valid_data, valid_target)
  test_dl = CreateDataloader(test_data, test_target)

  return train_dl, valid_dl, test_dl