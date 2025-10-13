import torch
import pandas as pd

def SaveCheckpoint(checkpoint):
   torch.save(checkpoint, 'data/checkpoint.pth.tar')

def SaveCSV(data, name):
   df_data = pd.DataFrame(data)
   df_data.to_csv("data/" + name + ".csv", index=False, header=False)