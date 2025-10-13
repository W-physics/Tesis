import pandas as pd
import numpy as np

def SaveCSV(data, name):
   df_data = pd.DataFrame(data)
   df_data.to_csv("data/" + name + ".csv", index=False, header=False)

def LoadCSV(path):
   return pd.read_csv(path, header=None).to_numpy
