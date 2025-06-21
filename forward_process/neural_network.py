import torch.nn as nn

class FeedForward(nn.Module):

  def __init__(self, input_size, output_size,n_hidden_layers,depht):
    super().__init__()

    #Setting input and output layers
    l = [0] * (2 * (n_hidden_layers + 2) - 1)
    l[0] = nn.Linear(input_size,depht)
    l[1] = nn.ReLU()
    l[-1] = nn.Linear(depht,output_size)

    #Assembling hidden layers
    i = 0
    while i < n_hidden_layers:
      l[2*(i+1)] = nn.Linear(depht,depht)
      l[2*(i+1) + 1] = nn.ReLU()
      i+=1
    #    try:
    self.model_list = nn.ModuleList(l)

  def forward(self,x):
    for layer in self.model_list:
      x = layer(x)
    return x