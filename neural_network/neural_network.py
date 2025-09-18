import torch.nn as nn

class FeedForward(nn.Module):

  def __init__(self, input_size, output_size,n_hidden_layers,depht):
    super().__init__()

    l = [nn.Linear(input_size,depht), nn.ReLU()]
    i = 0
    while i < n_hidden_layers:
      l.append(nn.Linear(depht, depht))
      l.append(nn.ReLU())
      i+=1
    last_layer = nn.Linear(depht, output_size)
    last_layer.bias.data.fill_(0)
    l.append(last_layer)

    #l.append(nn.Tanh())
    self.model_list = nn.ModuleList(l)

  def forward(self,x):
    for layer in self.model_list:
      x = layer(x)
    return x
