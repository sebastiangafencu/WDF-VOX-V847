import torch
from torch import nn

class BJTModel(nn.Module):
    def __init__(self,
                 input_size: int = 2,
                 output_size: int = 2,
                 num_layers: int = 4,
                 hidden_dim: int = 8,
                 activation: str = "relu",
                 device: str = 'cpu'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = []
        in_dim = self.input_size
        for i in range(self.num_layers):
            out_dim = self.hidden_dim
            self.layers.append(torch.nn.Linear(in_dim, out_dim, bias=True))
            if activation == "leakyRelu":
                self.layers.append(torch.nn.LeakyReLU())
            elif activation == "tanh":
                self.layers.append(torch.nn.Tanh())
            elif activation == "relu":
                self.layers.append(torch.nn.ReLU())
            elif activation == "elu":
                self.layers.append(torch.nn.ELU())
            in_dim = out_dim

        self.layers.append(torch.nn.Linear(in_dim, self.output_size))
        self.model = nn.Sequential(*self.layers).to(device)

    def forward(self, x):
        x = self.model(x)
        return x
