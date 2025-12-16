from torch import nn
from collections import OrderedDict

__all__ = [
    "MultiLayerPerceptron",
]

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation= nn.ReLU, dropout= 0.1, batch_norm=False):
        super().__init__()
        layers = OrderedDict()
        prev_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            layers[f'linear_{i}'] = nn.Linear(prev_dim, dim)
            
            if batch_norm:
                layers[f'batchnorm_{i}'] = nn.BatchNorm1d(dim)
            
            layers[f'activation_{i}'] = activation()

            if dropout > 0:
                layers[f'dropout_{i}'] = nn.Dropout(dropout)
            
            prev_dim = dim

        layers['output'] = nn.Linear(prev_dim, output_dim)
        self.network = nn.Sequential(layers)

    def forward(self, x):
        return self.network(x)
    
