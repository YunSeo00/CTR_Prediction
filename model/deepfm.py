import torch
from torch import nn
from model.ffm import FeaturesLinear

class FactorizationMachine(nn.Module):
    
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x): # equation 2
        square_of_sum = sum(x, dim=1) ** 2
        sum_of_square = sum(x ** 2, dim=1)
        fm_value = square_of_sum - sum_of_square
        if self.reduce_sum:
            fm_value = sum(fm_value, dim=1, keepdim=True)
        return 0.5 * fm_value
    
class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class DeepFactorizationMachineModel(nn.Module):
    def __init__(self, offsets, total_field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        # embedding
        self.embedding = nn.Embedding(total_field_dims, embed_dim)
        self.offsets = offsets
        nn.init.xavier_uniform_(self.embedding.weight.data)
        self.embed_output_dim = len(offsets) * embed_dim
        
        # fm part
        self.linear = FeaturesLinear(offsets, total_field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        
    def forward(self, x):
        embed_x = self.embedding(x + self.offsets.unsqueeze(0))
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))