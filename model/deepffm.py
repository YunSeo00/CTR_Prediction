import torch
import torch.nn as nn
from model.ffm import FeaturesLinear, FieldAwareFactorizationMachine
from model.deepfm import MultiLayerPerceptron

class DeepFieldAwareFactorizationMachineModel(nn.Module):
    def __init__(self, offsets, total_field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        # embedding
        self.embedding = nn.Embedding(total_field_dims, embed_dim)
        self.offsets = offsets
        nn.init.xavier_uniform_(self.embedding.weight.data)
        self.embed_output_dim = len(offsets) * embed_dim
        
        # fm part
        self.linear = FeaturesLinear(offsets, total_field_dims)
        self.ffm = FieldAwareFactorizationMachine(offsets, total_field_dims, embed_dim)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        
    def forward(self, x):
        embed_x = self.embedding(x + self.offsets.unsqueeze(0))
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))
    