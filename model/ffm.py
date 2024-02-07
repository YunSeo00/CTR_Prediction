import torch
from torch import nn

class FeaturesLinear(nn.Module):
    def __init__(self, offsets, total_field_dims, output_dim=1):
        super().__init__()
        self.offsets = offsets
        self.embedding = nn.Embedding(total_field_dims, output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        nn.init.xavier_uniform_(self.embedding.weight.data)
        
    def forward(self, x):
        x = x + self.offsets.unsqueeze(0)
        x = self.embedding(x)
        return torch.sum(x, dim=1) + self.bias
    
class FieldAwareFactorizationMachine(nn.Module):
    def __init__(self, offsets, total_field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(offsets)
        self.embeddings = nn.ModuleList([
            nn.Embedding(total_field_dims, embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = offsets
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        x = x + self.offsets.unsqueeze(0)
        w_f = [self.embeddings[i](x) for i in range(self.num_fields)] # weight for each field
        ffm_value = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ffm_value.append(w_f[j][:, i] * w_f[i][:, j])
        ffm_value = torch.stack(ffm_value, dim = 1)
        return ffm_value
    
class FieldAwareFactorizationMachineModel(nn.Module):
    def __init__(self, offsets, total_field_dims, embed_dim):
        super().__init__()
        self.linear = FeaturesLinear(offsets, total_field_dims)
        self.ffm = FieldAwareFactorizationMachine(offsets, total_field_dims, embed_dim)

    def forward(self, x):
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term
        return torch.sigmoid(x.squeeze(1))