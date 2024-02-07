import torch
import numpy as np

class LogisticRegressionWithEmbedding(torch.nn.Module):
    def __init__(self, offsets, total_field_dims, output_dim, embedding_dim):
        super(LogisticRegressionWithEmbedding, self).__init__()
        self.offsets = offsets
        self.embedding = torch.nn.Embedding(total_field_dims, embedding_dim)
        self.linear = torch.nn.Linear(embedding_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()
        
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        
        
    def forward(self, x):
        x = x + self.offsets.unsqueeze(0)
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = self.linear(x)
        y_pred = self.sigmoid(x)
        return y_pred.squeeze(1)
