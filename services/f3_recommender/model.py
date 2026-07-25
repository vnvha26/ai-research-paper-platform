"""Neural-network components used by F3 training."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(torch.nn.Module):
    """Two-layer GCN that turns paper features into graph-aware embeddings."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)


class EdgeDecoder(torch.nn.Module):
    """Scores a directed citation candidate by the dot product of its nodes."""

    def forward(self, embeddings, edge_index):
        source_vectors = embeddings[edge_index[0]]
        target_vectors = embeddings[edge_index[1]]
        return (source_vectors * target_vectors).sum(dim=-1)
