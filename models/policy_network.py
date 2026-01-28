# ================================================
# Pathogen/models/policy_network.py
# Graph Policy Network for Bacterial Adaptation
# ================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    class GCNConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels)
        def forward(self, x, edge_index):
            return self.linear(x)


class BioPolicyNet(nn.Module):
    """
    BioPolicyNet: A biologically grounded Graph Policy Network.
    Each node = bacterial isolate; edges = similarity relationships.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_actions: int = 3):
        super(BioPolicyNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # ---- Input embedding (maps raw features → latent biological space)
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # ---- Graph feature propagation (after embedding)
        self.gc1 = GCNConv(hidden_dim, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, hidden_dim)

        # ---- Regularization & normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.2)

        # ---- Policy head (produces action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        x: Node features (num_nodes × input_dim)
        edge_index: Graph edges (2 × num_edges)
        returns: Action probabilities for each node (num_nodes × num_actions)
        """

        # Step 1. Biological embedding
        x = F.relu(self.embedding(x))

        # Step 2. Graph message passing
        z1 = F.relu(self.gc1(x, edge_index))
        z1 = self.dropout(z1)

        # Step 3. Second propagation + residual normalization
        z2 = F.relu(self.gc2(z1, edge_index))
        z2 = self.layer_norm(z2 + z1)

        # Step 4. Policy projection
        logits = self.policy_head(z2)
        probs = F.softmax(logits, dim=-1)

        return probs
