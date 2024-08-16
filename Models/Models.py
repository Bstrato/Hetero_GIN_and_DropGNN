# adopted from https://github.com/KarolisMart/DropGNN

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GINConv

class HeteroGIN(nn.Module):
    def __init__(self, metadata, hidden_channels, num_layers, num_classes, in_channels_dict):
        super().__init__()
        self.convs = nn.ModuleList()
        self.lins = nn.ModuleDict()
        self.node_types, self.edge_types = metadata

        for node_type, in_channels in in_channels_dict.items():
            if in_channels > 0:
                self.lins[node_type] = nn.Linear(in_channels, hidden_channels)
            else:
                # For nodes without features, we'll use an embedding layer
                self.lins[node_type] = nn.Embedding(1, hidden_channels)  # Start with 1 node, expand later if needed

        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: GINConv(
                    nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                  nn.BatchNorm1d(hidden_channels), nn.ReLU(),
                                  nn.Linear(hidden_channels, hidden_channels)),
                    train_eps=True
                )
                for edge_type in self.edge_types
            }, aggr='sum')
            self.convs.append(conv)

        self.final_lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x_dict, edge_index_dict):
        h_dict = {}
        for node_type in self.node_types:
            if node_type in x_dict:
                x = x_dict[node_type]
                if isinstance(self.lins[node_type], nn.Embedding):
                    num_nodes = x.size(0)
                    if num_nodes > self.lins[node_type].num_embeddings:
                        self.lins[node_type] = nn.Embedding(num_nodes, self.lins[node_type].embedding_dim).to(x.device)
                    h = self.lins[node_type](torch.arange(num_nodes, device=x.device))
                else:
                    h = self.lins[node_type](x)
            else:
                # For node types without features, use the embedding layer
                num_nodes = max(edge_index[1].max().item() + 1
                                for edge_type, edge_index in edge_index_dict.items()
                                if edge_type[2] == node_type)
                if num_nodes > self.lins[node_type].num_embeddings:
                    self.lins[node_type] = nn.Embedding(num_nodes, self.lins[node_type].embedding_dim).to(
                        next(self.parameters()).device)
                h = self.lins[node_type](torch.arange(num_nodes, device=next(self.parameters()).device))
            h_dict[node_type] = h

        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {key: F.relu(h) for key, h in h_dict.items()}

        return self.final_lin(h_dict['author'])

class DropHeteroGIN(nn.Module):
    def __init__(self, metadata, hidden_channels, num_layers, num_classes, in_channels_dict, p, num_runs):
        super().__init__()
        self.convs = nn.ModuleList()
        self.lins = nn.ModuleDict()

        self.node_types, self.edge_types = metadata
        self.num_nodes_dict = {node_type: 0 for node_type in self.node_types}

        for node_type, in_channels in in_channels_dict.items():
            if in_channels > 0:
                self.lins[node_type] = nn.Linear(in_channels, hidden_channels)
            else:
                self.lins[node_type] = nn.Embedding(1, hidden_channels)  # Initialize with 1 node

        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in self.edge_types:
                conv_dict[edge_type] = GINConv(
                    nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                  nn.BatchNorm1d(hidden_channels), nn.ReLU(),
                                  nn.Linear(hidden_channels, hidden_channels)),
                    train_eps=True
                )
            conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(conv)

        self.final_lin = nn.Linear(hidden_channels, num_classes)
        self.p = p
        self.num_runs = num_runs

    def forward(self, x_dict, edge_index_dict):
        h_dict = {}
        for node_type in self.node_types:
            if node_type in x_dict:
                x = x_dict[node_type]
                if isinstance(self.lins[node_type], nn.Embedding):
                    num_nodes = x.size(0)
                    if num_nodes > self.lins[node_type].num_embeddings:
                        self.lins[node_type] = nn.Embedding(num_nodes, self.lins[node_type].embedding_dim).to(x.device)
                    h = self.lins[node_type](torch.arange(num_nodes, device=x.device))
                else:
                    h = self.lins[node_type](x)
            else:
                # For node types without features, use the embedding layer
                num_nodes = self.num_nodes_dict[node_type]
                if num_nodes > self.lins[node_type].num_embeddings:
                    self.lins[node_type] = nn.Embedding(num_nodes, self.lins[node_type].embedding_dim).to(
                        next(self.parameters()).device)
                h = self.lins[node_type](torch.arange(num_nodes, device=next(self.parameters()).device))
            h_dict[node_type] = h.unsqueeze(0).expand(self.num_runs, -1, -1).clone()

        for k in h_dict.keys():
            drop = torch.bernoulli(torch.ones_like(h_dict[k][:, :, 0]) * (1 - self.p)).bool()
            h_dict[k] = h_dict[k] * drop.unsqueeze(-1) / (1 - self.p)

        for conv in self.convs:
            h_dict_new = {}
            for edge_type in edge_index_dict.keys():
                src, _, dst = edge_type
                h_dict_new[dst] = conv.convs[edge_type](
                    (h_dict[src].view(-1, h_dict[src].size(-1)), h_dict[dst].view(-1, h_dict[dst].size(-1))),
                    edge_index_dict[edge_type]
                )
            for node_type in h_dict:
                if node_type in h_dict_new:
                    h_dict[node_type] = F.relu(
                        h_dict_new[node_type].view(self.num_runs, -1, h_dict[node_type].size(-1)))
                else:
                    h_dict[node_type] = F.relu(h_dict[node_type])

        h = h_dict['author'].mean(dim=0)
        return self.final_lin(h)
