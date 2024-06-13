import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(GAT, self).__init__()

        self.conv1 = GATConv(in_channels, hidden_channels //2 // num_heads, heads=num_heads)
        # self.conv2 = GATConv(hidden_channels, hidden_channels //2 // num_heads, heads=num_heads)
        self.conv3 = GATConv(hidden_channels//2, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        ## self.dropout
        # x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

class Pyg_SelfAttention(MessagePassing):
    def __init__(self, in_channels, out_channels, activation):
        super(SelfAttention, self).__init__(aggr='add')
        
        self.activation = activation
        self.lin_query = torch.nn.Linear(in_channels, out_channels)
        self.lin_key = torch.nn.Linear(in_channels, out_channels)
        self.lin_value = torch.nn.Linear(in_channels, out_channels)

    def forward(self, node_presentation, edge_index):
        # Calculate query, key and value
        query = self.lin_query(node_presentation)
        key = self.lin_key(node_presentation)
        value = self.lin_value(node_presentation)

        # Calculate attention weights
        alpha = self.propagate(edge_index, x=(query, key), size=(node_presentation.size(0), node_presentation.size(0)))
        alpha = self.activation(alpha)

        # Apply attention weights to values
        attended_values = self.propagate(edge_index, x=(alpha, value), size=(node_presentation.size(0), node_presentation.size(0)))
        
        return attended_values

    def message(self, x_i, x_j):
        # Calculate attention weights
        alpha = torch.matmul(x_i, x_j.transpose(0, 1))

        return alpha

    def update(self, aggr_out):
        return aggr_out
    


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()

        self.activation = nn.Softmax()

        self.query = nn.Linear(in_dim, in_dim, bias=False)
        self.key = nn.Linear(in_dim, in_dim, bias=False)
        self.value = nn.Linear(in_dim, in_dim, bias=False)

    def forward(self, node_presentation):
        query = self.query(node_presentation)
        key = self.key(node_presentation)
        value = self.value(node_presentation)

        # Calculate attention weights
        attention_weights = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = self.activation(attention_weights)

        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, value)
        node = attended_values.sum(-1)
        return node
