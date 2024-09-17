import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNNLayer, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * input_dim, output_dim),
            nn.ReLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim + output_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        bs, L, N, f = x.size()

        # Prepare the node pairs
        x_i = x.unsqueeze(3).repeat(1, 1, 1, N, 1)
        x_j = x.unsqueeze(2).repeat(1, 1, N, 1, 1)

        # Concatenate the pairs
        edge_features = torch.cat([x_i, x_j], dim=-1)

        # Apply the edge MLP
        edge_features = self.edge_mlp(edge_features)

        # Sum edge features for each node
        edge_sum = edge_features.sum(dim=3)

        # Concatenate with original node features
        node_features = torch.cat([x, edge_sum], dim=-1)

        # Apply the node MLP
        node_features = self.node_mlp(node_features)

        return node_features


class GRU_CM_model(nn.Module):
    def __init__(self, time_length, num_nodes, gru_hidden_dim=128):
        super(GRU_CM_model, self).__init__()
        hidden_dim = int(num_nodes/2)

        self.input_linear = nn.Linear(1, hidden_dim)
        self.dropout1 = nn.Dropout(0.2)

        self.gnn = GNNLayer(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.2)

        self.gru = nn.GRU(hidden_dim, gru_hidden_dim, batch_first=True)
        self.dropout3 = nn.Dropout(0.2)

        self.output_linear = nn.Linear(gru_hidden_dim*time_length, 1)

    def forward(self, x):
        bs, N, L = x.size()
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (bs, L, N, 1)

        # Initial linear transformation
        x = self.input_linear(x)  # (bs, L, N, input_dim)
        x = self.dropout1(x)

        # GNN Layer
        x = self.gnn(x)  # (bs, L, N, gnn_output_dim)
        x = self.dropout2(x)

        # max pooling over nodes
        x,_ = torch.max(x,2)  # (bs, L, gnn_output_dim)
        # GRU Layer
        x, _ = self.gru(x)  # (bs, L, gru_hidden_dim)
        x = self.dropout3(x)

        x = x.reshape(bs,-1)

        # Output linear layer
        x = self.output_linear(x)  # (bs, L, output_dim)

        return x


if __name__ == '__main__':
    # Example usage
    bs, N, L = 32, 14, 50  # batch size, number of nodes, sequence length
    input_dim = 16
    gnn_output_dim = 32
    gru_hidden_dim = 64
    output_dim = 1

    model = GRU_CM_model(L, N, gru_hidden_dim)
    inputs = torch.randn(bs, N, L)
    outputs = model(inputs)
    print(outputs.shape)  # should be (bs, L, output_dim)
