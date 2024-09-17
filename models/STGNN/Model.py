import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Gaussian kernel weight function to compute adjacency matrix
def compute_adjacency_matrix(input, top_k):
    bs, L, N, f = input.size()

    # Compute pairwise distances using broadcasting
    input_flat = input.reshape(bs * L, N, f)
    distances = torch.cdist(input_flat, input_flat, p=2)

    # Compute similarities using the Gaussian kernel
    similarities = torch.exp(-distances ** 2)
    similarities = similarities.view(bs, L, N, N)

    # Retain top-k similarities
    topk_indices = similarities.topk(k=top_k, dim=-1).indices
    topk_mask = torch.zeros_like(similarities)
    topk_mask.scatter_(-1, topk_indices, 1)
    adjacency_matrix = similarities * topk_mask

    return adjacency_matrix


# ChebNet class provided by user
class ChebNet(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super(ChebNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K

        self.filters = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.filters)

    def forward(self, x, adj_matrix):
        # x: (bs, N, in_channels)
        # adj_matrix: (bs, N, N)

        bs, N, in_channels = x.size()

        # Compute Chebyshev polynomials
        Tx_0 = x  # T0(x)
        Tx_1 = torch.bmm(adj_matrix, x)  # T1(x)

        out = torch.matmul(Tx_0, self.filters[0])
        if self.K > 1:
            out += torch.matmul(Tx_1, self.filters[1])

        for k in range(2, self.K):
            Tx_2 = 2 * torch.bmm(adj_matrix, Tx_1) - Tx_0  # T_k(x) = 2 * A * T_{k-1}(x) - T_{k-2}(x)
            out += torch.matmul(Tx_2, self.filters[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out


class STGNN_model(nn.Module):
    def __init__(self, patch_size, num_patch, num_nodes, hidden_dim, K, top_k):
        super(STGNN_model, self).__init__()
        self.num_patch = num_patch
        self.patch_size = patch_size

        self.top_k = top_k
        self.chebnet = ChebNet(patch_size, hidden_dim, K)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim*num_patch*num_nodes, 1)

    def forward(self, x):
        bs, num_node, time_length_origin = x.size()


        x = torch.reshape(x, [bs, num_node, self.num_patch, self.patch_size])
        x = torch.transpose(x, 1, 2)

        bs, L, N, f = x.size()

        # Compute adjacency matrix for each graph
        adj_matrix = compute_adjacency_matrix(x, self.top_k)

        # Reshape input for ChebNet
        x_reshaped = x.reshape(bs * L, N, f)
        adj_matrix_reshaped = adj_matrix.view(bs * L, N, N)

        # Process each graph with ChebNet
        chebnet_output = self.chebnet(x_reshaped, adj_matrix_reshaped)

        # Reshape output for GRU
        chebnet_output_reshaped = chebnet_output.view(bs, L, N, -1).permute(0, 2, 1, 3).contiguous()
        chebnet_output_reshaped = chebnet_output_reshaped.view(bs * N, L, -1)

        # Process sequences with GRU
        gru_output, _ = self.gru(chebnet_output_reshaped)

        # Take the last output of GRU
        final_output = gru_output.reshape(bs, -1)
        #
        # # Fully connected layer to predict RUL
        rul = self.fc(final_output)
        #
        return rul


if __name__ == '__main__':
    # Example usage
    bs, L, N, f = 32, 10, 14, 5  # batch size, sequence length, number of nodes, feature dimension
    top_k = 3
    in_channels = f
    hidden_channels = 64
    out_channels = 1
    K = 3
    input_data = torch.randn(bs, N, f*L)

    model = STGNN_model(in_channels, L, N, hidden_channels, K, top_k)
    output = model(input_data)
    print(output.shape)  # Expected output shape: (bs, N, out_channels)
