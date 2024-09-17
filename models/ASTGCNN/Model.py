import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm

torch.backends.cudnn.benchmark = True  # might be required to fasten TCN

# class Chomp1d(nn.Module):
#     def __init__(self, chomp_size):
#         super(Chomp1d, self).__init__()
#         self.chomp_size = chomp_size
#
#     def forward(self, x):
#         return x[:, :, :-self.chomp_size].contiguous()
#
# class TemporalBlock(nn.Module):
#     def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
#         super(TemporalBlock, self).__init__()
#         self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
#         self.chomp1 = Chomp1d(padding)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout)
#
#         self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
#         self.chomp2 = Chomp1d(padding)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(dropout)
#
#         self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
#                                  self.conv2, self.chomp2, self.relu2, self.dropout2)
#         self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
#         self.relu = nn.ReLU()
#         self.init_weights()
#
#     def init_weights(self):
#         self.conv1.weight.data.normal_(0, 0.01)
#         self.conv2.weight.data.normal_(0, 0.01)
#         if self.downsample is not None:
#             self.downsample.weight.data.normal_(0, 0.01)
#
#     def forward(self, x):
#         out = self.net(x)
#         res = x if self.downsample is None else self.downsample(x)
#         return self.relu(out + res)
#
# class TemporalConvNet(nn.Module):
#     def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
#         super(TemporalConvNet, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = num_inputs if i == 0 else num_channels[i - 1]
#             out_channels = num_channels[i]
#             layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
#                                      padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
#         self.network = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.network(x)



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
class TemporalConvNet(nn.Module):
    def __init__(self, input_channels, tcn_layers, kernel_size):
        super(TemporalConvNet, self).__init__()

        in_channels0 = input_channels
        out_channels0 = tcn_layers[1]
        kernel_size = kernel_size
        stride = 1
        dilation0 = 1
        padding0 = (kernel_size - 1) * dilation0

        self.net0 = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(out_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
        )

        self.downsample0 = nn.Conv1d(in_channels0, out_channels0, 1) if in_channels0 != out_channels0 else None
        self.relu = nn.ReLU()

        in_channels1 = tcn_layers[0]
        out_channels1 = tcn_layers[1]
        dilation1 = 2
        padding1 = (kernel_size - 1) * dilation1
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
            nn.Conv1d(out_channels1, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
        )
        self.downsample1 = nn.Conv1d(out_channels1, out_channels1, 1) if in_channels1 != out_channels1 else None

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False, padding=padding0,
                      dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),

            # nn.Conv1d(out_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False,
            #           padding=padding0, dilation=dilation0),
            # Chomp1d(padding0),
            # nn.BatchNorm1d(out_channels0),
            # nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(out_channels0, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),

            # nn.Conv1d(out_channels1, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
            #           padding=padding1, dilation=dilation1),
            # Chomp1d(padding1),
            # nn.BatchNorm1d(out_channels1),
            # nn.ReLU(),
        )

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x0 = self.conv_block1(inputs)
        res0 = inputs if self.downsample0 is None else self.downsample0(inputs)
        out_0 = self.relu(x0 + res0)

        x1 = self.conv_block2(out_0)
        res1 = out_0 if self.downsample1 is None else self.downsample1(out_0)
        out_1 = self.relu(x1 + res1)

        out = out_1[:, :, :]
        return out
# class TemporalConvNet(nn.Module):
#     def __init__(self, num_channels, out, kernel_size=2, dropout=0.2):
#         super(TemporalConvNet, self).__init__()
#         self.conv = nn.Conv1d(in_channels=num_channels,
#                               out_channels=num_channels,
#                               kernel_size=kernel_size,
#                               stride=1,
#                               padding=(kernel_size - 1) // 2,
#                               groups=num_channels)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         # x = torch.transpose(x,-1,-2)
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         # x = torch.transpose(x,-1,-2)
#
#         return x


class GatingMechanism(nn.Module):
    def __init__(self, num_channels, out_channels):
        super(GatingMechanism, self).__init__()
        self.theta = nn.Linear(num_channels, out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, tcn_output):
        # x = torch.transpose(x,-1,-2)

        z = torch.tanh(self.theta(x) + self.bias)
        # z = torch.transpose(z,-1,-2)

        return z * tcn_output


class construct_graph(nn.Module):
    def __init__(self, num_features):
        super(construct_graph, self).__init__()
        self.P = nn.Linear(num_features, num_features, bias=False)

    def forward(self, X):
        bs, N, f = X.size()
        X_flat = X.view(bs, N, f)
        P_X = self.P(X_flat)
        distances = torch.cdist(P_X, P_X, p=2)
        adj_matrix = torch.exp(-distances)  # Gaussian Kernel
        return adj_matrix


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


class ASTGCNN_model(nn.Module):
    def __init__(self, num_nodes, time_length, encoder_out_dim, output_dim, K):
        super(ASTGCNN_model, self).__init__()
        self.tcn = TemporalConvNet(num_nodes, [num_nodes,num_nodes], kernel_size=6)
        self.gate = GatingMechanism(time_length, encoder_out_dim)
        self.distance_module = construct_graph(encoder_out_dim)
        self.chebnet = ChebNet(encoder_out_dim, output_dim, K)
        self.fc = nn.Linear(output_dim, 1)

    def forward(self, X):
        bs, N, L = X.size()

        tcn_out = self.tcn(X)

        gated_output = self.gate(X, tcn_out)

        adj_matrix = self.distance_module(gated_output)

        chebnet_out = self.chebnet(gated_output, adj_matrix)
        chebnet_out = chebnet_out.mean(dim=1)

        return self.fc(chebnet_out)


if __name__ == '__main__':
    model = ASTGCNN_model(14, 50,50, 64,3)

    # Example input
    X = torch.randn(32, 14, 50)  # (bs, N, L)

    # Forward pass
    output = model(X)
    print(output.shape)  # should be (32, 1)

