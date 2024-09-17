import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_geometric.nn as pyg_nn
# from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
import numpy as np
from torch.nn.utils import weight_norm

def pcc_graph_construction(data):
    # Input shape: (bs, N, f)
    bs, N, f = data.size()

    # Center the features by subtracting the mean
    mean = data.mean(dim=-1, keepdim=True)  # Shape: (bs, N, 1)
    centered_tensor = data - mean  # Shape: (bs, N, f)

    # Compute the dot product between all pairs of feature vectors
    dot_product = torch.bmm(centered_tensor, centered_tensor.transpose(1, 2))  # Shape: (bs, N, N)

    # Compute the norms of the feature vectors
    norms = torch.norm(centered_tensor, dim=-1, keepdim=True)  # Shape: (bs, N, 1)
    norms_product = torch.bmm(norms, norms.transpose(1, 2))  # Shape: (bs, N, N)

    # Normalize by the product of the norms
    pcc = dot_product / norms_product

    return pcc

class MPNN_mk(nn.Module):
    def __init__(self, input_dimension, output_dinmension, k):
        ### In GCN, k means the size of receptive field. Different receptive fields can be concatnated or summed
        ### k=1 means the traditional MPNN
        super(MPNN_mk, self).__init__()
        self.k = k
        theta = []
        for kk in range(self.k):
            theta.append(nn.Linear(input_dimension, output_dinmension))
        self.theta = nn.ModuleList(theta)

    def forward(self, X, A):
        ## size of X is (bs, N, A)
        ## size of A is (bs, N, N)
        GCN_output_ = []
        for kk in range(self.k):
            if kk == 0:
                A_ = A
            else:
                A_ = torch.bmm(A_,A)
            out_k = self.theta[kk](torch.bmm(A_,X))
            GCN_output_.append(out_k)

        GCN_output_ = sum(GCN_output_)

        return F.leaky_relu(GCN_output_)


class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(CNNLayer, self).__init__()
        # padding = ((stride - 1) * 1 + kernel_size - stride) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same', stride=stride)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x = torch.transpose(x, -1,-2)
        x = self.conv(x)
        x = self.bn(x)
        # x = torch.transpose(x, -1,-2)

        return F.relu(x)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size):
        super(TemporalConvNet, self).__init__()

        in_channels0 = num_inputs
        out_channels0 = num_channels[1]
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

        in_channels1 = num_channels[0]
        out_channels1 = num_channels[1]
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

# class TCNLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation = 2, stride=1):
#         super(TCNLayer, self).__init__()
#         self.tcn = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
#                               padding='same', stride=stride, dilation=dilation)
#         self.bn = nn.BatchNorm1d(out_channels)
#
#     def forward(self, x):
#         # x = torch.transpose(x, -1,-2)
#
#         x = self.tcn(x)
#         x = self.bn(x)
#         # x = torch.transpose(x, -1,-2)
#
#         return F.relu(x)

class ST_Conv_model(nn.Module):
    def __init__(self, num_nodes, time_length, kernel_size):
        super(ST_Conv_model, self).__init__()
        self.gcn_layer_1 = MPNN_mk(time_length, time_length, k=1)
        self.cnn_layer_1 = CNNLayer(num_nodes, num_nodes, kernel_size)
        self.tcn_layer_1 = TemporalConvNet(num_nodes, [num_nodes,num_nodes], kernel_size)

        self.gcn_layer_2 = MPNN_mk(time_length, time_length, k=1)
        self.cnn_layer_2 = CNNLayer(num_nodes, num_nodes, kernel_size)
        self.tcn_layer_2 = TemporalConvNet(num_nodes, [num_nodes,num_nodes], kernel_size)


        self.theta1 = nn.Parameter(torch.randn(1))
        self.theta2 = nn.Parameter(torch.randn(1))
        self.theta3 = nn.Parameter(torch.randn(1))
        self.theta4 = nn.Parameter(torch.randn(1))


        self.fc = nn.Linear(num_nodes*time_length,1)


    def forward(self, x):
        bs, N, f = x.size()

        # GCN branch
        adj = pcc_graph_construction(x)
        gcn_output_1 = self.gcn_layer_1(x, adj)
        gcn_output_1 = self.cnn_layer_1(gcn_output_1)
        # TCN branch
        tcn_output_1 = self.tcn_layer_1(x)

        # GCN branch
        adj = pcc_graph_construction(x)
        gcn_output_2 = self.gcn_layer_1(x, adj)
        gcn_output_2 = self.cnn_layer_1(gcn_output_2)
        # TCN branch
        tcn_output_2 = self.tcn_layer_1(x)


        # Combine TCN and GCN outputs
        combined_output = torch.tanh(self.theta1 * tcn_output_1 + self.theta2 * gcn_output_1) * torch.sigmoid(self.theta3 * tcn_output_2 + self.theta4 * gcn_output_2)
        residual_output = combined_output+x

        ## Output

        residual_output = residual_output.view(bs,-1)

        predicted_rul = self.fc(residual_output)

        return predicted_rul
if __name__ == '__main__':

    batch_size, N, f = 32, 14, 50

    data = torch.randn((batch_size, N, f))

    model = ST_Conv_model(14,50,6)
    output = model(data)

    print(output.shape)