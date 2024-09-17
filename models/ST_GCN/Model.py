import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

def segment_and_compute_features(data):
    # Reshape data to (bs*num_windows, window_size)


    # Compute features in parallel
    max_val = torch.max(data, dim=1)[0]
    min_val = torch.min(data, dim=1)[0]
    ptp_val = max_val - min_val
    var_val = torch.var(data, dim=1)
    std_val = torch.std(data, dim=1)
    mean_val = torch.mean(data, dim=1)
    rms_val = torch.sqrt(torch.mean(data ** 2, dim=1))
    mean_abs_val = torch.mean(torch.abs(data), dim=1)
    skew_val = skew(data, dim=1)
    kurtosis_val = kurtosis(data, dim=1)

    # Stack features
    features = torch.stack([
        max_val,
        min_val,
        ptp_val,
        var_val,
        std_val,
        mean_val,
        rms_val,
        mean_abs_val,
        skew_val,
        kurtosis_val
    ], dim=-1)
    # print(features)
    # Reshape back to (bs, num_windows, num_features)
    return features


def skew(data, dim):
    mean = torch.mean(data, dim=dim, keepdim=True)
    std = torch.std(data, dim=dim, keepdim=True)
    skewness = torch.mean(((data - mean) / std) ** 3, dim=dim)
    return skewness


def kurtosis(data, dim):
    mean = torch.mean(data, dim=dim, keepdim=True)
    std = torch.std(data, dim=dim, keepdim=True)
    kurt = torch.mean(((data - mean) / std) ** 4, dim=dim) - 3
    return kurt
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
    def __init__(self, input_dimension, output_dimension, k):
        super(MPNN_mk, self).__init__()
        self.k = k
        self.theta = nn.ModuleList([nn.Linear(input_dimension, output_dimension) for _ in range(k)])

    def forward(self, X, A):
        GCN_output = []
        for kk in range(self.k):
            if kk == 0:
                A_ = A
            else:
                A_ = torch.bmm(A_, A)
            out_k = self.theta[kk](torch.bmm(A_, X))
            GCN_output.append(out_k)
        GCN_output = sum(GCN_output)
        return F.leaky_relu(GCN_output)

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


class SG_TCN(nn.Module):
    def __init__(self, in_features, num_patch, num_layers=5, dropout=0.2, k=1):
        super(SG_TCN, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                MPNN_mk(num_patch, num_patch, k),
                TemporalConvNet(in_features, [in_features,in_features], kernel_size=2),
                nn.Dropout(dropout)
            ]))

    def forward(self, x, adj):
        out = x
        for mpnn, tcn, dropout in self.layers:
            res = out
            out = mpnn(out, adj)
            out = tcn(out)
            out = dropout(out)
            out += res
        return out

class ST_GCN_model(nn.Module):
    def __init__(self, num_patch, patch_size, num_layers=2, dropout=0.5, k=1):
        super(ST_GCN_model, self).__init__()
        self.num_patch = num_patch
        self.patch_size = patch_size
        in_features = 10
        self.sg_tcn = SG_TCN(in_features, num_patch, num_layers, dropout, k)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(num_patch, num_patch)
        self.fc2 = nn.Linear(num_patch, 1)

    def forward(self, x):
        ## x size (bs, time_length)
        bs = x.size(0)
        x = x.reshape(bs, self.num_patch, self.patch_size)
        x = x.reshape(bs*self.num_patch, self.patch_size)
        x = segment_and_compute_features(x)
        x = x.reshape(bs, self.num_patch, -1)
        x = x.transpose(-1,-2)
        adj = pcc_graph_construction(x)
        out = self.sg_tcn(x, adj)
        out = out.permute(0, 2, 1)  # (bs, hidden_features, num_window)
        out = self.global_max_pool(out).squeeze(-1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    # Example usage
    x = torch.rand(30,2560)
    model = ST_GCN_model(64, 40)
    rul_prediction = model(x)
    print(rul_prediction.size())
    # rul_prediction shape: (bs, 1)
