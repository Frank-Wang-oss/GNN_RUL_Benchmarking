import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm



class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, A):
        # Add self-loops to the adjacency matrix

        A = A + torch.eye(A.size(1)).to(A.device)
        # Normalize the adjacency matrix
        D = torch.diag_embed(torch.sum(A, dim=-1) ** -0.5)
        A_hat = torch.bmm(D, torch.bmm(A, D))
        AX = torch.bmm(A_hat, X)
        out = self.linear(AX)
        return F.leaky_relu(out)



class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha=0.1):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.linear = nn.Linear(in_features, out_features)
        self.attention = nn.Linear(2 * out_features, 1)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = self.linear(h)  # bs, N, out_features
        e = self._prepare_attentional_mechanism_input(Wh)

        attention = F.softmax(e, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Multiply attention coefficients by the adjacency matrix

        attention = attention * adj

        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        bs, N, out_features = Wh.size()
        Wh1 = Wh.unsqueeze(2).repeat(1, 1, N, 1).view(bs, N * N, out_features)
        Wh2 = Wh.unsqueeze(1).repeat(1, N, 1, 1).view(bs, N * N, out_features)
        e = torch.cat([Wh1, Wh2], dim=2)
        e = self.leakyrelu(self.attention(e)).view(bs, N, N)

        return e

class GAT(nn.Module):
    def __init__(self, nfeat, nout, dropout, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nout, dropout=dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = torch.stack([att(x, adj) for att in self.attentions], dim=0)
        x = x.mean(0)
        return x
# Define the TCN layer
# Define Chomp1d to handle the padding in TemporalBlock
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


# Define the Multi-head Temporal Encoder
class MultiHeadTemporalEncoder(nn.Module):
    def __init__(self, num_heads, num_features):
        super(MultiHeadTemporalEncoder, self).__init__()
        self.num_heads = num_heads
        self.linears = nn.ModuleList([nn.Linear(num_features, 1) for _ in range(self.num_heads)])

    def forward(self, x):
        bs, N, L = x.size()
        outputs = []
        for i in range(self.num_heads):
            x_t = torch.transpose(x,-1,-2)
            temp = torch.sigmoid(self.linears[i](x_t)).view(bs, 1, L)
            temp = F.softmax(temp, dim=-1)
            output = temp * x
            outputs.append(output)
        outputs = torch.stack(outputs,0)
        return torch.mean(outputs,0)


# Define the main model
class STAGNN_model(nn.Module):
    def __init__(self, num_nodes, time_length, hidden_dim, output_dim, num_heads, threshold):
        super(STAGNN_model, self).__init__()
        self.threshold = threshold
        self.gcn1 = GCNLayer(time_length, hidden_dim)
        self.gat1 = GAT(hidden_dim, hidden_dim, dropout=0, nheads=num_heads)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gat2 = GAT(hidden_dim, hidden_dim, dropout=0, nheads=num_heads)


        self.tcn1 = TemporalConvNet(num_inputs=num_nodes, num_channels=[hidden_dim,hidden_dim], kernel_size=2)
        self.temporal_encoder1 = MultiHeadTemporalEncoder(num_heads, hidden_dim)
        self.tcn2 = TemporalConvNet(num_inputs=hidden_dim, num_channels=[output_dim,output_dim], kernel_size=2)
        self.temporal_encoder2 = MultiHeadTemporalEncoder(num_heads, output_dim)
        self.fc = nn.Linear(hidden_dim * output_dim, 1)

    def create_adjacency_matrix(self, x):
        bs, N, L = x.size()
        x = x.permute(0, 2, 1)  # (bs, L, N)
        mean_x = x.mean(dim=1, keepdim=True)
        x_centered = x - mean_x
        cov_matrix = torch.bmm(x_centered.permute(0, 2, 1), x_centered) / (L - 1)  # (bs, N, N)
        adj = (cov_matrix > self.threshold).float()
        return adj

    def forward(self, x):
        adj = self.create_adjacency_matrix(x)


        # First GCN and GAT layer
        x = self.gcn1(x, adj)
        x = self.gat1(x, adj)

        # Second GCN and GAT layer
        x = self.gcn2(x, adj)
        x = self.gat2(x, adj)

        # TCN and Temporal Encoder layers
        x = self.tcn1(x)
        x = self.temporal_encoder1(x)

        x = self.tcn2(x)
        x = self.temporal_encoder2(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

if __name__ == '__main__':
    bs, N, L = 32, 14, 50  # Batch size, number of nodes, time length
    hidden_dim = 64
    output_dim = 10
    num_heads = 4
    threshold = 0.5

    model = STAGNN_model(N, L, hidden_dim, output_dim, num_heads, threshold)
    input_sample = torch.randn(bs, N, L)
    output = model(input_sample)
    print(output.size())  # Expected output size: (bs, 1)