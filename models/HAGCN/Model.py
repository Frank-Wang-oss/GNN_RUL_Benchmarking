import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class GINLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GINLayer, self).__init__()
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, adj):
        # x is (bs, N, f)
        # adj is (bs, N, N)
        x_in = x
        adj_in = adj

        out = torch.bmm(adj_in, x_in) + (1 + self.eps) * x_in
        out = self.mlp(out)
        return out

class Bi_LSTM_Standard(nn.Module):
    def __init__(self, input_dim, num_hidden, time_length):
        super(Bi_LSTM_Standard, self).__init__()
        self.num_hidden = 16
        self.input_dim = input_dim

        self.time_length = time_length
        self.bi_lstm1 = nn.LSTM(input_size=self.input_dim,
                                hidden_size=num_hidden,
                                num_layers=1,
                                batch_first=True,
                                dropout=0,
                                bidirectional=True)
        self.drop1 = nn.Dropout(p=0.2)

        self.bi_lstm2 = nn.LSTM(input_size=num_hidden,
                                hidden_size=num_hidden*2,
                                num_layers=1,
                                batch_first=True,
                                dropout=0,
                                bidirectional=True)
        self.drop2 = nn.Dropout(p=0.2)

        self.bi_lstm3 = nn.LSTM(input_size=num_hidden*2,
                                hidden_size=num_hidden,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)

        self.drop3 = nn.Dropout(p=0.2)

    def forward(self, x):
        x, hidden = self.bi_lstm1(x)
        x_split = torch.split(x, (x.shape[2] // 2), 2)
        x = x_split[0] + x_split[1]
        x, hidden = self.bi_lstm2(x)
        x_split = torch.split(x, (x.shape[2] // 2), 2)
        x = x_split[0] + x_split[1]
        x = self.drop2(x)

        x2, hidden = self.bi_lstm3(x)
        x2_presp = x2

        x2_split = torch.split(x2_presp, (x2_presp.shape[2] // 2), 2)
        x2 = x2_split[0] + x2_split[1]
        x2 = self.drop3(x2)

        return F.leaky_relu(x2)

class SAGPool(nn.Module):
    def __init__(self, input_dimension, output_dimension, n):
        super(SAGPool, self).__init__()
        self.rank = nn.Linear(input_dimension, 1)
        self.model = nn.Linear(input_dimension, output_dimension)
        self.n = n

        # Define the MLP for computing prior self-attention scores
        self.mlp = nn.Sequential(
            nn.Linear(input_dimension, input_dimension // 2),
            nn.ReLU(),
            nn.Linear(input_dimension // 2, 1)
        )

    def forward(self, X, A):
        x_in = X
        A_in = A

        # Compute the output using the adjacency matrix
        x_out = torch.bmm(A_in, x_in)
        x_out = F.leaky_relu(self.model(x_out))

        # Compute prior self-attention scores using the MLP
        P = self.mlp(X)
        P = torch.softmax(P, dim=1)
        P = P.squeeze()

        # Compute scores using self.rank
        x_score = torch.bmm(A, X)
        score = torch.softmax(self.rank(x_score), 1)
        score = score.squeeze()

        # Compute KL divergence
        kl_div = F.kl_div(P.log(), score, reduction='batchmean')

        # Sort and select top-k nodes based on the score
        _, idx = torch.sort(score, descending=True, dim=1)
        topk = idx[:, :self.n]

        bat_id = torch.arange(X.size(0)).unsqueeze(1)
        x_out = x_out[bat_id, topk]
        A_out = A_in[bat_id, topk]
        A_out = torch.transpose(A_out, 1, 2)
        A_out = A_out[bat_id, topk]

        return x_out, A_out, kl_div

def cosine_distance(matrix1):
    matrix1_ = torch.matmul(matrix1, matrix1.transpose(-1, -2))
    matrix1_norm = torch.sqrt(torch.sum(matrix1 ** 2, -1))
    matrix1_norm = matrix1_norm.unsqueeze(-1)
    cosine_distance = matrix1_ / (torch.matmul(matrix1_norm, matrix1_norm.transpose(-1, -2)))
    return cosine_distance

class HAGCN_model(nn.Module):
    def __init__(self, patch_size, num_patch, encoder_hidden_dim, hidden_dim, output_dim):
        super(HAGCN_model, self).__init__()
        self.patch_size = patch_size
        self.num_patch = num_patch

        self.TD = Bi_LSTM_Standard(patch_size, encoder_hidden_dim, None)
        self.gin1 = GINLayer(encoder_hidden_dim, hidden_dim)
        self.gnn1 = SAGPool(hidden_dim, hidden_dim, 10)
        self.gin2 = GINLayer(hidden_dim, hidden_dim)
        self.gnn2 = SAGPool(hidden_dim, hidden_dim, 5)
        self.gin3 = GINLayer(hidden_dim, hidden_dim)
        self.gnn3 = SAGPool(hidden_dim, hidden_dim, 1)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3 * num_patch, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, 1)
        )

    def forward(self, X, train = False):
        bs, num_node, time_length_origin = X.size()

        X = torch.reshape(X, [bs, num_node, self.num_patch, self.patch_size])
        X = torch.transpose(X, 1, 2)

        bs, tlen, num_node, dimension = X.size()

        X = torch.transpose(X, 1, 2)
        X = torch.reshape(X, [bs * num_node, tlen, dimension])
        X = torch.transpose(X, 1, 0)

        TD_output = self.TD(X)
        X = torch.transpose(TD_output, 1, 0)
        X = torch.reshape(X, [bs, num_node, tlen, -1])
        X = torch.transpose(X, 1, 2)

        A_input_ = torch.reshape(X, [bs * tlen, num_node, -1])
        adj0 = cosine_distance(A_input_)

        # Apply GIN layer and SAGPool layer sequentially
        gin_output1 = self.gin1(A_input_, adj0)
        out1, Adj1, kl_div1 = self.gnn1(gin_output1, adj0)

        gin_output2 = self.gin2(out1, Adj1)
        out2, Adj2, kl_div2 = self.gnn2(gin_output2, Adj1)

        gin_output3 = self.gin3(out2, Adj2)
        out3, Adj3, kl_div3 = self.gnn3(gin_output3, Adj2)

        # Concatenate the outputs of the GIN+SAGPool layers

        out1 = torch.mean(out1,1)
        out2 = torch.mean(out2,1)
        out3 = torch.mean(out3,1)

        out = torch.cat([out1, out2, out3], dim=-1).squeeze()
        out = torch.reshape(out, [bs, -1])

        output = self.fc(out)

        # Sum the KL divergences from each SAGPool layer
        total_kl_div = kl_div1 + kl_div2 + kl_div3
        if train:
            return output, total_kl_div
        else:
            return output

if __name__ == '__main__':
    bs,num_nodes,L = 32, 14, 50
    X = torch.rand(bs,num_nodes,L)
    net = HAGCN_model(patch_size=10, num_patch=5, encoder_hidden_dim=60, hidden_dim=64, output_dim=32)
    net(X)