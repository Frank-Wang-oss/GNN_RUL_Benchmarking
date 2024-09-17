import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return F.relu(x)


def prior_knowledge_graph(device):
    adj_matrix = np.zeros((14, 14), dtype=int)

    # List of connections
    connections = [[1, 2], [1, 12], [1, 4], [1, 9], [1, 5], [1, 3],
                   [2, 4], [2, 7], [2, 8], [2, 13], [3, 14], [3, 13],
                   [3, 10], [3, 6], [4, 7], [4, 8], [5, 9], [5, 11],
                   [6, 10], [7, 8], [8, 13], [9, 11]]

    # Fill the adjacency matrix
    for connection in connections:
        i, j = connection
        adj_matrix[i-1][j-1] = 1
        adj_matrix[j-1][i-1] = 1

    adj_matrix = torch.from_numpy(adj_matrix).to(device)
    return adj_matrix



class STFA_model(nn.Module):
    def __init__(self, patch_size, num_patch, num_nodes, hidden_dim, output_dim, encoder_hidden_dim, device, num_heads, dropout):
        super(STFA_model, self).__init__()
        self.num_patch = num_patch
        self.patch_size = patch_size

        self.gat = GAT(patch_size, output_dim, nheads=num_heads, dropout=dropout)
        self.v = nn.Linear(output_dim*num_nodes, 1)
        self.lstm = nn.LSTM(output_dim * num_nodes + num_patch, encoder_hidden_dim, batch_first=True)
        self.fc = nn.Linear(encoder_hidden_dim, 1)
        self.adj = prior_knowledge_graph(device)

    def forward(self, x):
        bs, num_node, time_length_origin = x.size()
        x = torch.reshape(x, [bs, num_node, self.num_patch, self.patch_size])
        x = torch.transpose(x, 1, 2)


        bs, T, N, f = x.size()
        # print(x.size())
        adj = self.adj
        # GAT for each graph in the sequence
        x = x.reshape(bs * T, N, f)  # Reshape to (bs*T, N, f)
        # x = x.permute(1, 0, 2)  # Permute to (N, bs*T, f) for GAT
        adj = adj.unsqueeze(0).repeat(bs * T, 1, 1)  # Repeat adj for batch
        gat_output = self.gat(x, adj)  # (N, bs*T, out_features*num_heads)
        # gat_output = gat_output.permute(1, 0, 2)  # (bs*T, N, out_features*num_heads)
        gat_output = gat_output.view(bs, T, N, -1)  # (bs, T, N, out_features*num_heads)

        # Concatenate node features
        concat_features = gat_output.view(bs, T, -1)  # (bs, T, N*out_features*num_heads)

        # ASE Module
        tanh_features = torch.tanh(concat_features)  # (bs, T, N*out_features*num_heads)
        ase_weights = F.softmax(self.v(tanh_features),-1)  # (bs, T, N*out_features*num_heads)
        global_feature = ase_weights.view(bs,-1)  # (bs, N*out_features*num_heads)
        #
        # # Concatenate global feature with original GAT output
        final_features = torch.cat([global_feature.unsqueeze(1).repeat(1, T, 1), concat_features],
                                   dim=-1)  # (bs, T, 2*N*out_features*num_heads)
        #
        # # LSTM
        lstm_output, _ = self.lstm(final_features)  # (bs, T, lstm_hidden_size)
        final_output = self.fc(lstm_output[:, -1, :])  # (bs, 1)

        return final_output



if __name__ == '__main__':
    # Example usage
    bs, T, N, f = 32, 10, 14, 16  # Example dimensions
    adj = torch.zeros((N, N))  # Adjacency matrix

    x = torch.rand((bs, T, N, f)).to('cuda:0')  # Example input
    # model = GAT_RUL(in_features=f, out_features=32, hidden_size=64, lstm_hidden_size=128, num_heads=4, device='cuda:0').to('cuda:0')
    model = STFA_model(patch_size=f, num_patch= T, num_nodes=14, hidden_dim=32, output_dim=32, encoder_hidden_dim=32, device='cuda:0', num_heads=3, dropout=0.1).to('cuda:0')

    output = model(x)

    print(output.shape)  # Should print torch.Size([32, 1])

