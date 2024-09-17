import torch
import torch.nn as nn
import torch.nn.functional as F

# Graph Attention Layer
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

# PCC for graph construction
def pcc_graph_construction(data):
    # Input shape: (bs, T, N, f)
    bs, T, N, f = data.size()

    # Center the features by subtracting the mean
    mean = data.mean(dim=-1, keepdim=True)  # Shape: (bs, T, N, 1)
    centered_tensor = data - mean  # Shape: (bs, T, N, f)

    # Compute the dot product between all pairs of feature vectors
    dot_product = torch.bmm(centered_tensor.reshape(bs * T, N, f), centered_tensor.reshape(bs * T, f, N))  # Shape: (bs*T, N, N)

    # Compute the norms of the feature vectors
    norms = torch.norm(centered_tensor, dim=-1, keepdim=True)  # Shape: (bs, T, N, 1)
    norms_product = torch.bmm(norms.reshape(bs * T, N, 1), norms.reshape(bs * T, 1, N))  # Shape: (bs*T, N, N)

    # Normalize by the product of the norms
    pcc = dot_product / norms_product

    # Reshape PCC to (bs, T, N, N)
    pcc = pcc.reshape(bs, T, N, N)

    return pcc

# RUL prediction model
class GDAGDL_model(nn.Module):
    def __init__(self, num_patch, patch_size, num_nodes, nperseg, input_dim, gat_layer_dim, lstm_hidden_dim, autoencoder_hidden_dim, autoencoder_out_dim):
        super(GDAGDL_model, self).__init__()

        self.num_patch = num_patch
        self.patch_size = patch_size
        self.nperseg = nperseg

        gat_layer_dim = [input_dim] + gat_layer_dim

        # GAT layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(in_features=gat_layer_dim[i], out_features=gat_layer_dim[i+1], dropout=0.5) for i in range(len(gat_layer_dim)-1)
        ])

        # Linear layer for node importance
        self.node_importance_linear = nn.Linear(input_dim, 1)
        # Autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(gat_layer_dim[-1] * num_nodes, autoencoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(autoencoder_hidden_dim, autoencoder_hidden_dim//2),
            nn.ReLU(),
            nn.Linear(autoencoder_hidden_dim//2, autoencoder_hidden_dim//4),
            nn.ReLU(),
            nn.Linear(autoencoder_hidden_dim//4, autoencoder_out_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(autoencoder_out_dim, autoencoder_hidden_dim//4),
            nn.ReLU(),
            nn.Linear(autoencoder_hidden_dim//4, autoencoder_hidden_dim//2),
            nn.ReLU(),
            nn.Linear(autoencoder_hidden_dim//2, autoencoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(autoencoder_hidden_dim, gat_layer_dim[-1] * num_nodes)
        )

        # LSTM for RUL prediction
        self.lstm = nn.LSTM(input_size=autoencoder_out_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.linear = nn.Linear(lstm_hidden_dim * num_patch, 1)  # RUL prediction

    def forward(self, x, train=False):
        bs = x.size(0)  # batch size
        x = x.reshape(bs, self.num_patch, self.patch_size)
        x = x.reshape(bs * self.num_patch, self.patch_size)

        # STFT parameters
        nperseg = self.nperseg  # Length of each segment for STFT
        window = torch.hann_window(nperseg, periodic=True).to(x.device)
        nfft = nperseg

        # Compute STFT for the batch in parallel
        Zxx = torch.stft(x, n_fft=nfft, hop_length=nperseg, win_length=nperseg, window=window, return_complex=True)

        # Compute magnitude
        x = Zxx.abs()
        N, f = x.size(-2), x.size(-1)
        x = x.reshape(bs, self.num_patch, N, f)

        bs, T, N, f = x.shape

        # Graph construction using PCC
        # x = x.reshape(bs*T,N,f)
        adj = pcc_graph_construction(x)  # (bs, T, N, N)
        # x = x.reshape(bs,T,N,f)
        adj = adj.reshape(bs,T,N,N)

        # Linear layer for node importance
        node_importance = self.node_importance_linear(x.reshape(bs * T * N, f)).reshape(bs* T, N, 1)
        node_importance = adj.reshape(bs* T, N, N) @ node_importance  # (bs, T, N, N)
        node_importance = node_importance.reshape(bs, T, N)

        # Improve node importance based on connections
        high_importance_nodes = (node_importance > 0).float()
        adj = high_importance_nodes.unsqueeze(-1) * high_importance_nodes.unsqueeze(-2)


        # GAT processing
        h = x.reshape(bs * T, N, f)
        for gat_layer in self.gat_layers:
            h = gat_layer(h, adj.view(bs * T, N, N))
            h = F.elu(h)

        h = h.view(bs, T, N, -1)

        # Concatenate node features
        Y_o = h.view(bs, T, -1)  # (bs, T, N*f)

        # Autoencoder
        H = self.encoder(Y_o)  # (bs, T, f)
        Y_o_prime = self.decoder(H).view(bs, T, N, -1)  # (bs, T, N, f)

        # Reconstruction loss
        Y_o = Y_o.view(bs, T, N, -1)  # (bs, T, N*f)
        reconstruction_loss = F.mse_loss(Y_o, Y_o_prime)  # Use Y_o for MSE loss

        # RUL prediction
        lstm_out, _ = self.lstm(H)  # (bs, T, lstm_hidden_dim)
        rul_pred = self.linear(lstm_out.reshape(bs, -1))  # (bs, 1)

        if train:
            return rul_pred, reconstruction_loss
        else:
            return rul_pred

if __name__ == '__main__':
    # Example usage
    bs, num_patch, patch_size = 32, 10, 256
    node_feature_dim = 17
    gat_hidden_dim = [300,150,50]
    lstm_hidden_dim = 10
    autoencoder_hidden_dim = 256

    model = GDAGDL_model(num_patch, patch_size, 9, 16, node_feature_dim, gat_hidden_dim, lstm_hidden_dim, autoencoder_hidden_dim, 50)
    x = torch.randn(bs, 2560)  # Example input
    rul_pred, reconstruction_loss = model(x, True)

    print(rul_pred.size())
