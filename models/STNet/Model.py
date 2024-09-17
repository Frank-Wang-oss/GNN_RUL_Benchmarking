import torch
import torch.nn as nn
import torch.nn.functional as F


# ChebNet class as provided
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


# RUL prediction model
class STNet_model(nn.Module):
    def __init__(self, num_patch, patch_size, num_nodes, nperseg, input_dim, Cheb_layers, lstm_hidden_dim, autoencoder_hidden_dim):
        super(STNet_model, self).__init__()

        self.num_patch = num_patch
        self.patch_size = patch_size
        self.nperseg = nperseg

        # input_dim =

        Cheb_layers = [input_dim] + Cheb_layers

        # CNN to compute node weights
        # self.cnn = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=(1, 1, 1))
        self.cnn = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 1))
        # ChebNet layers
        self.chebnets = nn.ModuleList(
            [ChebNet(in_channels=Cheb_layers[i], out_channels=Cheb_layers[i+1], K=3) for i in range(len(Cheb_layers)-1)])

        # Autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(Cheb_layers[-1] * num_nodes, autoencoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(autoencoder_hidden_dim, autoencoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(autoencoder_hidden_dim, autoencoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(autoencoder_hidden_dim, autoencoder_hidden_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(autoencoder_hidden_dim, autoencoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(autoencoder_hidden_dim, autoencoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(autoencoder_hidden_dim, autoencoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(autoencoder_hidden_dim, Cheb_layers[-1] * num_nodes)
        )

        # LSTM for RUL prediction
        self.lstm = nn.LSTM(input_size=autoencoder_hidden_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.linear = nn.Linear(lstm_hidden_dim*num_patch, 1)  # RUL prediction

    def forward(self, x, train=False):
        bs = x.size(0)  # batch size
        x = x.reshape(bs, self.num_patch, self.patch_size)
        x = x.reshape(bs * self.num_patch, self.patch_size)

        # STFT parameters
        nperseg = self.nperseg  # Length of each segment for STFT
        window = torch.hann_window(nperseg, periodic=True).to(x.device)
        nfft = nperseg

        # Compute STFT for the batch in parallel
        Zxx = torch.stft(x, n_fft=nfft, hop_length=nperseg, win_length=nperseg, window=window,
                         return_complex=True)

        # Compute magnitude
        x = Zxx.abs()
        N, f = x.size(-2), x.size(-1)
        x = x.reshape(bs, self.num_patch, N, f)
        # print(f,N)

        bs, T, N, f = x.shape

        # Compute mean and max values for each node
        mean_vals = x.mean(dim=-1, keepdim=True)
        max_vals = x.max(dim=-1, keepdim=True).values
        node_features = torch.cat([mean_vals, max_vals], dim=-1)  # (bs, T, N, 2)

        ### Construct V1

        # # Reshape to compute adjacency matrix
        # node_features_i = node_features.unsqueeze(3)  # (bs, T, N, 1, 2)
        # node_features_j = node_features.unsqueeze(2)  # (bs, T, 1, N, 2)
        # combined_features = torch.cat([node_features_i.repeat(1, 1, 1, N, 1), node_features_j.repeat(1, 1, N, 1, 1)],
        #                               dim=-1)  # (bs, T, N, N, 4)
        #
        # # Compute node weights using CNN
        # node_weights = self.cnn(combined_features.permute(0, 4, 1, 2, 3)).squeeze(1)  # (bs, T, N, N)
        # adjacency_matrix = (node_weights > 0.7).float()  # (bs, T, N, N)



        ### Construct V2
        # Compute node weights using CNN
        node_weights = self.cnn(node_features.permute(0, 3, 1, 2)).squeeze(1)  # (bs, T, N)

        # Identify nodes with weights larger than 0.7
        high_weight_nodes = (node_weights > 0.7).float()  # (bs, T, N)

        # Create adjacency matrix in parallel
        high_weight_nodes_expanded = high_weight_nodes.unsqueeze(-1)  # (bs, T, N, 1)
        adjacency_matrix = high_weight_nodes_expanded @ high_weight_nodes_expanded.transpose(-1, -2)  # (bs, T, N, N)

        adjacency_matrix = adjacency_matrix.clamp(max=1.0)  # Ensure values are between 0 and 1


        # ChebNet processing
        # x = x.permute(0, 2, 1, 3)  # (bs, N, T, f)
        x = x.reshape(bs* T,N,-1)
        adjacency_matrix = adjacency_matrix.reshape(bs* T,N,-1)

        for chebnet in self.chebnets:
            x = chebnet(x, adjacency_matrix)

        # Concatenate node features
        Y_o = x.view(bs, T, -1)  # (bs, T, N*f)

        # Autoencoder
        H = self.encoder(Y_o)  # (bs, T, f)
        Y_o_prime = self.decoder(H).view(bs, T, N, -1)  # (bs, T, N, f)

        # Reconstruction loss
        Y_o = Y_o.view(bs, T, N, -1)  # (bs, T, N*f)

        reconstruction_loss = F.mse_loss(Y_o, Y_o_prime)  # Use Y_o for MSE loss

        # RUL prediction
        # print(H.size())
        lstm_out, _ = self.lstm(H)  # (bs, T, lstm_hidden_dim)
        rul_pred = self.linear(lstm_out.reshape(bs,-1))  # (bs, 1)

        if train:
            return rul_pred, reconstruction_loss
        else:
            return rul_pred


if __name__ == '__main__':
    # Usage example
    bs, num_patch, patch_size = 32, 10, 256

    node_feature_dim = 17
    chebnet_layers = 3
    lstm_hidden_dim = 10
    lstm_layers = 2
    autoencoder_hidden_dim = 50

    model = STNet_model(num_patch, patch_size, 9, 16, node_feature_dim, [300,200,100], lstm_hidden_dim, autoencoder_hidden_dim)
    x = torch.randn(bs, 2560)  # Example input
    rul_pred, reconstruction_loss = model(x,True)

    print(rul_pred.size())
