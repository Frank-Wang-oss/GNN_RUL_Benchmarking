import torch
import torch.nn as nn
import torch.nn.functional as F



def SED_features(input_data, interval, band_width):


    # input_data shape: (bs, time_length)
    bs, time_length = input_data.size()

    # Perform FFT
    freq_spectrum = torch.fft.fft(input_data, dim=-1)

    # Calculate Spectral Difference (SD)
    X_t = freq_spectrum[:, interval:]
    X_t_m = freq_spectrum[:, :-interval]
    SD = X_t - X_t_m  # shape: (bs, time_length - interval)

    # Calculate Spectral Energy Difference (SED)
    squared_SD = SD.real ** 2 + SD.imag ** 2  # energy is the sum of squares of real and imaginary parts

    # print(squared_SD.size())
    # Calculate number of bands
    squared_SD = squared_SD.view(bs, -1, band_width)

    # Sum over the band width dimension to get the SED
    SED = squared_SD.sum(dim=-1)

    return SED


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
        out = F.leaky_relu(out)
        return out


class GRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GRULayer, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # GRU layer computation
        out, _ = self.gru(x)
        return out  # Get the last time step's output


class STMSGCN_model(nn.Module):
    def __init__(self, num_patch, patch_size, interval, band_width, gcn_dims, gru_hidden_dim):
        super(STMSGCN_model, self).__init__()
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.interval = interval
        self.band_width = band_width

        gcn_dims = [1] + gcn_dims

        self.gcn_dims = gcn_dims

        # Define GCN layers with different dimensions
        self.gcn_layers = nn.ModuleList([GCNLayer(gcn_dims[i],gcn_dims[i+1]) for i in range(len(gcn_dims) - 1)])

        # GRU input dimension is the sum of all GCN outputs and input graph features
        self.gru_layer = GRULayer(sum(gcn_dims), gru_hidden_dim, 1)

        # Final fully connected layer for RUL prediction
        self.fc = nn.Linear(gru_hidden_dim*num_patch, 1)

    def forward(self, x):
        bs = x.size(0)
        x = x.reshape(bs, self.num_patch, self.patch_size)
        x = x.reshape(bs * self.num_patch, self.patch_size)
        x = SED_features(x, self.interval, self.band_width)

        x = x.reshape(bs * self.num_patch, -1, 1)
        N = x.size(1)
        gcn_outputs = [x]

        # Pass through each GCN layer
        for gcn in self.gcn_layers:
            adj = torch.bmm(x, x.transpose(-1, -2))
            x = gcn(x, adj)
            gcn_outputs.append(x)

        # Concatenate the outputs of all GCN layers and the input graph features
        gcn_outputs = torch.cat(gcn_outputs, dim=-1)
        gcn_outputs = gcn_outputs.reshape(bs, self.num_patch, N, -1)
        gcn_outputs = gcn_outputs.transpose(1,2)
        gcn_outputs = gcn_outputs.reshape(bs* N, self.num_patch, -1)
        # Pass through GRU layer
        gru_out = self.gru_layer(gcn_outputs)

        gru_out = gru_out.reshape(bs, N, self.num_patch, -1)
        # Final prediction
        gru_out = gru_out.mean(1)
        output = self.fc(gru_out.reshape(bs,-1))
        return output

if __name__ == '__main__':
    # Example input parameters
    num_patch, patch_size = 10, 256
    interval, band_width = 6, 5
    batch_size = 32
    gcn_dims = [16, 64, 16, 1]  # Dimensions for each GCN layer
    gru_hidden_dim = 8

    # Random input data
    x = torch.randn(batch_size, num_patch* patch_size)

    # Instantiate and run the model
    model = STMSGCN_model(num_patch, patch_size, interval, band_width, gcn_dims, gru_hidden_dim)
    output = model(x)
    print(output.shape)  # Should print: torch.Size([32, 1])
