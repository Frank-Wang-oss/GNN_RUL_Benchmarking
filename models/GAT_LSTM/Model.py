

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the feature extraction function
def extract_features(data):
    """
    Extracts the following features for each sample in the batch:
    - Mean
    - Standard Deviation
    - Root Mean Square Amplitude
    - Root Mean Square
    - Peak-to-Peak Value
    - Skewness
    - Kurtosis
    - Crest Factor
    - Clearance Factor
    - Shape Factor
    - Impulse Factor

    Parameters:
    data (torch.Tensor): Input data of shape (bs, Time_length)

    Returns:
    torch.Tensor: Extracted features of shape (bs, 11)
    """
    bs, m = data.shape

    # Mean
    mean = torch.mean(data, dim=-1)

    # Standard Deviation
    std_dev = torch.std(data, dim=-1)

    # Root Mean Square Amplitude
    rms_amplitude = torch.mean(torch.sqrt(torch.abs(data)), dim=-1) ** 2

    # Root Mean Square
    rms = torch.sqrt(torch.mean(data ** 2, dim=-1))

    # Peak-to-Peak Value
    peak_to_peak = 0.5*(torch.max(data, dim=-1).values - torch.min(data, dim=-1).values)

    # Skewness
    mean_diff = data - mean.unsqueeze(-1)
    coeffi_skewness = m / ((m - 1) * (m - 2))
    skewness = coeffi_skewness * torch.sum(mean_diff ** 3, dim=-1) / (std_dev ** 3)

    # Kurtosis
    coeffi_Kuritosis = (m * (m + 1) - 3 * (m - 1) ** 3) / ((m - 1) * (m - 2) * (m - 3))
    kurtosis = coeffi_Kuritosis*torch.sum(mean_diff ** 4, dim=-1) / (std_dev ** 4)

    # Crest Factor
    crest_factor = torch.max(torch.abs(data), dim=-1).values / rms

    # Clearance Factor
    clearance_factor = torch.max(torch.abs(data), dim=-1).values / rms_amplitude

    # Shape Factor
    shape_factor = rms / torch.mean(torch.abs(data), dim=-1)

    # Impulse Factor
    impulse_factor = torch.max(torch.abs(data), dim=-1).values / torch.mean(torch.abs(data), dim=-1)

    # Combine all features into a single tensor
    features = torch.stack(
        [mean, std_dev, rms_amplitude, rms, peak_to_peak, skewness, kurtosis, crest_factor, clearance_factor,
         shape_factor, impulse_factor], dim=-1)

    return features


# Provided Graph Attention Layer
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

        return F.leaky_relu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        bs, N, out_features = Wh.size()
        Wh1 = Wh.unsqueeze(2).repeat(1, 1, N, 1).view(bs, N * N, out_features)
        Wh2 = Wh.unsqueeze(1).repeat(1, N, 1, 1).view(bs, N * N, out_features)
        e = torch.cat([Wh1, Wh2], dim=2)
        e = self.leakyrelu(self.attention(e)).view(bs, N, N)

        return e

# GAT + LSTM + Fully Connected Network for RUL Prediction
class GAT_LSTM_model(nn.Module):
    def __init__(self, num_patch, patch_size, hidden_dim, lstm_hidden_dim, dropout=0.1, alpha=0.1):
        super(GAT_LSTM_model, self).__init__()
        input_dim = 11
        self.num_patch = num_patch
        self.patch_size = patch_size

        hidden_dim = [input_dim] + hidden_dim
        lstm_hidden_dim = [hidden_dim[-1]] + lstm_hidden_dim

        # Three layers of GAT
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim[i], hidden_dim[i+1], dropout, alpha)
            for i in range(len(hidden_dim)-1)
        ])

        # Two layers of LSTM
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(lstm_hidden_dim[i], lstm_hidden_dim[i+1], num_layers=1, batch_first=True)
            for i in range(len(lstm_hidden_dim)-1)
        ])
        # Fully connected layer
        self.fc = nn.Linear(lstm_hidden_dim[-1]*num_patch, 1)

    def forward(self, x):
        ## x size (bs, time_length)
        bs = x.size(0)
        x = x.reshape(bs, self.num_patch, self.patch_size)
        x = x.reshape(bs * self.num_patch, self.patch_size)
        x = extract_features(x)
        x = x.reshape(bs, self.num_patch, -1)
        bs, time_length, _ = x.size()

        # Construct path graph adjacency matrix in parallel
        adj = torch.eye(time_length, device=x.device).unsqueeze(0).repeat(bs, 1, 1)
        idx = torch.arange(time_length - 1, device=x.device)
        adj[:, idx, idx + 1] = 1
        adj[:, idx + 1, idx] = 1

        # Apply GAT layers
        for gat in self.gat_layers:
            x = gat(x, adj)

        # Apply LSTM
        for lstm in self.lstm_layers:
            x, _ = lstm(x)

        # Use the last hidden state of LSTM
        x = x.reshape(bs,-1)

        # Fully connected layer for RUL prediction
        x = self.fc(x)

        return x

if __name__ == '__main__':
    # Example usage
    bs, time_length = 32, 2560
    num_patch, patch_size = 32, 80
    hidden_features = [400, 300, 200]
    lstm_hidden_size = [30, 20]
    dropout = 0.1

    model = GAT_LSTM_model(num_patch, patch_size, hidden_features, lstm_hidden_size, dropout)
    data = torch.randn(bs, time_length)
    rul_predictions = model(data)
    print(rul_predictions.shape)  # Should be (bs, 1)
