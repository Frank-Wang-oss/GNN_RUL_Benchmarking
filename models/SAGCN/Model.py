import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F


def generate_cumulative_features(signals):
    bs, L, f = signals.shape
    x_prime = torch.zeros(bs, L, f).to(signals.device)  # Initialize output tensor

    for p in range(1, L + 1):  # Iterate over each position p from 1 to L
        x_cumsum = torch.cumsum(signals[:, :p, :], dim=1)  # Cumulative sum up to position p
        x_abs_sum = torch.abs(x_cumsum)  # Absolute sum of cumulative sums

        # Compute x'_{p, i} for each feature dimension i
        for i in range(f):
            x_prime[:, p - 1, i] = x_cumsum[:, p - 1, i] / torch.sqrt(x_abs_sum[:, p - 1, i].clamp_min(1e-12))

    return x_prime

def extract_temporal_features(signals):
    max_vals = torch.max(signals, dim=-1)[0]
    min_vals = torch.min(signals, dim=-1)[0]
    std_vals = torch.std(signals, dim=-1)
    rms_vals = torch.sqrt(torch.mean(signals ** 2, dim=-1))
    mean_vals = torch.mean(signals, dim=-1)
    ptp_vals = max_vals - min_vals
    var_vals = torch.var(signals, dim=-1)
    entropy_vals = -torch.sum(F.softmax(signals, dim=-1) * F.log_softmax(signals, dim=-1), dim=-1)
    std_inv_sin_vals = torch.std(torch.arcsin(torch.clamp(signals, -1 + 1e-7, 1 - 1e-7)), dim=-1)
    std_inv_tan_vals = torch.std(torch.atan(signals), dim=-1)

    kurtosis_vals = torch.mean((signals - mean_vals.unsqueeze(-1)) ** 4, dim=-1) / (std_vals ** 4) - 3
    skewness_vals = torch.mean((signals - mean_vals.unsqueeze(-1)) ** 3, dim=-1) / (std_vals ** 3)

    return torch.stack([max_vals, min_vals, std_vals, rms_vals, mean_vals, ptp_vals,
                        var_vals, entropy_vals, std_inv_sin_vals, std_inv_tan_vals,
                        kurtosis_vals, skewness_vals], dim=-1)


def extract_frequency_features(signals, fs=1.0):
    n = signals.shape[-1]
    freqs = torch.fft.fftfreq(n, d=1 / fs).to(signals.device)
    fft_vals = torch.fft.fft(signals, dim=-1)
    psd = torch.abs(fft_vals) ** 2 / n

    mean_freq = torch.sum(freqs * psd, dim=-1) / torch.sum(psd, dim=-1)
    median_freq = freqs[torch.argsort(psd, dim=-1)[:, n // 2]]
    band_power = torch.sum(psd, dim=-1)
    occupied_bw = torch.sum(psd * (freqs < fs / 2), dim=-1) / torch.sum(psd, dim=-1)
    power_bw = torch.sqrt(torch.sum(psd ** 2, dim=-1) / torch.sum(psd, dim=-1))
    max_psd = torch.max(psd, dim=-1)[0]
    max_amp = torch.max(torch.abs(fft_vals), dim=-1)[0]
    freq_max_amp = freqs[torch.argmax(torch.abs(fft_vals), dim=-1)]

    return torch.stack([mean_freq, median_freq, band_power, occupied_bw, power_bw,
                        max_psd, max_amp, freq_max_amp], dim=-1)


def extract_features(signals, fs=1.0):
    bs, num_patch, patch_size = signals.size()
    signals = signals.reshape(bs*num_patch,-1)
    temporal_features = extract_temporal_features(signals)
    frequency_features = extract_frequency_features(signals, fs)

    features = torch.cat([temporal_features, frequency_features], dim=-1)
    features = features.reshape(bs, num_patch,-1)
    cumulative_features = generate_cumulative_features(features)
    features = torch.cat([features, cumulative_features], -1)
    features_norm = torch.norm(features, dim=(1, 2), keepdim=True)
    features = features / features_norm
    return features

def cosine_distance(matrix1):
    matrix1_ = torch.matmul(matrix1, matrix1.transpose(-1, -2))
    matrix1_norm = torch.sqrt(torch.sum(matrix1 ** 2, -1))
    matrix1_norm = matrix1_norm.unsqueeze(-1)
    cosine_distance = matrix1_ / (torch.matmul(matrix1_norm, matrix1_norm.transpose(-1, -2)))
    return cosine_distance

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
        return F.relu(out)



class GraphProjectionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(GraphProjectionLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.project_matrices = nn.Linear(num_nodes, num_nodes)

    def forward(self, x):
        ## Input size (bs, N, f)

        x_w_proj = self.project_matrices(x.transpose(-1,-2))
        x = self.linear(x_w_proj.transpose(-1,-2))


        return F.relu(x)


class SelfAttentionLayer(nn.Module):
    def __init__(self, num_nodes, attention_hidden_dim):
        super(SelfAttentionLayer, self).__init__()
        self.tanh_layer = nn.Linear(num_nodes, attention_hidden_dim)
        self.softmax_layer = nn.Linear(attention_hidden_dim, num_nodes)

    def forward(self, x):
        attn_scores = torch.tanh(self.tanh_layer(x.transpose(-1,-2)))
        attn_scores = F.softmax(self.softmax_layer(attn_scores), dim=-1)
        return attn_scores.transpose(-1,-2)


class SAGCN_model(nn.Module):
    def __init__(self, num_patch, patch_size, gcn_hidden_dim, attention_hidden_dim):
        super(SAGCN_model, self).__init__()
        self.num_patch = num_patch
        self.patch_size = patch_size
        input_dim = 40
        self.gcn1 = GCNLayer(input_dim, gcn_hidden_dim)
        self.proj1 = GraphProjectionLayer(gcn_hidden_dim, gcn_hidden_dim, num_patch)
        self.proj2 = GraphProjectionLayer(gcn_hidden_dim, gcn_hidden_dim, num_patch)
        self.attn = SelfAttentionLayer(num_patch, attention_hidden_dim)
        self.fc = nn.Linear(gcn_hidden_dim*num_patch, 1)

    def forward(self, x):
        bs = x.size(0)
        x = x.reshape(bs, self.num_patch, self.patch_size)

        # Compute adjacency matrix using cosine similarity
        x = extract_features(x)

        adj = cosine_distance(x)

        # First GCN layer with projection
        x = self.gcn1(x, adj)
        x = self.proj1(x)
        x = self.proj2(x)

        # Self-attention mechanism
        attn_scores = self.attn(x)
        # print(x.size())
        # print(attn_scores.size())
        x = x * attn_scores

        # Fully connected layer for final prediction
        x = x.reshape(bs,-1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # Example usage
    bs = 32  # Batch size
    num_patch = 20  # Number of nodes
    patch_size = 128
    f = 40  # Number of features
    gcn_hidden_features = 100
    attention_hidden_dim = 32
    output_features = 1

    # Example input features with shape (bs, N, f)
    signals = torch.randn(bs, num_patch*patch_size)

    # Initialize and forward pass through the GCN for RUL prediction
    model = GCNForRUL(f,num_patch,patch_size,gcn_hidden_features,attention_hidden_dim)

    output = model(signals)
    print(output.shape)  # Expected output shape: (bs, N, output_features)
