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


class MPNN_mk(nn.Module):
    def __init__(self, input_dimension, output_dimension, k):
        super(MPNN_mk, self).__init__()
        self.way_multi_field = 'sum'  # two choices: 'cat' (concatenate) or 'sum' (sum up)
        self.k = k
        theta = []
        for kk in range(self.k):
            theta.append(nn.Linear(input_dimension, output_dimension))
        self.theta = nn.ModuleList(theta)

    def forward(self, X, A):
        # X: (bs, N, input_dimension)
        # A: (bs, N, N)
        MPNN_output = []
        A_ = A
        for kk in range(self.k):
            out_k = self.theta[kk](torch.bmm(A_, X))
            MPNN_output.append(out_k)
            if kk > 0:
                A_ = torch.bmm(A_, A)

        if self.way_multi_field == 'cat':
            MPNN_output = torch.cat(MPNN_output, -1)
        elif self.way_multi_field == 'sum':
            MPNN_output = sum(MPNN_output)

        return F.leaky_relu(MPNN_output)


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, H):
        bs, N, d_model = H.shape

        Q = self.W_q(H)  # (bs, N, d_model)
        K = self.W_k(H)  # (bs, N, d_model)
        V = self.W_v(H)  # (bs, N, d_model)

        QK_T = torch.bmm(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(d_model, dtype=torch.float32))  # (bs, N, N)
        attn_weights = F.softmax(QK_T, dim=-1)
        attn_output = torch.bmm(attn_weights, V)  # (bs, N, d_model)

        return attn_output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([SelfAttention(d_model) for _ in range(num_heads)])

    def forward(self, H):
        attn_outputs = [head(H) for head in self.heads]  # list of (bs, N, d_model)
        attn_outputs_concat = torch.cat(attn_outputs, dim=-1)  # (bs, N, d_model * num_heads)
        return attn_outputs_concat


class AGCN_TF_model(nn.Module):
    def __init__(self, num_patch, patch_size, hidden_adj_dim, hidden_gnn_dim, num_heads=1):
        super(AGCN_TF_model, self).__init__()
        self.num_patch = num_patch
        self.patch_size = patch_size

        input_dim = 40
        self.attention_spa_adj = nn.Sequential(
            nn.Linear(num_patch, hidden_adj_dim),
            nn.Tanh(),
            nn.Linear(hidden_adj_dim, input_dim)
        )

        self.attention_tem_adj = nn.Sequential(
            nn.Linear(input_dim, hidden_adj_dim),
            nn.Tanh(),
            nn.Linear(hidden_adj_dim, num_patch)
        )

        self.spatial_gnn = MPNN_mk(num_patch, hidden_gnn_dim, k=1)
        self.temporal_gnn = MPNN_mk(input_dim, hidden_gnn_dim, k=1)
        self.self_attention = MultiHeadSelfAttention(hidden_gnn_dim, num_heads)
        self.fc = nn.Linear(hidden_gnn_dim*num_heads*(num_patch+input_dim), 1)


    def forward(self, X):
        bs = X.size(0)
        X = X.reshape(bs, self.num_patch, self.patch_size)

        # Compute adjacency matrix using cosine similarity
        X = extract_features(X)

        # Compute adjacency matrices
        A_s = self.attention_spa_adj(X.transpose(-1, -2))  # shape (bs, N, N)
        A_t = self.attention_tem_adj(X)  # shape (bs, L, L)

        # Process with MPNN
        H_s = self.spatial_gnn(X.transpose(1, 2), A_s)
        H_t = self.temporal_gnn(X, A_t)


        # Combine H_s and H_t
        H = torch.cat((H_s,H_t),1)

        # Apply multi-head self-attention
        H_attn = self.self_attention(H)
        # print()

        # Flatten and predict RUL
        H_flat = H_attn.view(bs, -1)
        output = self.fc(H_flat)

        return output


if __name__ == '__main__':
    # Example usage
    bs = 32  # Batch size
    num_patch = 20  # Number of nodes
    patch_size = 128
    f = 40  # Number of features

    hidden_adj_dim, hidden_gnn_dim, num_heads = 100, 100, 1
    # Example input features with shape (bs, N, f)
    signals = torch.randn(bs, num_patch*patch_size)

    # Initialize and forward pass through the GCN for RUL prediction
    model = AGCN_TF(num_patch,patch_size,hidden_adj_dim, hidden_gnn_dim, num_heads)

    output = model(signals)
    print(output.shape)  # Expected output shape: (bs, N, output_features)
