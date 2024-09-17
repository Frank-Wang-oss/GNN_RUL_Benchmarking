import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.nn.utils.rnn import pack_padded_sequence
import math
from torch.nn.functional import pad
from collections import OrderedDict

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
def corrcoef_generation_full(data):
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



def dot_graph_construction(node_features, prior = False):
    ## node features size is (bs, N, dimension)
    ## output size is (bs, N, N)
    bs, N, dimen = node_features.size()

    node_features_1 = torch.transpose(node_features, 1, 2)

    Adj = torch.bmm(node_features, node_features_1)

    eyes_like = torch.eye(N).repeat(bs, 1, 1).cuda()
    eyes_like_inf = eyes_like*1e8
    Adj = F.leaky_relu(Adj-eyes_like_inf)
    Adj = F.softmax(Adj, dim = -1)
    Adj = Adj+eyes_like

    return Adj

def Graph_regularization_loss(X, Adj, gamma):
    ### X size is (bs, N, dimension)
    ### Adj size is (bs, N, N)
    X_0 = X.unsqueeze(-3)
    X_1 = X.unsqueeze(-2)

    X_distance = torch.sum((X_0 - X_1)**2, -1)

    Loss_GL_0 = X_distance*Adj
    Loss_GL_0 = torch.mean(Loss_GL_0)

    Loss_GL_1 = torch.sqrt(torch.mean(Adj**2))

    Loss_GL = Loss_GL_0 + gamma*Loss_GL_1

    return Loss_GL



class Bi_LSTM_Standard(nn.Module):
    def __init__(self,  input_dim, num_hidden,time_length):
        super(Bi_LSTM_Standard, self).__init__()
        # num_hidden = 64
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
        ### size of x is (time_length, Bs, dimension)
        x, hidden = self.bi_lstm1(x)
        x_split = torch.split(x, (x.shape[2] // 2), 2)
        x = x_split[0] + x_split[1]
        x, hidden = self.bi_lstm2(x)
        x_split = torch.split(x, (x.shape[2] // 2), 2)
        x = x_split[0] + x_split[1]
        x = self.drop2(x)

        x2, hidden = self.bi_lstm3(x)
        # x2_presp = torch.split(x2, 1, 1)
        x2_presp = x2

        x2_split = torch.split(x2_presp, (x2_presp.shape[2] // 2), 2)
        x2 = x2_split[0] + x2_split[1]
        x2 = self.drop3(x2)


        return F.leaky_relu(x2)



class MPNN_mk(nn.Module):
    def __init__(self, input_dimension, outpuut_dinmension, k):
        ### In GCN, k means the size of receptive field. Different receptive fields can be concatnated or summed
        ### k=1 means the traditional GCN
        super(MPNN_mk, self).__init__()
        self.way_multi_field = 'sum' ## two choices 'cat' (concatnate) or 'sum' (sum up)
        self.k = k
        theta = []
        for kk in range(self.k):
            theta.append(nn.Linear(input_dimension, outpuut_dinmension))
        self.theta = nn.ModuleList(theta)

    def forward(self, X, A):
        ## size of X is (bs, N, A)
        ## size of A is (bs, N, N)
        GCN_output_ = []
        for kk in range(self.k):
            if kk == 0:
                A_ = A
            else:
                A_ = torch.bmm(A_,A)
            out_k = self.theta[kk](torch.bmm(A_,X))
            GCN_output_.append(out_k)

        if self.way_multi_field == 'cat':
            GCN_output_ = torch.cat(GCN_output_, -1)

        elif self.way_multi_field == 'sum':
            GCN_output_ = sum(GCN_output_)

        return F.leaky_relu(GCN_output_)


class Graph_atten_block(nn.Module):
    def __init__(self, num_node, out_dimension):
        super(Graph_atten_block, self).__init__()

        self.W_Z_T = nn.Linear(num_node, out_dimension)
        self.W_Z_G = nn.Linear(num_node, out_dimension)
        self.W_R_T = nn.Linear(num_node, num_node)
        self.W_R_G = nn.Linear(num_node, num_node)
        self.W_h_T = nn.Linear(num_node, num_node)
        self.W_h = nn.Linear(num_node, num_node)

    def forward(self, A_T, A_G):

        bs, N, N = A_T.size()

        z = torch.sigmoid(self.W_Z_T(A_T)+ self.W_Z_G(A_G))

        r = torch.sigmoid(self.W_R_T(A_T)+ self.W_R_G(A_G))

        A_hat = torch.tanh(self.W_h_T(A_G) + self.W_h(r))

        A_final = (1-z)*A_T + z*A_hat

        eyes_like = torch.eye(N).repeat(bs, 1, 1).cuda()

        eyes_like_inf = eyes_like * 1e8

        A_final = A_final - eyes_like_inf

        A_final = torch.softmax(A_final, dim = -1)

        A_final = A_final+eyes_like

        return A_final

class LOGO_bearing_model(nn.Module):
    def __init__(self, patch_size, num_patch, input_dim, num_nodes, nperseg, hidden_dim):
        super(LOGO_bearing_model, self).__init__()
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.nperseg = nperseg


        self.nonlin_map = nn.Linear(input_dim, 2*input_dim)

        self.MPNN = MPNN_mk(2*input_dim, input_dim*3, k=1)

        self.TD = Bi_LSTM_Standard(input_dim*3, 3*hidden_dim, None)

        self.graph_attn_blk = Graph_atten_block(num_nodes, num_nodes)

        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(3* hidden_dim * num_patch*num_nodes, 16)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(16, 8)),
            ('relu2', nn.ReLU(inplace=True))
        ]))

        self.cls = nn.Linear(8, 1)

    def forward(self, X, GL = False, gamma = 1):
        bs = X.size(0)  # batch size
        x = X.reshape(bs, self.num_patch, self.patch_size)
        x = x.reshape(bs * self.num_patch, self.patch_size)
        # print(x.size())

        # STFT parameters
        nperseg = self.nperseg  # Length of each segment for STFT
        window = torch.hann_window(nperseg, periodic=True).to(x.device)
        nfft = nperseg

        # Compute STFT for the batch in parallel
        Zxx = torch.stft(x, n_fft=nfft, hop_length=nperseg, win_length=nperseg, window=window, return_complex=True)

        # Compute magnitude
        x = Zxx.abs()
        N, f = x.size(-2), x.size(-1)
        X = x.reshape(bs, self.num_patch, N, f)
        # print(X.size())

        X_full = X.transpose(1,2)
        global_correlations = corrcoef_generation_full(X_full.reshape(bs,N,-1))


        # X = torch.reshape(X, [bs, N, self.num_window, self.window_size])
        # X = torch.transpose(X, 1, 2)
        global_correlations = global_correlations.unsqueeze(1).repeat(1, self.num_patch, 1, 1)

        bs, tlen, num_node, dimension = X.size() ### tlen = 1
        # print(X.size())


        A_input = torch.reshape(X, [bs*tlen,num_node, dimension])
        A_input_ = self.nonlin_map(A_input)
        local_correlations = dot_graph_construction(A_input_)

        global_correlations = torch.reshape(global_correlations, [bs*tlen,num_node, num_node])
        correlations = self.graph_attn_blk(local_correlations,global_correlations)

        MPNN_output = self.MPNN(A_input_, correlations)

        MPNN_output = torch.reshape(MPNN_output, [bs, tlen,num_node, -1])

        TD_input = torch.reshape(MPNN_output, [bs, num_node*tlen, -1])

        TD_input = torch.transpose(TD_input,0,1)

        TD_output = self.TD(TD_input)

        TD_output = torch.transpose(TD_output, 0, 1)

        FC_input = torch.reshape(TD_output, [bs, -1])

        FC_output = self.fc(FC_input)

        output = self.cls(FC_output)

        if GL:
            return output,Graph_regularization_loss(A_input, correlations, gamma)
        else:
            return output
