import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
        return out


class SCL(nn.Module):
    def __init__(self, hidden_dim):
        super(SCL, self).__init__()
        self.gcn1 = GCNLayer(1, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.conv1d = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, X, A):
        bs, N, L = X.size()
        X = X.permute(0, 2, 1).reshape(bs * L, N, 1)  # (bs*L, N, 1)
        X = F.relu(self.gcn1(X, A))
        X = F.relu(self.gcn2(X, A))
        X = self.dropout(X)

        X = self.conv1d(X.permute(0, 2, 1)).permute(0, 2, 1)  # (bs*L, N, 1)
        X = X.reshape(bs, L, N).permute(0, 2, 1)  # (bs, N, L)
        return X


class TDL(nn.Module):
    def __init__(self, num_nodes, encoder_hidden_dim):
        super(TDL, self).__init__()
        self.lstm = nn.LSTM(num_nodes, encoder_hidden_dim, batch_first=True)

    def forward(self, X):
        X, _ = self.lstm(X.permute(0, 2, 1))  # (bs, f, L)
        return X


class FusionModule(nn.Module):
    def __init__(self, num_nodes, encoder_hidden_dim, kernel_size, time_length):
        super(FusionModule, self).__init__()
        self.cnn1 = nn.Conv1d(num_nodes, encoder_hidden_dim, kernel_size=1)
        self.cnn2 = nn.Conv1d(encoder_hidden_dim, encoder_hidden_dim, kernel_size=kernel_size, padding='same')
        self.fc1 = nn.Linear(encoder_hidden_dim * time_length, 1)  # Assuming L=50
        self.fc2 = nn.Linear(encoder_hidden_dim * time_length, 1)  # Assuming L=50

    def forward(self, X, residual):
        X = self.cnn1(X)  # (bs, f, L)
        X = torch.transpose(X, -1, -2)

        M = X + residual  # Combine the features
        M = torch.transpose(M, -1, -2)

        M = self.cnn2(M)
        # print(M.size())
        M = M.view(M.size(0), -1)
        pre = self.fc1(M)
        std = self.fc2(M)

        return pre, std

class adj_construction(nn.Module):
    def __init__(self, num_nodes, time_length, alpha):
        super(adj_construction, self).__init__()
        self.alpha = alpha
        self.trainable_theta1 = nn.Linear(time_length, num_nodes)
        self.trainable_theta2 = nn.Linear(time_length, num_nodes)
    def forward(self, X):
        bs, N, L = X.size()
        alpha = self.alpha
        A1 = torch.tanh(alpha * self.trainable_theta1(X))
        A2 = torch.tanh(alpha * self.trainable_theta2(X))
        A = F.relu(torch.tanh(alpha * (torch.bmm(A1, A2.permute(0, 2, 1)) - torch.bmm(A2, A1.permute(0, 2, 1)))))

        return A


class RGCNU_model(nn.Module):
    def __init__(self, num_nodes, time_length, hidden_dim, encoder_hidden_dim, kernel_size, alpha):
        super(RGCNU_model, self).__init__()
        self.time_length = time_length
        self.adj = adj_construction(num_nodes, time_length, alpha)
        self.scl = SCL(hidden_dim)
        self.tdl = TDL(num_nodes, encoder_hidden_dim)
        self.fusion = FusionModule(num_nodes, encoder_hidden_dim, kernel_size, time_length)

    def forward(self, X, train = False):
        bs, N, L = X.size()
        # Generate adjacency matrix

        A = self.adj(X)

        A = A.repeat(self.time_length,1,1)

        # Spatial correlation layer
        spatial_features = self.scl(X, A)
        # Temporal dependency layer
        temporal_features = self.tdl(spatial_features)
        # Fusion module
        pre, std = self.fusion(X, temporal_features)
        if train:
            return pre, std
        else:
            return pre

if __name__ == '__main__':
    # Sample usage
    # Assuming data is a tensor with shape (bs, N, L)
    data = torch.randn(32, 14, 50)  # Example input: batch size 32, 10 sensors, sequence length 50
    model = RGCNU_model(14,50,32,3,3,0.1)


    output = model(data)

