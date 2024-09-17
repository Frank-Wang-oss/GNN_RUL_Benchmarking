import torch
import torch.nn as nn
import math


def PCC_dist(data):
    # Input shape: (bs, dim1, dim2)

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


class TVGTformer(nn.Module):
    def __init__(self, num_nodes, time_length, d_model, num_heads, lambda_param, d_ff, dropout):
        super(TVGTformer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.lambda_param = lambda_param

        self.linears_Q_temp = nn.ModuleList([nn.Linear(num_nodes + 1, d_model) for _ in range(self.num_heads)])
        self.linears_K_temp = nn.ModuleList([nn.Linear(num_nodes + 1, d_model) for _ in range(self.num_heads)])
        self.linears_V_temp = nn.ModuleList([nn.Linear(num_nodes + 1, d_model) for _ in range(self.num_heads)])

        self.W_O_temp = nn.Linear(d_model * num_heads, num_nodes + 1)

        self.layer_norm1_temp = nn.LayerNorm(num_nodes + 1)
        self.layer_norm2_temp = nn.LayerNorm(num_nodes + 1)

        self.feed_forward_temp = nn.Sequential(
            nn.Linear(num_nodes + 1, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, num_nodes + 1)
        )
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, X, A_temp):
        out_temp = []
        for idx in range(self.num_heads):
            Q_temp_i = self.linears_Q_temp[idx](X)
            K_temp_i = self.linears_K_temp[idx](X)
            V_temp_i = self.linears_V_temp[idx](X)
            scores_temp_i = torch.bmm(Q_temp_i, K_temp_i.transpose(-2, -1)) / math.sqrt(self.d_model)
            attention_temp_i = (1 - self.lambda_param) * torch.softmax(scores_temp_i,
                                                                       dim=-1) + self.lambda_param * torch.softmax(
                nn.ReLU()(A_temp), dim=-1)
            head_v = torch.bmm(torch.softmax(attention_temp_i, -1), V_temp_i)
            out_temp.append(head_v)
        attention_temp = torch.cat(out_temp, -1)
        output_temp = self.W_O_temp(attention_temp)
        output_temp = self.layer_norm1_temp(output_temp) + X
        output_temp = self.dropout1(output_temp)
        ff_output_temp = self.feed_forward_temp(output_temp)
        ff_output_temp = self.layer_norm2_temp(ff_output_temp) + output_temp
        return ff_output_temp


class SVGTformer(nn.Module):
    def __init__(self, num_nodes, time_length, d_model, num_heads, lambda_param, d_ff, dropout):
        super(SVGTformer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.lambda_param = lambda_param

        self.linears_Q_spat = nn.ModuleList([nn.Linear(time_length + 1, d_model) for _ in range(self.num_heads)])
        self.linears_K_spat = nn.ModuleList([nn.Linear(time_length + 1, d_model) for _ in range(self.num_heads)])
        self.linears_V_spat = nn.ModuleList([nn.Linear(time_length + 1, d_model) for _ in range(self.num_heads)])

        self.W_O_spat = nn.Linear(d_model * num_heads, time_length + 1)

        self.layer_norm1_spat = nn.LayerNorm(time_length + 1)
        self.layer_norm2_spat = nn.LayerNorm(time_length + 1)

        self.feed_forward_spat = nn.Sequential(
            nn.Linear(time_length + 1, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, time_length + 1)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, X, A_spat):
        out_spat = []
        for idx in range(self.num_heads):
            Q_spat_i = self.linears_Q_spat[idx](X)
            K_spat_i = self.linears_K_spat[idx](X)
            V_spat_i = self.linears_V_spat[idx](X)
            scores_spat_i = torch.bmm(Q_spat_i, K_spat_i.transpose(-2, -1)) / math.sqrt(self.d_model)
            attention_spat_i = (1 - self.lambda_param) * torch.softmax(scores_spat_i,
                                                                       dim=-1) + self.lambda_param * torch.softmax(
                nn.ReLU()(A_spat), dim=-1)
            head_v_spat = torch.bmm(torch.softmax(attention_spat_i, -1), V_spat_i)
            out_spat.append(head_v_spat)
        attention_spat = torch.cat(out_spat, -1)
        output_spat = self.W_O_spat(attention_spat)
        output_spat = self.layer_norm1_spat(output_spat) + X
        ff_output_spat = self.feed_forward_spat(output_spat)
        ff_output_spat = self.layer_norm2_spat(ff_output_spat) + output_spat
        return ff_output_spat


class DVGTformer_model(nn.Module):
    def __init__(self, num_nodes, time_length, d_model, num_heads, lambda_param, d_ff, dropout, num_blocks):
        super(DVGTformer_model, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.lambda_param = lambda_param
        self.num_blocks = num_blocks

        self.linear_t = nn.Linear(time_length, time_length)
        self.linear_x = nn.Linear(num_nodes, num_nodes)

        self.t_v = nn.Parameter(torch.randn(1, 1, num_nodes))
        self.x_v = nn.Parameter(torch.randn(1, time_length + 1, 1))

        self.positional_encoding = self.create_positional_encoding(time_length + 1, num_nodes + 1)

        self.tvgtformer_blocks = nn.ModuleList(
            [TVGTformer(num_nodes, time_length, d_model[0], num_heads, lambda_param, d_ff[0], dropout) for _ in
             range(num_blocks)])
        self.svgtformer_blocks = nn.ModuleList(
            [SVGTformer(num_nodes, time_length, d_model[1], num_heads, lambda_param, d_ff[1], dropout) for _ in
             range(num_blocks)])


        self.output_layer = nn.Sequential(
            nn.Linear(((time_length + 1) * (num_nodes + 1)), 100),
            nn.GELU(),
            nn.Linear(100, 1)
        )

    def create_positional_encoding(self, N, d_model):
        pos_encoding = torch.zeros(N, d_model)
        for pos in range(N):
            for i in range(0, d_model - 1, 2):
                pos_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pos_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
        return pos_encoding.unsqueeze(0)

    def forward(self, X):
        bs = X.size(0)
        X = self.linear_t(X)
        X = torch.transpose(X, -1, -2)  # (bs, L, N)
        X = self.linear_x(X)

        X = torch.cat([X, self.t_v.expand(bs, -1, -1)], dim=1)  # (bs, L, N+1)
        X = torch.cat([X, self.x_v.expand(bs, -1, -1)], dim=-1)  # (bs, L+1, N+1)
        A_temp = PCC_dist(X)
        A_spat = PCC_dist(X.transpose(-1, -2))

        X += self.positional_encoding[:, :X.size(1), :X.size(2)].to(X.device)  # (bs, L+1, N+1)

        for block_i in range(self.num_blocks):
            # print(X.size())
            X = self.tvgtformer_blocks[block_i](X, A_temp)
            X = torch.transpose(X, 1, 2)  # Transpose for spatial module input
            X = self.svgtformer_blocks[block_i](X, A_spat)
            X = torch.transpose(X, 1, 2)  # Transpose for spatial module input


        X = X.reshape(bs, -1)  # Flatten the tensor
        output = self.output_layer(X)
        return output


if __name__ == '__main__':
    bs = 4  # Batch size
    N = 14  # Number of nodes
    L = 50  # Time length
    d_model = 32
    num_heads = 4
    lambda_param = 0.5
    d_ff = 64

    # Generate synthetic input data
    X = torch.randn(bs, N, L)

    # Instantiate the model
    model = DVGTformer(num_nodes=N, time_length=L, d_model=d_model, num_heads=num_heads, lambda_param=lambda_param, d_ff=d_ff, dropout=0.5, num_blocks=2)

    # Forward pass through the model
    output = model(X)

    print(output.size())