
from .Model_Base import *



class HierCorrPool_bearing_model(nn.Module):
    def __init__(self, patch_size, num_patch, input_dim, hidden_dim, embedding_dim, num_nodes, nperseg, encoder_conv_kernel, num_nodes_out):
        super(HierCorrPool_bearing_model, self).__init__()
        num_nodes_in = num_nodes

        self.nperseg = nperseg
        self.patch_size = patch_size
        self.num_patch = num_patch
        self.encoder_conv_kernel = encoder_conv_kernel
        self.Time_Preprocessing = Feature_extractor_1DCNN(input_dim * num_nodes_in,
                                                                     hidden_dim * num_nodes_in,
                                                                     embedding_dim * num_nodes_in, 8, 1, 0.35)

        self.gc1 = Graph_Classification_block(embedding_dim * self.encoder_conv_kernel, embedding_dim * self.encoder_conv_kernel * 3,
                                   num_nodes_in, num_nodes_out)

        self.fc_0 = nn.Linear(self.encoder_conv_kernel * num_nodes_out * embedding_dim * 3, embedding_dim * 3)
        self.fc_1 = nn.Linear(embedding_dim * 3, 1)

    def forward(self, X):

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


        bs, tlen, num_node, dimension = X.size()

        X = torch.reshape(X, [bs, tlen, dimension * num_node])
        TD_input = torch.transpose(X, 2, 1)  ### size is (bs, dimension*num_node, tlen)
        TD_output = self.Time_Preprocessing(TD_input)  ### size is (bs, out_dimension*num_node, tlen)
        TD_output = torch.transpose(TD_output, 2, 1)  ### size is (bs, tlen, out_dimension*num_node

        GC_input = torch.reshape(TD_output, [bs, self.encoder_conv_kernel, num_node, -1])
        GC_input = torch.transpose(GC_input, 1, 2)

        GC_input = torch.reshape(GC_input, [bs, num_node, -1])  ## size is (bs*tlen, num_node, embedding_size)

        A_output = dot_graph_construction(GC_input)  ## size is (bs*tlen, num_node, num_node)

        A_output, GC_output = self.gc1(A_output, GC_input)


        GC_output = torch.reshape(GC_output, [bs, -1])

        out = F.leaky_relu(self.fc_1(F.leaky_relu(self.fc_0(GC_output))))

        return out



