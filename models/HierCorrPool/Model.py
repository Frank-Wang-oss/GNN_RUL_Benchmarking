
from .Model_Base import *



class HierCorrPool_model(nn.Module):
    def __init__(self, patch_size, num_patch, input_dim, hidden_dim, embedding_dim, num_nodes, encoder_conv_kernel, num_nodes_out):
        super(HierCorrPool_model, self).__init__()
        num_nodes_in = num_nodes

        kernel = 8
        stride = 1
        self.patch_size = patch_size
        self.num_patch = num_patch
        self.encoder_conv_kernel = encoder_conv_kernel
        self.Time_Preprocessing = Feature_extractor_1DCNN(patch_size * num_nodes_in,
                                                                     hidden_dim * num_nodes_in,
                                                                     embedding_dim * num_nodes_in, kernel, stride, 0.35)

        self.gc1 = Graph_Classification_block(embedding_dim * self.encoder_conv_kernel, embedding_dim * self.encoder_conv_kernel * 3,
                                   num_nodes_in, num_nodes_out)

        self.fc_0 = nn.Linear(self.encoder_conv_kernel * num_nodes_out * embedding_dim * 3, embedding_dim * 3)
        self.fc_1 = nn.Linear(embedding_dim * 3, 1)

    def forward(self, X):
        bs, num_node, time_length_origin = X.size()
        X = torch.reshape(X, [bs, num_node, self.num_patch, self.patch_size])
        X = torch.transpose(X, 1, 2)

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



