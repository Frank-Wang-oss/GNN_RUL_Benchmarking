
from .Model_Base import *


class FC_STGNN_RUL(nn.Module):
    def __init__(self, patch_size, num_patch, encoder_time_out,
                 encoder_hidden_dim, encoder_out_dim, encoder_conv_kernel,
                 hidden_dim, num_sequential, num_node, num_windows):
        super(FC_STGNN_RUL, self).__init__()
        # graph_construction_type = args.graph_construction_type
        pooling_choice = 'mean'
        decay = 0.7
        moving_window = [2, 2]
        stride = [1, 2]
        self.patch_size = patch_size
        self.num_patch = num_patch

        self.nonlin_map = Feature_extractor_1DCNN_RUL(1, encoder_hidden_dim, encoder_out_dim,kernel_size=encoder_conv_kernel)
        self.nonlin_map2 = nn.Sequential(
            nn.Linear(encoder_out_dim*encoder_time_out, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim)
        )

        self.positional_encoding = PositionalEncoding(2*hidden_dim,0.1,max_len=5000)

        self.MPNN1 = GraphConvpoolMPNN_block_v6(2*hidden_dim, hidden_dim, num_node, num_sequential, time_window_size=moving_window[0], stride=stride[0], decay = decay, pool_choice=pooling_choice)
        self.MPNN2 = GraphConvpoolMPNN_block_v6(2*hidden_dim, hidden_dim, num_node, num_sequential, time_window_size=moving_window[1], stride=stride[1], decay = decay, pool_choice=pooling_choice)


        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(hidden_dim * num_windows * num_node, 2*hidden_dim)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(2*hidden_dim, 2*hidden_dim)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(2*hidden_dim, hidden_dim)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc4', nn.Linear(hidden_dim, 1)),

        ]))



    def forward(self, X):
        ## Input X size is (bs, channel, time_length)
        bs, num_node, time_length_origin = X.size()
        X = torch.reshape(X, [bs, num_node, self.num_patch, self.patch_size])
        X = torch.transpose(X, 1, 2)

        bs, tlen, num_node, dimension = X.size() ### tlen = 1




        ### Graph Generation
        A_input = torch.reshape(X, [bs*tlen*num_node, dimension, 1])
        A_input_ = self.nonlin_map(A_input)
        A_input_ = torch.reshape(A_input_, [bs*tlen*num_node,-1])
        A_input_ = self.nonlin_map2(A_input_)
        A_input_ = torch.reshape(A_input_, [bs, tlen,num_node,-1])

        ## positional encoding before mapping starting
        X_ = torch.reshape(A_input_, [bs,tlen,num_node, -1])
        X_ = torch.transpose(X_,1,2)
        X_ = torch.reshape(X_,[bs*num_node, tlen, -1])
        X_ = self.positional_encoding(X_)
        X_ = torch.reshape(X_,[bs,num_node, tlen, -1])
        X_ = torch.transpose(X_,1,2)
        A_input_ = X_

        ## positional encoding before mapping ending

        # print(A_input_.size())

        MPNN_output1 = self.MPNN1(A_input_)
        MPNN_output2 = self.MPNN2(A_input_)


        features1 = torch.reshape(MPNN_output1, [bs, -1])
        features2 = torch.reshape(MPNN_output2, [bs, -1])

        features = torch.cat([features1,features2],-1)

        features = self.fc(features)

        return features