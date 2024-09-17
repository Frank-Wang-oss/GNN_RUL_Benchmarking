import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import OrderedDict


def dot_graph_construction(node_features):
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




class Feature_extractor_1DCNN(nn.Module):
    def __init__(self, input_channels, num_hidden, output_dimension, kernel_size, stride, dropout):
        super(Feature_extractor_1DCNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_hidden, num_hidden*2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(num_hidden*2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(num_hidden*2, num_hidden*4, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(num_hidden*4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )


    def forward(self, x_in):
        # print('input size is {}'.format(x_in.size()))
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # print(x.size())
        return x

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





class Clustering_Assignment_Matrix_cat_proj_prop(nn.Module):
    def __init__(self, in_num_node, input_dimension,hidden_dimension, out_num_node):
        super(Clustering_Assignment_Matrix_cat_proj_prop, self).__init__()

        self.dimension_mapping = nn.Linear(input_dimension, hidden_dimension)
        self.matrix = nn.Linear(in_num_node+1*hidden_dimension, out_num_node)

    def forward(self, X, in_A):
        ## size of in_A is (bs, N, N)
        ## size of X is (bs, N, d)
        input_0 = torch.bmm(in_A,X)
        output_0 = torch.sigmoid(self.dimension_mapping(input_0))
        input_ = torch.cat([in_A,output_0],-1)
        out_A = self.matrix(input_)
        out_A = F.softmax(out_A, -2)   ### should use softmax to the first dimension instead of the second dimension
        return out_A




class Graph_Classification_block(nn.Module):
    def __init__(self, input_dimension, out_dimension, in_nodes, out_nodes):
        super(Graph_Classification_block, self).__init__()
        self.out_node = out_nodes
        self.Message_Passing = MPNN_mk(input_dimension, out_dimension, 1)


        self.Graph_Clustering = Clustering_Assignment_Matrix_cat_proj_prop(in_nodes, input_dimension, out_nodes, out_nodes)

    def forward(self, A, X):
        ## Size of X is (bs, time_length, num_nodes, dimension)
        ## Size of A is (bs, time_length, num_nodes, num_nodes)
        X_CAM_input = X
        A_input = A

        Clu_Ass_Mat = self.Graph_Clustering(X_CAM_input, A_input) # (bs*time_length, num_nodes, out_nodes)

        X_MPNN_input = torch.bmm(torch.transpose(Clu_Ass_Mat, -1, -2), X_CAM_input)

        A_MPNN_input = torch.bmm(torch.bmm(torch.transpose(Clu_Ass_Mat, -1, -2), A_input), Clu_Ass_Mat)

        X_MPNN_output = self.Message_Passing(X_MPNN_input, A_MPNN_input)

        return A_MPNN_input, X_MPNN_output

