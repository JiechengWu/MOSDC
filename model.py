import torch
import torch.nn as nn
# import paddle.nn as pnn
import numpy as np
import torch.nn.functional as F
import time
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
from utils import knn_adjacency_matrix, gcn_norm
from torch_geometric.nn import norm


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dimen1):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hid_dimen1),
            nn.Tanh()
        )

    def forward(self, x):
        encode = self.encoder(x)
        return encode


class Decoder(nn.Module):
    def __init__(self, input_dim, hid_dimen1):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hid_dimen1),
            nn.Tanh()
        )

    def forward(self, x):
        decode = self.decoder(x)
        return decode


class MLP(nn.Module):
    def __init__(self, input_dim, hid_dimen1, hid_dimen2, hid_dimen3):
        super(MLP, self).__init__()
        self.Mlp = nn.Sequential(
            nn.Linear(input_dim, hid_dimen1),
            nn.Tanh(),
            nn.Linear(hid_dimen1, hid_dimen2),
            nn.Tanh(),
            nn.Linear(hid_dimen2, hid_dimen3),
            nn.Tanh(),
        )

    def forward(self, x):
        mlp = self.Mlp(x)
        return mlp


class GraphConvolution(nn.Module):
    def __init__(self, infeas, outfeas, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = infeas
        self.out_features = outfeas
        self.weight = Parameter(torch.FloatTensor(infeas, outfeas))
        if bias:
            self.bias = Parameter(torch.FloatTensor(outfeas))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        x1 = torch.mm(x, self.weight)
        output = torch.mm(adj, x1)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nclass)
        self.bn = norm.BatchNorm(nhid1)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return x


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, z):
        w = self.project(z)
        beta = F.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta0 = torch.softmax(w, dim=0)
        # print(beta0)
        beta = beta0.expand((z.shape[0],) + beta0.shape)
        return (beta * z).sum(1), beta0


class MOSDC(nn.Module):
    def __init__(self, num_classes, num_views, dimension, hid_d, device, k_neighbors=3, dropout=0.1):
        super(MOSDC, self).__init__()
        self.device = device
        self.dimension = dimension
        self.num_views = num_views
        self.num_classes = num_classes
        self.mv_module = nn.ModuleList()
        self.k_neighbors = k_neighbors
        self.hid_dimen1 = hid_d[0]
        self.hid_dimen2 = hid_d[1]

        total_dimen = 0
        for i in range(self.num_views):  # specific feature encoder for each view

            self.mv_module.append(Encoder(dimension[i], self.hid_dimen1))
            total_dimen = total_dimen + dimension[i]

        self.mv_module.append(Encoder(total_dimen, self.hid_dimen1))  # shared feature encoder for each view

        self.mv_module.append(Encoder(self.hid_dimen1, self.hid_dimen2))  # shared layer encoder

        self.mv_module.append(Decoder(self.hid_dimen2, self.hid_dimen1))  # shared layer decoder

        for i in range(self.num_views):
            self.mv_module.append(Decoder(self.hid_dimen1, dimension[i]))

        self.view_class_dimension1 = hid_d[2]
        self.view_class_dimension2 = hid_d[3]
        self.classifier_view = MLP(self.hid_dimen2, self.view_class_dimension1, self.view_class_dimension2, self.num_views)
        self.label_class_dimension1 = hid_d[4]
        self.label_class_dimension2 = hid_d[5]

        # 注意力权重参数，用于每个视图的特征加权
        self.attention_weights = nn.Parameter(torch.FloatTensor(self.num_views, 1))
        nn.init.xavier_uniform_(self.attention_weights)  # 初始化权重
        self.semanticAttention = SemanticAttention(self.hid_dimen2)

        self.gcn_share_classifier_label = GCN(self.hid_dimen2, hid_d[4], num_classes, dropout)
        self.gcn_specific_classifier_label = GCN(self.hid_dimen2, hid_d[4], num_classes, dropout)


    def forward(self, fea):
        # encoder beign
        specific_fea_en_lay1 = []
        specific_fea_en_lay2 = []
        fea_con = fea[0]
        # print(fea_con.shape)
        time1 = time.time()
        for i in range(self.num_views):
            tmp = self.mv_module[i](fea[i])
            specific_fea_en_lay1.append(tmp)
            if i == 0:
                continue
            fea_con = torch.cat((fea_con, fea[i]), 1)
        share_fea_en_lay1 = self.mv_module[self.num_views](fea_con)
        time2 = time.time()
        # print(time2 - time1)
        for i in range(self.num_views):
            tmp = self.mv_module[self.num_views + 1](specific_fea_en_lay1[i])
            specific_fea_en_lay2.append(tmp)
        share_fea_en_lay2 = self.mv_module[self.num_views + 1](share_fea_en_lay1)
        # encoder end
        # decoder begin
        specific_fea_de_lay1 = []
        specific_fea_de_lay2 = []
        for i in range(self.num_views):
            tmp = self.mv_module[self.num_views + 2](specific_fea_en_lay2[i] + share_fea_en_lay2)
            specific_fea_de_lay1.append(tmp)
        for i in range(self.num_views):
            tmp = self.mv_module[self.num_views + 3 + i](specific_fea_de_lay1[i])
            specific_fea_de_lay2.append(tmp)
        # decoder end
        # view classfier begin
        view_class_specific_res = []
        for i in range(self.num_views):
            tmp = self.classifier_view(specific_fea_en_lay2[i])
            view_class_specific_res.append(tmp)
        view_class_share_res = self.classifier_view(share_fea_en_lay2)

        # Attention fusion
        specific_fea_stack = torch.stack(specific_fea_en_lay2, dim=1)  # 形状为 (batch_size, num_views, feature_dim)
        specific_con, att = self.semanticAttention(specific_fea_stack)

        # adj
        edge_index_specific = knn_adjacency_matrix(specific_con.cpu().detach().numpy(), self.k_neighbors).to(self.device)
        edge_index_share = knn_adjacency_matrix(share_fea_en_lay2.cpu().detach().numpy(), self.k_neighbors).to(self.device)

        # GCN分类
        label_class_specific_res = self.gcn_specific_classifier_label(specific_con, edge_index_specific)
        label_class_share_res = self.gcn_share_classifier_label(share_fea_en_lay2, edge_index_share)

        return specific_fea_de_lay2, view_class_specific_res, view_class_share_res, label_class_specific_res, label_class_share_res, specific_con, share_fea_en_lay2, edge_index_specific, edge_index_share

