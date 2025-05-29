from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch
import scipy.io as sio
import scipy.sparse as ss
from sklearn.preprocessing import normalize
import pandas as pd


def load_data(name, device):
    data = sio.loadmat('./datasets/{0}/{1}'.format(name, name) + '.mat')

    num_omics = len(data['X'][0])
    fea = []
    dimension = []

    for i in range(num_omics):
        feature = data['X'][0][i]
        feature = np.array(feature)
        feature = np.squeeze(feature)
        feature = normalize(feature)
        if ss.isspmatrix(feature):
            feature = feature.todense()
        feature = torch.from_numpy(feature).float().to(device)
        fea.append(feature)
        dimension.append(feature.shape[1])
        del feature
    Y = np.array(data['Y'])
    Y = Y - min(Y)
    Y = torch.from_numpy(Y).long()
    return fea, Y, num_omics, dimension


def knn_adjacency_matrix(X, k):
    # 初始化KNN模型
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='cosine').fit(X)

    # 获取K近邻
    distances, indices = knn.kneighbors(X)

    # 构建邻接矩阵
    n_samples = X.shape[0]
    adj_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in indices[i]:
            adj_matrix[i, j] = 1
    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float)

    return gcn_norm(adj_matrix)


def gcn_norm(adj):  # 归一化操作，低频增强
    I = torch.eye(adj.shape[0])
    adj = adj + I  # 自环
    degrees = torch.sum(adj, 1)
    degrees1 = torch.diag(degrees)  # A+I的度矩阵
    degrees = torch.pow(degrees, -0.5)
    D = torch.diag(degrees)    #  A+I的度矩阵的 -1/2次方
    adj = (adj + degrees1) / 2

    return torch.matmul(torch.matmul(D, adj), D)






