import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from modules import *


# GCN model
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# Deep Set model
class rFF_pool(nn.Module):
    def __init__(self, in_features=200, pooling_method='max'):
        super(rFF_pool, self).__init__()
        self.in_features = in_features
        self.pooling_method = pooling_method

        self.ll1 = nn.Linear(in_features, 256)
        self.ll2 = nn.Linear(256, 128)
        self.ll3 = nn.Linear(128, 64)
        self.d3 = nn.Dropout(p=0.5)

        self.fc = nn.Linear(64, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()

    def forward(self, input):

        x = input

        x = [(F.relu(self.ll1(x_))) for x_ in x]
        x = [(F.relu(self.ll2(x_))) for x_ in x]
        x = [self.d3(F.relu(self.ll3(x_))) for x_ in x]

        if self.pooling_method == 'max':
            x = [torch.unsqueeze(torch.max(x_, axis=0)[0], 0) for x_ in x]
        elif self.pooling_method == 'mean':
            x = [torch.unsqueeze(x_.mean(dim=0), 0) for x_ in x]
        elif self.pooling_method == 'sum':
            x = [torch.unsqueeze(x_.sum(dim=0), 0) for x_ in x]
        else:
            print('Invalid Pooling method!!!!!!')
            exit(0)

        x = torch.cat(x, axis=0)
        embedding = x.cpu().detach().numpy()

        x = torch.sigmoid(self.fc(x))

        return x, embedding


# Deep Set GCN model
class rFF_pool_GCN(nn.Module):
    def __init__(self, in_features=200, pooling_method='max'):
        super(rFF_pool_GCN, self).__init__()
        self.in_features = in_features
        self.pooling_method = pooling_method

        self.ll1 = nn.Linear(in_features, 256)
        self.ll2 = nn.Linear(256, 128)
        self.ll3 = nn.Linear(128, 64)
        self.d3 = nn.Dropout(p=0.5)

        self.gc = GraphConvolution(64, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()

    def forward(self, input, adj):

        x = input

        x = [(F.relu(self.ll1(x_))) for x_ in x]
        x = [(F.relu(self.ll2(x_))) for x_ in x]
        x = [self.d3(F.relu(self.ll3(x_))) for x_ in x]

        if self.pooling_method == 'max':
            x = [torch.unsqueeze(torch.max(x_, axis=0)[0], 0) for x_ in x]
        elif self.pooling_method == 'mean':
            x = [torch.unsqueeze(x_.mean(dim=0), 0) for x_ in x]
        elif self.pooling_method == 'sum':
            x = [torch.unsqueeze(x_.sum(dim=0), 0) for x_ in x]
        else:
            print('Invalid Pooling method!!!!!!')
            exit(0)

        x = torch.cat(x, axis=0)
        embedding = x.cpu().detach().numpy()

        x = torch.sigmoid(self.gc(x, adj))
        return x, embedding


# Set Transformer model
class SetTransformer(nn.Module):
    def __init__(self, in_features=200, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=in_features, dim_out=64, num_heads=num_heads, ln=ln),
            SAB(dim_in=64, dim_out=64, num_heads=num_heads, ln=ln)
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=num_heads, num_seeds=1, ln=ln)
        )
        self.fc = nn.Linear(in_features=64, out_features=1)

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()

    def forward(self, x):
        x = [self.enc(torch.unsqueeze(x_, 0)) for x_ in x]
        x = [self.dec(x_).squeeze() for x_ in x]
        x = [torch.unsqueeze(x_, 0) for x_ in x]
        x = torch.cat(x, axis=0)
        embedding = x.cpu().detach().numpy()

        x = torch.sigmoid(self.fc(x))
        return x, embedding


# Set Transformer GCN model
class STGCN(nn.Module):
    def __init__(self, in_features=200, num_heads=4, ln=False):
        super(STGCN, self).__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=in_features, dim_out=64, num_heads=num_heads, ln=ln),
            SAB(dim_in=64, dim_out=64, num_heads=num_heads, ln=ln)
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=num_heads, num_seeds=1, ln=ln)
        )
        self.gc = GraphConvolution(64, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()

    def forward(self, x, adj):
        x = [self.enc(torch.unsqueeze(x_, 0)) for x_ in x]
        x = [self.dec(x_).squeeze() for x_ in x]
        x = [torch.unsqueeze(x_, 0) for x_ in x]
        x = torch.cat(x, axis=0)
        embedding = x.cpu().detach().numpy()

        x = torch.sigmoid(self.gc(x, adj))
        return x, embedding


# Deep Set model
class res_pool(nn.Module):
    def __init__(self, in_features=200, pooling_method='max'):
        super(res_pool, self).__init__()
        self.in_features = in_features
        self.pooling_method = pooling_method

        self.ll1 = nn.Linear(in_features, 128)
        self.ll2 = nn.Linear(128, 128)
        self.ll3 = nn.Linear(128, 128)
        self.d1 = nn.Dropout(p=0.5)
        self.d2 = nn.Dropout(p=0.5)
        self.d3 = nn.Dropout(p=0.5)

        self.fc = nn.Linear(128, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()

    def forward(self, input):

        x = input

        x1 = [(F.relu(self.ll1(x_))) for x_ in x]
        x2 = [(F.relu(self.ll2(x_))) for x_ in x1]
        x3 = [(F.relu(self.ll3(x_))) for x_ in x2]

        if self.pooling_method == 'max':
            x1 = [torch.unsqueeze(torch.max(self.d1(x_), axis=0)[0], 0) for x_ in x1]
            x2 = [torch.unsqueeze(torch.max(self.d2(x_), axis=0)[0], 0) for x_ in x2]
            x3 = [torch.unsqueeze(torch.max(self.d3(x_), axis=0)[0], 0) for x_ in x3]
        elif self.pooling_method == 'mean':
            x1 = [torch.unsqueeze(self.d1(x_).mean(dim=0), 0) for x_ in x1]
            x2 = [torch.unsqueeze(self.d2(x_).mean(dim=0), 0) for x_ in x2]
            x3 = [torch.unsqueeze(self.d3(x_).mean(dim=0), 0) for x_ in x3]
        elif self.pooling_method == 'sum':
            x1 = [torch.unsqueeze(self.d1(x_).sum(dim=0), 0) for x_ in x1]
            x2 = [torch.unsqueeze(self.d2(x_).sum(dim=0), 0) for x_ in x2]
            x3 = [torch.unsqueeze(self.d3(x_).sum(dim=0), 0) for x_ in x3]
        else:
            print('Invalid Pooling method!!!!!!')
            exit(0)

        x1 = torch.cat(x1, axis=0)
        x2 = torch.cat(x2, axis=0)
        x3 = torch.cat(x3, axis=0)

        x = x1 + x2 + x3

        embedding = x.cpu().detach().numpy()

        x = torch.sigmoid(self.fc(x))

        return x, embedding


# Deep Set model
class res_pool_GCN(nn.Module):
    def __init__(self, in_features=200, pooling_method='max'):
        super(res_pool_GCN, self).__init__()
        self.in_features = in_features
        self.pooling_method = pooling_method

        self.ll1 = nn.Linear(in_features, 128)
        self.ll2 = nn.Linear(128, 128)
        self.ll3 = nn.Linear(128, 128)
        self.d1 = nn.Dropout(p=0.5)
        self.d2 = nn.Dropout(p=0.5)
        self.d3 = nn.Dropout(p=0.5)

        self.gc = GraphConvolution(128, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()

    def forward(self, input, adj):

        x = input

        x1 = [(F.relu(self.ll1(x_))) for x_ in x]
        x2 = [(F.relu(self.ll2(x_))) for x_ in x1]
        x3 = [(F.relu(self.ll3(x_))) for x_ in x2]

        if self.pooling_method == 'max':
            x1 = [torch.unsqueeze(torch.max(self.d1(x_), axis=0)[0], 0) for x_ in x1]
            x2 = [torch.unsqueeze(torch.max(self.d2(x_), axis=0)[0], 0) for x_ in x2]
            x3 = [torch.unsqueeze(torch.max(self.d3(x_), axis=0)[0], 0) for x_ in x3]
        elif self.pooling_method == 'mean':
            x1 = [torch.unsqueeze(self.d1(x_).mean(dim=0), 0) for x_ in x1]
            x2 = [torch.unsqueeze(self.d2(x_).mean(dim=0), 0) for x_ in x2]
            x3 = [torch.unsqueeze(self.d3(x_).mean(dim=0), 0) for x_ in x3]
        elif self.pooling_method == 'sum':
            x1 = [torch.unsqueeze(self.d1(x_).sum(dim=0), 0) for x_ in x1]
            x2 = [torch.unsqueeze(self.d2(x_).sum(dim=0), 0) for x_ in x2]
            x3 = [torch.unsqueeze(self.d3(x_).sum(dim=0), 0) for x_ in x3]
        else:
            print('Invalid Pooling method!!!!!!')
            exit(0)

        x1 = torch.cat(x1, axis=0)
        x2 = torch.cat(x2, axis=0)
        x3 = torch.cat(x3, axis=0)

        x = x1 + x2 + x3

        embedding = x.cpu().detach().numpy()

        x = torch.sigmoid(self.gc(x, adj))

        return x, embedding


# MINetDS model
class DSNet(nn.Module):
    def __init__(self, in_features=200, pooling_method='max'):
        super(DSNet, self).__init__()
        self.in_features = in_features
        self.pooling_method = pooling_method

        self.ll1 = nn.Linear(in_features, 256)
        self.d1 = nn.Dropout(p=0.5)
        self.ll2 = nn.Linear(256, 128)
        self.d2 = nn.Dropout(p=0.5)
        self.ll3 = nn.Linear(128, 64)
        self.d3 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(256, 1)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(64, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()

    def forward(self, input):
        x = input

        x1 = [(F.relu(self.ll1(x_))) for x_ in x]
        x2 = [(F.relu(self.ll2(x_))) for x_ in x1]
        x3 = [(F.relu(self.ll3(x_))) for x_ in x2]

        if self.pooling_method == 'max':
            x1 = [torch.unsqueeze(torch.max(self.d1(x_), axis=0)[0], 0) for x_ in x1]
            x2 = [torch.unsqueeze(torch.max(self.d2(x_), axis=0)[0], 0) for x_ in x2]
            x3 = [torch.unsqueeze(torch.max(self.d3(x_), axis=0)[0], 0) for x_ in x3]
        elif self.pooling_method == 'mean':
            x1 = [torch.unsqueeze(self.d1(x_).mean(dim=0), 0) for x_ in x1]
            x2 = [torch.unsqueeze(self.d2(x_).mean(dim=0), 0) for x_ in x2]
            x3 = [torch.unsqueeze(self.d3(x_).mean(dim=0), 0) for x_ in x3]
        elif self.pooling_method == 'sum':
            x1 = [torch.unsqueeze(self.d1(x_).sum(dim=0), 0) for x_ in x1]
            x2 = [torch.unsqueeze(self.d2(x_).sum(dim=0), 0) for x_ in x2]
            x3 = [torch.unsqueeze(self.d3(x_).sum(dim=0), 0) for x_ in x3]
        else:
            print('Invalid Pooling method!!!!!!')
            exit(0)

        x1 = torch.cat(x1, axis=0)
        x2 = torch.cat(x2, axis=0)
        x3 = torch.cat(x3, axis=0)

        embedding = torch.cat((x1, x2, x3), -1).cpu().detach().numpy()

        x1 = torch.sigmoid(self.fc1(x1))
        x2 = torch.sigmoid(self.fc2(x2))
        x3 = torch.sigmoid(self.fc3(x3))
        x4 = (x1 + x2 + x3) / 3.0

        return x1, x2, x3, x4, embedding


# MINetDS model
class DSNetGCN(nn.Module):
    def __init__(self, in_features=200, pooling_method='max'):
        super(DSNetGCN, self).__init__()
        self.in_features = in_features
        self.pooling_method = pooling_method

        self.ll1 = nn.Linear(in_features, 256)
        self.d1 = nn.Dropout(p=0.5)
        self.ll2 = nn.Linear(256, 128)
        self.d2 = nn.Dropout(p=0.5)
        self.ll3 = nn.Linear(128, 64)
        self.d3 = nn.Dropout(p=0.5)

        self.gc1 = GraphConvolution(256, 1)
        self.gc2 = GraphConvolution(128, 1)
        self.gc3 = GraphConvolution(64, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()

    def forward(self, input, adj):
        x = input

        x1 = [(F.relu(self.ll1(x_))) for x_ in x]
        x2 = [(F.relu(self.ll2(x_))) for x_ in x1]
        x3 = [(F.relu(self.ll3(x_))) for x_ in x2]

        if self.pooling_method == 'max':
            x1 = [torch.unsqueeze(torch.max(self.d1(x_), axis=0)[0], 0) for x_ in x1]
            x2 = [torch.unsqueeze(torch.max(self.d2(x_), axis=0)[0], 0) for x_ in x2]
            x3 = [torch.unsqueeze(torch.max(self.d3(x_), axis=0)[0], 0) for x_ in x3]
        elif self.pooling_method == 'mean':
            x1 = [torch.unsqueeze(self.d1(x_).mean(dim=0), 0) for x_ in x1]
            x2 = [torch.unsqueeze(self.d2(x_).mean(dim=0), 0) for x_ in x2]
            x3 = [torch.unsqueeze(self.d3(x_).mean(dim=0), 0) for x_ in x3]
        elif self.pooling_method == 'sum':
            x1 = [torch.unsqueeze(self.d1(x_).sum(dim=0), 0) for x_ in x1]
            x2 = [torch.unsqueeze(self.d2(x_).sum(dim=0), 0) for x_ in x2]
            x3 = [torch.unsqueeze(self.d3(x_).sum(dim=0), 0) for x_ in x3]
        else:
            print('Invalid Pooling method!!!!!!')
            exit(0)

        x1 = torch.cat(x1, axis=0)
        x2 = torch.cat(x2, axis=0)
        x3 = torch.cat(x3, axis=0)

        embedding = torch.cat((x1, x2, x3), -1).cpu().detach().numpy()

        x1 = torch.sigmoid(self.gc1(x1, adj))
        x2 = torch.sigmoid(self.gc2(x2, adj))
        x3 = torch.sigmoid(self.gc3(x3, adj))
        x4 = (x1 + x2 + x3) / 3.0

        return x1, x2, x3, x4, embedding
