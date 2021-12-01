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


#Deep Set Model
class DeepSet(nn.Module):

    def __init__(self, in_features=10, set_features=25, nhid=64, dropout=0.3):
        super(DeepSet, self).__init__()
        self.in_features = in_features
        self.out_features = set_features
        self.fc = nn.Linear(nhid, 1)
        self.dropout = dropout
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 25),
            nn.ELU(inplace=True),
            nn.Linear(25, 25),
            nn.ELU(inplace=True),
            nn.Linear(25, set_features)
        )

        self.regressor = nn.Sequential(
            nn.Linear(set_features, 25),
            nn.ELU(inplace=True),
            nn.Linear(25, 25),
            nn.ELU(inplace=True),
            nn.Linear(25, nhid),
            nn.ELU(inplace=True),
            nn.Linear(nhid, nhid),
        )
        self.reset_parameters()

        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()

    def forward(self, input):
        x = input
        x = self.feature_extractor(x)
        x = x.sum(dim=1)
        x = self.regressor(x)
        embedding = x.cpu().detach().numpy()
        x = F.dropout(x, self.dropout)
        x = self.fc(x)
        return x, embedding

    # def __repr__(self):
    #     return self.__class__.__name__ + '(' \
    #            + 'Feature Exctractor=' + str(self.feature_extractor) \
    #            + '\n Set Feature' + str(self.regressor) + ')'
    

# Graph Deep Set Model
class DSGCN(nn.Module):

    def __init__(self, in_features=10, set_features=25, nhid=64, dropout=0.3):
        super(DSGCN, self).__init__()
        self.in_features = in_features
        self.out_features = set_features
        self.gc = GraphConvolution(nhid, 1)
        self.dropout = dropout
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 25),
            nn.ELU(inplace=True),
            nn.Linear(25, 25),
            nn.ELU(inplace=True),
            nn.Linear(25, set_features)
        )

        self.regressor = nn.Sequential(
            nn.Linear(set_features, 25),
            nn.ELU(inplace=True),
            nn.Linear(25, 25),
            nn.ELU(inplace=True),
            nn.Linear(25, nhid),
            nn.ELU(inplace=True),
            nn.Linear(nhid, nhid),
        )
        self.reset_parameters()

        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()

    def forward(self, input, adj):
        x = input
        x = self.feature_extractor(x)
        x = x.sum(dim=1)
        x = self.regressor(x)
        embedding = x.cpu().detach().numpy()
        x = F.dropout(x, self.dropout)
        x = self.gc(x, adj)
        return x, embedding

