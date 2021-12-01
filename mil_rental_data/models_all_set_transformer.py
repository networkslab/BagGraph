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


# Set Transformer model
class SetTransformer(nn.Module):
    def __init__(self, in_features=200, num_heads=4, ln=True):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=in_features, dim_out=64, num_heads=num_heads, ln=ln),
            # SAB(dim_in=64, dim_out=64, num_heads=num_heads, ln=ln)
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=num_heads, num_seeds=1, ln=ln),
            # PMA(dim=64, num_heads=num_heads, num_seeds=1, ln=ln)
        )
        self.last_layer = nn.Linear(in_features=64, out_features=1)

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x).squeeze()
        embedding = x.cpu().detach().numpy()
        x = self.last_layer(x)
        return x, embedding


# Set Transformer GCN model
class STGCN(nn.Module):
    def __init__(self, in_features=200, num_heads=4, ln=True):
        super(STGCN, self).__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=in_features, dim_out=64, num_heads=num_heads, ln=ln),
            # SAB(dim_in=64, dim_out=64, num_heads=num_heads, ln=ln)
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=num_heads, num_seeds=1, ln=ln),
            # PMA(dim=64, num_heads=num_heads, num_seeds=1, ln=ln)
        )
        self.last_layer = GraphConvolution(in_features=64, out_features=1)

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()

    def forward(self, x, adj):
        x = self.enc(x)
        x = self.dec(x).squeeze()
        embedding = x.cpu().detach().numpy()
        x = self.last_layer(x, adj)
        return x, embedding


