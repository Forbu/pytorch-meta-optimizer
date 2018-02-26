#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:48:22 2018

@author: adrienbufort
"""

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn

class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    def forward(self, matrix1, matrix2):
        self.save_for_backward(matrix1, matrix2)
        return torch.mm(matrix1, matrix2)

    def backward(self, grad_output):
        matrix1, matrix2 = self.saved_tensors
        grad_matrix1 = grad_matrix2 = None

        if self.needs_input_grad[0]:
            grad_matrix1 = torch.mm(grad_output, matrix2.t())

        if self.needs_input_grad[1]:
            grad_matrix2 = torch.mm(matrix1.t(), grad_output)

        return grad_matrix1, grad_matrix2


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = SparseMM()(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, n_output, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc3 = GraphConvolution(nhid, 2)
        self.dropout = dropout

    def forward(self, x, adj):

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.sigmoid(self.gc3(x, adj))
        return x
    
class GraphConvolution_double(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution_double, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_1 = Parameter(torch.Tensor(in_features, out_features))
        self.weight_2 = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight_1.data.uniform_(-stdv, stdv)
        self.weight_2.data.uniform_(-stdv, stdv)
        
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj_1, adj_2):
        support_1 = torch.mm(input, self.weight_1)
        output_1 = SparseMM()(adj_1, support_1)
        
        support_2 = torch.mm(input, self.weight_2)
        output_2 = SparseMM()(adj_2, support_2)
        
        if self.bias is not None:
            return output_1 + output_2 + self.bias
        else:
            return output_1 + output_2

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
        
class GCN_double(nn.Module):
    def __init__(self, nfeat, nhid, n_output, dropout):
        super(GCN_double, self).__init__()

        self.gc1 = GraphConvolution_double(nfeat, nhid)
        self.gc3 = GraphConvolution_double(nhid, 2)
        self.dropout = dropout

    def forward(self, x, adj_1, adj_2):

        x = F.relu(self.gc1(x, adj_1, adj_2))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.sigmoid(self.gc3(x, adj_1, adj_2))
        return x

class GraphConvolution_LSTM(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features + out_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # parameter LSTM 1
        self.weight_f = Parameter(torch.Tensor(in_features + out_features, out_features))
        
        self.bias_f = Parameter(torch.Tensor(out_features))        
        
        self.weight_i = Parameter(torch.Tensor(in_features + out_features, out_features))

        self.bias_i = Parameter(torch.Tensor(out_features))  

        self.weight_c = Parameter(torch.Tensor(in_features + out_features, out_features))

        self.bias_c = Parameter(torch.Tensor(out_features))          
        
        self.reset_parameters()
        
        # register buffer for c and h

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
        self.weight_f.data.uniform_(-stdv, stdv)
        self.weight_i.data.uniform_(-stdv, stdv)
        self.weight_c.data.uniform_(-stdv, stdv)
        
        self.bias_f.data.uniform_(-stdv, stdv)
        self.bias_i.data.uniform_(-stdv, stdv)
        self.bias_c.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, c, h):
        """
        Define input caract
        sequence length - N number of node - m number of features 
        """
        #
        if c == None:
            c = self.c
        if h == None:
            h = self.h
        # concat input and h
        input_concat = torch.cat((input,h),1) 
        
        # LSTM equation :
        f_i = F.sigmoid(torch.mm(self.weight_f, input_concat) + self.bias_f)
        
        i = F.sigmoid(torch.mm(self.weight_i, input_concat) + self.bias_i)
        
        c_t = F.tanh(torch.mm(self.weight_c, input_concat) + self.bias_c)
        
        self.c = torch.dot(c,f_i) + torch.dot(c_t,i)
        
        support = torch.mm(input, self.weight)
        output = SparseMM()(adj, support)

        if self.bias is not None:
            output = output + self.bias

        output = torch.dot(F.tanh(c),output)

        return output, c, h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN_LSTM(nn.Module):
    def __init__(self, nfeat, nhid, n_output, dropout):
        super(GCN_LSTM, self).__init__()

        self.gc1 = GraphConvolution_LSTM(nfeat, nhid)
        self.gc3 = GraphConvolution_LSTM(nhid, 2)
        self.dropout = dropout

    def forward(self, x, adj):

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.sigmoid(self.gc3(x, adj))
        return x
    



