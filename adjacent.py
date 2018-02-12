#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:51:54 2018

@author: adrienbufort
"""

"""
Code to get the adjacent matrix from the model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from operator import mul
from model import Model
import networkx as nx

import scipy.sparse as sp
import numpy as np
    
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def get_adjacent_matrix(model_input):
    
    caract = []
    list_index = []
    
    for module in model_input.children():
    
        n_output = module.out_features
        n_input = module.in_features
        
        caract.append((n_input, n_output))
        list_index.append((n_input+1) * n_output)
        
    
    n_size = np.sum(list_index)
    n_size_part = np.cumsum(list_index)
    
    # init of the matrix
    idx = np.array(np.arange(0,n_size), dtype=np.int32)
    
    edges = []
    
    # creating the matrix
    index_global = 0
    for i, layer_struct in enumerate(caract[:-1]):
        for input_neuron in range(layer_struct[1]):
            for output_neuron in range(layer_struct[0]):
                
                for p in range(caract[i+1][1]):
                    edges.append((index_global,n_size_part[i] + caract[i+1][0]*p + input_neuron))
                
                index_global += 1
           
        # bias layer
        for input_neuron in range(layer_struct[1]): 
            for p in range(caract[i+1][1]):
                edges.append((index_global,n_size_part[i] + caract[i+1][0]*p + input_neuron))
            index_global += 1
                
    G=nx.DiGraph()
    G.add_nodes_from(idx)
    G.add_edges_from(edges)
    
    # initiale matrix
    A = nx.adjacency_matrix(G)
    #A.dtype = np.float32
    
    adjacent_matrix = A + A.transpose()
    adjacent_matrix.dtype = float
    A_directed_fw = A
    A_directed_fw.dtype = float
    
    A_directed_bw = A.transpose()
    A_directed_bw.dtype = float
    
    adjacent_matrix = normalize(adjacent_matrix + sp.eye(adjacent_matrix.shape[0]))
    adj_undirected = sparse_mx_to_torch_sparse_tensor(adjacent_matrix)
    
    adjacent_matrix_directed_fw = normalize(A_directed_fw + sp.eye(A_directed_fw.shape[0]))
    adjacent_matrix_directed_fw_ = sparse_mx_to_torch_sparse_tensor(adjacent_matrix_directed_fw)
    
    adjacent_matrix_directed_bw = normalize(A_directed_bw + sp.eye(A_directed_bw.shape[0]))
    adjacent_matrix_directed_bw_ = sparse_mx_to_torch_sparse_tensor(adjacent_matrix_directed_bw)

    return adj_undirected, adjacent_matrix_directed_fw_, adjacent_matrix_directed_bw_


