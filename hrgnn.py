#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/7/6 3:31
# @Author : ZM7
# @File : hrgnn.py
# @Software: PyCharm
import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.nn import RelGraphConv, GATConv, GraphConv
import torch.nn.functional as F
from dgl.sampling import sample_neighbors
from TKG.utils import decoder_sorce, comp_deg_norm
import math

class HRGNN(nn.Module):
    def __init__(self, graph, num_nodes, num_rels, time_length, time_idx, h_dim, out_dim, max_length=10, a_layer_num=2,
                 d_layer_num=1, encoder='regcn', decoder='rgat_r1', attn_drop=0.3, feat_drop=0.3, score='mlp', last=True,
                 ori=True, norm=False, relation_prediction=True, filter=False, low_memory=True):
        super(HRGNN, self).__init__()
        self.g = graph
        self.num_nodes = num_nodes
        self.num_rels = num_rels * 2
        self.time_rels = time_length
        self.time_length = time_length  # 总的时间跨度
        self.time_idx = time_idx        # 不同时间下节点在graph中的索引
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.a_layer_num = a_layer_num
        self.d_layer_num = d_layer_num
        self.en_embedding = None
        self.max_length = max_length
        self.relation_prediction = relation_prediction
        self.low_memory = low_memory
        self.encoder = encoder
        self.decoder = decoder
        self.attn_drop = attn_drop
        self.feat_drop = feat_drop
        self.last = last
        self.ori = ori
        self.norm = norm
        if self.norm:
            self.norm_layer = torch.nn.LayerNorm(self.h_dim)
        else:
            self.norm_layer = torch.nn.Identity()
        self.pos_decoder = TimeEncode(self.h_dim)
        self.rel_embedding = None
        self.score = score
        self.filter = filter
        dim = 1
        if self.last:
            dim +=1
        if self.ori:
            dim +=1
        self.linear_1 = nn.Linear(self.out_dim * dim, self.out_dim, bias=False)
        self.reset_parameters()
        # 初始化local-level
        self.aggregator = None
        # 初始化global-level
        self.decoder_f = GNN(self.h_dim, self.h_dim, layer_num=self.d_layer_num, gnn=self.decoder,
                                 attn_drop=self.attn_drop, feat_drop=self.feat_drop)

    def forward(self, data_list, node_id_new=None, time_gap=None, device=None, mode='test'):
        out_triple = data_list['triple']
        h = F.normalize(self.en_embedding(self.g.ndata['id']))
        #h = self.norm_layer(self.en_embedding(self.g.ndata['id']))
        # sub_graph level
        if self.encoder == 'ori':
            pass
        else:
            sub_e_graph = data_list['sub_e_graph'].to(device)
            pre_id = data_list['pre_e_nid']
            if self.encoder == 'regcn':
                norm = comp_deg_norm(sub_e_graph)
                sub_e_graph.ndata.update({'norm': norm.view(-1,1)})
                sub_e_graph.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
                h[pre_id] = self.aggregator(sub_e_graph, h[pre_id], [self.rel_embedding.weight[0:self.num_rels*2],self.rel_embedding.weight[0:self.num_rels*2]], 1)
        #new_feature = self.norm_layer(h)
        new_feature = F.normalize(h)  # layer
        #new_feature = h
        # global-graph level
        if self.decoder == 'regcn':
            sub_d_graph = data_list['sub_d_graph'].to(device)
            pre_id = data_list['pre_d_nid']
            norm = comp_deg_norm(sub_d_graph)
            sub_d_graph.ndata.update({'norm': norm.view(-1, 1)})
            sub_d_graph.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
            new_feature[pre_id] = self.decoder_f(sub_d_graph, h[pre_id], [self.rel_embedding.weight] * self.d_layer_num)
        elif self.decoder in ['rgat_r1']:
            sub_d_graph = data_list['sub_d_graph'].to(device)
            pre_id = data_list['pre_d_nid']
            sub_d_graph.edata['r_h'] = self.rel_embedding(sub_d_graph.edata['etype']) + self.pos_decoder(
                sub_d_graph.edata['e_r'])
            new_feature[pre_id] = self.decoder_f(sub_d_graph, h[pre_id])
        elif self.decoder == 'ori':
            new_feature = h

        # 如果历史交互只有1，则不进行decoder的rgat,防止过拟合：
        if self.filter:
            list_length = data_list['list_length']
            one_idx = torch.where(list_length == 1)[0]
            if len(one_idx) > 0:
                new_feature[node_id_new[one_idx]] = h[node_id_new[one_idx]]
        new_list = [new_feature[node_id_new]]
        if self.last:
            #new_list.append(self.norm_layer(h[node_id_new]))
            new_list.append(F.normalize(h[node_id_new]))
        if self.ori:
            #new_list.append(self.norm_layer(self.en_embedding.weight))
            new_list.append(F.normalize(self.en_embedding.weight))
        new_embedding = self.linear_1(torch.cat(new_list,1))
        return new_embedding

    def reset_parameters(self):
        # stdv = 1.0 / math.sqrt(self.hidden_size)
        # for weight in self.parameters():
        #     weight.data.uniform_(-stdv, stdv)
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight, gain=gain)

class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, layer_num, gnn='rgcn', num_rels=None,
                 attn_drop=0.3, feat_drop=0.3, num_head=None, low_memory=False):
        super(GNN, self).__init__()
        self.h_dim = in_dim
        self.out_dim = out_dim
        self.layer_num = layer_num
        self.num_rels = num_rels
        self.gnn = gnn
        self.attn_drop = attn_drop
        self.feat_drop = feat_drop

        if self.gnn == 'rgcn':
            self.layer = nn.ModuleList(RelGraphConv(self.h_dim, self.h_dim, num_rels=self.num_rels, regularizer='basis',
                                                    num_bases=100, low_mem=low_memory, dropout=0.5, activation=F.relu)
                                       for _ in range(self.layer_num))
        elif self.gnn == 'gat':
            self.layer = nn.ModuleList(
                GATConv(self.h_dim, int(self.h_dim / num_head), num_head, feat_drop=self.feat_drop, attn_drop=self.attn_drop,
                        activation=F.elu)
                for _ in range(self.layer_num))
        elif self.gnn == 'gcn':
            self.layer = nn.ModuleList(GraphConv(self.h_dim, self.h_dim, norm='both', activation=F.relu)
                                       for _ in range(self.layer_num))
        elif self.gnn == 'rgat_r1':
            self.layer = nn.ModuleList(RGATLayer(self.h_dim, self.h_dim, self.feat_drop, self.attn_drop, self.gnn) for _ in range(self.layer_num))

    def forward(self, graph, feature, etypes=None):
        for conv in self.layer:
            if self.gnn == 'rgcn':
                feature = conv(graph, feature, etypes)
            elif self.gnn in ['rgat','rgat_r','rgat_x','rgat1','rgat_r1']:
                feature = conv(graph, feature)
        return feature



class RGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_drop=0.3, attn_drop=0.3, gnn='rgat_r'):
        super(RGATLayer, self).__init__()
        self.gnn = gnn
        if self.gnn in ['rgat', 'rgat_r','rgat1','rgat_r1']:
            self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)
            self.fc = nn.Linear(in_dim, out_dim, bias=False)
            self.fc_r = nn.Linear(in_dim, out_dim, bias=False)
        elif self.gnn in ['rgat_x']:
            self.w1 = nn.Linear(in_dim, out_dim, bias=False)
            self.w2 = nn.Linear(in_dim, out_dim, bias=False)
        self.loop_weight = nn.Parameter(torch.Tensor(out_dim, out_dim))
        self.reset_parameters()
        self.feat_drop = nn.Dropout(feat_drop)
        self.atten_drop = nn.Dropout(attn_drop)
        self.h_dim = out_dim

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if self.gnn in ['rgat', 'rgat_r', 'rgat1','rgat_r1']:
            nn.init.xavier_uniform_(self.fc.weight, gain=gain)
            nn.init.xavier_uniform_(self.attn_fc.weight, gain=gain)
        elif self.gnn in ['rgat_x']:
            nn.init.xavier_uniform_(self.w1.weight, gain=gain)
            nn.init.xavier_uniform_(self.w2.weight, gain=gain)
        nn.init.xavier_uniform_(self.loop_weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        if self.gnn in ['rgat', 'rgat_r','rgat1','rgat_r1']:
            z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['r_h']], dim=1)
            a = self.attn_fc(z2)
            return {'e': F.leaky_relu(a)}
        elif self.gnn in ['rgat_x']:
            e = ((self.w1(edges.src['z']) + edges.data['r_h'])*self.w2(edges.dst['z'])).sum(1)/torch.sqrt(torch.tensor(self.h_dim).float())
            return {'e':F.leaky_relu(e.unsqueeze(1))}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e'], 'r_h': edges.data['r_h']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = self.atten_drop(F.softmax(nodes.mailbox['e'], dim=1))
        # equation (4)
        h = self.feat_drop(torch.sum(alpha * (nodes.mailbox['z'] + nodes.mailbox['r_h']), dim=1) + torch.mm(nodes.data['z'], self.loop_weight))
        return {'h': h}

    def forward(self, g, h, edge_update=False):
        # equation (1)
        if self.gnn == 'rgat_r1':
            z = self.fc(h)
        else:
            z = h
        with g.local_scope():
            g.ndata['z'] = z
            g.edata['r_h'] = self.fc_r(g.edata['r_h'])
            # equation (2)
            g.apply_edges(self.edge_attention)
            # equation (3) & (4)
            g.update_all(self.message_func, self.reduce_func)
            return F.relu(g.ndata.pop('h'))


class TimeEncode(torch.nn.Module):  #time encoder type = 1
    #INDUCTIVE REPRESENTATION LEARNING ON TEMPORAL GRAPHS
    def __init__(self, expand_dim):
        super(TimeEncode, self).__init__()

        time_dim = expand_dim
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        # print(self.basis_freq.shape) #[50]
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        # print(self.phase.shape)#[50]

    def forward(self, ts):
        map_ts = ts.unsqueeze(1) * self.basis_freq  # [N, L, time_dim]
        map_ts += self.phase
        harmonic = torch.cos(map_ts)
        return harmonic  # self.dense(harmonic)








