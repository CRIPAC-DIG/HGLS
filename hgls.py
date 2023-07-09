#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/9/14 4:28
# @Author : ZM7
# @File : hgls
# @Software: PyCharm

import torch
import torch.nn as nn
from rrgcn import RecurrentRGCN
from hrgnn import HRGNN
from rgcn.utils import build_sub_graph
import torch.nn.functional as F
from decoder import ConvTransE, ConvTransR
from rrgcn import RGCNCell
from hrgnn import GNN

class HGLS(nn.Module):
    def __init__(self, graph, num_nodes, num_rels, h_dim, task, relation_prediction, short=True, long=True, fuse='con',
                 r_fuse='re', short_con=None, long_con=None):
        super(HGLS, self).__init__()
        self.g = graph
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.h_dim = h_dim
        self.sequence_len = short_con['sequence_len']
        self.task = task
        self.relation_prediction = relation_prediction
        self.short = short
        self.long = long
        self.fuse = fuse
        self.r_fuse = r_fuse
        self.en_embedding = nn.Embedding(self.num_nodes, self.h_dim)
        self.rel_embedding = nn.Embedding(self.num_rels * 2 + 1, self.h_dim)
        torch.nn.init.normal_(self.en_embedding.weight)
        torch.nn.init.xavier_normal_(self.rel_embedding.weight)
        self.gnn = long_con['encoder']
        # GNN 初始化
        if self.gnn == 'regcn':
            self.rgcn = RGCNCell(num_nodes,
                                 h_dim,
                                 h_dim,
                                 num_rels * 2,
                                 short_con['num_bases'],
                                 short_con['num_basis'],
                                 long_con['a_layer_num'],
                                 short_con['dropout'],
                                 short_con['self_loop'],
                                 short_con['skip_connect'],
                                 short_con['encoder'],
                                 short_con['opn'])
        elif self.gnn == 'rgat':
            self.rgcn = GNN(self.h_dim, self.h_dim, layer_num=long_con['a_layer_num'], gnn=self.gnn, attn_drop=0.0, feat_drop=0.2)
        if self.short:
            self.model_r = RecurrentRGCN(num_ents=num_nodes, num_rels=num_rels, gnn=self.gnn, **short_con)
            self.model_r.rgcn = self.rgcn
            self.model_r.dynamic_emb = self.en_embedding.weight
            self.model_r.emb_rel = self.rel_embedding.weight

        if self.long:
            self.model_t = HRGNN(graph=graph, num_nodes=num_nodes, num_rels=num_rels, **long_con)
            self.model_t.aggregator = self.rgcn
            self.model_t.en_embedding = self.en_embedding
            self.model_t.rel_embedding = self.rel_embedding
        if self.short and self.long:
            if self.fuse == 'con':
                self.linear_fuse = nn.Linear(self.h_dim * 2, self.h_dim, bias=False)
            elif self.fuse == 'att':
                self.linear_l = nn.Linear(self.h_dim, self.h_dim, bias=True)
                self.linear_s = nn.Linear(self.h_dim, self.h_dim, bias=True)
            elif self.fuse == 'att1':
                self.linear_l = nn.Linear(self.h_dim, self.h_dim, bias=True)
                self.linear_s = nn.Linear(self.h_dim, self.h_dim, bias=True)
                self.fuse_f = nn.Linear(self.h_dim, 1, bias=True)
            elif self.fuse == 'gate':
                self.gate = GatingMechanism(self.num_nodes, self.h_dim)
            else:
                print('no fuse function')
            if self.r_fuse == 'con':
                self.linear_fuse_r = nn.Linear(self.h_dim * 2, self.h_dim, bias=False)
            elif self.r_fuse == 'att1':
                self.linear_l_r = nn.Linear(self.h_dim, self.h_dim, bias=True)
                self.linear_s_r = nn.Linear(self.h_dim, self.h_dim, bias=True)
                self.fuse_f_r = nn.Linear(self.h_dim, 1, bias=True)
            elif self.r_fuse == 'gate':
                self.gate_r = GatingMechanism(self.num_rels *2 , self.h_dim)
            else:
                print('no fuse_r function')
        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()
        self.decoder_ob = ConvTransE(num_nodes, h_dim, short_con['input_dropout'], short_con['hidden_dropout'], short_con['feat_dropout'])
        self.rdecoder = ConvTransR(num_rels, h_dim, short_con['input_dropout'], short_con['hidden_dropout'], short_con['feat_dropout'])

    def forward(self, total_list, data_list, node_id_new=None, time_gap=None, device=None, mode='test'):
        # RE-GCN的更新
        t = data_list['t'][0].to(device)
        all_triples = data_list['triple'].to(device)
        #output = total_list[t]
        if self.short:
            if t - self.sequence_len < 0:
                input_list = total_list[0:t]
            else:
                input_list = total_list[t-self.sequence_len: t]
            history_glist = [build_sub_graph(self.num_nodes, self.num_rels, snap, device) for snap in input_list]
            evolve_embs, static_emb, r_emb, _, _ = self.model_r(history_glist, device=device)
            pre_emb = F.normalize(evolve_embs[-1])
        if self.long:
            new_embedding = F.normalize(self.model_t(data_list, node_id_new, time_gap, device, mode))
            new_r_embedding = self.model_t.rel_embedding.weight[0:self.num_rels*2]

        if self.long and self.short:
            # entity embedding fusion
            if self.fuse == 'con':
                pre_emb = self.linear_fuse(torch.cat((pre_emb, new_embedding), 1))
            elif self.fuse == 'att':
                pre_emb, e_cof = self.fuse_attention(pre_emb, new_embedding, self.en_embedding.weight)
            elif self.fuse == 'att1':
                pre_emb, e_cof = self.fuse_attention1(pre_emb, new_embedding)
            elif self.fuse == 'gate':
                pre_emb, e_cof = self.gate(pre_emb, new_embedding)
            # relation embedding fusion
            if self.r_fuse == 'short':
                r_emb = r_emb
            elif self.r_fuse == 'long':
                r_emb = new_r_embedding
            elif self.r_fuse == 'con':
                r_emb = self.linear_fuse_r(torch.cat((r_emb, new_r_embedding), 1))
            elif self.r_fuse == 'att1':
                r_emb, r_cof = self.fuse_attention_r(r_emb, new_r_embedding)
            elif self.r_fuse == 'gate':
                r_emb, r_cof = self.gate_r(r_emb, new_r_embedding)
        elif self.long and not self.short:
            pre_emb = new_embedding
            r_emb = new_r_embedding

        # 构造loss
        loss_ent = torch.zeros(1).to(device)
        loss_rel = torch.zeros(1).to(device)
        scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples, mode).view(-1, self.num_nodes)
        loss_ent += self.loss_e(scores_ob, all_triples[:, 2])

        if self.relation_prediction:
            score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode).view(-1, self.num_rels *2)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])
        loss = self.task * loss_ent + (1 - self.task) * loss_rel
        if mode == 'test':
            return scores_ob, 0, loss
        else:
            return loss_ent, 0, loss

    def fuse_attention(self, s_embedding, l_embedding, o_embedding):
        w1 = (o_embedding * torch.tanh(self.linear_s(s_embedding))).sum(1)
        w2 = (o_embedding * torch.tanh(self.linear_l(l_embedding))).sum(1)
        aff = F.softmax(torch.cat((w1.unsqueeze(1),w2.unsqueeze(1)),1), 1)
        en_embedding = aff[:,0].unsqueeze(1) * s_embedding + aff[:, 1].unsqueeze(1) * l_embedding
        return en_embedding, aff

    def fuse_attention1(self, s_embedding, l_embedding):
        w1 = self.fuse_f(torch.tanh(self.linear_s(s_embedding)))
        w2 = self.fuse_f(torch.tanh(self.linear_l(l_embedding)))
        aff = F.softmax(torch.cat((w1,w2),1), 1)
        en_embedding = aff[:,0].unsqueeze(1) * s_embedding + aff[:, 1].unsqueeze(1) * l_embedding
        return en_embedding, aff

    def fuse_attention_r(self, s_embedding, l_embedding):
        w1 = self.fuse_f_r(torch.tanh(self.linear_s_r(s_embedding)))
        w2 = self.fuse_f_r(torch.tanh(self.linear_l_r(l_embedding)))
        aff = F.softmax(torch.cat((w1,w2),1), 1)
        en_embedding = aff[:,0].unsqueeze(1) * s_embedding + aff[:, 1].unsqueeze(1) * l_embedding
        return en_embedding, aff



class GatingMechanism(nn.Module):
    def __init__(self, entity_num, hidden_dim):
        super(GatingMechanism, self).__init__()
        # gating 的参数
        self.gate_theta = nn.Parameter(torch.empty(entity_num, hidden_dim))
        nn.init.xavier_uniform_(self.gate_theta)
        # self.dropout = nn.Dropout(self.params.dropout)

    def forward(self, X: torch.FloatTensor, Y: torch.FloatTensor):
        '''
        :param X:   LSTM 的输出tensor   |E| * H
        :param Y:   Entity 的索引 id    |E|,
        :return:    Gating后的结果      |E| * H
        '''
        gate = torch.sigmoid(self.gate_theta)
        output = torch.mul(gate, X) + torch.mul(-gate + 1, Y)
        return output, gate