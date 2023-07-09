import pdb
import pickle
from typing import SupportsAbs
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import functools
import torch as th
import dgl.function as fn
from dgl.nn.functional import edge_softmax


class AURGCN(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.code_embedding = nn.Embedding(args.n_codes, args.emb_dim)
        self.relation_embedding = nn.Embedding(args.n_rels, args.emb_dim)
        nn.init.uniform_(self.code_embedding.weight, -1.0, 1.0)
        nn.init.uniform_(self.relation_embedding.weight, -1.0, 1.0)


        self.layers = nn.ModuleList()
        n_rels = args.n_rels
        in_feats = args.emb_dim
        n_hidden = args.emb_dim
        activation = F.relu
        Layer = AURGCNLayer
        for i in range(args.n_layers):
            if i == 0:
                self.layers.append(Layer(in_feats, n_hidden, num_rels=n_rels, activation=activation, self_loop=True, dropout=args.dropout))
            else:
                self.layers.append(Layer(n_hidden, n_hidden, num_rels=n_rels, activation=activation, self_loop=True, dropout=args.dropout))

 
    def forward(self, g, state):

        features = self.code_embedding(g.ndata['id'])
        g.ndata['h'] = features
        rel_emb = self.relation_embedding.weight

        for layer in self.layers:
            layer(g, rel_emb, state)
        
        emb = g.ndata.pop('h')

        g.ndata['emb'] = emb
        x = dgl.readout_nodes(g, 'emb', weight='readout_weight', op='sum')
        # l, b = mask.shape
        # x = x.reshape(b, l, -1).swapaxes(0, 1)
        return x

class AURGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels,  bias=None,
                 activation=None, self_loop=False, dropout=0.0, feat_drop=0.,
                 attn_drop=0., negative_slope=0.2, num_heads=4):
        super(AURGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.fc = nn.Linear(self.in_feat, out_feat * num_heads, bias=False)
        self.num_heads = num_heads
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feat)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feat)))
        self.attn_s = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feat)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()


    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_s, gain=gain)


    def forward(self, g, rel_emb, state):

        h = self.feat_drop(g.ndata['h'])
        feat = self.fc(h).view(-1, self.num_heads, self.out_feat)
        g.ndata['ft'] = feat
        g.edata['a'] = self.cal_attention(g, feat, state)

        self.rel_emb = rel_emb
      
        if self.self_loop:
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),
                (g.in_degrees(range(g.number_of_nodes())) > 0))
            loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
            loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]

        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'))
        node_repr = g.ndata['h']

        if self.self_loop:
            node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr

    def cal_attention(self, graph, feat, state):
        with graph.local_scope():

            # linear transformation
            state = self.fc(state).view(-1, self.num_heads, self.out_feat)
           
            # FNN
            el = (feat * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
            es = (state * self.attn_s).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat, 'el': el, 'es': es})
            graph.dstdata.update({'er': er})

            # compute edge attention
            graph.apply_edges(fn.u_add_v('el', 'er', 'elr')) # elr = el + er
            graph.apply_edges(fn.u_add_e('es', 'elr', 'e')) # e = el + er + es
            e = self.leaky_relu(graph.edata.pop('e'))

            # compute softmax
            # graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            return self.attn_drop(edge_softmax(graph, e))

    def msg_func(self, edges):
        relation = self.rel_emb.index_select(0, edges.data['rel_type']).view(-1, self.out_feat)
        feat = edges.src['ft'] # (E, num_heads, d)
        node = (edges.data['a'] * feat).sum(dim=1)
        # msg = node + relation
        msg = node + torch.mm(relation, self.weight_neighbor)
        return {'msg': msg}