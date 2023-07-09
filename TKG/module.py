#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/8/23 11:20
# @Author : ZM7
# @File : module
# @Software: PyCharm
import torch
import torch.nn as nn
from dgl.nn import RelGraphConv, GATConv, GraphConv
from TKG.rgcn.decoder import ConvTransE, ConvTransR
import torch.nn.functional as F
from dgl.sampling import sample_neighbors
from TKG.utils import decoder_sorce
import dgl.function as fn


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, layer_num, low_memory=False, decoder='rgat'):
        super(Decoder, self).__init__()
        self.h_dim = in_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.layer_num = layer_num
        self.decoder = decoder
        if self.decoder == 'rgcn':
            self.decoder_f = GNN(self.h_dim, self.h_dim, layer_num=self.d_layer_num, gnn=self.decoder,
                                  num_rels=self.num_rels, low_memory=low_memory)
        elif self.decoder == 'rgat':
            self.decoder_f = GNN(self.h_dim, self.h_dim, layer_num=self.d_layer_num, gnn=self.decoder)
        elif self.decoder == 'gat':
            self.decoder_f = GAT(self.h_dim, self.h_dim, num_head=2, layer_num=self.d_layer_num)
    def forward(self, graph, feature, edge_type=None):
        if self.decoder == 'rgat':
            return self.decoder_f(graph, feature)
        elif self.decoder == 'rgcn':
            return self.decoder_f(graph, feature,edge_type)




class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, layer_num, low_memory=False, encoder_f='rgat'):
        super(Encoder, self).__init__()
        self.h_dim = in_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.layer_num = layer_num
        self.encoder_f = encoder_f
        if self.encoder == 'rgcn':
            self.aggregator = GNN(self.h_dim, self.h_dim, layer_num=self.a_layer_num, gnn=self.encoder,
                                  num_rels=self.num_rels, low_memory=low_memory)
        elif self.encoder == 'gcn':
            self.aggregator = GNN(self.h_dim, self.h_dim, layer_num=self.a_layer_num, gnn=self.encoder)
        elif self.encoder == 'rgat':
            self.aggregator = GNN(self.h_dim, self.h_dim, layer_num=self.a_layer_num, gnn=self.encoder)

    def forward(self, graph, feature, edge_type=None):
        if self.encoder_f == 'rgcn':
            return self.aggregator(graph, feature, edge_type)
        elif self.encoder_f == 'gcn':
            return  self.aggregator(graph, feature)
        elif self.encoder_f == 'rgat':
            return self.aggregator(graph, feature)


# RGCN net
class RGCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, layer_num, low_memory=False):
        super(RGCN, self).__init__()
        self.h_dim = in_dim
        self.num_rels = num_rels
        self.layer_num = layer_num
        self.layer = nn.ModuleList(RelGraphConv(self.h_dim, self.h_dim, num_rels=self.num_rels, regularizer='basis',
                                                low_mem=low_memory, dropout=0.2, activation=F.relu)
                                   for _ in range(self.layer_num))

    def forward(self, graph, features, etypes):
        for conv in self.layer:
            features = conv(graph, features, etypes)
        return features


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_head, layer_num):
        super(GAT, self).__init__()
        self.h_dim = in_dim
        self.layer_num = layer_num
        self.layer = nn.ModuleList(GATConv(self.h_dim, int(self.h_dim/num_head), num_head, feat_drop=0.2, attn_drop=0.2, activation=F.elu)
                                   for _ in range(self.layer_num))

    def forward(self, graph, features):
        for conv in self.layer:
            features = conv(graph, features).flatten(1)
        return features


class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, layer_num, gnn='rgcn', num_rels=None, num_head=None, low_memory=False):
        super(GNN, self).__init__()
        self.h_dim = in_dim
        self.out_dim = out_dim
        self.layer_num = layer_num
        self.num_rels = num_rels
        self.gnn = gnn
        if self.gnn == 'rgcn':
            self.layer = nn.ModuleList(RelGraphConv(self.h_dim, self.h_dim, num_rels=self.num_rels, regularizer='basis',
                                                    num_bases=100, low_mem=low_memory, dropout=0.5, activation=F.relu)
                                       for _ in range(self.layer_num))
        elif self.gnn == 'gat':
            self.layer = nn.ModuleList(
                GATConv(self.h_dim, int(self.h_dim / num_head), num_head, feat_drop=0.2, attn_drop=0.2,
                        activation=F.elu)
                for _ in range(self.layer_num))
        elif self.gnn == 'gcn':
            self.layer = nn.ModuleList(GraphConv(self.h_dim, self.h_dim, norm='both', activation=F.relu)
                                       for _ in range(self.layer_num))
        elif self.gnn == 'rgat':
            self.layer = nn.ModuleList(RGATLayer(self.h_dim, self.h_dim) for _ in range(self.layer_num))

    def forward(self, graph, features, etypes=None):
        for conv in self.layer:
            if self.gnn == 'rgcn':
                features = conv(graph, features, etypes)
            elif self.gnn == 'gat':
                features = conv(graph, features).flatten(1)
            elif self.gnn == 'gcn':
                features = conv(graph, features)
            elif self.gnn == 'rgat':
                features = conv(graph, features)
        return features


class RGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RGATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)
        self.loop_weight = nn.Parameter(torch.Tensor(out_dim, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['r_h']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1) + torch.mm(nodes.data['z'], self.loop_weight)
        return {'h': h}

    def forward(self, g, h, edge_update=False):
        # equation (1)
        z = self.fc(h)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return F.relu(g.ndata.pop('h'))


class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, encoder_name="", opn="sub", rel_emb=None, use_cuda=False, analysis=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_basis = num_basis
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.skip_connect = skip_connect
        self.self_loop = self_loop
        self.encoder_name = encoder_name
        self.use_cuda = use_cuda
        self.run_analysis = analysis
        self.skip_connect = skip_connect
        print("use layer :{}".format(encoder_name))
        self.rel_emb = rel_emb
        self.opn = opn
        # create rgcn layers
        self.build_model()
        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):

            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        print("h before GCN message passing")
        print(g.ndata['h'])
        print("h behind GCN message passing")
        for layer in self.layers:
            layer(g)
        print(g.ndata['h'])
        return g.ndata.pop('h')



class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            # node_id = g.ndata['id'].squeeze()
            # g.ndata['h'] = init_ent_emb[node_id]
            g.ndata['h'] = init_ent_emb
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')

class UnionRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1,  bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
        super(UnionRGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect
        self.ob = None
        self.sub = None

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
            nn.init.xavier_uniform_(self.skip_connect_weight,gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def forward(self, g, prev_h, emb_rel):
        self.rel_emb = emb_rel
        # self.sub = sub
        # self.ob = ob
        if self.self_loop:
            #loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            # masked_index = torch.masked_select(torch.arange(0, g.number_of_nodes(), dtype=torch.long), (g.in_degrees(range(g.number_of_nodes())) > 0))
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),
                (g.in_degrees(range(g.number_of_nodes())) > 0))
            loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
            loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1

        # calculate the neighbor message with weight_neighbor
        self.propagate(g)
        node_repr = g.ndata['h']

        # print(len(prev_h))
        if len(prev_h) != 0 and self.skip_connect:  # 两次计算loop_message的方式不一样，前者激活后再加权
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr

    def msg_func(self, edges):
        # if reverse:
        #     relation = self.rel_emb.index_select(0, edges.data['type_o']).view(-1, self.out_feat)
        # else:
        #     relation = self.rel_emb.index_select(0, edges.data['type_s']).view(-1, self.out_feat)
        relation = self.rel_emb.index_select(0, edges.data['etype']).view(-1, self.out_feat)
        edge_type = edges.data['etype']
        edge_num = edge_type.shape[0]
        node = edges.src['h'].view(-1, self.out_feat)
        # node = torch.cat([torch.matmul(node[:edge_num // 2, :], self.sub),
        #                  torch.matmul(node[edge_num // 2:, :], self.ob)])
        # node = torch.matmul(node, self.sub)

        # after add inverse edges, we only use message pass when h as tail entity
        # 这里计算的是每个节点发出的消息，节点发出消息时其作为头实体
        # msg = torch.cat((node, relation), dim=1)
        msg = node + relation
        # calculate the neighbor message with weight_neighbor
        msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}